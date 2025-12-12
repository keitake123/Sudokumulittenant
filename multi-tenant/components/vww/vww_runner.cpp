#include "vww/vww.h"

#include <cstring>
#include <cmath>

#include "esp_camera.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define CAMERA_MODEL_XIAO_ESP32S3
#include "vww/camera_pins.h"

#include "vww/NeuralNetwork.h"

static const char* TAG = "vww_sudoku";

// ================= CONFIG =================
#define GRID_SIZE 4

// 🔒 FIXED BOARD CROP (DO NOT CHANGE)
#define X_LEFT   77
#define X_RIGHT  253
#define Y_TOP    57
#define Y_BOTTOM 233

#define CROP_W (X_RIGHT - X_LEFT)
#define CROP_H (Y_BOTTOM - Y_TOP)

// ================= GLOBAL BUFFERS =================
static uint8_t cropped_gray[CROP_W * CROP_H];
static uint8_t cell_raw[64 * 64];
static uint8_t cell28[28 * 28];

static int preds[GRID_SIZE][GRID_SIZE];

static NeuralNetwork* s_nn = nullptr;
static bool s_camera_ok = false;

// ------------------------------------------------------------
// PREPROCESS ONE CELL: contrast + resize to 28x28
// (PASTED FROM YOUR FRIEND — DO NOT CHANGE)
// ------------------------------------------------------------
static void preprocess_cell_to_28x28(
    const uint8_t *cell_in,
    int cell_w,
    int cell_h,
    uint8_t *cell_out
) {
    uint8_t minv = 255;
    uint8_t maxv = 0;

    for (int i = 0; i < cell_w * cell_h; i++) {
        uint8_t v = cell_in[i];
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
    }

    int range = maxv - minv;
    if (range < 30) {
        memset(cell_out, 0, 28 * 28);
        return;
    }

    for (int oy = 0; oy < 28; oy++) {
        for (int ox = 0; ox < 28; ox++) {

            int x0 = (ox * cell_w) / 28;
            int x1 = ((ox + 1) * cell_w) / 28;
            int y0 = (oy * cell_h) / 28;
            int y1 = ((oy + 1) * cell_h) / 28;

            if (x1 <= x0) x1 = x0 + 1;
            if (y1 <= y0) y1 = y0 + 1;

            int sum = 0;
            int count = 0;

            for (int y = y0; y < y1; y++) {
                for (int x = x0; x < x1; x++) {
                    int v = cell_in[y * cell_w + x];
                    int c = (v - minv) * 255 / range;
                    if (c < 0) c = 0;
                    if (c > 255) c = 255;
                    sum += c;
                    count++;
                }
            }

            uint8_t out = sum / count;
            cell_out[oy * 28 + ox] = out;
        }
    }
}

// ------------------------------------------------------------
// Flip vertically (PASTED FROM YOUR FRIEND — DO NOT CHANGE)
// ------------------------------------------------------------
static void flip_vertical(uint8_t *img, int w, int h) {
    for (int y = 0; y < h / 2; y++) {
        for (int x = 0; x < w; x++) {
            int top = y * w + x;
            int bottom = (h - 1 - y) * w + x;

            uint8_t tmp = img[top];
            img[top] = img[bottom];
            img[bottom] = tmp;
        }
    }
}

// ------------------------------------------------------------
// UINT8 → INT8 normalization (PASTED FROM YOUR FRIEND — DO NOT CHANGE)
// ------------------------------------------------------------
static void normalize_uint8_to_int8(
    const uint8_t *in,
    int8_t *out,
    int n,
    int zero_point,
    float scale
) {
    for (int i = 0; i < n; i++) {
        // Match training preprocessing EXACTLY
        float f = in[i] / 255.0f;

        int q = (int)round(f / scale) + zero_point;

        if (q < -128) q = -128;
        if (q > 127)  q = 127;

        out[i] = (int8_t)q;
    }
}

// ------------------------------------------------------------
// EXTRACT ONE CELL FROM CROPPED BOARD (PASTED FROM YOUR FRIEND — DO NOT CHANGE)
// ------------------------------------------------------------
static void extract_cell(
    const uint8_t *board,
    int board_w,
    int board_h,
    int row,
    int col,
    uint8_t *cell_out,
    int &cell_w,
    int &cell_h
) {
    cell_w = board_w / GRID_SIZE;
    cell_h = board_h / GRID_SIZE;

    int x0 = col * cell_w;
    int y0 = row * cell_h;

    int idx = 0;
    for (int y = 0; y < cell_h; y++) {
        for (int x = 0; x < cell_w; x++) {
            cell_out[idx++] = board[(y0 + y) * board_w + (x0 + x)];
        }
    }
}

// ---------------------- IDF Camera config ----------------------
static camera_config_t MakeCameraConfig() {
    camera_config_t c = {};
    c.ledc_channel = LEDC_CHANNEL_0;
    c.ledc_timer   = LEDC_TIMER_0;

    c.pin_d0 = Y2_GPIO_NUM;
    c.pin_d1 = Y3_GPIO_NUM;
    c.pin_d2 = Y4_GPIO_NUM;
    c.pin_d3 = Y5_GPIO_NUM;
    c.pin_d4 = Y6_GPIO_NUM;
    c.pin_d5 = Y7_GPIO_NUM;
    c.pin_d6 = Y8_GPIO_NUM;
    c.pin_d7 = Y9_GPIO_NUM;

    c.pin_xclk = XCLK_GPIO_NUM;
    c.pin_pclk = PCLK_GPIO_NUM;
    c.pin_vsync = VSYNC_GPIO_NUM;
    c.pin_href  = HREF_GPIO_NUM;
    c.pin_sccb_sda = SIOD_GPIO_NUM;
    c.pin_sccb_scl = SIOC_GPIO_NUM;
    c.pin_pwdn  = PWDN_GPIO_NUM;
    c.pin_reset = RESET_GPIO_NUM;

    c.xclk_freq_hz = 20000000;
    c.pixel_format = PIXFORMAT_GRAYSCALE;
    c.frame_size   = FRAMESIZE_QVGA;
    c.fb_count     = 1;

    // optional but fine:
    c.grab_mode   = CAMERA_GRAB_WHEN_EMPTY;
    c.fb_location = CAMERA_FB_IN_PSRAM;

    return c;
}

static void RunSudokuOnce() {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        ESP_LOGE(TAG, "Frame capture failed");
        return;
    }

    // sanity: expect GRAYSCALE
    if (fb->format != PIXFORMAT_GRAYSCALE) {
        ESP_LOGE(TAG, "Unexpected format %d (expected GRAYSCALE)", fb->format);
        esp_camera_fb_return(fb);
        return;
    }

    uint8_t* img = fb->buf;

    // Crop board (EXACT)
    for (int y = 0; y < CROP_H; y++) {
        for (int x = 0; x < CROP_W; x++) {
            int sx = X_LEFT + x;
            int sy = Y_TOP  + y;
            cropped_gray[y * CROP_W + x] = img[sy * fb->width + sx];
        }
    }

    ESP_LOGI(TAG, "=== SUDOKU PREDICTION ===");

    for (int r = 0; r < GRID_SIZE; r++) {
        for (int c = 0; c < GRID_SIZE; c++) {

            int cell_w, cell_h;
            extract_cell(cropped_gray, CROP_W, CROP_H, r, c, cell_raw, cell_w, cell_h);

            preprocess_cell_to_28x28(cell_raw, cell_w, cell_h, cell28);
            flip_vertical(cell28, 28, 28);

            // Debug only for first cell
            if (r == 0 && c == 0) {
                uint8_t mn = 255, mx = 0;
                for (int i = 0; i < 28 * 28; i++) {
                    uint8_t v = cell28[i];
                    if (v < mn) mn = v;
                    if (v > mx) mx = v;
                }
                ESP_LOGI(TAG, "CELL28 after preprocess+flip+invert: min=%u max=%u range=%u",
                         mn, mx, (unsigned)(mx - mn));
            }

            TfLiteTensor* input = s_nn->getInput();

            normalize_uint8_to_int8(
                cell28,
                input->data.int8,
                28 * 28,
                input->params.zero_point,
                input->params.scale
            );

            if (r == 0 && c == 0) {
                int8_t mn = 127, mx = -128;
                for (int i = 0; i < 28 * 28; i++) {
                    int8_t v = input->data.int8[i];
                    if (v < mn) mn = v;
                    if (v > mx) mx = v;
                }
                ESP_LOGI(TAG, "INPUT int8: min=%d max=%d (scale=%f zp=%d)",
                         mn, mx, input->params.scale, input->params.zero_point);
            }

            if (s_nn->predict() != kTfLiteOk) {
                ESP_LOGE(TAG, "Inference failed");
                esp_camera_fb_return(fb);
                return;
            }

            int pred = s_nn->getPredictedClass();
            preds[GRID_SIZE - 1 - r][c] = pred;
        }
    }

    ESP_LOGI(TAG, "=== FINAL PREDICTION GRID ===");
    for (int r = 0; r < GRID_SIZE; r++) {
        char line[32];
        int pos = 0;
        for (int c = 0; c < GRID_SIZE; c++) {
            int v = preds[r][c];
            pos += snprintf(line + pos, sizeof(line) - pos, "%c ",
                            (v == 0) ? '_' : ('0' + v));
        }
        ESP_LOGI(TAG, "%s", line);
    }

    esp_camera_fb_return(fb);
}

extern "C" esp_err_t vww_init(void) {
    if (s_nn) return ESP_OK;

    camera_config_t cfg = MakeCameraConfig();
    esp_err_t err = esp_camera_init(&cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed: 0x%x", err);
        s_camera_ok = false;
        return err;
    }
    s_camera_ok = true;

    s_nn = new NeuralNetwork();
    if (!s_nn || !s_nn->getInput() || !s_nn->getOutput()) {
        ESP_LOGE(TAG, "NN init failed");
        delete s_nn;
        s_nn = nullptr;
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "Camera + model initialized");
    return ESP_OK;
}

extern "C" void vww_task(void* param) {
    (void)param;

    bool ran_once = false;

    while (true) {
        if (s_camera_ok && s_nn && !ran_once) {
            vTaskDelay(pdMS_TO_TICKS(5000));
            RunSudokuOnce();
            ran_once = true;
        }
        vTaskDelay(pdMS_TO_TICKS(200));
    }
}
