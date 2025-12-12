#include "vww/vww.h"

#include <cstdint>
#include <cstring>

#include "driver/gpio.h"
#include "driver/ledc.h"
#include "esp_camera.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "soc/rtc_cntl_reg.h"
#include "soc/soc.h"

#include "vww/NeuralNetwork.h"
#define CAMERA_MODEL_XIAO_ESP32S3
#include "vww/camera_pins.h"
#include "tensorflow/lite/c/common.h"

static bool s_camera_ok = false;

namespace {
constexpr int kInputWidth = 96;
constexpr int kInputHeight = 96;
constexpr gpio_num_t kLedPin = GPIO_NUM_21;

NeuralNetwork *s_network = nullptr;
const char *TAG = "vww_runner";

uint32_t Rgb565ToRgb888(uint16_t color) {
  const uint8_t lb = (color >> 8) & 0xFF;
  const uint8_t hb = color & 0xFF;

  const uint32_t r = (lb & 0x1F) << 3;
  const uint32_t g = ((hb & 0x07) << 5) | ((lb & 0xE0) >> 3);
  const uint32_t b = hb & 0xF8;

  return (r << 16) | (g << 8) | b;
}

esp_err_t FillInputTensor(camera_fb_t *fb, TfLiteTensor *input) {
  if (!fb || !input) return ESP_ERR_INVALID_ARG;

  if (fb->format != PIXFORMAT_RGB565) {
    ESP_LOGE(TAG, "Unexpected fb format %d", fb->format);
    return ESP_ERR_INVALID_STATE;
  }

  if (!input->dims || input->dims->size != 4) {
    ESP_LOGE(TAG, "Unexpected input dims");
    return ESP_ERR_INVALID_STATE;
  }

  const int inN = input->dims->data[0];
  const int inH = input->dims->data[1];
  const int inW = input->dims->data[2];
  const int inC = input->dims->data[3];

  if (inN != 1) {
    ESP_LOGE(TAG, "Unexpected batch %d", inN);
    return ESP_ERR_INVALID_STATE;
  }

  // Your digit model should be 28x28x1
  const bool want_gray28 = (inH == 28 && inW == 28 && inC == 1);

  const uint16_t *src = reinterpret_cast<const uint16_t *>(fb->buf);

  auto rgb565_to_rgb888 = [](uint16_t c, uint8_t &r, uint8_t &g, uint8_t &b) {
    r = ((c >> 11) & 0x1F) << 3;
    g = ((c >> 5)  & 0x3F) << 2;
    b = ( c        & 0x1F) << 3;
  };

  auto rgb_to_gray_u8 = [](uint8_t r, uint8_t g, uint8_t b) -> uint8_t {
    // cheap luma
    return (uint8_t)((r * 30 + g * 59 + b * 11) / 100);
  };

  // Helper: safe quantize a 0..255 value using tensor params
  auto quant_u8 = [&](int x255) -> int {
    const float scale = input->params.scale;
    const int zp = input->params.zero_point;

    if (scale <= 0.0f) {
      // If scale is broken, fall back to raw centered mapping (best-effort)
      // This is NOT ideal, but prevents divide-by-zero.
      return x255 - 128;
    }

    int q = (int)lroundf((float)x255 / scale) + zp;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    return q;
  };

  if (want_gray28) {
    // Downsample fb->width/height to 28x28, grayscale, fill tensor
    if (input->type == kTfLiteInt8) {
      int8_t *dst = input->data.int8;
      for (int y = 0; y < 28; y++) {
        int sy = (y * fb->height) / 28;
        for (int x = 0; x < 28; x++) {
          int sx = (x * fb->width) / 28;
          uint8_t r,g,b;
          rgb565_to_rgb888(src[sy * fb->width + sx], r,g,b);
          uint8_t gray = rgb_to_gray_u8(r,g,b);
          *dst++ = (int8_t)quant_u8(gray);
        }
      }
      return ESP_OK;
    }

    if (input->type == kTfLiteUInt8) {
      uint8_t *dst = input->data.uint8;
      // For uint8 models, quantization is typically identity-ish, but still use params if you want.
      for (int y = 0; y < 28; y++) {
        int sy = (y * fb->height) / 28;
        for (int x = 0; x < 28; x++) {
          int sx = (x * fb->width) / 28;
          uint8_t r,g,b;
          rgb565_to_rgb888(src[sy * fb->width + sx], r,g,b);
          uint8_t gray = rgb_to_gray_u8(r,g,b);
          *dst++ = gray;
        }
      }
      return ESP_OK;
    }

    if (input->type == kTfLiteFloat32) {
      float *dst = input->data.f;
      for (int y = 0; y < 28; y++) {
        int sy = (y * fb->height) / 28;
        for (int x = 0; x < 28; x++) {
          int sx = (x * fb->width) / 28;
          uint8_t r,g,b;
          rgb565_to_rgb888(src[sy * fb->width + sx], r,g,b);
          uint8_t gray = rgb_to_gray_u8(r,g,b);
          *dst++ = gray / 255.0f; // typical float model normalization
        }
      }
      return ESP_OK;
    }

    ESP_LOGE(TAG, "Unsupported tensor type %d for 28x28x1", input->type);
    return ESP_ERR_NOT_SUPPORTED;
  }

  // If you ever swap back to a 96x96x3 VWW model, you can keep your old path,
  // but DO NOT assume fb->len == kInputWidth*kInputHeight*2; use fb->width/height.
  ESP_LOGE(TAG, "Unexpected input shape [%d,%d,%d,%d] (expected 28x28x1)",
           inN, inH, inW, inC);
  return ESP_ERR_INVALID_STATE;
}



camera_config_t CreateCameraConfig() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;

  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;

  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  config.xclk_freq_hz = 16000000;              // ✅ lower clock
  config.frame_size = FRAMESIZE_96X96;
  config.pixel_format = PIXFORMAT_RGB565;

  config.grab_mode = CAMERA_GRAB_LATEST;       // ✅ safer
  config.fb_location = CAMERA_FB_IN_PSRAM;      // ✅ critical
  config.jpeg_quality = 12;
  config.fb_count = 1;

  return config;
}


void ConfigureLed() {
  gpio_config_t cfg = {};
  cfg.mode = GPIO_MODE_OUTPUT;
  cfg.pin_bit_mask = 1ULL << kLedPin;
  gpio_config(&cfg);
  gpio_set_level(kLedPin, 1);
}

void ProcessFrame() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    ESP_LOGW(TAG, "Camera capture failed");
    return;
  }

  const uint64_t start_prep = esp_timer_get_time();
  if (FillInputTensor(fb, s_network->getInput()) != ESP_OK) {
    esp_camera_fb_return(fb);
    return;
  }
  const uint64_t dur_prep = esp_timer_get_time() - start_prep;

  const uint64_t start_infer = esp_timer_get_time();
  if (s_network->predict() != kTfLiteOk) {
    ESP_LOGE(TAG, "Inference failed");
    esp_camera_fb_return(fb);
    return;
  }
  const uint64_t dur_infer = esp_timer_get_time() - start_infer;

  esp_camera_fb_return(fb);

  const float prob = s_network->getOutput()->data.f[0];
  ESP_LOGI(TAG, "Prep: %llums, Infer: %llums, prob=%.3f",
           dur_prep / 1000ULL, dur_infer / 1000ULL, prob);

  const bool person_detected = prob >= 0.5f;
  gpio_set_level(kLedPin, person_detected ? 0 : 1);
}
}  // namespace

extern "C" esp_err_t vww_init(void) {
  if (s_network != nullptr) {
    ESP_LOGI(TAG, "vww_init: network already created");
    return ESP_OK;
  }

  ESP_LOGI(TAG, "vww_init: start");

  // leave brownout hack OFF while debugging
  // WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

  ESP_LOGI(TAG, "vww_init: configuring LED");
  ConfigureLed();
  ESP_LOGI(TAG, "vww_init: LED configured");

  // ---- ENABLE CAMERA INIT ----
  camera_config_t config = CreateCameraConfig();
  ESP_LOGI(TAG, "vww_init: calling esp_camera_init...");
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "vww_init: Camera init failed: 0x%x", err);
    s_camera_ok = false;
    return err;
  }
  ESP_LOGI(TAG, "vww_init: Camera init OK");
  s_camera_ok = true;
  // ----------------------------

  ESP_LOGI(TAG, "vww_init: creating NeuralNetwork");
  s_network = new NeuralNetwork();
  if (!s_network) {
    ESP_LOGE(TAG, "vww_init: Failed to allocate NeuralNetwork");
    s_camera_ok = false;
    return ESP_ERR_NO_MEM;
  }

  TfLiteTensor* in = s_network->getInput();
  TfLiteTensor* out = s_network->getOutput();
  if (!in || !out) {
    ESP_LOGE(TAG, "vww_init: Neural network tensors unavailable (in=%p out=%p)", in, out);
    delete s_network;
    s_network = nullptr;
    s_camera_ok = false;
    return ESP_FAIL;
  }

  ESP_LOGI(TAG, "vww_init: done (camera ok, network ok)");
  return ESP_OK;
}


extern "C" void vww_task(void *param) {
  (void)param;
  ESP_LOGI(TAG, "VWW task started");

  while (true) {
    if (!s_camera_ok || !s_network) {
      ESP_LOGW(TAG, "VWW loop: not ready (camera_ok=%d network=%p)",
               (int)s_camera_ok, s_network);
      vTaskDelay(pdMS_TO_TICKS(500));
      continue;
    }

    ProcessFrame();                 // ✅ runs capture + ML + output
    vTaskDelay(pdMS_TO_TICKS(50));  // adjust speed
  }
}


