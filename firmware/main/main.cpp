// AgroLens - ESP32-S3 plant disease inference over USB serial
// Protocol: laptop sends 0xAA 0x55 + 96*96*3 raw RGB bytes (0-255).
//           ESP replies with one line: "RESULT <class> <confidence>\n"
#include <stdio.h>
#include <math.h>
#include "driver/uart.h"
#include "esp_heap_caps.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "model_data.h"

constexpr int kArenaSize = 1024 * 1024;
constexpr int kImgW = 96, kImgH = 96;
constexpr int kImgBytes = kImgW * kImgH * 3;
constexpr int kNumClasses = 10;

static const char* kClasses[kNumClasses] = {
    "Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold",
    "Septoria_leaf_spot", "Spider_mites", "Target_Spot",
    "Yellow_Leaf_Curl_Virus", "Mosaic_virus", "healthy"};

static void read_exact(uint8_t* dst, int n) {
  int got = 0;
  while (got < n) {
    int r = uart_read_bytes(UART_NUM_0, dst + got, n - got, portMAX_DELAY);
    if (r > 0) got += r;
  }
}

extern "C" void app_main() {
  uart_driver_install(UART_NUM_0, 4096, 0, 0, NULL, 0);

  uint8_t* arena = (uint8_t*)heap_caps_malloc(kArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  uint8_t* buf = (uint8_t*)heap_caps_malloc(kImgBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!arena || !buf) {
    printf("ERROR PSRAM allocation failed (check SPIRAM octal mode in sdkconfig)\n");
    return;
  }

  const tflite::Model* model = tflite::GetModel(agrolens_tflite);

  static tflite::MicroMutableOpResolver<14> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddAdd();
  resolver.AddMul();
  resolver.AddPad();
  resolver.AddMean();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddAveragePool2D();
  resolver.AddRelu6();
  resolver.AddRelu();

  static tflite::MicroInterpreter interp(model, resolver, arena, kArenaSize);
  if (interp.AllocateTensors() != kTfLiteOk) {
    printf("ERROR AllocateTensors failed (arena too small or unsupported op)\n");
    return;
  }
  TfLiteTensor* in = interp.input(0);
  TfLiteTensor* out = interp.output(0);
  printf("READY arena_used=%u bytes in_scale=%f in_zp=%d\n",
         (unsigned)interp.arena_used_bytes(), in->params.scale, (int)in->params.zero_point);

  while (true) {
    uint8_t prev = 0, cur = 0;
    do { prev = cur; read_exact(&cur, 1); } while (!(prev == 0xAA && cur == 0x55));
    read_exact(buf, kImgBytes);

    const float scale = in->params.scale;
    const int zp = in->params.zero_point;
    for (int i = 0; i < kImgBytes; i++) {
      int q = (int)roundf(buf[i] / scale) + zp;
      in->data.int8[i] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
    }

    if (interp.Invoke() != kTfLiteOk) {
      printf("ERROR Invoke failed\n");
      continue;
    }

    int best = 0;
    for (int i = 1; i < kNumClasses; i++)
      if (out->data.int8[i] > out->data.int8[best]) best = i;
    float conf = (out->data.int8[best] - out->params.zero_point) * out->params.scale;
    printf("RESULT %s %.3f\n", kClasses[best], conf);
  }
}
