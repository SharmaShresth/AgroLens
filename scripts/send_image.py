"""Send a leaf image to the ESP32-S3 and print the prediction.
Usage:  python send_image.py leaf.jpg COM5
Needs:  pip install pyserial pillow numpy
"""
import sys
import numpy as np
import serial
from PIL import Image

img_path = sys.argv[1]
port = sys.argv[2] if len(sys.argv) > 2 else "COM5"

# BILINEAR matches Keras image_dataset_from_directory's default resize
img = Image.open(img_path).convert("RGB").resize((96, 96), Image.BILINEAR)
payload = b"\xAA\x55" + np.asarray(img, dtype=np.uint8).tobytes()   # 2 + 27648 bytes

ser = serial.Serial(port, 921600, timeout=15)
ser.reset_input_buffer()
ser.write(payload)

while True:
    line = ser.readline().decode(errors="ignore").strip()
    if not line:
        print("Timed out waiting for a reply from the ESP32.")
        break
    if line.startswith(("RESULT", "ERROR")):
        print(line)
        break
