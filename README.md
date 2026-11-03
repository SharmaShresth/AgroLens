@'

\# AgroLens: Offline Plant Leaf Disease Detection



AgroLens classifies plant leaf diseases from a photo using MobileNetV2 transfer learning. The trained models run in a browser-based web app, and the tomato model is quantized to INT8 for an ESP32-S3 microcontroller (TensorFlow Lite Micro) so it can work fully offline in farming regions with no internet.



B.Tech Major Project-I, KIIT Deemed to be University (School of Electronics Engineering), supervised by Prof. Sumit Dass.



\## Results



| Crop | Classes | Test accuracy | Macro F1 |

|------|---------|---------------|----------|

| Tomato (baseline) | 10 | 83.28 % | 0.78 |

| Rice | 4 | 99.10 % | 0.99 |



Radish and potato models are also included in the web app. See the report in `docs/` for details and limitations.



\## Project structure



\- `notebooks/`: training notebooks (tomato, rice, radish, potato, guava)

\- `webapp/`: Flask server (`app.py`), web page (`index.html`) and trained models (`saved\_models/`)

\- `firmware/`: ESP-IDF project for the ESP32-S3 (TensorFlow Lite Micro)

\- `scripts/`: `send\_image.py`, which sends a leaf image to the board over serial

\- `docs/`: project report



\## Run the web app



pip install flask flask-cors tensorflow pillow numpy

cd webapp

python app.py





Then open http://localhost:5000, choose a plant, upload a leaf image and click Analyze.



How it works:



Browser (index.html) -> Flask server (app.py) -> Keras model (.keras)

choose plant resize + scale image class probabilities

upload leaf image returned as JSON





\## Datasets



Public datasets, not included in this repository: PlantVillage (tomato, potato), a rice leaf disease dataset and a radish leaf disease dataset, all from Kaggle. Check each dataset page for its licence.



\## Team



\- Asmit Lahiri (2330224): hardware

\- Rishika Ghosh (2330394): frontend and documentation

\- Shresth Sharma (2330402): ML pipeline, backend and server

\- Sumita Saha (2330413): dataset collection and splitting



\## Note



AgroLens gives model estimates, not a diagnosis. Consult an agronomist for treatment decisions.



\## License



MIT. See `LICENSE`.

'@ | Set-Content README.md -Encoding UTF8

