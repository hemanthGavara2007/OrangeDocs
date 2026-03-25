# 🍊 OrangeDocs - Privacy First Intelligent Document Auditing 🏆
**Consolation Prize Winner (5K) at the JNTUK Artemis Hackathon!**

OrangeDocs is a local, self-hosted solution to digitize handwritten documents and perform Machine Learning-based intelligent auditing of the documents without sending any sensitive information to the cloud.

## 👥 Contributors
  Hemanth Gavara
  Snehal GSS
  Rashmi P

## 🛠️ Tech Stack
* **Backend:** Python (Flask, Flask-CORS)
* **AI & Machine Learning:** Local `glm-ocr` via Ollama, NumPy, Statistical IQR Anomaly math.
* **Frontend:** HTML5 (ARTEMIS Dashboard), Chart.js for visualization.
* **Outputs:** Multi-page PDF Generator using ReportLab.

## 🚀 How to Run Locally
1. Clone the repo
2. Run `pip install flask flask-cors ollama pillow reportlab pymupdf numpy`
3. Pull the weights of the local AI model `ollama pull glm-ocr`
4. Run `python app.py`
