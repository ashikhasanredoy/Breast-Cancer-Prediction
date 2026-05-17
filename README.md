# breast cancer prediction.

A highly optimized end-to-end Machine Learning pipeline and interactive inference system. Powered by a Support Vector Classifier (SVC), the system achieves an outstanding **99.12% test accuracy score** on Stratified Breast Cancer Wisconsin (Diagnostic) features.

---

## 🌟 Key Highlights
* **99.12% Test Accuracy**: Validated model performance using stratified train-test splits.
* **Modular Pipeline Architecture**: Separates Ingestion, Transformation, Training, and Prediction steps.
* **Interactive Day Mode Web UI**: Responsive light theme dashboard with preloaded malignant/benign Presets, real-time AJAX predictions, and an interactive dataset information modal overlay.
* **Dual Serializer Core**: Robust model loading utilizing `dill` with transparent fallback to standard `pickle` formats.

---

## 📊 Dataset Specifications
The system evaluates **26 preprocessed numerical attributes** mapping morphological cell nuclei characteristics extracted from digitized fine needle aspirates (FNA).

### Input Features Map (26 Attributes)
* **Mean Features**: `texture_mean`, `area_mean`, `smoothness_mean`, `compactness_mean`, `concavity_mean`, `concave points_mean`, `symmetry_mean`, `fractal_dimension_mean`
* **Standard Error (SE)**: `texture_se`, `perimeter_se`, `area_se`, `smoothness_se`, `compactness_se`, `concavity_se`, `concave points_se`, `symmetry_se`, `fractal_dimension_se`
* **Worst (Extreme) Features**: `texture_worst`, `perimeter_worst`, `area_worst`, `smoothness_worst`, `compactness_worst`, `concavity_worst`, `concave points_worst`, `symmetry_worst`, `fractal_dimension_worst`

### Output Variable
* **`diagnosis`**: target outcome mapped to:
  * `1` (Malignant / High Risk)
  * `0` (Benign / Low Risk)

---

## ⚙️ Setup & Installations

### 1. Initialize Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Execute End-to-End Training & Serializing
Run the data ingestion module to trigger training, evaluate estimators, and serialize modern artifacts to `/artifact`:
```bash
python -m src.components.data_ingestion
```

### Start Web Inference Server
To spin up the web interface server on port `5001`:
```bash
python app.py
```
👉 Once active, visit: **[http://localhost:5001/](http://localhost:5001/)**

---

## 📂 Directory Blueprint
```
├── app.py                      # Flask API Gateway & Server
├── templates/
│   └── index.html              # Day Mode Glassmorphism Dashboard Page
├── static/
│   └── css/
│       └── style.css           # Premium Day Mode styling
├── artifact/
│   ├── model.pkl               # Trained SVM Classifier Estimator
│   └── preprocessor.pkl        # Data scaler column transformer
└── src/
    ├── pipeline/
    │   └── predict_pipeline.py # Data shape mapper & scoring executor
    ├── components/
    │   ├── data_ingestion.py   # Dataset load & train-test split module
    │   ├── data_transformation.py # Scaling & robust Label Mapping step
    │   └── model_train.py      # RandomizedSearchCV model evaluator
    ├── utils.py                # Dual Pickle/Dill Fallback serializer
    └── exeception.py           # Custom exception handler
```
