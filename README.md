# Breast Cancer Prediction

This project aims to predict breast cancer using machine learning models based on the Breast Cancer Wisconsin (Diagnostic) Dataset.

## Dataset Overview

The dataset used in this project is the **Breast Cancer Wisconsin (Diagnostic) Dataset**, available on [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data). It contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.

### Attribute Information:
1. **ID number**
2. **Diagnosis**: (M = malignant, B = benign) - This is our target variable.
   - **Distribution**: 357 Benign (B), 212 Malignant (M).

Ten real-valued features are computed for each cell nucleus:
- **Radius**: Mean of distances from center to points on the perimeter.
- **Texture**: Standard deviation of gray-scale values.
- **Perimeter**
- **Area**
- **Smoothness**: Local variation in radius lengths.
- **Compactness**: `perimeter^2 / area - 1.0`
- **Concavity**: Severity of concave portions of the contour.
- **Concave points**: Number of concave portions of the contour.
- **Symmetry**
- **Fractal dimension**: "Coastline approximation" - 1.

The mean, standard error, and "worst" (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, and field 23 is Worst Radius.

## Setup Instructions

1. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   ```

2. **Activate the virtual environment:**
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Dependencies

- Flask
- Numpy
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib (Visualization)
- Seaborn (Visualization)
- Statsmodels (Statistical Analysis)
- Imbalanced-learn (Handling imbalanced data)
- Ipykernel (Notebook support)
- Openpyxl (Excel support)
