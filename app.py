from flask import Flask, request, jsonify, render_template
import sys
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.exeception import CustomBreasctCancerException

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        custom_data = CustomData(
            texture_mean=data.get('texture_mean'),
            area_mean=data.get('area_mean'),
            smoothness_mean=data.get('smoothness_mean'),
            compactness_mean=data.get('compactness_mean'),
            concavity_mean=data.get('concavity_mean'),
            concave_points_mean=data.get('concave points_mean'),
            symmetry_mean=data.get('symmetry_mean'),
            fractal_dimension_mean=data.get('fractal_dimension_mean'),
            texture_se=data.get('texture_se'),
            perimeter_se=data.get('perimeter_se'),
            area_se=data.get('area_se'),
            smoothness_se=data.get('smoothness_se'),
            compactness_se=data.get('compactness_se'),
            concavity_se=data.get('concavity_se'),
            concave_points_se=data.get('concave points_se'),
            symmetry_se=data.get('symmetry_se'),
            fractal_dimension_se=data.get('fractal_dimension_se'),
            texture_worst=data.get('texture_worst'),
            perimeter_worst=data.get('perimeter_worst'),
            area_worst=data.get('area_worst'),
            smoothness_worst=data.get('smoothness_worst'),
            compactness_worst=data.get('compactness_worst'),
            concavity_worst=data.get('concavity_worst'),
            concave_points_worst=data.get('concave points_worst'),
            symmetry_worst=data.get('symmetry_worst'),
            fractal_dimension_worst=data.get('fractal_dimension_worst')
        )
        
        features_df = custom_data.get_data_as_data_frame()
        
        predict_pipeline = PredictPipeline()
        preds = predict_pipeline.predict(features_df)
        prediction = int(preds[0])
        
        return jsonify({
            'prediction': prediction,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
