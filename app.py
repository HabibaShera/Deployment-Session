from flask import Flask, request, jsonify
import joblib
import pandas as pd

from flask_core import CORS

with open('gbrt_pipeline.pkl', 'rb') as f:
    GBR_pipeline = joblib.load(f)


# Create a Flask app
app = Flask(__name__)
CORS(app, resources={r"/predict":{"origins":"http://127.0.0.1:5500"}})

@app.route('/')
def home():
    return "Now Run Successfully......"


# Define an API endpoint for image classification
@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_data = request.get_json()
        data = pd.DataFrame(json_data)
            
        new_data_transformed = GBR_pipeline.named_steps['preprocessor'].transform(data)
            
        prediction = GBR_pipeline.named_steps['regressor'].predict(new_data_transformed)

        return jsonify({'Prediction': prediction.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)})
    


if __name__ == '__main__':
    app.run()
