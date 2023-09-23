import joblib
import pandas as pd


# Loading our pipeline
try:
    with open('gbrt_pipeline.pkl', 'rb') as f:
        GBR_pipeline = joblib.load(f)
    
except Exception as err:
    print(f"Unexpected {err=}, {type(err)=}")

new_data = pd.DataFrame({
    'Brand': ['Hyundai', 'Fiat'],
    'Model': ['Accent', 'Punto'],
    'Body': ['Sedan', 'Hatchback'],
    'Color': ['Blue', 'Red'],
    'Fuel': ['Gasoline', 'Diesel'],
    'Kilometers': ['40000 to 159999', '80000 to 199999'],
    'Engine': ['1.6L', '1.3L'],
    'Transmission': ['Automatic', 'Manual'],
    'Gov': ['Yes', 'No'],
    'Year': [2018, 2017],  
})

# Transform the new data using the preprocessor
new_data_transformed = GBR_pipeline.named_steps['preprocessor'].transform(new_data)

# Use the trained regressor to make predictions
predictions = GBR_pipeline.named_steps['regressor'].predict(new_data_transformed)


# Print the predicted prices
print(f"Predicted Prices: {predictions}")
