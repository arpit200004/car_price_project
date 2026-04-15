import pickle
import pandas as pd
import numpy as np

# Load model bundle
with open('car_price_model_v3.pkl', 'rb') as f:
    bundle = pickle.load(f)

model = bundle['model']

# 2023 Maruti Suzuki Ertiga ZXi Plus Petrol (Manual)
# Actual Price in Delhi (2026 check on 2023 model): Rs. 9.8 Lakh
row = pd.DataFrame([{
    "brand": "maruti",
    "model": "maruti ertiga",
    "car_name": "maruti ertiga zxi plus", # matches cleaned variant logic
    "seller_type": "dealer",
    "fuel_type": "petrol",
    "transmission_type": "manual",
    "owner_type": "first",
    "body_type": "muv",
    "drive_type": "2wd",
    "car_age": float(2026 - 2023), # Age 3 in 2026
    "km_driven": 43000.0,
    "km_per_year": 43000.0 / 3.0,
    "mileage": 20.5, # approx
    "engine_cc": 1462.0,
    "max_power": 101.65,
    "max_torque": 136.8,
    "seats": 7.0,
    "length": 4395.0,
    "width": 1735.0,
    "height": 1690.0
}])

prediction = model.predict(row)[0]
# Inverse log transform if needed? No, the pipeline handles it (assuming it's in the pipeline)
# Wait, let's check train_model.py again to see if log transform is in the pipeline or manual.
# In app.py, it calls model.predict(row).
price = np.expm1(prediction)

print(f"Predicted Price: ₹{price:,.0f}")
