import pickle
import pandas as pd
import numpy as np

# Load model bundle
with open('car_price_model_v3.pkl', 'rb') as f:
    bundle = pickle.load(f)

model = bundle['model']

# 2017 Maruti Suzuki Vitara Brezza (Diesel, Manual)
# This car is 7 years old in our new 2024 baseline.
row = pd.DataFrame([{
    "brand": "maruti",
    "model": "maruti vitara brezza",
    "car_name": "maruti vitara brezza vdi",
    "seller_type": "dealer",
    "fuel_type": "diesel",
    "transmission_type": "manual",
    "owner_type": "first",
    "body_type": "suv",
    "drive_type": "fwd",
    "car_age": float(2024 - 2017), # Age 7 in 2024
    "km_driven": 60000.0,
    "km_per_year": 60000.0 / 7.0,
    "mileage": 24.3,
    "engine_cc": 1248.0,
    "max_power": 88.5,
    "max_torque": 200.0,
    "seats": 5.0,
    "length": 3995.0,
    "width": 1790.0,
    "height": 1640.0
}])

prediction = model.predict(row)[0]
price = np.expm1(prediction)

print(f"Predicted Price (2017 Brezza in 2024): ₹{price:,.0f}")
