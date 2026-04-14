import pickle
import pandas as pd
import numpy as np

# Load model bundle
with open('car_price_model_v3.pkl', 'rb') as f:
    bundle = pickle.load(f)

model = bundle['model']

# Features used in training (from model_report.json)
# ["brand", "model", "car_name", "seller_type", "fuel_type", "transmission_type", "owner_type", "body_type", "drive_type", "car_age", "km_driven", "km_per_year", "mileage", "engine_cc", "max_power", "max_torque", "seats", "length", "width", "height"]

# Case: 2017 Maruti Vitara Brezza VXi, 120,000 km, First Owner
# Note: In the app, Brezza is likely 'Maruti Vitara Brezza'
# We'll use median values for specs to be safe

row = pd.DataFrame([{
    "brand": "maruti",
    "model": "maruti vitara brezza",
    "car_name": "maruti vitara brezza vxi",
    "seller_type": "individual",
    "fuel_type": "diesel",
    "transmission_type": "manual",
    "owner_type": "first",
    "body_type": "suv",
    "drive_type": "fwd",
    "car_age": float(2024 - 2017),
    "km_driven": 120000.0,
    "km_per_year": 120000.0 / 7,
    "mileage": 24.3, # typical
    "engine_cc": 1248.0,
    "max_power": 88.5,
    "max_torque": 200.0,
    "seats": 5.0,
    "length": 3995.0,
    "width": 1790.0,
    "height": 1640.0
}])

price_log = model.predict(row)[0]
price = np.expm1(price_log)

print(f"Predicted Price: ₹{price:,.0f}")
