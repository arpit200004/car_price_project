import pandas as pd
import numpy as np
import random
from pathlib import Path

def generate_symmetric_cars(n_samples=5000):
    np.random.seed(42)
    random.seed(42)

    # Dictionary of modern 2025-2026 Indian cars with base specs & variants
    models = [
        {"brand": "Tata", "model": "Tata Punch", "body": "suv", "fuel": "petrol", "engine": "1199 cc", "power": "86.63 bhp", "torque": "115 Nm", "seats": 5, "length": "3827", "width": "1742", "height": "1615", "price_base": 700000, 
         "variants": ["Pure", "Adventure", "Accomplished", "Creative"]},
        {"brand": "Tata", "model": "Tata Nexon", "body": "suv", "fuel": "petrol", "engine": "1198 cc", "power": "118.27 bhp", "torque": "170 Nm", "seats": 5, "length": "3993", "width": "1811", "height": "1606", "price_base": 1100000,
         "variants": ["Smart", "Pure", "Creative", "Fearless"]},
        {"brand": "Tata", "model": "Tata Safari", "body": "suv", "fuel": "diesel", "engine": "1956 cc", "power": "167.62 bhp", "torque": "350 Nm", "seats": 7, "length": "4668", "width": "1922", "height": "1795", "price_base": 2200000,
         "variants": ["Smart", "Pure", "Adventure", "Accomplished"]},
        
        {"brand": "Maruti", "model": "Maruti Fronx", "body": "suv", "fuel": "petrol", "engine": "1197 cc", "power": "88.50 bhp", "torque": "113 Nm", "seats": 5, "length": "3995", "width": "1765", "height": "1550", "price_base": 950000,
         "variants": ["Sigma", "Delta", "Delta+", "Zeta", "Alpha"]},
        {"brand": "Maruti", "model": "Maruti Grand Vitara", "body": "suv", "fuel": "petrol", "engine": "1462 cc", "power": "101.64 bhp", "torque": "136.8 Nm", "seats": 5, "length": "4345", "width": "1795", "height": "1645", "price_base": 1500000,
         "variants": ["Sigma", "Delta", "Zeta", "Alpha"]},
        {"brand": "Maruti", "model": "Maruti Brezza", "body": "suv", "fuel": "petrol", "engine": "1462 cc", "power": "101.65 bhp", "torque": "136.8 Nm", "seats": 5, "length": "3995", "width": "1790", "height": "1685", "price_base": 1200000,
         "variants": ["LXi", "VXi", "ZXi", "ZXi+"]},
        
        {"brand": "Hyundai", "model": "Hyundai Exter", "body": "suv", "fuel": "petrol", "engine": "1197 cc", "power": "81.80 bhp", "torque": "113.8 Nm", "seats": 5, "length": "3815", "width": "1710", "height": "1631", "price_base": 850000,
         "variants": ["EX", "S", "SX", "SX (O)"]},
        {"brand": "Hyundai", "model": "Hyundai Creta", "body": "suv", "fuel": "petrol", "engine": "1497 cc", "power": "113.18 bhp", "torque": "143.8 Nm", "seats": 5, "length": "4330", "width": "1790", "height": "1635", "price_base": 1600000,
         "variants": ["E", "EX", "S", "SX", "SX (O)"]},
        
        {"brand": "Mahindra", "model": "Mahindra Scorpio-N", "body": "suv", "fuel": "diesel", "engine": "2198 cc", "power": "172.45 bhp", "torque": "400 Nm", "seats": 7, "length": "4662", "width": "1917", "height": "1857", "price_base": 2400000,
         "variants": ["Z2", "Z4", "Z6", "Z8", "Z8L"]},
        {"brand": "Mahindra", "model": "Mahindra XUV700", "body": "suv", "fuel": "petrol", "engine": "1997 cc", "power": "197.13 bhp", "torque": "380 Nm", "seats": 7, "length": "4695", "width": "1890", "height": "1755", "price_base": 2300000,
         "variants": ["MX", "AX3", "AX5", "AX7", "AX7L"]},
        {"brand": "Mahindra", "model": "Mahindra Thar", "body": "suv", "fuel": "diesel", "engine": "2184 cc", "power": "130 bhp", "torque": "300 Nm", "seats": 4, "length": "3985", "width": "1820", "height": "1844", "price_base": 1800000,
         "variants": ["AX (O)", "LX"]},
        
        {"brand": "Kia", "model": "Kia Seltos", "body": "suv", "fuel": "petrol", "engine": "1482 cc", "power": "157.81 bhp", "torque": "253 Nm", "seats": 5, "length": "4365", "width": "1800", "height": "1645", "price_base": 1700000,
         "variants": ["HTE", "HTK", "HTX", "HTX+", "GTLine"]},
        
        {"brand": "Toyota", "model": "Toyota Innova Hycross", "body": "muv", "fuel": "petrol", "engine": "1987 cc", "power": "183.72 bhp", "torque": "188 Nm", "seats": 7, "length": "4755", "width": "1845", "height": "1795", "price_base": 3000000,
         "variants": ["GX", "VX", "ZX", "ZX (O)"]},
    ]

    records = []
    colors = ["White", "Silver", "Grey", "Black", "Red", "Blue"]
    
    for _ in range(n_samples):
        car = random.choice(models)
        year = random.choices([2025, 2026], weights=[0.4, 0.6])[0]
        
        # New cars have low km
        if year == 2026:
            km = int(np.random.normal(5000, 2000))
        else:
            km = int(np.random.normal(15000, 5000))
        km = max(km, 500)
        
        # Calculate realistically depreciated price
        age = 2026 - year
        depreciation = 0.85 if age == 1 else 0.95
        
        # Add market variance to pricing (-10% to +10%)
        variance = random.uniform(0.9, 1.1)
        final_price = int(car["price_base"] * depreciation * variance)
        
        # Seller & Transmission
        seller = random.choices(["dealer", "individual"], weights=[0.75, 0.25])[0]
        trans = random.choices(["manual", "automatic"], weights=[0.6, 0.4])[0]
        
        # Pick real variant
        variant = random.choice(car["variants"])
        
        # Mileage varies slightly around 15-22 kmpl
        mileage = f"{random.uniform(15.0, 22.0):.2f} kmpl"
        
        # Generate the row perfectly matching kaggle headers
        row = {
            "brand_name": car["brand"],
            "model_new": car["model"],
            "variant_name": f"{car['model']} {variant}", 
            "seller_type_new": seller,
            "fuel_type": car["fuel"],
            "transmission_type_new": trans,
            "owner_type": "first" if year==2026 else random.choices(["first", "second"], weights=[0.9, 0.1])[0],
            "body_type_new": car["body"],
            "Drive Type": "fwd",
            "model_year": year,
            "km_driven": km,
            "pu": f"{final_price:,}",  # Formatted like '15,00,000'
            "mileage_new": mileage,
            "Displacement": car["engine"],
            "Max Power": car["power"],
            "Max Torque": car["torque"],
            "seating_capacity_new": car["seats"],
            "Length": car["length"],
            "Width": car["width"],
            "Height": car["height"],
            "Color": random.choice(colors)
        }
        records.append(row)

    df = pd.DataFrame(records)
    out_path = Path(__file__).parent / "synthetic_2026_cars.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Generated {len(df)} newer car records at {out_path}")

if __name__ == "__main__":
    generate_symmetric_cars(5000)
