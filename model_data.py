from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_FILE = Path("data/used-cars-dataset-cardekho/cars_details_merges.csv.gz")
SYNTHETIC_FILE = Path("data/synthetic_2024_cars.csv")


def extract_first_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0],
        errors="coerce",
    )


def clean_text(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.lower()
    )


# ────────────────────────────────────────────────
#  Price calibration: CarDekho prices are dealer
#  asking prices. We apply a small markdown to
#  bring them closer to actual transaction prices.
#  Research shows Indian used car transaction prices
#  average ~8% below the listed/asking price.
# ────────────────────────────────────────────────
DEALER_DISCOUNT    = 0.90   # dealer listings: 10% markdown (balanced)
INDIVIDUAL_DISCOUNT = 0.95  # individual listings: 5% markdown (balanced)


def load_market_data() -> pd.DataFrame:
    raw_main = pd.read_csv(DATA_FILE, low_memory=False)
    raw_main["listing_year"] = 2021
    
    # ── Inject Modern 2023-2024 Cars ──
    try:
        raw_synth = pd.read_csv(SYNTHETIC_FILE, low_memory=False)
        raw_synth["listing_year"] = 2024
        raw = pd.concat([raw_main, raw_synth], ignore_index=True)
    except FileNotFoundError:
        raw = raw_main

    # ── Step 1: parse and rename fields ──────────────────────────────
    df = pd.DataFrame(
        {
            "brand":             clean_text(raw["brand_name"]),
            "model":             clean_text(raw["model_new"]),
            "variant_name":      clean_text(raw["variant_name"]),
            "seller_type":       clean_text(raw["seller_type_new"]),
            "fuel_type":         clean_text(raw["fuel_type"]),
            "transmission_type": clean_text(raw["transmission_type_new"]),
            "owner_type":        clean_text(raw["owner_type"]),
            "body_type":         clean_text(raw["body_type_new"]),
            "drive_type":        clean_text(raw["Drive Type"]),
            "model_year":        pd.to_numeric(raw["model_year"], errors="coerce"),
            "listing_year":      pd.to_numeric(raw["listing_year"], errors="coerce"),
            "km_driven":         pd.to_numeric(raw["km_driven"], errors="coerce"),
            # Raw asking price  ─ will be calibrated below
            "price_raw":         pd.to_numeric(
                raw["pu"].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            ),
            "mileage":      extract_first_number(raw["mileage_new"]),
            "engine_cc":    extract_first_number(raw["Displacement"]).fillna(
                pd.to_numeric(raw["max_engine_capacity_new"], errors="coerce")
            ),
            "max_power":   extract_first_number(raw["Max Power"]),
            "max_torque":  extract_first_number(raw["Max Torque"]),
            "seats":       pd.to_numeric(raw["seating_capacity_new"], errors="coerce").fillna(
                pd.to_numeric(raw["Seating Capacity"], errors="coerce")
            ),
            "length": extract_first_number(raw["Length"]),
            "width":  extract_first_number(raw["Width"]),
            "height": extract_first_number(raw["Height"]),
            "color":  clean_text(raw["Color"]),
        }
    )

    df["car_name"] = (
        raw["variant_name"]
        .fillna(raw["model_new"])
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # ── Step 2: standardise nulls, drop rows missing critical fields ──
    required = [
        "brand", "model", "seller_type", "fuel_type",
        "transmission_type", "owner_type", "body_type", "drive_type",
        "model_year", "km_driven", "price_raw",
        "mileage", "engine_cc", "max_power", "max_torque",
        "seats", "length", "width", "height",
    ]
    df = df.replace({"nan": pd.NA, "none": pd.NA, "": pd.NA})
    df = df.dropna(subset=required).copy()
    df = df.drop_duplicates().reset_index(drop=True)

    # ── Step 3: hard-range sanity filters ────────────────────────────
    # Exclude price extremes: below ₹60K (scrap) and above ₹1.2 Cr (ultra-luxury)
    # These outliers distort training without helping everyday use-cases.
    price_cap   = min(float(df["price_raw"].quantile(0.995)), 12_000_000)
    km_cap      = float(df["km_driven"].quantile(0.995))
    power_cap   = float(df["max_power"].quantile(0.995))
    torque_cap  = float(df["max_torque"].quantile(0.995))

    df = df[
        df["price_raw"].between(60_000, price_cap)
        & df["km_driven"].between(500, km_cap)
        & df["model_year"].between(2000, 2024)    # tighter: pre-2000 cars are rare & noisy
        & df["mileage"].between(5, 35)             # tighter upper: >35kmpl is implausible for most cars
        & df["engine_cc"].between(500, 6_500)
        & df["max_power"].between(25, power_cap)
        & df["max_torque"].between(25, torque_cap)
        & df["seats"].between(2, 9)               # tighter: buses/trucks filtered
        & df["length"].between(2800, 5_500)        # tighter dimensional bounds
        & df["width"].between(1300, 2_300)
        & df["height"].between(1200, 2_200)
    ].copy()

    # ── Step 4: remove purely round-number prices (likely fake listings)
    # Prices that are exact multiples of ₹50,000 with no decimals are
    # overwhelmingly aspirational asking prices, not transaction values.
    # We keep them but flag suspicious ones to down-weight later.
    df["is_round_price"] = (df["price_raw"] % 50_000 == 0).astype(int)

    # ── Step 5: calibrate asking → transaction price ─────────────────
    # Dealers in India typically negotiate 8-12% off the asking price.
    # Individual sellers negotiate ~3% off. We apply conservative discounts.
    dealer_mask = df["seller_type"] == "dealer"
    # For round prices (likely inflated aspirational), apply an extra 2% haircut
    df["price"] = df["price_raw"].astype(float).copy()
    df.loc[dealer_mask,  "price"] = df.loc[dealer_mask,  "price_raw"].astype(float) * DEALER_DISCOUNT
    df.loc[~dealer_mask, "price"] = df.loc[~dealer_mask, "price_raw"].astype(float) * INDIVIDUAL_DISCOUNT

    # For round prices (likely inflated aspirational), apply an extra 2% haircut
    round_mask = df["is_round_price"] == 1
    df.loc[round_mask, "price"] *= 0.98

    # ── Step 5.1: 2024 Inflation Normalization ───────────────────────
    # The CarDekho data is mostly from 2021. New and used car prices have 
    # We are using a balanced inflation scaling to hit the 'goldilocks' pricing zone
    # between historic lows and aggressive modern bounds.
    annual_inflation = 1.025
    df["price"] = df["price"] * (annual_inflation ** (2024 - df["listing_year"]))

    # ── Step 6: engineered features ──────────────────────────────────
    # Calculate the age at the time of listing, NOT the age today!
    # This teaches the model the true depreciation curve.
    df["car_age"]    = df["listing_year"] - df["model_year"]
    df["km_per_year"] = df["km_driven"] / df["car_age"].clip(lower=1)

    # Remove implausibly high km_per_year (>40k km/year is likely data error)
    df = df[df["km_per_year"] <= 40_000].copy()

    return df.reset_index(drop=True)
