#!/usr/bin/env python3
import os

# Test file paths
MODEL_PATH = 'attached_assets/model_1753234187394.pkl'
SCALER_PATH = 'attached_assets/df_1753234187393.pkl'
DATA_PATH = 'attached_assets/USA_Housing_1753234187392.csv'

print("Model file exists:", os.path.exists(MODEL_PATH))
print("Scaler file exists:", os.path.exists(SCALER_PATH))
print("Data file exists:", os.path.exists(DATA_PATH))

if os.path.exists(DATA_PATH):
    import pandas as pd
    df = pd.read_csv(DATA_PATH)
    print("CSV shape:", df.shape)
    print("Columns:", list(df.columns))
    print("First few rows:")
    print(df.head())