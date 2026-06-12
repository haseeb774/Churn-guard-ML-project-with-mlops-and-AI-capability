import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import OneHotEncoder
from src.logging import logging
from src.exception import CustomException


# These exact column names must match what app.py sends to the model
BINARY_MAP = {
    "gender":          {"Female": 0, "Male": 1},
    "Partner":         {"Yes": 1, "No": 0},
    "Dependents":      {"Yes": 1, "No": 0},
    "PhoneService":    {"Yes": 1, "No": 0},
    "PaperlessBilling":{"Yes": 1, "No": 0},
}

MULTI_MAP = {
    "MultipleLines":    {"Yes": 1, "No": 0, "No phone service": 0},
    "InternetService":  {"DSL": 1, "Fiber optic": 2, "No": 0},
    "OnlineSecurity":   {"Yes": 1, "No": 0, "No internet service": 0},
    "OnlineBackup":     {"Yes": 1, "No": 0, "No internet service": 0},
    "DeviceProtection": {"Yes": 1, "No": 0, "No internet service": 0},
    "TechSupport":      {"Yes": 1, "No": 0, "No internet service": 0},
    "StreamingTV":      {"Yes": 1, "No": 0, "No internet service": 0},
    "StreamingMovies":  {"Yes": 1, "No": 0, "No internet service": 0},
    "Contract":         {"Month-to-month": 0, "One year": 1, "Two year": 2},
}

# Expected OHE columns for PaymentMethod — must match app.py
PAYMENT_COLS = [
    "PaymentMethod_Bank transfer (automatic)",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
]


class Datatransform:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def transform_data(self) -> pd.DataFrame:
        try:
            df = self.data.copy()

            # Drop unused columns
            for col in ["customerID", "CLV_proxy"]:
                if col in df.columns:
                    df.drop(columns=col, inplace=True)

            # Fix TotalCharges (whitespace → NaN)
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            df.dropna(inplace=True)
            df.reset_index(drop=True, inplace=True)

            # Encode target
            df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

            # Binary mappings
            for col, mapping in BINARY_MAP.items():
                if col in df.columns:
                    df[col] = df[col].map(mapping)

            # Multi-value mappings
            for col, mapping in MULTI_MAP.items():
                if col in df.columns:
                    df[col] = df[col].replace(mapping)

            # One-hot encode PaymentMethod
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded = ohe.fit_transform(df[["PaymentMethod"]])
            encoded_cols = ohe.get_feature_names_out(["PaymentMethod"])
            encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)

            df = df.drop(columns=["PaymentMethod"])
            df = pd.concat([df, encoded_df], axis=1)

            # Ensure all expected payment columns exist (fill 0 if missing)
            for col in PAYMENT_COLS:
                if col not in df.columns:
                    df[col] = 0.0

            df = df.dropna()
            logging.info(f"Transformation complete — shape: {df.shape}")

            # Save processed copy
            proc_path = "data/processed/processed_churn.csv"
            if not os.path.exists(proc_path):
                os.makedirs(os.path.dirname(proc_path), exist_ok=True)
                df.to_csv(proc_path, index=False)
                logging.info(f"Processed data saved to {proc_path}")

            return df

        except Exception as e:
            logging.error(f"Error in data_transform: {e}")
            raise CustomException(e, sys)