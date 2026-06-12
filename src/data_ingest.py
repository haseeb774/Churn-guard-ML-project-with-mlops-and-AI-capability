import pandas as pd
import sys
import os
from src.logging import logging
from src.exception import CustomException


class DataIngest:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def data_import(self) -> pd.DataFrame:
        try:
            logging.info(f"Loading data from: {self.file_path}")
            df = pd.read_csv(self.file_path)
            logging.info(f"Data loaded successfully — shape: {df.shape}")

            # Save raw copy only if it doesn't exist yet
            raw_path = "data/raw/churn.csv"
            if not os.path.exists(raw_path):
                os.makedirs(os.path.dirname(raw_path), exist_ok=True)
                df.to_csv(raw_path, index=False)
                logging.info(f"Raw data saved to {raw_path}")

            return df

        except Exception as e:
            logging.error(f"Error in data_ingest: {e}")
            raise CustomException(e, sys)