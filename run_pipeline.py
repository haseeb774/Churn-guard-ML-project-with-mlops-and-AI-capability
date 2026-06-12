"""
run_pipeline.py — ChurnGuard AI Training Pipeline

Usage:
    python run_pipeline.py --data data/raw/churn.csv

    Or with a custom path:
    python run_pipeline.py --data /path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv
"""

import sys
import os
import argparse

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_ingest import DataIngest
from src.data_transform import Datatransform
from src.model_train import TrainModel
from src.logging import logging


def run_pipeline(data_path: str):
    print("\n" + "=" * 50)
    print("  CHURNGUARD AI — PIPELINE START")
    print("=" * 50)

    # Step 1: Ingest
    print("\n[1/3] Data Ingestion...")
    ingestor = DataIngest(data_path)
    raw_data = ingestor.data_import()
    print(f"      ✅ Loaded {len(raw_data):,} rows")

    # Step 2: Transform
    print("\n[2/3] Data Transformation...")
    transformer = Datatransform(raw_data)
    transformed_data = transformer.transform_data()
    print(f"      ✅ Processed shape: {transformed_data.shape}")

    # Step 3: Train
    print("\n[3/3] Model Training + MLflow Logging...")
    trainer = TrainModel(transformed_data)
    model, metrics = trainer.train_model()

    print("\n✅ Pipeline complete. Run the app:\n")
    print("   uvicorn app:app --reload")
    print("   streamlit run streamlit_app.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChurnGuard AI Training Pipeline")
    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/churn.csv",
        help="Path to the Telco Customer Churn CSV file"
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"\n❌ Data file not found: {args.data}")
        print("   Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        print("   Then run: python run_pipeline.py --data /your/path/to/churn.csv\n")
        sys.exit(1)

    run_pipeline(args.data)