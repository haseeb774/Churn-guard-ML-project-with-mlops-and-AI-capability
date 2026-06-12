import pandas as pd
import numpy as np
import sys
import os
import time
import warnings
import joblib
import mlflow
import mlflow.xgboost
import optuna
import matplotlib.pyplot as plt
import seaborn as sns

from dotenv import load_dotenv
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    f1_score, recall_score, precision_score,
    confusion_matrix, roc_curve
)

from src.logging import logging
from src.exception import CustomException

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ANALYSIS_DIR = "outputs/analysis"
MODEL_PATH   = "outputs/model.pkl"
PLT_KWARGS   = dict(dpi=120, bbox_inches="tight")


class TrainModel:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    # ── EDA Plots ────────────────────────────────────────────────────────────

    def _generate_eda_plots(self, df: pd.DataFrame):
        os.makedirs(ANALYSIS_DIR, exist_ok=True)
        sns.set_theme(style="darkgrid")

        # 1. Churn Distribution
        fig, ax = plt.subplots(figsize=(7, 4))
        counts = df["Churn"].value_counts()
        ax.bar(["No Churn", "Churn"], counts.values,
               color=["#3fb950", "#f85149"], edgecolor="none")
        ax.set_title("Churn Distribution", fontsize=14, fontweight="bold")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 30, f"{v:,} ({v/len(df)*100:.1f}%)", ha="center", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{ANALYSIS_DIR}/01_churn_distribution.png", **PLT_KWARGS)
        plt.close()

        # 2. Churn by Contract
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = {0: "Month-to-Month", 1: "One Year", 2: "Two Year"}
        contract_churn = df.groupby("Contract")["Churn"].mean()
        ax.bar([labels.get(i, i) for i in contract_churn.index],
               contract_churn.values, color=["#f85149", "#d29922", "#3fb950"])
        ax.set_title("Churn Rate by Contract Type", fontsize=14, fontweight="bold")
        ax.set_ylabel("Churn Rate")
        for i, v in enumerate(contract_churn.values):
            ax.text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{ANALYSIS_DIR}/02_churn_by_contract.png", **PLT_KWARGS)
        plt.close()

        # 3. Tenure Distribution by Churn
        fig, ax = plt.subplots(figsize=(10, 4))
        df[df["Churn"] == 0]["tenure"].hist(bins=30, alpha=0.7, color="#3fb950",
                                             label="No Churn", ax=ax)
        df[df["Churn"] == 1]["tenure"].hist(bins=30, alpha=0.7, color="#f85149",
                                             label="Churn", ax=ax)
        ax.set_title("Tenure Distribution by Churn", fontsize=14, fontweight="bold")
        ax.set_xlabel("Tenure (months)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{ANALYSIS_DIR}/03_tenure_by_churn.png", **PLT_KWARGS)
        plt.close()

        # 4. Monthly Charges by Churn
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(x="Churn", y="MonthlyCharges", data=df,
                    palette=["#3fb950", "#f85149"], ax=ax)
        ax.set_xticklabels(["No Churn", "Churn"])
        ax.set_title("Monthly Charges by Churn", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{ANALYSIS_DIR}/04_monthly_charges.png", **PLT_KWARGS)
        plt.close()

        # 5. Senior Citizen churn rate
        fig, ax = plt.subplots(figsize=(6, 4))
        senior_churn = df.groupby("SeniorCitizen")["Churn"].mean()
        ax.bar(["Non-Senior", "Senior"], senior_churn.values,
               color=["#388bfd", "#f85149"])
        ax.set_title("Churn Rate: Senior vs Non-Senior", fontsize=14, fontweight="bold")
        ax.set_ylabel("Churn Rate")
        for i, v in enumerate(senior_churn.values):
            ax.text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{ANALYSIS_DIR}/05_senior_churn.png", **PLT_KWARGS)
        plt.close()

        logging.info(f"Generated {len(os.listdir(ANALYSIS_DIR))} EDA plots")

    # ── Post-training plots ───────────────────────────────────────────────────

    def _generate_model_plots(self, model, X_test, y_test):
        # Confusion Matrix
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Churn", "Churn"],
                    yticklabels=["No Churn", "Churn"], ax=ax)
        ax.set_title("Confusion Matrix — XGBoost", fontsize=13, fontweight="bold")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(f"{ANALYSIS_DIR}/06_confusion_matrix.png", **PLT_KWARGS)
        plt.close()

        # ROC Curve
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fpr, tpr, color="#388bfd", lw=2, label=f"ROC Curve (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], color="#484f58", lw=1, linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — XGBoost", fontsize=13, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{ANALYSIS_DIR}/07_roc_curve.png", **PLT_KWARGS)
        plt.close()

        # Feature Importance
        fi = pd.Series(model.feature_importances_,
                       index=X_test.columns).sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(10, 6))
        fi.sort_values().plot(kind="barh", color="#388bfd", ax=ax)
        ax.set_title("Top 15 Feature Importances", fontsize=13, fontweight="bold")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        plt.savefig(f"{ANALYSIS_DIR}/08_feature_importance.png", **PLT_KWARGS)
        plt.close()

    # ── Main training method ─────────────────────────────────────────────────

    def train_model(self):
        try:
            load_dotenv()

            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            experiment   = os.getenv("MLFLOW_EXPERIMENT_NAME", "churnguard-ai")

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment)

            # Generate EDA plots first
            logging.info("Generating EDA plots...")
            self._generate_eda_plots(self.data)

            # Prepare data
            X = self.data.drop(columns=["Churn"])
            y = self.data["Churn"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            logging.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

            # Optuna hyperparameter search
            def objective(trial):
                params = {
                    "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
                    "max_depth":         trial.suggest_int("max_depth", 3, 8),
                    "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3),
                    "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "scale_pos_weight":  trial.suggest_float("scale_pos_weight", 1.0, 3.0),
                }
                model = XGBClassifier(
                    **params,
                    eval_metric="logloss",
                    random_state=42,
                    verbosity=0,
                )
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
                return f1_score(y_test, model.predict(X_test))

            logging.info("Starting Optuna hyperparameter search (30 trials)...")
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=30, show_progress_bar=False)
            best_params = study.best_params
            logging.info(f"Best params: {best_params}")

            # Train final model
            with mlflow.start_run(run_name=f"XGBoost_{int(time.time())}"):
                final_model = XGBClassifier(
                    **best_params,
                    eval_metric="logloss",
                    random_state=42,
                    verbosity=0,
                )
                final_model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    verbose=False
                )

                y_pred  = final_model.predict(X_test)
                y_proba = final_model.predict_proba(X_test)[:, 1]

                metrics = {
                    "roc_auc":   roc_auc_score(y_test, y_proba),
                    "f1_score":  f1_score(y_test, y_pred),
                    "recall":    recall_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                }

                mlflow.log_params(best_params)
                mlflow.log_metrics(metrics)
                mlflow.xgboost.log_model(final_model, "model")

                logging.info(f"Metrics: {metrics}")
                logging.info(f"\n{classification_report(y_test, y_pred)}")

            # Generate model evaluation plots
            self._generate_model_plots(final_model, X_test, y_test)

            # Save model
            os.makedirs("outputs", exist_ok=True)
            joblib.dump(final_model, MODEL_PATH)
            logging.info(f"Model saved to {MODEL_PATH}")

            print("\n" + "=" * 50)
            print("  TRAINING COMPLETE")
            print("=" * 50)
            for k, v in metrics.items():
                print(f"  {k:12s}: {v:.4f}")
            print(f"  Model saved: {MODEL_PATH}")
            print(f"  EDA plots  : {ANALYSIS_DIR}/")
            print("=" * 50)
            print("\nNext:")
            print("  uvicorn app:app --reload")
            print("  streamlit run streamlit_app.py")

            return final_model, metrics

        except Exception as e:
            logging.error(f"Training failed: {e}")
            raise CustomException(e, sys)