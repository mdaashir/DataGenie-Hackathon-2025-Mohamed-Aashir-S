import os
import pandas as pd
import logging
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    hamming_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    jaccard_score,
    log_loss,
    top_k_accuracy_score,
    zero_one_loss,
)
from imblearn.over_sampling import SMOTE

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        n_jobs=-1,
        random_state=42,
        eval_metric="mlogloss",
    ),
    "LightGBM": LGBMClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
}

label_encoder = LabelEncoder()
scaler = StandardScaler()


def setup_logging(output_dir, log_name="classifier_selection.log"):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_name)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
        ],
    )
    logging.info("Classifier selection logging initialized.")
    return log_path


def load_dataset_and_labels(folder_path):
    files = glob(os.path.join(folder_path, "*.csv"))
    data = []

    for file in tqdm(files, desc="Extracting features"):
        try:
            df = pd.read_csv(file)
            if "Label" not in df.columns:
                continue

            data.append(df)
        except Exception as fe:
            logging.error(f"Error processing {file}: {fe}")

    if not data:
        raise ValueError("No valid CSV files with 'Label' column found.")

    full_data = pd.concat(data, ignore_index=True)
    full_data.dropna(inplace=True)
    full_data["Label"] = label_encoder.fit_transform(full_data["Label"])
    return full_data.drop(columns=["Label"]), full_data["Label"]


def evaluate_model(x_train, x_test, y_train, y_test):
    metrics_list = []

    for name, model in models.items():
        logging.info(f"\nTraining and evaluating: {name}...")
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        try:
            y_prob = model.predict_proba(x_test)
            roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
            top3_acc = top_k_accuracy_score(y_test, y_prob, k=3)
            logloss = log_loss(y_test, y_prob)
        except Exception:
            roc_auc, top3_acc, logloss = None, None, None

        report = classification_report(y_test, y_pred, zero_division=0)

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
            "Precision (macro)": precision_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "Recall (macro)": recall_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "F1 Score (macro)": f1_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "Precision (weighted)": precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "Recall (weighted)": recall_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "F1 Score (weighted)": f1_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "Cohen Kappa": cohen_kappa_score(y_test, y_pred),
            "MCC": matthews_corrcoef(y_test, y_pred),
            "Hamming Loss": hamming_loss(y_test, y_pred),
            "Jaccard (macro)": jaccard_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "Jaccard (weighted)": jaccard_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "Top-3 Accuracy": top3_acc,
            "ROC AUC": roc_auc,
            "Log Loss": logloss,
            "Zero-One Loss": zero_one_loss(y_test, y_pred),
            "Confusion Matrix": confusion_matrix(y_test, y_pred),
            "Classification Report": report,
        }

        metrics_list.append(metrics)

        logging.info(f"{name} Results:")
        for k, v in metrics.items():
            if k not in ["Model", "Confusion Matrix", "Classification Report"]:
                logging.info(
                    f"   {k:<20}: {v:.4f}"
                    if isinstance(v, float)
                    else f"   {k:<20}: {v}"
                )

    return metrics_list


def plot_feature_importance(model, features_name, top_n=20):
    if hasattr(model, "feature_importance"):
        importance = model.feature_importances_
        indices = importance.argsort()[-top_n:][::-1]
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importance[indices], y=[features_name[i] for i in indices])
        plt.title("Top Feature Importance")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    log_file = setup_logging("Datasets/logs")

    logging.info("Starting classifier selection...")
    try:
        logging.info("Loading dataset and labels...")
        x, y = load_dataset_and_labels("Datasets/synthetic_data")
        feature_names = x.columns

        logging.info("Normalizing features...")
        x = scaler.fit_transform(x)

        X_train, X_test, Y_train, Y_test = train_test_split(
            x, y, test_size=0.2, stratify=y, random_state=42
        )

        logging.info("Applying SMOTE to balance classes...")
        smote = SMOTE(random_state=42)
        X_train, Y_train = smote.fit_resample(X_train, Y_train)

        summary_metrics = evaluate_model(X_train, X_test, Y_train, Y_test)

        df_results = pd.DataFrame(summary_metrics)
        df_results_sorted = df_results.sort_values(
            by="F1 Score (weighted)", ascending=False
        )

        print("\nOverall Model Comparison:")
        print(
            df_results_sorted[
                [
                    "Model",
                    "F1 Score (weighted)",
                    "Accuracy",
                    "Precision (macro)",
                    "Recall (macro)",
                    "ROC AUC",
                    "Top-3 Accuracy",
                    "Log Loss",
                ]
            ].to_string(index=False)
        )

        df_results_sorted.to_csv("model_comparison_results.csv", index=False)
        logging.info("Results saved to 'model_comparison_results.csv'.")

        best_model_name = df_results_sorted.iloc[0]["Model"]
        best_model = models[best_model_name]
        logging.info(f"Plotting feature importance for best model: {best_model_name}")
        plot_feature_importance(best_model, feature_names)

    except Exception as e:
        logging.critical(f"Pipeline failed: {e}")

    logging.info(f"\nDone! Classifier selection completed successfully.  Log: '{log_file}'")
