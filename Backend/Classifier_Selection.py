import os
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
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
)

label_encoder = LabelEncoder()


def load_dataset_and_labels(folder_path):
    files = glob(os.path.join(folder_path, "*.csv"))
    data = []

    for file in tqdm(files, desc="Extracting features"):
        try:
            df = pd.read_csv(file)
            if "Label" not in df.columns:
                continue

            data.append(df)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    full_data = pd.concat(data, ignore_index=True)
    full_data["Label"] = label_encoder.fit_transform(full_data["Label"])
    return full_data.drop(columns=["Label"]), full_data["Label"]


x, y = load_dataset_and_labels("Datasets/synthetic_data")

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, random_state=42, eval_metric="mlogloss"),
    "LightGBM": LGBMClassifier(n_estimators=200, random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
}


def evaluate_model():
    metrics_list = []

    for name, model in models.items():
        print(f"\nTraining and evaluating: {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

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
            "Confusion Matrix": confusion_matrix(y_test, y_pred),
            "Classification Report": report,
        }

        metrics_list.append(metrics)

        print(f"{name} Results:")
        for k, v in metrics.items():
            if k not in ["Model", "Confusion Matrix", "Classification Report"]:
                print(f"   {k:<20}: {v:.4f}")

    return metrics_list


if __name__ == "__main__":
    summary_metrics = evaluate_model()
    print("\nOverall Model Comparison:")
    df_results = pd.DataFrame(summary_metrics)
    df_results_sorted = df_results.sort_values(
        by="F1 Score (weighted)", ascending=False
    )
    print(
        df_results_sorted[
            [
                "Model",
                "F1 Score (weighted)",
                "Accuracy",
                "Precision (macro)",
                "Recall (macro)",
            ]
        ].to_string(index=False)
    )
