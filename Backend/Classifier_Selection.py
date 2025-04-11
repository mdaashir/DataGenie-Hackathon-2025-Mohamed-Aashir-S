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

    for name, model in models.items():
        print(f"\nTraining and evaluating: {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred)

        print(f"{name} Results:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
        print(f"Weighted F1: {report['weighted avg']['f1-score']:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    evaluate_model()
