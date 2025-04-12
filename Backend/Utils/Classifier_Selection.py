import os
from itertools import cycle
import pandas as pd
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
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
    zero_one_loss, roc_curve, auc,
)
from imblearn.over_sampling import SMOTE
from Backend import RESULTS_DIR, DATASET_DIR
from Backend.Utils.Logger import setup_logging

# Models
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, n_jobs=-1, random_state=42, eval_metric="mlogloss"
    ),
    "LightGBM": LGBMClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
}

label_encoder = LabelEncoder()
scaler = StandardScaler()
log_file, logging = setup_logging(log_name="classifier_selection.log")

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


def evaluate_model(
    x_train, x_test, y_train, y_test, label_classes, output_dir=f"{RESULTS_DIR}/reports"
):
    os.makedirs(output_dir, exist_ok=True)
    metrics_list = []

    for name, model in tqdm(list(models.items()), desc="Training models", ncols=100):
        logging.info(f"\nTraining and evaluating: {name}...")
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        try:
            y_prob = model.predict_proba(x_test)
            roc_auc = roc_auc_score(
                y_test, y_prob, multi_class="ovr", average="weighted"
            )
            plot_multiclass_roc(y_test, y_prob, label_classes, name)
            top3_acc = top_k_accuracy_score(
                y_test, y_prob, k=3, labels=np.unique(y_test)
            )
            logloss = log_loss(y_test, y_prob, labels=np.unique(y_test))
        except Exception:
            roc_auc, top3_acc, logloss = None, None, None

        try:
            cv_scores = cross_val_score(
                model, x_train, y_train, cv=5, scoring="f1_weighted"
            )
            cv_f1_weighted = np.mean(cv_scores)
        except Exception:
            cv_f1_weighted = None

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
            "CV F1 Score (weighted)": cv_f1_weighted,
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

        logging.info("Saving confusion matrix and classification report...")
        plot_confusion_matrix(metrics["Confusion Matrix"], label_classes, name)
        with open(f"{output_dir}/{name}_classification_report.txt", "w") as f:
            f.write(report)

        logging.info(f"{name} Results:")
        for k, v in metrics.items():
            if k not in ["Model", "Confusion Matrix", "Classification Report"]:
                logging.info(
                    f"   {k:<25}: {v:.4f}"
                    if isinstance(v, float)
                    else f"   {k:<25}: {v}"
                )

    return metrics_list


def plot_feature_importance(model, features_name, top_n=20, output_dir=f"{RESULTS_DIR}/plots"):
    os.makedirs(output_dir, exist_ok=True)
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim == 1:
            importance = np.abs(coef)
        else:
            importance = np.mean(np.abs(coef), axis=0)
    else:
        logging.warning(
            f"Feature importance not available for model: {type(model).__name__}"
        )
        return

    indices = np.argsort(importance)[-top_n:][::-1]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance[indices], y=[features_name[i] for i in indices])
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    logging.info("Saving feature importance plot...")
    plt.savefig(f"{output_dir}/feature_importance.png")
    plt.close()


def plot_confusion_matrix(cm, labels, model_name, output_dir=f"{RESULTS_DIR}/images"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    logging.info("Saving confusion matrix plot...")
    plt.savefig(f"{output_dir}/{model_name}_confusion_matrix.png")
    plt.close()


def plot_multiclass_roc(y_true, y_score, class_names, model_name, output_dir=f"{RESULTS_DIR}/plots"):
    os.makedirs(output_dir, exist_ok=True)
    n_classes = len(class_names)

    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = cycle(plt.cm.tab10.colors)

    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"Class {class_names[i]} (AUC = {roc_auc[i]:.2f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    logging.info("Saving ROC curve plot...")
    plt.savefig(f"{output_dir}/{model_name}_multiclass_ROC.png")
    plt.close()


def classifier_selection():
    logging.info("Starting classifier selection...")

    try:
        logging.info("Loading dataset and labels...")
        x, y = load_dataset_and_labels(f"{DATASET_DIR}/synthetic_data")
        feature_names = x.columns

        logging.info("Normalizing features...")
        x = scaler.fit_transform(x)

        stratify_option = y if len(set(y)) > 1 else None
        X_train, X_test, Y_train, Y_test = train_test_split(
            x, y, test_size=0.2, stratify=stratify_option, random_state=42
        )

        logging.info("Applying SMOTE to balance classes...")
        smote = SMOTE(random_state=42)
        X_train, Y_train = smote.fit_resample(X_train, Y_train)

        summary_metrics = evaluate_model(
            X_train, X_test, Y_train, Y_test, label_encoder.classes_
        )

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

        os.makedirs(RESULTS_DIR, exist_ok=True)
        df_results_sorted.to_csv(f"{RESULTS_DIR}/model_comparison_results.csv", index=False)
        logging.info("Results saved to 'Results/model_comparison_results.csv'.")

        best_model_name = df_results_sorted.iloc[0]["Model"]
        best_model = models[best_model_name]
        best_model.fit(X_train, Y_train)

        logging.info(f"Saving best model: {best_model_name}")
        joblib.dump(best_model, f"{RESULTS_DIR}/best_model.pkl")

        logging.info(f"Plotting feature importance for best model: {best_model_name}")
        plot_feature_importance(best_model, feature_names)

    except Exception as e:
        logging.critical(f"Pipeline failed: {e}")

    logging.info(
        f"\nDone! Classifier selection completed successfully.  Log: '{log_file}'"
    )


if __name__ == "__main__":
    classifier_selection()
