import os
import glob
import joblib
import pandas as pd
from tqdm import tqdm
from Backend import MODELS_DIR, DATASET_DIR, MODEL_LIST
from Backend.Utils.Extract_Feature import extract_features
from Backend.Utils.Logger import setup_logging

log_file, logging = setup_logging(log_name="add_feature.log")

try:
    os.makedirs(MODELS_DIR, exist_ok=True)
    transform = joblib.load(f"{MODELS_DIR}/feature_extractor.pkl")
except FileNotFoundError:
    joblib.dump(extract_features, f"{MODELS_DIR}/feature_extractor.pkl")


def add_extracted_features(output_dir=f"{DATASET_DIR}/synthetic_data"):
    os.makedirs(output_dir, exist_ok=True)

    for model in tqdm(MODEL_LIST, desc="Processing models"):
        model_df = pd.DataFrame()
        files = glob.glob(f"{output_dir}/{model}/{model}_series_*.csv")
        logging.info(f"\n--- Processing model: {model} | Found {len(files)} files ---")

        success_count = 0
        fail_count = 0

        for file in tqdm(files, desc=f"{model}", leave=False):
            try:
                df = pd.read_csv(file)
                features = transform(df)
                features["Label"] = model
                if isinstance(features, pd.DataFrame):
                    model_df = pd.concat([model_df, features], ignore_index=True)
                else:
                    model_df = pd.concat([model_df, pd.DataFrame([features])], ignore_index=True)
                success_count += 1
                logging.info(f"Successfully extracted features from: {file}")
            except Exception as e:
                fail_count += 1
                logging.error(f"Error processing file {file}: {e}", exc_info=True)
                tqdm.write(f"Error in {file}: {e}")

        model_df.dropna(inplace=True)

        save_path = f"{output_dir}/{model}.csv"
        if not model_df.empty:
            model_df.to_csv(save_path, index=False)
            tqdm.write(f"\nSaved {save_path} with shape {model_df.shape}")
            logging.info(f"Saved features for {model} to {save_path}")
        else:
            tqdm.write(f"No valid data for {model}. Skipping save.")
            logging.warning(f"No valid data extracted for model: {model}")

        logging.info(
            f"Finished {model} | Success: {success_count} | Failures: {fail_count}"
        )

    logging.info(f"\nDone! All model features extracted and saved.  Log: '{log_file}'")


if __name__ == "__main__":
    add_extracted_features()
