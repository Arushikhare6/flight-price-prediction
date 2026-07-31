from pathlib import Path
import pandas as pd
import joblib


def load_split_data(split_dir="data/processed/split"):
    """
    Load train, val, and test DataFrames from split_dir.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    path = Path(split_dir)
    if not path.exists():
        path = Path("../data/processed/split")

    train_df = pd.read_csv(path / "train.csv")
    val_df = pd.read_csv(path / "val.csv")
    test_df = pd.read_csv(path / "test.csv")

    return train_df, val_df, test_df


def save_pipeline(pipeline, filepath):
    """
    Save a trained scikit-learn pipeline object using joblib.
    """
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_path)
    print(f"Pipeline saved to {out_path}")


def load_pipeline(filepath):
    """
    Load a saved scikit-learn pipeline object using joblib.
    """
    return joblib.load(filepath)


def format_results_table(results_list, sort_by="R2 Score", ascending=False):
    """
    Convert results list to DataFrame, sort, and return formatted table.
    """
    df = pd.DataFrame(results_list)
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)
    return df
