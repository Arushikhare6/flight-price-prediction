from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def create_data_splits(
    featured_csv_path="data/processed/featured_train.csv",
    output_dir="data/processed/split",
    test_size=0.15,
    val_size=0.15,
    random_state=42,
):
    """
    Split processed dataset into train (70%), validation (15%), and test (15%) sets.
    Saves outputs to output_dir/train.csv, val.csv, test.csv.
    """
    featured_path = Path(featured_csv_path)
    if not featured_path.exists():
        # Fallback path if running from src directory
        featured_path = Path("../data/processed/featured_train.csv")

    df = pd.read_csv(featured_path)

    # 1. Split off test set first (15%)
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    # 2. Split remaining train_val into train (70% overall) and val (15% overall)
    # val_relative_size = 0.15 / (1.0 - 0.15) = 0.17647
    val_relative_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_relative_size, random_state=random_state
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_path / "train.csv", index=False)
    val_df.to_csv(out_path / "val.csv", index=False)
    test_df.to_csv(out_path / "test.csv", index=False)

    print("Data splits created successfully:")
    print(f"  Train samples : {len(train_df)} ({len(train_df)/len(df):.1%})")
    print(f"  Val samples   : {len(val_df)} ({len(val_df)/len(df):.1%})")
    print(f"  Test samples  : {len(test_df)} ({len(test_df)/len(df):.1%})")

    return train_df, val_df, test_df


if __name__ == "__main__":
    create_data_splits()
