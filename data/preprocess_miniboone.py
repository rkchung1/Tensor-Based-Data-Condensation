import sys
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess(input_path: str, train_path: str, test_path: str) -> None:
    # Load data
    df = pd.read_csv(input_path)

    # Convert “false”/“true” in signal to 0/1
    df['signal'] = (
        df['signal']
        .astype(str)
        .str.strip().str.lower()
        .map({'false': 0, 'true': 1})
    )

    # Rename signal column to label
    df.rename(columns={'signal': 'label'}, inplace=True)

    # Split 80% train / 20% test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Write out
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Wrote train set to {train_path}")
    print(f"Wrote test set to  {test_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python preprocess.py  input.csv  train_out.csv  test_out.csv")
        sys.exit(1)

    in_file, train_file, test_file = sys.argv[1], sys.argv[2], sys.argv[3]
    preprocess(in_file, train_file, test_file)