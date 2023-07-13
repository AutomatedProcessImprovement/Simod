from pathlib import Path

import pandas as pd


def main():
    base_dir = Path("input/logs")

    for log_path in base_dir.glob("*.csv.gz"):
        print(log_path)
        df = pd.read_csv(log_path)
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        df.to_csv(log_path, index=False, compression="gzip")


if __name__ == "__main__":
    main()
