import pandas as pd


def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
        print("Dataset shape:", df.shape)

        return df

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        print(f"No data: {file_path} is empty.")
        raise

    except Exception as e:
        print(f"Error while loading dataset: {e}")
        raise