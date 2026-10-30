import pandas as pd


def load_dataset(filepath):

    """
    Reads CSV, Excel, or ODS dataset
    and returns a pandas DataFrame.
    """

    if "." not in filepath:
        raise ValueError("File has no extension.")

    extension = filepath.rsplit(".", 1)[1].lower()

    try:

        if extension == "csv":
            df = pd.read_csv(filepath)

        elif extension in ["xlsx", "xls", "xlsm", "xlsb"]:
            df = pd.read_excel(filepath)

        elif extension == "ods":
            df = pd.read_excel(filepath, engine="odf")

        else:
            raise ValueError("Unsupported file format.")

    except ValueError:
        raise

    except Exception as e:
        raise ValueError(f"Failed to read file: {e}")

    df = fix_numeric_columns(df)

    return df


def fix_numeric_columns(df):
    """
    Detects object-dtype columns that are actually numeric
    (e.g. blank strings instead of NaN) and converts them.
    """
    for col in df.select_dtypes(include="object").columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() >= 0.99 * df[col].notna().sum():
            df[col] = converted
    return df