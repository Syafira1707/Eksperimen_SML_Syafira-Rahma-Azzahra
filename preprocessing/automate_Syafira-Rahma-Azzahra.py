import pandas as pd
import numpy as np

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_titanic(
    input_path="train_raw.csv",
    output_path="titanic_preprocessed.csv"
):
    """
    Fungsi untuk melakukan preprocessing dataset Titanic.
    Dataset input berupa data mentah (train_raw.csv)
    Output berupa dataset siap digunakan untuk training model.
    """

    # =========================
    # 1. Load Dataset
    # =========================
    df = pd.read_csv(input_path)

    # =========================
    # 2. Informasi Awal Dataset
    # =========================
    print("Informasi Dataset:")
    print(df.info())

    print("\nUkuran Dataset:")
    print(df.shape)

    # =========================
    # 3. Exploratory Data Analysis (EDA)
    # =========================
    print("\nStatistik Deskriptif:")
    print(df.describe())

    print("\nDistribusi Survived:")
    print(df["Survived"].value_counts())

    print("\nDistribusi Sex:")
    print(df["Sex"].value_counts())

    print("\nDistribusi Pclass:")
    print(df["Pclass"].value_counts())

    print("\nKorelasi Fitur Numerik:")
    print(df.corr(numeric_only=True))

    # =========================
    # 4. Data Preprocessing
    # =========================
    missing_values = df.isnull().sum()
    duplicate_count = df.duplicated().sum()

    # Handling missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Drop kolom tidak relevan
    df = df.drop(columns=["Name", "Ticket", "Cabin"])

    # =========================
    # 5. Encoding Data Kategorikal
    # =========================
    le = LabelEncoder()
    df["Sex"] = le.fit_transform(df["Sex"])

    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    # =========================
    # 6. Normalisasi Fitur Numerik
    # =========================
    scaler = StandardScaler()
    fitur_numerik = ["Age", "Fare", "SibSp", "Parch"]

    df[fitur_numerik] = scaler.fit_transform(df[fitur_numerik])

    # =========================
    # 7. Simpan Dataset
    # =========================
    df.to_csv(output_path, index=False)

    print("\nMissing Values per Kolom:")
    print(missing_values)

    print("\nJumlah Data Duplikat:")
    print(duplicate_count)

    print("\nDataset hasil preprocessing (5 baris pertama):")
    print(df.head())

    print(f"\nFile '{output_path}' berhasil dibuat dan siap digunakan.")

    # Khusus untuk Google Colab
    if IN_COLAB:
    try:
        files.download(output_path)
    except Exception as e:
        print("Gagal download di Colab:", e)

    return df


if __name__ == "__main__":
    preprocess_titanic()

