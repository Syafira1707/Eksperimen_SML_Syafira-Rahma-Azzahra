import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os


def preprocess_titanic(
    input_path="train.csv",
    output_path="titanic_preprocessed.csv"
):
    """
    Fungsi untuk melakukan preprocessing dataset Titanic secara otomatis.
    Output berupa dataset siap digunakan untuk pelatihan model.
    """

    # =========================
    # 1. Load Dataset
    # =========================
    df = pd.read_csv(input_path)

    # =========================
    # 2. Analisis Awal Dataset
    # =========================
    missing_values = df.isnull().sum()
    duplicate_count = df.duplicated().sum()

    # =========================
    # 3. Menangani Missing Values
    # =========================
    # Age → median
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # Embarked → modus
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # =========================
    # 4. Menghapus Kolom Tidak Relevan
    # =========================
    df = df.drop(columns=['Name', 'Ticket', 'Cabin'])

    # =========================
    # 5. Encoding Data Kategorikal
    # =========================
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])

    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

    # =========================
    # 6. Normalisasi Fitur Numerik
    # =========================
    scaler = StandardScaler()

    fitur_numerik = ['Age', 'Fare', 'SibSp', 'Parch']
    df[fitur_numerik] = scaler.fit_transform(df[fitur_numerik])

    # =========================
    # 7. Simpan Dataset
    # =========================
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    # =========================
    # 8. Output Ringkasan
    # =========================
    print("Preprocessing selesai.")
    print("\nRingkasan Analisis Dataset:")
    print("Missing Values per Kolom:")
    print(missing_values)

    print("\nJumlah Data Duplikat:")
    print(duplicate_count)

    print(f"\nDataset hasil preprocessing disimpan sebagai: {output_path}")

    return df


if __name__ == "__main__":
    preprocess_titanic()
