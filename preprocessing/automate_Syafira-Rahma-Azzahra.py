import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def preprocess_iris(
    input_path='iris_raw/Iris.csv',
    output_path='preprocessing/iris_preprocessing/iris_clean.csv'
):
    df = pd.read_csv(input_path)

    if 'Id' in df.columns:
        df.drop(columns=['Id'], inplace=True)

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    encoder = LabelEncoder()
    df['Species'] = encoder.fit_transform(df['Species'])

    features = df.drop(columns=['Species'])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    df_scaled = pd.DataFrame(scaled_features, columns=features.columns)
    df_scaled['Species'] = df['Species'].values

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_scaled.to_csv(output_path, index=False)

    return df_scaled

if __name__ == "__main__":
    preprocess_iris()
