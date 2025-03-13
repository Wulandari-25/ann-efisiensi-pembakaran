import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib

# Load dataset
file_path = "Dataset_nilai_kalor_cleaned.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Menghapus kolom yang tidak relevan
irrelevant_columns = ["Jenis Bahan Bakar", "Kategori Efisiensi", "0% - 35%", "sangat rendah (sangat tidak efisien)"]
df_cleaned = df.drop(columns=irrelevant_columns, errors='ignore')

# Encode kategori efisiensi
label_encoder = LabelEncoder()
df_cleaned["Kategori Efisiensi"] = label_encoder.fit_transform(df["Kategori Efisiensi"])

# Simpan Label Encoder
joblib.dump(label_encoder, "label_encoder.pkl")

# Normalisasi fitur numerik
scaler = MinMaxScaler()
X = df_cleaned.drop(columns=["Kategori Efisiensi"])
y = df_cleaned["Kategori Efisiensi"]

X_scaled = scaler.fit_transform(X)

# Simpan scaler
joblib.dump(scaler, "scaler.pkl")

# Simpan data yang telah diproses
pd.DataFrame(X_scaled, columns=X.columns).to_csv("X_scaled.csv", index=False)
pd.DataFrame(y, columns=["Kategori Efisiensi"]).to_csv("y.csv", index=False)