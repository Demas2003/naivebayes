import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Membaca data dari file Excel
file_path = 'handphone_data_1000.xlsx'
df = pd.read_excel(file_path)

# Encode fitur kategorikal
label_encoder = LabelEncoder()
df['merk'] = label_encoder.fit_transform(df['merk'])
df['layak_beli'] = label_encoder.fit_transform(df['layak_beli'])

# Memisahkan fitur dan label
X = df.drop(columns=['layak_beli'])
y = df['layak_beli']

# Membagi data menjadi data training dan data testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model Naive Bayes
model = GaussianNB()

# Melatih model
model.fit(X_train, y_train)

# Memprediksi data testing
y_pred = model.predict(X_test)

# Evaluasi model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report: \n{classification_report(y_test, y_pred, zero_division=1)}")

# Fungsi untuk update LabelEncoder
def update_label_encoder(merk, label_encoder):
    if merk not in label_encoder.classes_:
        label_encoder.classes_ = np.append(label_encoder.classes_, merk)

# Daftar merk HP yang tersedia dengan keterangan
merk_dict = {code: brand for code, brand in enumerate(label_encoder.classes_)}

# Tampilkan merk yang tersedia kepada pengguna
print("Merk HP yang tersedia untuk diprediksi:")
for code, brand in merk_dict.items():
    print(f"{brand} (kode: {code})")

# Minta input pengguna
harga = int(input("Masukkan harga handphone (1-5): "))
baterai = int(input("Masukkan kapasitas baterai (1-5): "))
kamera = int(input("Masukkan kualitas kamera (1-5): "))
RAM = int(input("Masukkan kapasitas RAM (1-5): "))
memori_internal = int(input("Masukkan kapasitas memori internal (1-5): "))
tahun_rilis = int(input("Masukkan tahun rilis handphone: "))
kondisi_fisik = int(input("Masukkan kondisi fisik handphone (1-5): "))
merk_input = input("Masukkan merk handphone: ")
rating_pengguna = float(input("Masukkan rating pengguna (1.0-5.0): "))

# Perbarui LabelEncoder jika merk tidak dikenal
update_label_encoder(merk_input, label_encoder)

# Transformasi nilai merk menggunakan label_encoder
merk_transformed = label_encoder.transform([merk_input])[0]

# Buat data handphone baru untuk prediksi
new_phone = [[harga, baterai, kamera, RAM, memori_internal, tahun_rilis, kondisi_fisik, merk_transformed, rating_pengguna]]

# Prediksi kelayakan
prediksi = model.predict(new_phone)

# Inverse transform untuk mendapatkan label kelayakan
kelayakan = label_encoder.inverse_transform(prediksi)

print(f"Handphone baru ini {'layak' if kelayakan[0] == 1 else 'tidak layak'} dibeli.")
