import pandas as pd

# Ganti 'nama_file.csv' dengan nama file kamu
file_path = './diabetes_012_health_indicators_BRFSS2015.csv'

# Load dataset
df = pd.read_csv(file_path)

# 1. Informasi umum tentang dataset
print("=== Informasi Umum ===")
print(df.info())
print()

# 2. Cek nilai yang hilang
print("=== Jumlah Nilai Kosong per Kolom ===")
print(df.isnull().sum())
print()

# 3. Statistik deskriptif untuk kolom numerik
print("=== Statistik Deskriptif (Numerik) ===")
print(df.describe())
print()

# 4. Statistik deskriptif untuk kolom non-numerik (kategorikal)
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
if len(categorical_cols) > 0:
    print("=== Statistik Deskriptif (Kategorikal) ===")
    print(df[categorical_cols].describe())
    print()
else:
    print("=== Tidak ada kolom kategorikal di dataset ===\n")

# 5. Jumlah nilai unik dan contoh distribusi tiap kolom
print("=== Nilai Unik dan Contoh Distribusi per Kolom ===")
for col in df.columns:
    print(f"\nKolom: {col}")
    print(f"Jumlah nilai unik: {df[col].nunique()}")
    print("Contoh distribusi nilai:")
    print(df[col].value_counts().head())

# Ganti 'label' dengan nama kolom target yang sesuai
if 'Class' in df.columns:
    label_column = 'Class'
elif 'Label' in df.columns:
    label_column = 'Label'
elif 'Diabetes_012' in df.columns:
    label_column = 'Diabetes_012'
elif 'Diabetes_binary' in df.columns:
    label_column = 'Diabetes_binary'
else:
    raise ValueError("Tidak ada kolom label yang ditemukan.")


print("=== Distribusi Label (Keseimbangan Kelas) ===")
print(df[label_column].value_counts())
print()
print("=== Persentase Tiap Kelas ===")
print(df[label_column].value_counts(normalize=True) * 100)

