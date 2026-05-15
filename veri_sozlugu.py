import pandas as pd

df = pd.read_csv("data/World Energy Consumption.csv")

# Tüm sütunlar ve ilk değerleri
print("=" * 60)
print("TÜM SÜTUNLAR VE ÖRNEK VERİLER")
print("=" * 60)

for sutun in df.columns:
    ornek = df[sutun].dropna().iloc[0] if df[sutun].dropna().shape[0] > 0 else "boş"
    bos_sayi = df[sutun].isna().sum()
    print(f"\n📌 {sutun}")
    print(f"   Örnek değer : {ornek}")
    print(f"   Boş değer   : {bos_sayi} satır")