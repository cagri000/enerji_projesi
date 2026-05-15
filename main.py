import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

print("=" * 50)
print("   DÜNYA ENERJİ TÜKETİMİ ANALİZİ VE TAHMİNİ")
print("=" * 50)

# ── VERİYİ YÜKLE ──
df = pd.read_csv("data/World Energy Consumption.csv")
sutunlar = ['country', 'year', 'population', 'gdp',
            'primary_energy_consumption', 'fossil_fuel_consumption', 'renewables_consumption']
df = df[sutunlar].dropna()
print(f"\n✅ Veri yüklendi: {len(df)} satır, {df['country'].nunique()} ülke")

# ════════════════════════════════════════
# GRAFİK 1 — Ülke Karşılaştırması
# ════════════════════════════════════════
ulkeler = ['Turkey', 'Germany', 'France', 'United States', 'China']
renkler = ['red', 'black', 'blue', 'purple', 'orange']

plt.figure(figsize=(12, 5))
for ulke, renk in zip(ulkeler, renkler):
    veri = df[df['country'] == ulke]
    plt.plot(veri['year'], veri['primary_energy_consumption'],
             label=ulke, color=renk, marker='o', markersize=3)

plt.title('Ülkelerin Yıllık Enerji Tüketimi (1965-2018)')
plt.xlabel('Yıl')
plt.ylabel('Enerji Tüketimi (TWh)')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/grafik1_ulkeler.png')
plt.show()
print("\n✅ Grafik 1 kaydedildi — Ülke karşılaştırması")

# ════════════════════════════════════════
# GRAFİK 2 — 2018 Top 10 Ülke
# ════════════════════════════════════════
son_yil = df[df['year'] == 2018].sort_values(
    'primary_energy_consumption', ascending=False).head(10)

plt.figure(figsize=(12, 5))
plt.bar(son_yil['country'], son_yil['primary_energy_consumption'], color='steelblue')
plt.title('2018 Yılında En Çok Enerji Tüketen 10 Ülke')
plt.xlabel('Ülke')
plt.ylabel('Enerji Tüketimi (TWh)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('outputs/grafik2_top10.png')
plt.show()
print("✅ Grafik 2 kaydedildi — Top 10 ülke")

# ════════════════════════════════════════
# GRAFİK 3 — Türkiye Fosil vs Yenilenebilir
# ════════════════════════════════════════
turkey = df[df['country'] == 'Turkey']

plt.figure(figsize=(12, 5))
plt.plot(turkey['year'], turkey['fossil_fuel_consumption'],
         label='Fosil Yakıt', color='red', linewidth=2)
plt.plot(turkey['year'], turkey['renewables_consumption'],
         label='Yenilenebilir', color='green', linewidth=2)
plt.title('Türkiye: Fosil Yakıt vs Yenilenebilir Enerji')
plt.xlabel('Yıl')
plt.ylabel('Enerji Tüketimi (TWh)')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/grafik3_turkiye.png')
plt.show()
print("✅ Grafik 3 kaydedildi — Türkiye fosil vs yenilenebilir")

# ════════════════════════════════════════
# GRAFİK 4 — Model Doğruluğu (Gerçek vs Tahmin)
# ════════════════════════════════════════
turkey = df[df['country'] == 'Turkey'].copy()
X = turkey[['year', 'population', 'gdp', 'fossil_fuel_consumption', 'renewables_consumption']]
y = turkey['primary_energy_consumption']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
tahmin = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, tahmin)
r2  = r2_score(y_test, tahmin)
print(f"\n📊 Model Performansı:")
print(f"   Ortalama Hata (MAE) : {mae:.2f} TWh")
print(f"   Doğruluk Skoru (R²) : %{r2*100:.1f}")

plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Gerçek Değer', color='blue', marker='o')
plt.plot(tahmin,        label='Tahmin',       color='red',  marker='x')
plt.title(f'Gerçek vs Tahmin — Doğruluk: %{r2*100:.1f}')
plt.xlabel('Test Verisi Noktaları')
plt.ylabel('Enerji Tüketimi (TWh)')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/grafik4_tahmin.png')
plt.show()
print("✅ Grafik 4 kaydedildi — Gerçek vs Tahmin")

# ════════════════════════════════════════
# GRAFİK 5 — Gelecek Tahmini (2019-2030)
# ════════════════════════════════════════
X2     = turkey[['year']]
y2     = turkey['primary_energy_consumption']
poly   = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X2)

poly_model = LinearRegression()
poly_model.fit(X_poly, y2)

gelecek_yillar          = pd.DataFrame({'year': range(2019, 2031)})
gelecek_yillar['tahmin'] = poly_model.predict(poly.transform(gelecek_yillar))

gecmis_tahmin = poly_model.predict(poly.transform(turkey[['year']]))

plt.figure(figsize=(12, 5))
plt.plot(turkey['year'], turkey['primary_energy_consumption'],
         label='Gerçek Veri (1965-2018)', color='blue', marker='o', markersize=4)
plt.plot(turkey['year'], gecmis_tahmin,
         label='Model Eğrisi', color='orange', linewidth=2)
plt.plot(gelecek_yillar['year'], gelecek_yillar['tahmin'],
         label='Tahmin (2019-2030)', color='red',
         marker='o', markersize=5, linestyle='--')
plt.axvline(x=2018, color='gray', linestyle=':', label='Tahmin başlangıcı')
plt.title('Türkiye Enerji Tüketimi: Geçmiş + Gelecek Tahmini (2019-2030)')
plt.xlabel('Yıl')
plt.ylabel('Enerji Tüketimi (TWh)')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/grafik5_gelecek.png')
plt.show()
print("✅ Grafik 5 kaydedildi — Gelecek tahmini")

print("\n" + "=" * 50)
print("   PROJE TAMAMLANDI! 🎉")
print("   outputs/ klasörüne 5 grafik kaydedildi.")
print("=" * 50)