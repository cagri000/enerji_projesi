# ⚡ Dünya Enerji Tüketimi Analizi ve Tahmini

Bu proje; 1965-2018 yılları arasında 77 ülkeye ait gerçek enerji tüketim verilerini kullanarak örüntüler bulan, anomalileri tespit eden ve **2030 yılına kadar gelecek tahminleri** üreten yapay zeka destekli interaktif bir web uygulamasıdır. **Python ve Streamlit** kullanılarak geliştirilmiştir.

## 🚀 Özellikler (Uygulama Sekmeleri)

Uygulama 4 ana bölümden oluşmaktadır:

* **📚 Genel Bakış (Tab 1):** Nüfus, GSYİH (GDP) ve iklim gibi enerji tüketimini etkileyen temel faktörlerin ve veri setinin özeti.
* **📊 Analiz ve Isı Haritası (Tab 2):** Bölgesel karşılaştırmalar, tüketim yoğunluğunu yıllara göre gösteren ısı haritaları ve en çok tüketen ülkeler listesi.
* **🔮 Tahmin (Tab 3):** Polinomsal Regresyon algoritması kullanılarak 2030 yılına kadar yapılan ülke bazlı enerji tüketimi tahminleri.
* **🤖 Yapay Zeka Modelleri (Tab 4):** 3 farklı modelin (Doğrusal Regresyon, Random Forest, Sinir Ağları) karşılaştırılması, canlı simülasyonlar (What-If), anomali (kriz/savaş dönemi) tespiti ve ülkelerin enerji profillerine göre K-Means ile kümelenmesi.

## 🧠 Kullanılan Yapay Zeka Algoritmaları

| Algoritma | Tür | Amaç |
| :--- | :--- | :--- |
| **Doğrusal Regresyon** | Gözetimli Öğrenme | Temel tahmin ve simülasyon duyarlılığı |
| **Polinomsal Regresyon** | Gözetimli Öğrenme | 2030 yılına kadar zaman serisi trend tahmini |
| **Random Forest** | Topluluk (Ensemble) | Yüksek doğruluklu tahmin ve özellik önemi analizi |
| **Sinir Ağları (MLP)** | Derin Öğrenme | Karmaşık veri örüntülerini tanıma |
| **K-Means** | Gözetimsiz Öğrenme | Ülkeleri tüketim profillerine göre otomatik gruplama |
| **Isolation Forest** | Anomali Tespiti | Savaş/kriz gibi olağandışı tüketim yıllarını yakalama |

## 📁 Proje Yapısı


enerji_projesi/
│
├── data/
│   └── World Energy Consumption.csv   # Veri seti (Kaggle'dan indirilecek)
│
├── uygulama.py                        # Ana Streamlit uygulaması
├── README.md                          # Proje dokümantasyonu
└── requirements.txt                   # Gerekli kütüphaneler
📊 Veri Seti ve Önemli Çıktılar
Kaggle'dan alınan veri seti temizlendikten sonra 3.112 satır ve 77 ülke içermektedir.

💡 En Önemli İçgörüler:

Enerji tüketimi zamanla düzenli artmaktadır (Yıl en önemli faktör).

Fosil yakıt tüketimi, ülkelerin genel enerji profillerini belirleyen en güçlü tahminsel faktördür.

Ülkelerin yaşadığı savaşlar, ekonomik krizler veya radikal politika değişiklikleri, algoritmalar tarafından net birer "anomali" olarak tespit edilmektedir.


🛠️ Kurulum ve Çalıştırma
Projeyi kendi bilgisayarında test etmek istersen şu adımları izleyebilirsin:
```text
1. Repoyu bilgisayarına klonla ve klasöre gir:

Bash
git clone [https://github.com/cagri000/enerji_projesi.git](https://github.com/cagri000/enerji_projesi.git)
cd enerji_projesi
2. Gerekli kütüphaneleri yükle:

Bash
pip install -r requirements.txt
3. Veri setini ekle:

Kaggle üzerinden "World Energy Consumption" veri setini indirin.

Projenin içindeki data/ klasörüne World Energy Consumption.csv adıyla yerleştirin.

4. Uygulamayı başlat:

Bash
streamlit run uygulama.py