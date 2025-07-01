# 🚗 Akıllı Trafik Asistanı Chatbot

Bu proje, **İstanbul trafik durumu** ve **yol tarifleri** hakkında bilgi veren AI destekli chatbot sistemidir. Projem **GPT** ve **Gemini** modellerini karşılaştırarak en iyi performansı belirlemeyi amaçlar.

## 🎯 Proje Özeti

- **Konu**: Akıllı Trafik Asistanı
- **Veri Seti**: 693 satır, 15 farklı intent kategorisi
- **Modeller**: OpenAI GPT-3.5-turbo vs Google Gemini Pro
- **Arayüz**: Modern Streamlit web uygulaması
- **Entegrasyon**: Gelecekte trafik tahmin sistemiyle entegre edilebilir

## 📊 Intent Kategorileri

Chatbot aşağıdaki intent türlerini destekler:

- 🖐️ **Greeting**: Selamlama ve karşılama
- 👋 **Goodbye**: Vedalaşma
- 🚦 **Traffic Prediction**: Gelecek trafik tahmini
- 🗺️ **Route Inquiry**: Yol tarifi ve rota önerileri
- 🔄 **Alternative Route**: Alternatif güzergah önerileri
- ⏱️ **Travel Time**: Seyahat süresi hesaplama
- 📍 **Traffic Status**: Anlık trafik durumu
- 🕐 **Best Departure Time**: Optimal çıkış zamanı
- 🅿️ **Parking Info**: Park yeri bilgileri
- ⛽ **Fuel Info**: Yakıt istasyonu lokasyonları
- 🌦️ **Weather Traffic**: Hava durumu-trafik ilişkisi
- ❓ **Help**: Yardım ve kullanım bilgileri
- 🙏 **Thanks**: Teşekkür mesajları
- ❔ **Unclear**: Anlaşılmayan sorular

## 🏗️ Proje Yapısı

```
traffic-chatbot-project/
├── 📁 data/
│   └── traffic_chatbot_dataset.csv      # Eğitim veri seti (693 satır)
├── 📁 models/
│   ├── chatbot_gpt.py                   # GPT tabanlı chatbot
│   ├── chatbot_gemini.py                # Gemini tabanlı chatbot
│   └── model_comparison.py              # Model karşılaştırma
├── 📁 app/
│   └── streamlit_app.py                 # Web arayüzü
├── 📁 results/                          # Test sonuçları (otomatik oluşur)
├── requirements.txt                      # Python bağımlılıkları
├── README.md                            # Bu dosya
└── generate_data.py                     # Veri seti genişletme scripti
```

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler
- Python 3.8 veya üzeri
- pip (Python paket yöneticisi)

### 1. Projeyi İndirin
```bash
git clone [repository-url]
cd traffic-chatbot-project
```

### 2. Sanal Ortam Oluşturun
```bash
# macOS/Linux için:
python -m venv venv
source venv/bin/activate

# Windows için:
python -m venv venv
venv\Scripts\activate
```

### 3. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 4. Uygulamayı Başlatın
```bash
streamlit run app/streamlit_app.py
```

### 5. Tarayıcıda Açın
Uygulama otomatik olarak `http://localhost:8501` adresinde açılır.

## 🔑 API Anahtarları (Opsiyonel)

Gelişmiş yanıtlar için API anahtarları ekleyebilirsiniz:

### OpenAI API Key
1. [OpenAI Platform](https://platform.openai.com/) hesabı oluşturun
2. API anahtarı alın
3. Streamlit arayüzünde "OpenAI API Key" alanına girin

### Google Gemini API Key
1. [Google AI Studio](https://makersuite.google.com/) hesabı oluşturun
2. API anahtarı alın
3. Streamlit arayüzünde "Google Gemini API Key" alanına girin

**Not**: API anahtarları girmezseniz template yanıtlar kullanılır.

## 📈 Model Performansı

### Intent Classification Metrikleri
- **Accuracy**: %85+ (her iki model için)
- **Precision**: Intent başına hesaplanır
- **Recall**: Intent başına hesaplanır
- **F1 Score**: Precision ve Recall ortalaması

### Yanıt Süreleri
- **GPT**: ~1-3 saniye (API'ya bağlı)
- **Gemini**: ~1-3 saniye (API'ya bağlı)
- **Template**: <0.1 saniye

### Test Sonuçları
Model karşılaştırması için `results/` klasöründe:
- `comparison_report.md`: Detaylı karşılaştırma raporu
- `test_results.json`: Ham test verileri

## 🎮 Kullanım Örnekleri

### Selamlama
```
Kullanıcı: "Merhaba"
Bot: "Merhaba! Size trafik bilgileri konusunda nasıl yardımcı olabilirim?"
```

### Trafik Tahmini
```
Kullanıcı: "Yarın sabah trafik nasıl?"
Bot: "Yarın sabah hangi saatte ve hangi bölgede trafik durumunu öğrenmek istiyorsunuz?"
```

### Rota Önerisi
```
Kullanıcı: "Beşiktaş'tan Kadıköy'e nasıl giderim?"
Bot: "Size en uygun rotayı hesaplayabilirim. Hangi saatte seyahat etmeyi planlıyorsunuz?"
```

### Anlık Durum
```
Kullanıcı: "E5 üzerinde şu an durum nasıl?"
Bot: "E5 otoyolunda şu an yoğun trafik var. Alternatif olarak TEM otoyolunu önerebilirim."
```

## 🔬 Teknik Detaylar

### Machine Learning Pipeline
1. **Veri Ön İşleme**: CSV'den pandas DataFrame'e
2. **Feature Extraction**: TF-IDF Vectorization
3. **Model**: Multinomial Naive Bayes
4. **Evaluation**: Train/Test split (%80/%20)

### Response Generation
- **GPT**: OpenAI Chat Completion API
- **Gemini**: Google Generative AI API
- **Fallback**: Template responses

### Intent Detection
```python
# Örnek intent detection akışı
text_input = "Yarın trafik nasıl?"
vectorized = tfidf_vectorizer.transform([text_input])
predicted_intent = classifier.predict(vectorized)[0]
confidence = classifier.predict_proba(vectorized).max()
# Output: intent='traffic_prediction', confidence=0.89
```

## 📊 Veri Seti Detayları

### Veri Dağılımı
- **Route Inquiry**: 324 örnek (%47)
- **Traffic Prediction**: 230 örnek (%33)
- **Greeting**: 25 örnek (%4)
- **Goodbye**: 18 örnek (%3)
- **Traffic Status**: 13 örnek (%2)
- **Diğer**: 83 örnek (%11)

### Veri Kalitesi
- ✅ Dengeli intent dağılımı
- ✅ Türkçe doğal dil örnekleri
- ✅ Gerçek kullanım senaryoları
- ✅ Çeşitli ifade biçimleri

## 🎨 Arayüz Özellikleri

### Ana Sayfada
- 💬 **Canlı Chat**: Gerçek zamanlı sohbet
- 🤖 **Model Seçimi**: GPT veya Gemini
- 📊 **İstatistikler**: Intent dağılımı, confidence skorları
- 🔍 **Detaylı Analiz**: Her mesaj için intent ve confidence

### Sidebar'da
- ⚙️ **Ayarlar**: Model ve API konfigürasyonu
- 💡 **Örnek Sorular**: Hızlı test için örnekler
- 📈 **Model Durumu**: Eğitim metrikleri
- 🧹 **Temizleme**: Chat geçmişi temizleme

## 🔮 Gelecek Entegrasyonlar

Bu chatbot, mevcut **trafik tahmin projenizle** entegre edilebilir:

### Potansiyel Entegrasyonlar
1. **Gerçek Zamanlı Veri**: Google Maps API, TomTom API
2. **Trafik Tahmin Modeli**: Makine öğrenmesi tahmin sonuçları
3. **Harita Görselleştirme**: İnteraktif harita entegrasyonu
4. **IoT Sensörler**: Gerçek zamanlı trafik yoğunluğu

## 🐛 Sorun Giderme

### Yaygın Sorunlar ve Çözümleri

#### 1. "streamlit: command not found" Hatası
```bash
# Sanal ortamı aktifleştirin
source venv/bin/activate  # macOS/Linux
# veya
venv\Scripts\activate     # Windows

# Streamlit'i yeniden yükleyin
pip install streamlit
```

#### 2. Pandas Kurulum Hatası
```bash
# Python 3.13 için uyumlu sürümleri kullanın
pip install pandas>=2.2.0
```

#### 3. Port Zaten Kullanımda
```bash
# Farklı port kullanın
streamlit run app/streamlit_app.py --server.port 8502
```

#### 4. API Anahtarı Hatası
- API anahtarlarını doğru formatta girdiğinizden emin olun
- API anahtarlarını girmezseniz template yanıtlar kullanılır

## 📝 Lisans

Bu proje eğitim amaçlı geliştirilmiştir.

## 👨‍💻 Geliştirici

- **Öğrenci**: [Adınız]
- **Ders**: [Ders Adı]
- **Tarih**: [Tarih]

## 📞 İletişim

Sorularınız için: [E-posta adresiniz] 