import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import time
import json
from datetime import datetime

from chatbot_gpt import GPTTrafficChatbot
from chatbot_gemini import GeminiTrafficChatbot

class ModelComparison:
    def __init__(self):
        self.gpt_bot = GPTTrafficChatbot()
        self.gemini_bot = GeminiTrafficChatbot()
        self.test_results = {}
        
    def setup_models(self, data_path, gpt_api_key=None, gemini_api_key=None):
        """Modelleri hazırla"""
        print("🤖 Modelleri hazırlıyor...")
        
        # GPT Model
        if gpt_api_key:
            self.gpt_bot.api_key = gpt_api_key
        
        success_gpt = self.gpt_bot.load_data(data_path)
        if success_gpt:
            self.gpt_results = self.gpt_bot.train_intent_classifier()
            print(f"✅ GPT Model hazır - Accuracy: {self.gpt_results['accuracy']:.4f}")
        
        # Gemini Model  
        if gemini_api_key:
            self.gemini_bot.api_key = gemini_api_key
            
        success_gemini = self.gemini_bot.load_data(data_path)
        if success_gemini:
            self.gemini_results = self.gemini_bot.train_intent_classifier()
            print(f"✅ Gemini Model hazır - Accuracy: {self.gemini_results['accuracy']:.4f}")
            
        return success_gpt and success_gemini
    
    def create_test_dataset(self):
        """Test için örnekler oluştur"""
        test_messages = [
            # Greeting
            "Merhaba", "Selam bot", "İyi günler",
            
            # Traffic Prediction
            "Yarın sabah trafik nasıl?", "Bu akşam yoğunluk var mı?", "Pazartesi sabahı durum",
            
            # Route Inquiry
            "Beşiktaş'tan Kadıköy'e nasıl giderim?", "En iyi rota hangisi?", "Sisli Maslak arası yol",
            
            # Alternative Route
            "Alternatif yol var mı?", "Başka güzergah öner", "Farklı rota",
            
            # Travel Time
            "Ne kadar sürer?", "Kaç dakika?", "Gidiş süresi",
            
            # Traffic Status
            "Şu an trafik durumu", "Canlı durum", "E5 üzerinde nasıl?",
            
            # Best Departure Time
            "Ne zaman çıkmalı?", "En iyi saat", "Hangi saatte",
            
            # Parking Info
            "Park yeri var mı?", "Otopark nerede?", "Park edebilirim",
            
            # Weather Traffic
            "Yağmurda trafik", "Kar etkisi", "Hava durumu",
            
            # Goodbye
            "Teşekkürler", "Hoşça kal", "Güle güle",
            
            # Help
            "Yardım", "Ne yapabilirsin?", "Nasıl kullanırım?",
            
            # Unclear
            "Anlamadım", "Ne demek istiyorsun?", "Karışık"
        ]
        
        return test_messages
    
    def run_performance_test(self):
        """Performans testi çalıştır"""
        print("📊 Performans testi başlayıyor...")
        
        test_messages = self.create_test_dataset()
        
        gpt_times = []
        gemini_times = []
        gpt_responses = []
        gemini_responses = []
        
        for msg in test_messages:
            # GPT Test
            start_time = time.time()
            gpt_response = self.gpt_bot.chat(msg)
            gpt_time = time.time() - start_time
            gpt_times.append(gpt_time)
            gpt_responses.append(gpt_response)
            
            # Gemini Test
            start_time = time.time()
            gemini_response = self.gemini_bot.chat(msg)
            gemini_time = time.time() - start_time
            gemini_times.append(gemini_time)
            gemini_responses.append(gemini_response)
            
            time.sleep(0.1)  # API rate limit için
        
        # Sonuçları sakla
        self.test_results = {
            'messages': test_messages,
            'gpt_responses': gpt_responses,
            'gemini_responses': gemini_responses,
            'gpt_times': gpt_times,
            'gemini_times': gemini_times,
            'gpt_accuracy': self.gpt_results['accuracy'],
            'gemini_accuracy': self.gemini_results['accuracy']
        }
        
        return self.test_results
    
    def calculate_detailed_metrics(self):
        """Detaylı metrikler hesapla"""
        if not self.test_results:
            print("Önce performans testi çalıştırın!")
            return None
        
        # Intent accuracy karşılaştırması
        gpt_intents = [r['intent'] for r in self.test_results['gpt_responses']]
        gemini_intents = [r['intent'] for r in self.test_results['gemini_responses']]
        
        # Confidence scores
        gpt_confidences = [r['confidence'] for r in self.test_results['gpt_responses']]
        gemini_confidences = [r['confidence'] for r in self.test_results['gemini_responses']]
        
        # Response times
        gpt_avg_time = np.mean(self.test_results['gpt_times'])
        gemini_avg_time = np.mean(self.test_results['gemini_times'])
        
        # Intent distribution
        gpt_intent_dist = pd.Series(gpt_intents).value_counts()
        gemini_intent_dist = pd.Series(gemini_intents).value_counts()
        
        metrics = {
            'model_accuracy': {
                'GPT': self.test_results['gpt_accuracy'],
                'Gemini': self.test_results['gemini_accuracy']
            },
            'response_time': {
                'GPT_avg': gpt_avg_time,
                'Gemini_avg': gemini_avg_time,
                'GPT_total': sum(self.test_results['gpt_times']),
                'Gemini_total': sum(self.test_results['gemini_times'])
            },
            'confidence_scores': {
                'GPT_avg': np.mean(gpt_confidences),
                'Gemini_avg': np.mean(gemini_confidences),
                'GPT_std': np.std(gpt_confidences),
                'Gemini_std': np.std(gemini_confidences)
            },
            'intent_distribution': {
                'GPT': gpt_intent_dist.to_dict(),
                'Gemini': gemini_intent_dist.to_dict()
            }
        }
        
        return metrics
    
    def create_comparison_report(self):
        """Karşılaştırma raporu oluştur"""
        if not self.test_results:
            print("Önce testleri çalıştırın!")
            return None
        
        metrics = self.calculate_detailed_metrics()
        
        report = f"""
# 🤖 Trafik Chatbot Model Karşılaştırması

## 📊 Genel Performans Metrikleri

### Model Accuracy
- **GPT**: {metrics['model_accuracy']['GPT']:.4f}
- **Gemini**: {metrics['model_accuracy']['Gemini']:.4f}

### Yanıt Süreleri (saniye)
- **GPT Ortalama**: {metrics['response_time']['GPT_avg']:.3f}s
- **Gemini Ortalama**: {metrics['response_time']['Gemini_avg']:.3f}s

### Güven Skorları
- **GPT Ortalama**: {metrics['confidence_scores']['GPT_avg']:.3f} ± {metrics['confidence_scores']['GPT_std']:.3f}
- **Gemini Ortalama**: {metrics['confidence_scores']['Gemini_avg']:.3f} ± {metrics['confidence_scores']['Gemini_std']:.3f}

## 🎯 Detaylı Analiz

### Intent Tespit Dağılımı
**GPT Model:**
"""
        
        for intent, count in metrics['intent_distribution']['GPT'].items():
            report += f"- {intent}: {count}\n"
        
        report += "\n**Gemini Model:**\n"
        for intent, count in metrics['intent_distribution']['Gemini'].items():
            report += f"- {intent}: {count}\n"
        
        report += f"""

## 📋 Test Sonuçları Özeti

- **Toplam Test Mesajı**: {len(self.test_results['messages'])}
- **GPT Toplam Süre**: {metrics['response_time']['GPT_total']:.2f}s
- **Gemini Toplam Süre**: {metrics['response_time']['Gemini_total']:.2f}s

## 🏆 Sonuç ve Öneriler

"""
        
        # Winner belirleme
        if metrics['model_accuracy']['GPT'] > metrics['model_accuracy']['Gemini']:
            report += "- **Accuracy Kazananı**: GPT\n"
        else:
            report += "- **Accuracy Kazananı**: Gemini\n"
        
        if metrics['response_time']['GPT_avg'] < metrics['response_time']['Gemini_avg']:
            report += "- **Hız Kazananı**: GPT\n"
        else:
            report += "- **Hız Kazananı**: Gemini\n"
        
        if metrics['confidence_scores']['GPT_avg'] > metrics['confidence_scores']['Gemini_avg']:
            report += "- **Güven Skoru Kazananı**: GPT\n"
        else:
            report += "- **Güven Skoru Kazananı**: Gemini\n"
        
        report += f"""

## 📈 Model Seçim Önerisi

Her iki model de trafik chatbot görevi için uygun performans sergiliyor. 
Seçim kriterleri:

- **Accuracy öncelikli**: {"GPT" if metrics['model_accuracy']['GPT'] > metrics['model_accuracy']['Gemini'] else "Gemini"}
- **Hız öncelikli**: {"GPT" if metrics['response_time']['GPT_avg'] < metrics['response_time']['Gemini_avg'] else "Gemini"}
- **Maliyet öncelikli**: API maliyetleri karşılaştırılmalı

Rapor Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return report, metrics
    
    def save_results(self, output_dir="results"):
        """Sonuçları kaydet"""
        import os
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Test sonuçlarını kaydet
        with open(f"{output_dir}/test_results.json", 'w', encoding='utf-8') as f:
            # JSON serializable hale getir
            serializable_results = {
                'messages': self.test_results['messages'],
                'gpt_accuracy': float(self.test_results['gpt_accuracy']),
                'gemini_accuracy': float(self.test_results['gemini_accuracy']),
                'gpt_times': [float(t) for t in self.test_results['gpt_times']],
                'gemini_times': [float(t) for t in self.test_results['gemini_times']],
                'timestamp': datetime.now().isoformat()
            }
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        # Rapor kaydet
        report, metrics = self.create_comparison_report()
        with open(f"{output_dir}/comparison_report.md", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ Sonuçlar kaydedildi: {output_dir}/")
        return True

# Test için örnek kullanım
if __name__ == "__main__":
    # Model karşılaştırması oluştur
    comparison = ModelComparison()
    
    # Modelleri hazırla (API anahtarları isteğe bağlı)
    if comparison.setup_models('../data/traffic_chatbot_dataset.csv'):
        print("🚀 Testler başlıyor...")
        
        # Performans testi
        comparison.run_performance_test()
        
        # Rapor oluştur
        report, metrics = comparison.create_comparison_report()
        print(report)
        
        # Sonuçları kaydet
        comparison.save_results()
    else:
        print("❌ Model hazırlama başarısız!") 