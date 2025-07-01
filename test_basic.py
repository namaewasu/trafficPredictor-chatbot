#!/usr/bin/env python3
"""
Temel chatbot test scripti - API kütüphanelerine bağımlı olmadan çalışır
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
import os
from datetime import datetime

class BasicTrafficChatbot:
    def __init__(self):
        """
        Temel trafik chatbot sınıfı - sadece intent classification
        """
        self.vectorizer = None
        self.classifier = None
        self.intents = []
        self.responses = {}
        self.model_trained = False
        self.df = None
    
    def load_data(self, csv_path):
        """Veri setini yükle"""
        try:
            self.df = pd.read_csv(csv_path)
            print(f"✅ Veri seti yüklendi: {len(self.df)} satır")
            
            # Intent'leri ve response'ları hazırla
            self.intents = self.df['intent'].unique().tolist()
            print(f"📊 Toplam intent sayısı: {len(self.intents)}")
            
            # Her intent için response template'lerini kaydet
            for intent in self.intents:
                intent_data = self.df[self.df['intent'] == intent]
                self.responses[intent] = intent_data['response_template'].iloc[0]
            
            # Intent dağılımını göster
            intent_counts = self.df['intent'].value_counts()
            print("\n🎯 Intent Dağılımı:")
            for intent, count in intent_counts.head(10).items():
                print(f"  {intent}: {count}")
            
            return True
        except Exception as e:
            print(f"❌ Veri yükleme hatası: {e}")
            return False
    
    def train_intent_classifier(self):
        """Intent classification modeli eğit"""
        if self.df is None:
            print("❌ Önce veri setini yükleyin!")
            return False
        
        try:
            print("🤖 Model eğitimi başlıyor...")
            
            # Metin ve label'ları hazırla
            X = self.df['text'].values
            y = self.df['intent'].values
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"📚 Eğitim seti: {len(X_train)} örnek")
            print(f"🧪 Test seti: {len(X_test)} örnek")
            
            # TF-IDF Vectorization
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words=None
            )
            
            X_train_tfidf = self.vectorizer.fit_transform(X_train)
            X_test_tfidf = self.vectorizer.transform(X_test)
            
            # Naive Bayes Classifier
            self.classifier = MultinomialNB()
            self.classifier.fit(X_train_tfidf, y_train)
            
            # Test performance
            y_pred = self.classifier.predict(X_test_tfidf)
            
            self.accuracy = accuracy_score(y_test, y_pred)
            self.classification_report = classification_report(y_test, y_pred, output_dict=True)
            self.confusion_matrix = confusion_matrix(y_test, y_pred)
            
            print(f"🎉 Model eğitildi! Accuracy: {self.accuracy:.4f}")
            self.model_trained = True
            
            # Detaylı rapor
            print("\n📊 Performans Metrikleri:")
            print(f"  Accuracy: {self.accuracy:.4f}")
            print(f"  Macro Avg Precision: {self.classification_report['macro avg']['precision']:.4f}")
            print(f"  Macro Avg Recall: {self.classification_report['macro avg']['recall']:.4f}")
            print(f"  Macro Avg F1-Score: {self.classification_report['macro avg']['f1-score']:.4f}")
            
            return {
                'accuracy': self.accuracy,
                'classification_report': self.classification_report,
                'test_size': len(X_test)
            }
            
        except Exception as e:
            print(f"❌ Model eğitimi hatası: {e}")
            return False
    
    def predict_intent(self, text):
        """Metinden intent tahmin et"""
        if not self.model_trained:
            return 'unclear', 0.5
        
        try:
            text_tfidf = self.vectorizer.transform([text])
            predicted_intent = self.classifier.predict(text_tfidf)[0]
            confidence = np.max(self.classifier.predict_proba(text_tfidf))
            
            return predicted_intent, confidence
        except:
            return 'unclear', 0.5
    
    def chat(self, user_input):
        """Ana chat fonksiyonu"""
        # Intent prediction
        intent, confidence = self.predict_intent(user_input)
        
        # Template response
        response = self.responses.get(intent, "Size nasıl yardımcı olabilirim?")
        
        return {
            'response': response,
            'intent': intent,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
    
    def interactive_test(self):
        """İnteraktif test modu"""
        print("\n🚗 Akıllı Trafik Asistanı Test Modu")
        print("Çıkmak için 'quit' yazın\n")
        
        while True:
            user_input = input("👤 Sen: ")
            
            if user_input.lower() in ['quit', 'çıkış', 'exit']:
                print("👋 Görüşmek üzere!")
                break
            
            response = self.chat(user_input)
            print(f"🤖 Bot: {response['response']}")
            print(f"📊 Intent: {response['intent']} (Confidence: {response['confidence']:.2f})")
            print("-" * 50)

def main():
    """Ana test fonksiyonu"""
    print("🚗 Akıllı Trafik Asistanı - Test Scripti")
    print("=" * 50)
    
    # Chatbot oluştur
    chatbot = BasicTrafficChatbot()
    
    # Veri setini yükle
    data_path = 'data/traffic_chatbot_dataset.csv'
    if not os.path.exists(data_path):
        print(f"❌ Veri dosyası bulunamadı: {data_path}")
        return
    
    if not chatbot.load_data(data_path):
        print("❌ Veri yükleme başarısız!")
        return
    
    # Modeli eğit
    results = chatbot.train_intent_classifier()
    if not results:
        print("❌ Model eğitimi başarısız!")
        return
    
    # Otomatik test
    print("\n🧪 Otomatik Test Başlıyor...")
    test_messages = [
        "Merhaba",
        "Yarın sabah trafik nasıl?",
        "Beşiktaş'tan Kadıköy'e nasıl giderim?",
        "Alternatif yol var mı?",
        "E5 üzerinde şu an durum nasıl?",
        "Ne zaman çıkmalıyım?",
        "Park yeri var mı?",
        "Teşekkürler",
        "Hoşça kal"
    ]
    
    for msg in test_messages:
        response = chatbot.chat(msg)
        print(f"\n👤 User: {msg}")
        print(f"🤖 Bot: {response['response']}")
        print(f"📊 Intent: {response['intent']} (Confidence: {response['confidence']:.2f})")
    
    # İnteraktif test
    print("\n" + "="*50)
    try:
        chatbot.interactive_test()
    except KeyboardInterrupt:
        print("\n👋 Test sonlandırıldı!")

if __name__ == "__main__":
    main() 