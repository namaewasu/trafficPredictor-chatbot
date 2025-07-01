import openai
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
import os
from datetime import datetime
import random

class GPTTrafficChatbot:
    def __init__(self, api_key=None):
        """
        GPT tabanlı trafik chatbot sınıfı
        """
        self.api_key = api_key
        if api_key:
            openai.api_key = api_key
        
        self.vectorizer = None
        self.classifier = None
        self.intents = []
        self.responses = {}
        self.model_trained = False
        
        # Trafik tahmini için örnek veriler (gerçek projenizle entegre edilecek)
        self.traffic_data = {
            'heavy': ['E5', 'TEM', 'Boğaz Köprüsü', 'FSM Köprüsü'],
            'moderate': ['Kennedy Caddesi', 'Mecidiyeköy', 'Levent'],
            'light': ['Eminönü', 'Fatih', 'Beşiktaş']
        }
    
    def load_data(self, csv_path):
        """Veri setini yükle"""
        try:
            self.df = pd.read_csv(csv_path)
            print(f"Veri seti yüklendi: {len(self.df)} satır")
            
            # Intent'leri ve response'ları hazırla
            self.intents = self.df['intent'].unique().tolist()
            
            # Her intent için response template'lerini kaydet
            for intent in self.intents:
                intent_data = self.df[self.df['intent'] == intent]
                self.responses[intent] = intent_data['response_template'].iloc[0]
            
            return True
        except Exception as e:
            print(f"Veri yükleme hatası: {e}")
            return False
    
    def train_intent_classifier(self):
        """Intent classification modeli eğit"""
        if self.df is None:
            print("Önce veri setini yükleyin!")
            return False
        
        try:
            # Metin ve label'ları hazırla
            X = self.df['text'].values
            y = self.df['intent'].values
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # TF-IDF Vectorization
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words=None  # Türkçe için custom stop words eklenebilir
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
            
            print(f"Model eğitildi! Accuracy: {self.accuracy:.4f}")
            self.model_trained = True
            
            return {
                'accuracy': self.accuracy,
                'classification_report': self.classification_report,
                'test_size': len(X_test)
            }
            
        except Exception as e:
            print(f"Model eğitimi hatası: {e}")
            return False
    
    def predict_intent(self, text):
        """Metinden intent tahmin et"""
        if not self.model_trained:
            return 'unclear'
        
        try:
            text_tfidf = self.vectorizer.transform([text])
            predicted_intent = self.classifier.predict(text_tfidf)[0]
            confidence = np.max(self.classifier.predict_proba(text_tfidf))
            
            return predicted_intent, confidence
        except:
            return 'unclear', 0.5
    
    def generate_gpt_response(self, user_input, intent, confidence=0.5):
        """GPT ile response üret"""
        if not self.api_key:
            # API key yoksa template response döndür
            return self.responses.get(intent, "Size nasıl yardımcı olabilirim?")
        
        try:
            # Intent'e göre context hazırla
            context = self._get_context_for_intent(intent)
            
            prompt = f"""
            Sen bir trafik asistanı chatbot'usun. Kullanıcının sorusu: "{user_input}"
            Tespit edilen intent: {intent}
            Güven skoru: {confidence:.2f}
            
            Trafik bilgileri context: {context}
            
            Kullanıcıya Türkçe, yardımcı ve samimi bir yanıt ver. Trafik durumu, rota önerileri veya ilgili bilgiler ver.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Sen trafik konusunda uzman, yardımcı bir asistansın."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"GPT API hatası: {e}")
            return self.responses.get(intent, "Size nasıl yardımcı olabilirim?")
    
    def _get_context_for_intent(self, intent):
        """Intent'e göre context bilgisi hazırla"""
        contexts = {
            'traffic_status': f"Şu an yoğun trafik: {', '.join(self.traffic_data['heavy'])}",
            'traffic_prediction': "Trafik tahmini için zaman ve lokasyon gerekli",
            'route_inquiry': "Rota önerisi için başlangıç ve bitiş noktası gerekli",
            'alternative_route': "Alternatif rotalar mevcut trafik durumuna göre öneriliyor"
        }
        return contexts.get(intent, "Genel trafik bilgisi")
    
    def chat(self, user_input):
        """Ana chat fonksiyonu"""
        # Intent prediction
        intent, confidence = self.predict_intent(user_input)
        
        # GPT ile response generation
        response = self.generate_gpt_response(user_input, intent, confidence)
        
        return {
            'response': response,
            'intent': intent,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_model(self, path):
        """Modeli kaydet"""
        if not self.model_trained:
            print("Model henüz eğitilmedi!")
            return False
        
        try:
            model_data = {
                'vectorizer': self.vectorizer,
                'classifier': self.classifier,
                'intents': self.intents,
                'responses': self.responses,
                'accuracy': self.accuracy,
                'classification_report': self.classification_report
            }
            
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"Model kaydedildi: {path}")
            return True
        except Exception as e:
            print(f"Model kaydetme hatası: {e}")
            return False
    
    def load_model(self, path):
        """Modeli yükle"""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.vectorizer = model_data['vectorizer']
            self.classifier = model_data['classifier']
            self.intents = model_data['intents']
            self.responses = model_data['responses']
            self.accuracy = model_data['accuracy']
            self.classification_report = model_data['classification_report']
            self.model_trained = True
            
            print(f"Model yüklendi: {path}")
            return True
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            return False

# Test için örnek kullanım
if __name__ == "__main__":
    # Chatbot oluştur
    chatbot = GPTTrafficChatbot()
    
    # Veri setini yükle
    if chatbot.load_data('../data/traffic_chatbot_dataset.csv'):
        # Modeli eğit
        results = chatbot.train_intent_classifier()
        if results:
            print(f"Model Performance:")
            print(f"Accuracy: {results['accuracy']:.4f}")
            print(f"Test size: {results['test_size']}")
            
            # Test
            test_messages = [
                "Merhaba",
                "Yarın sabah trafik nasıl?",
                "Beşiktaş'tan Kadıköy'e nasıl giderim?",
                "Alternatif yol var mı?",
                "Teşekkürler"
            ]
            
            print("\nTest Messages:")
            for msg in test_messages:
                response = chatbot.chat(msg)
                print(f"User: {msg}")
                print(f"Bot: {response['response']}")
                print(f"Intent: {response['intent']} (Confidence: {response['confidence']:.2f})")
                print("-" * 50) 