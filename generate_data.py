import csv
import random

# Mevcut veri setini oku
with open('data/traffic_chatbot_dataset.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    data = list(reader)

print(f'Mevcut veri sayısı: {len(data)}')

# Veri setini genişletmek için çeşitlemeler ekle
extra_data = []

# Her intent için ekstra varyasyonlar
intents = {
    'greeting': [
        'selam nasılsın', 'hey merhaba', 'günaydın', 'selamlar', 'hello', 'hi there',
        'naber', 'ne haber', 'merhaba bot', 'selam robot', 'hoş bulduk', 'selam canım',
        'merhabalar', 'selamün aleyküm', 'günaydın bot', 'iyi günler arkadaş',
        'hey dostum', 'selam kardeşim', 'merhaba asistan', 'hello bot',
        'selamlar canım', 'merhaba trafik botu', 'hey trafik asistanı'
    ],
    'goodbye': [
        'kendine iyi bak', 'elveda', 'allaha ısmarladık', 'sonra görüşürüz', 'güle güle',
        'iyi akşamlar', 'iyi geceler', 'hoşça kal dostum', 'görüşmek üzere',
        'iyi yolculuklar', 'dikkatli ol', 'güvenli sürüş', 'hoşça kal bot',
        'görüşürüz dostum', 'iyi günler bot', 'elveda arkadaşım', 'sonra konuşuruz'
    ],
    'route_inquiry': [
        'Beylikdüzünden Şişliye nasıl giderim', 'Kadıköyden Beşiktaşa rota',
        'Bakırköyden Üsküdara yol', 'Sisli Maslak arası rota', 'Taksimden İtüye nasıl',
        'Boğaziçi köprüsünden geçer miyim', 'TEM mi E5 mi daha iyi', 'şehir içi rota',
        'Anadolu yakasından Avrupa yakasına', 'Avrupa yakasından Anadolu yakasına',
        'Fatih Sultan Mehmet köprüsü güzergahı', 'Bosphorus köprüsü rotası',
        'Marmaray ile nasıl giderim', 'minibüs rotası', 'otobüs güzergahı'
    ],
    'traffic_prediction': [
        'perşembe akşamı trafik', 'cumartesi sabahı durum', 'bayram günü trafik',
        'okul çıkışı saatlerinde', 'mesai saatlerinde yoğunluk', 'cuma namazı trafik',
        'maç günü trafik durumu', 'konser sonrası trafik', 'festival zamanı',
        'pazartesi sendromu trafik', 'cuma kaçışı', 'hafta sonu trafiği',
        'yağmurlu günde trafik', 'karlı havada durum', 'okullar açık mı'
    ],
    'travel_time': [
        'Anadolu yakasından Avrupa yakasına kaç dakika', 'havalimanına süre',
        'şehir merkezine ne kadar', 'otogar mesafesi', 'gidip gelme süresi',
        'köprü geçiş süresi', 'metro süresi', 'otobüs yolculuğu',
        'taksi ile ne kadar', 'yürüyerek kaç dakika', 'bisikletle süre'
    ],
    'traffic_status': [
        'E5 üzerinde durum', 'TEM otoyolunda trafik', 'Boğaz köprüsü durumu',
        'FSM köprüsü trafiği', 'çevre yolu durumu', 'kennedy caddesi trafik',
        'Mecidiyeköy durumu', 'Maslak trafiği', 'Levent yoğunluğu',
        'Şişli trafiği', 'Beşiktaş durumu', 'Kadıköy trafiği'
    ],
    'alternative_route': [
        'başka yol önerisi', 'alternatif güzergah', 'farklı rota seçeneği',
        'yan yollar', 'dolama yolu', 'kestirme yol', 'trafik olmayan yol',
        'sakin yol', 'hızlı alternatif', 'güvenli rota', 'ekonomik yol'
    ],
    'best_departure_time': [
        'hangi saatte çıkayım', 'en iyi çıkış zamanı', 'optimal saat',
        'ne zaman yola çıkmalı', 'trafik olmayan saat', 'ideal zaman',
        'en uygun çıkış', 'boş saatler', 'yoğunluk olmayan zaman'
    ],
    'parking_info': [
        'park yeri nerede', 'otopark lokasyonu', 'park edebileceğim yer',
        'ücretsiz park', 'ücretli otopark', 'açık park alanı', 'kapalı otopark',
        'valet park', 'park durumu', 'boş park yeri var mı'
    ],
    'fuel_info': [
        'yakın benzinlik', 'akaryakıt istasyonu', 'petrol ofisi nerede',
        'shell istasyonu', 'opet lokasyonu', 'bp benzinlik', 'total istasyon',
        'lpg istasyonu', 'motorin', 'benzin fiyatı'
    ],
    'weather_traffic': [
        'yağmurda trafik', 'karlı havada durum', 'sisli hava trafik',
        'fırtınada yol durumu', 'buzlanma tehlikesi', 'hava durumu etkisi',
        'rüzgarlı hava', 'dolu yağışı', 'aşırı sıcakta trafik'
    ]
}

templates = {
    'greeting': 'Merhaba! Size trafik bilgileri konusunda nasıl yardımcı olabilirim?',
    'goodbye': 'Güvenli yolculuklar dilerim!',
    'route_inquiry': 'Size en uygun rotayı hesaplayabilirim. Detayları belirtir misiniz?',
    'traffic_prediction': 'Trafik tahmini için tarih ve saati belirtir misiniz?',
    'travel_time': 'Süre hesaplaması için rotanızı belirtir misiniz?',
    'traffic_status': 'Hangi güzergahın durumunu kontrol etmek istiyorsunuz?',
    'alternative_route': 'Alternatif rota seçeneklerini gösterebilirim.',
    'best_departure_time': 'En iyi çıkış zamanını hesaplayabilirim.',
    'parking_info': 'Park yeri bilgisi için lokasyon belirtir misiniz?',
    'fuel_info': 'Yakıt istasyonu bilgisi için bölge belirtir misiniz?',
    'weather_traffic': 'Hava durumunun trafik etkisini analiz edebilirim.'
}

# Her intent için ekstra veriler üret
for intent, texts in intents.items():
    for text in texts:
        extra_data.append({
            'intent': intent,
            'text': text,
            'response_template': templates.get(intent, 'Size nasıl yardımcı olabilirim?')
        })

# Rastgele kombinasyonlar ve çeşitlemeler ekle
locations = ['Beylikdüzü', 'Şişli', 'Kadıköy', 'Beşiktaş', 'Üsküdar', 'Bakırköy', 'Maslak', 'Taksim', 'Levent', 'Mecidiyeköy', 'Fatih', 'Eminönü', 'Bostancı', 'Kartal', 'Maltepe']
times = ['sabah', 'akşam', 'öğle', 'gece', 'mesai saati', 'akşam üstü', 'sabah erken', 'gece geç']
days = ['pazartesi', 'salı', 'çarşamba', 'perşembe', 'cuma', 'cumartesi', 'pazar', 'hafta içi', 'hafta sonu']

# Rastgele kombinasyonlarla veri üret
for i in range(300):
    loc1 = random.choice(locations)
    loc2 = random.choice(locations)
    if loc1 != loc2:
        patterns = [
            f'{loc1} {loc2} arası yol',
            f'{loc1}den {loc2}ye nasıl giderim',
            f'{loc1} ile {loc2} arasında rota',
            f'{loc1} çıkışı {loc2} yolu',
            f'{loc1} {loc2} güzergahı'
        ]
        extra_data.append({
            'intent': 'route_inquiry',
            'text': random.choice(patterns),
            'response_template': 'Size en uygun rotayı hesaplayabilirim.'
        })

for i in range(200):
    day = random.choice(days)
    time = random.choice(times)
    patterns = [
        f'{day} {time} trafik durumu',
        f'{day} günü {time} trafik',
        f'{day} {time} yoğunluk',
        f'{day} {time} durum nasıl'
    ]
    extra_data.append({
        'intent': 'traffic_prediction',
        'text': random.choice(patterns),
        'response_template': 'Trafik tahmini için lokasyon belirtir misiniz?'
    })

# Daha fazla çeşitli veri ekle
misc_data = [
    {'intent': 'help', 'text': 'yardım et', 'response_template': 'Size nasıl yardımcı olabilirim?'},
    {'intent': 'help', 'text': 'ne yapabilirsin', 'response_template': 'Trafik bilgileri, rota önerileri ve daha fazlası için buradayım.'},
    {'intent': 'help', 'text': 'özelliklerin neler', 'response_template': 'Trafik tahmini, rota planlama, anlık durum kontrolü yapabilirim.'},
    {'intent': 'help', 'text': 'kullanım kılavuzu', 'response_template': 'Trafik soruları sorabilirsiniz.'},
    {'intent': 'help', 'text': 'nasıl çalışırsın', 'response_template': 'Trafik verilerini analiz ederim.'},
    {'intent': 'thanks', 'text': 'süpersin', 'response_template': 'Teşekkürler! Güvenli yolculuklar.'},
    {'intent': 'thanks', 'text': 'çok yardımcı oldun', 'response_template': 'Memnun oldum! Dikkatli sürün.'},
    {'intent': 'thanks', 'text': 'harikasın', 'response_template': 'Teşekkürler! İyi yolculuklar.'},
    {'intent': 'thanks', 'text': 'mükemmelsin', 'response_template': 'Çok teşekkürler!'},
    {'intent': 'thanks', 'text': 'başarılısın', 'response_template': 'Güvenli sürüşler!'},
    {'intent': 'unclear', 'text': 'ne diyorsun', 'response_template': 'Sorunuzu daha açık belirtir misiniz?'},
    {'intent': 'unclear', 'text': 'anlamıyorum', 'response_template': 'Hangi konuda yardım istiyorsunuz?'},
    {'intent': 'unclear', 'text': 'kafam karıştı', 'response_template': 'Sorunuzu netleştirebilir misiniz?'},
    {'intent': 'unclear', 'text': 'belirsiz', 'response_template': 'Daha açık ifade edebilir misiniz?'},
    {'intent': 'unclear', 'text': 'garip', 'response_template': 'Sorunuzu yeniden sorabilir misiniz?'},
    {'intent': 'complaint', 'text': 'kötü çalışıyorsun', 'response_template': 'Özür dilerim, nasıl daha iyi olabilirim?'},
    {'intent': 'complaint', 'text': 'yanlış bilgi verdin', 'response_template': 'Üzgünüm, doğru bilgiyi vermeye çalışayım.'},
    {'intent': 'complaint', 'text': 'beğenmedim', 'response_template': 'Geri bildiriminiz için teşekkürler.'},
    {'intent': 'complaint', 'text': 'berbat', 'response_template': 'Nasıl gelişebilirim?'},
    {'intent': 'complaint', 'text': 'işe yaramıyorsun', 'response_template': 'Size daha iyi hizmet vermeye çalışırım.'}
]

extra_data.extend(misc_data)

# Yeni veri ile birleştir
all_data = data + extra_data

print(f'Toplam veri sayısı: {len(all_data)}')

# Veriyi karıştır
random.shuffle(all_data)

# Yeni dosyaya yaz
with open('data/traffic_chatbot_dataset.csv', 'w', encoding='utf-8', newline='') as f:
    fieldnames = ['intent', 'text', 'response_template']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    # Sadece geçerli alanları olan verileri yaz
    valid_data = []
    for item in all_data:
        if all(key in item and item[key] is not None for key in fieldnames):
            valid_data.append(item)
    writer.writerows(valid_data)
    all_data = valid_data

print('Veri seti genişletildi ve karıştırıldı!')

# İstatistikleri göster
intent_counts = {}
for item in all_data:
    intent = item['intent']
    intent_counts[intent] = intent_counts.get(intent, 0) + 1

print("\nIntent dağılımı:")
for intent, count in sorted(intent_counts.items()):
    print(f"{intent}: {count}") 