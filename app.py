import time
import atexit
import threading
from collections import deque
from flask import Flask, jsonify, send_file, request
from tensorflow.keras.models import load_model
from eeg_stream import EEGStream
import numpy as np

app = Flask(__name__)

# ── EEG CONNECTION ────────────────────────────────────────────────────────────
eeg_stream = EEGStream(
    client_id='YOUR_CLIENT_ID',
    client_secret='YOUR_CLIENT_SECRET',
    # headset_id="EPOCX-XXXXXXXX",  # optional
    debug=False,
)

# ── MODEL LOAD ────────────────────────────────────────────────────────────────
print("=" * 55)
print("  NöroART — Model yükleniyor...")
model = load_model("neuro_art_regression_tanh.keras")
print("  Model hazir!")
print("=" * 55)

# ── THREAD & DATA STATES ──────────────────────────────────────────────────────
data_lock = threading.Lock()
dummy_mode = False

# Son 5 saniyenin V ve A değerlerini tutan kayan pencere (Eski veriyi otomatik siler)
recent_raw_v = deque(maxlen=5)
recent_raw_a = deque(maxlen=5)

# Baseline Değişkenleri
baseline_v         = None
baseline_a         = None
baseline_recording = False
baseline_v_list    = []
baseline_a_list    = []

# ── CONTINUOUS PROCESSING THREAD ──────────────────────────────────────────────
def process_eeg_continuously():
    """
    Arka planda saniye saniye sürekli çalışarak EEG tamponunu (buffer) temizler,
    modeli çalıştırır ve son 5 saniyelik ortalama için verileri hazırlar.
    """
    global dummy_mode
    print("  [Arka Plan] EEG İzleme Motoru Devrede (1Hz)...")
    while True:
        # 1 saniyelik veri dolana kadar bekler
        if dummy_mode:
            eeg = np.random.randn(128, 14)
            time.sleep(1)
        else:
            eeg = eeg_stream.get_window(window_size=128)
        if eeg is None:
            continue

        # Normalizasyon
        eeg_norm = (eeg - np.mean(eeg, axis=0)) / (np.std(eeg, axis=0) + 1e-8)
        eeg_ready = eeg_norm.reshape(1, 128, 14)
        
        # Model Tahmini
        pred = model.predict(eeg_ready, verbose=0)
        v = float(pred[0][0])
        a = float(pred[0][1])

        # Verileri güvenli bir şekilde listelere yazdır
        with data_lock:
            recent_raw_v.append(v)
            recent_raw_a.append(a)

            # Eğer o an kalibrasyon yapılıyorsa, kalibrasyon listesine de ekle
            if baseline_recording:
                baseline_v_list.append(v)
                baseline_a_list.append(a)


# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return send_file("code.html")


@app.route("/baseline", methods=["POST"])
def record_baseline():
    """
    Arka plan işçisinin topladığı verileri dinleyerek kalibrasyonu tamamlar.
    """
    global baseline_v, baseline_a, baseline_recording, baseline_v_list, baseline_a_list

    duration = int(request.args.get("duration", 15))

    with data_lock:
        if baseline_recording:
            return jsonify({"error": "Kalibrasyon zaten devam ediyor."}), 409
        baseline_recording = True
        baseline_v_list = []
        baseline_a_list = []

    print("\n" + "=" * 55)
    print(f"  KALİBRASYON BAŞLADI — {duration} saniye")
    print(f"  Lütfen hareketsiz durun ve gözlerinizi kapatın.")
    print("=" * 55, flush=True)

    # İşçinin (thread) veriyi doldurmasını bekle
    start_time = time.time()
    while time.time() - start_time < duration:
        time.sleep(1) # Saniyede bir ekrana log basmak için
        with data_lock:
            count = len(baseline_v_list)
        print(f"  Kalibre ediliyor: {count}/{duration} saniye tamamlandı...", flush=True)

    # Süre doldu, ortalamayı al
    with data_lock:
        baseline_recording = False
        if len(baseline_v_list) > 0:
            baseline_v = sum(baseline_v_list) / len(baseline_v_list)
            baseline_a = sum(baseline_a_list) / len(baseline_a_list)
        else:
            baseline_v, baseline_a = 0.0, 0.0

    print("=" * 55)
    print(f"  KALİBRASYON TAMAMLANDI!")
    print(f"  Kişisel Nötr Nokta  ->  V: {baseline_v:+.4f}  |  A: {baseline_a:+.4f}")
    print("=" * 55 + "\n", flush=True)

    return jsonify({
        "status":     "ok",
        "duration_sec": duration,
        "baseline_v": baseline_v,
        "baseline_a": baseline_a,
    })


@app.route("/baseline", methods=["GET"])
def get_baseline_status():
    with data_lock:
        if baseline_v is None or baseline_a is None:
            return jsonify({"status": "not_recorded"})
        return jsonify({
            "status":     "recorded",
            "baseline_v": baseline_v,
            "baseline_a": baseline_a,
        })


@app.route("/baseline/clear", methods=["POST"])
def clear_baseline():
    global baseline_v, baseline_a
    with data_lock:
        baseline_v = None
        baseline_a = None
    print("[Baseline] Sifirlandi.")
    return jsonify({"status": "cleared"})


@app.route("/predict")
def predict():
    """
    Arka planda toplanan SON 5 SANİYENİN ortalamasını anında verir.
    """
    with data_lock:
        # Eğer sistem yeni açıldıysa ve 5 saniye dolmadıysa
        if len(recent_raw_v) == 0:
            return jsonify({"error": "Sistem analiz yapiyor, lutfen bikac saniye bekleyin..."})

        # Son 5 saniyenin ham ortalaması alınır
        avg_raw_v = sum(recent_raw_v) / len(recent_raw_v)
        avg_raw_a = sum(recent_raw_a) / len(recent_raw_a)

        bv = baseline_v
        ba = baseline_a

    # Baseline (Tare) Uygulaması
    if bv is not None and ba is not None:
        
        # --- VALENCE ÖLÇEKLEME ---
        if avg_raw_v >= bv:
            # Pozitif bölgedeyse: Kalan dar tavan boşluğunu (1.0 - bv) genişlet
            final_v = (avg_raw_v - bv) / (1.0 - bv + 1e-8) 
        else:
            # Negatif bölgedeyse: Kalan geniş taban boşluğunu (bv + 1.0) daralt
            final_v = (avg_raw_v - bv) / (bv + 1.0 + 1e-8) 
            
        # --- AROUSAL ÖLÇEKLEME ---
        if avg_raw_a >= ba:
            final_a = (avg_raw_a - ba) / (1.0 - ba + 1e-8)
        else:
            final_a = (avg_raw_a - ba) / (ba + 1.0 + 1e-8)

        # İşlem sonrası olası taşmaları kesin olarak engelle (clipping)
        final_v = max(-1.0, min(1.0, final_v))
        final_a = max(-1.0, min(1.0, final_a))
    else:
        final_v, final_a = avg_raw_v, avg_raw_a

    return jsonify({
        "valence":          final_v,
        "arousal":          final_a,
        "raw_valence":      avg_raw_v,
        "raw_arousal":      avg_raw_a,
        "baseline_applied": bv is not None,
        "samples_averaged": len(recent_raw_v) # Ön yüzde kaç saniyenin ortalaması alındığını görmek istersen
    })


# ── SHUTDOWN ──────────────────────────────────────────────────────────────────
@atexit.register
def _on_exit():
    print("\n[NöroART] Kapatiliyor — Cortex baglantisi sonlandiriliyor...")
    eeg_stream.disconnect()


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  NöroART Sistemi Baslatiliyor...")
    try:
        eeg_stream.connect(timeout=5)
    except Exception as e:
        print("  [!] EEG başlığı bağlanamadı. Dummy (simülasyon) modunda devam ediliyor...")
        dummy_mode = True
    
    # Arka plan işçisini (Thread) başlat
    processing_thread = threading.Thread(target=process_eeg_continuously, daemon=True)
    processing_thread.start()
    
    print("\n  Flask Sunucusu Hazir: http://127.0.0.1:5000n")
    app.run(debug=False)