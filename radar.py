import streamlit as st
import cv2
import numpy as np
import time
from collections import defaultdict, deque
from ultralytics import YOLO

st.set_page_config(page_title="Anlık Hız ve Mesafe Tespiti", layout="wide")
st.title("🏇/🚗 Kamera ile Anlık Mesafe ve Hız Tespiti")

# Yan Menü: Kalibrasyon ve Kaynak Ayarları
st.sidebar.header("Kalibrasyon Ayarları")
hedef_tipi = st.sidebar.selectbox("Hedef Türü", ["At Yarışı (At)", "Yol Trafiği (Araç)"])

if hedef_tipi == "At Yarışı (At)":
    hedef_siniflar = [17]  # COCO: horse
    varsayilan_boyut = 1.65  # Ortalama at yüksekliği (metre)
else:
    hedef_siniflar = [2, 5, 7]  # COCO: car, bus, truck
    varsayilan_boyut = 1.50  # Ortalama araç yüksekliği (metre)

gercek_yukseklik = st.sidebar.number_input(
    "Nesnenin Gerçek Yüksekliği (m)", min_value=0.5, max_value=4.0, value=varsayilan_boyut, step=0.05
)
odak_uzakligi = st.sidebar.slider(
    "Kamera Odak Uzaklığı (Piksel)", min_value=300, max_value=2000, value=850, step=10
)
pencere_boyutu = st.sidebar.slider(
    "Hız Yumuşatma Penceresi (Kare Sayısı)", min_value=3, max_value=30, value=8
)

kaynak_tipi = st.sidebar.radio("Görüntü Kaynağı", ["Video Yükle", "Telefon IP Kamera (URL)"])

@st.cache_resource
def model_yukle():
    return YOLO("yolo11n.pt")

model = model_yukle()

def hiz_ve_mesafe_isle(video_kaynagi):
    cap = cv2.VideoCapture(video_kaynagi)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    # Her Track ID için geçmiş (zaman, mesafe, x_merkez_m) kayıtlarını tutar
    gecmis_veriler = defaultdict(lambda: deque(maxlen=pencere_boyutu))

    col1, col2 = st.columns([3, 1])
    goruntu_alani = col1.empty()
    tablo_alani = col2.empty()
    durdur = st.button("İşlemi Durdur")

    while cap.isOpened() and not durdur:
        ret, frame = cap.read()
        if not ret:
            break

        zaman = time.time()
        # ByteTrack ile nesne takibi
        sonuclar = model.track(frame, persist=True, classes=hedef_siniflar, verbose=False)
        anlik_liste = []

        if sonuclar[0].boxes.id is not None:
            kutular = sonuclar[0].boxes.xyxy.cpu().numpy()
            idler = sonuclar[0].boxes.id.int().cpu().numpy()

            for kutu, track_id in zip(kutular, idler):
                x1, y1, x2, y2 = map(int, kutu)
                piksel_yukseklik = max(y2 - y1, 1)
                x_merkez_piksel = (x1 + x2) / 2.0

                # 1. Kameraya olan dikey/radyal uzaklık (Metre)
                mesafe_m = (gercek_yukseklik * odak_uzakligi) / piksel_yukseklik

                # 2. Yanal konumun metre karşılığı (yandan geçen hedefler için)
                metre_piksel_orani = gercek_yukseklik / piksel_yukseklik
                x_merkez_m = x_merkez_piksel * metre_piksel_orani

                gecmis_veriler[track_id].append((zaman, mesafe_m, x_merkez_m))

                # 3. Hız hesabı (Radyal + Yanal bileşke hız)
                hiz_kmh = 0.0
                if len(gecmis_veriler[track_id]) >= 2:
                    t0, d0, x0 = gecmis_veriler[track_id][0]
                    t1, d1, x1 = gecmis_veriler[track_id][-1]
                    dt = t1 - t0 if (t1 - t0) > 0 else (len(gecmis_veriler[track_id]) / fps)

                    delta_d = d1 - d0
                    delta_x = x1 - x0
                    toplam_yer_degistirme = np.sqrt(delta_d**2 + delta_x**2)
                    hiz_ms = toplam_yer_degistirme / dt
                    hiz_kmh = hiz_ms * 3.6

                anlik_liste.append({
                    "ID": f"#{track_id}",
                    "Uzaklık (m)": round(mesafe_m, 1),
                    "Hız (km/s)": round(hiz_kmh, 1)
                })

                # Görüntü üzerine çizim
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                etiket = f"ID:{track_id} | {mesafe_m:.1f}m | {hiz_kmh:.1f} km/s"
                cv2.putText(frame, etiket, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        goruntu_alani.image(frame_rgb, channels="RGB", use_container_width=True)
        if anlik_liste:
            tablo_alani.dataframe(anlik_liste, use_container_width=True)

    cap.release()

if kaynak_tipi == "Video Yükle":
    yuklenen_video = st.file_uploader("Test videosu seçin (MP4, MOV)", type=["mp4", "mov", "avi"])
    if yuklenen_video is not None:
        with open("gecici_video.mp4", "wb") as f:
            f.write(yuklenen_video.read())
        hiz_ve_mesafe_isle("gecici_video.mp4")
else:
    ip_url = st.text_input("IP Kamera URL'si girin", placeholder="http://192.168.1.35:8080/video")
    if ip_url and st.button("Kameraya Bağlan"):
        hiz_ve_mesafe_isle(ip_url)