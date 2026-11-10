import streamlit as st
import cv2
import numpy as np
import time
import av
from collections import defaultdict, deque
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.set_page_config(page_title="iPhone WebRTC Hız Radarı", layout="wide")
st.title("🏇/🚗 iPhone Safari (WebRTC) & Homography Hız Radarı")

# --- YAN MENÜ: IPHONE DONANIM VE KALİBRASYON AYARLARI ---
st.sidebar.header("1. iPhone Kamera ve Hedef")
kaynak_tipi = st.sidebar.radio(
    "Görüntü Kaynağı",
    ["iPhone Canlı Kamera (Safari WebRTC)", "Video Yükle"]
)

hedef_tipi = st.sidebar.selectbox("Hedef Türü", ["At Yarışı (At)", "Yol Trafiği (Araç)"])

lens_secimi = st.sidebar.selectbox(
    "iPhone Aktif Kamera Lensi",
    ["1x Ana Kamera (24mm)", "3x Telefoto (77mm)", "5x Telefoto (120mm)"]
)

lens_odak_haritasi = {
    "1x Ana Kamera (24mm)": 1050,
    "3x Telefoto (77mm)": 3350,
    "5x Telefoto (120mm)": 5250
}

if hedef_tipi == "At Yarışı (At)":
    hedef_siniflar = [17]
    varsayilan_boyut = 1.65
else:
    hedef_siniflar = [2, 5, 7]
    varsayilan_boyut = 1.50

gercek_yukseklik = st.sidebar.number_input("Hedef Yüksekliği (m)", 0.5, 4.0, varsayilan_boyut, 0.05)
odak_uzakligi = st.sidebar.slider("Odak Uzaklığı (Piksel)", 500, 6000, lens_odak_haritasi[lens_secimi], 50)

st.sidebar.header("2. Homography (Kuşbakışı Bölge)")
olcum_modu = st.sidebar.radio(
    "Hız Hesaplama Yöntemi",
    ["Hibrit (Homography + Optik Boyut)", "Sadece Homography (Kuşbakışı)", "Sadece Optik Boyut (Pinhole)"]
)

koridor_genislik_m = st.sidebar.number_input("Seçili Alanın Gerçek Genişliği (m)", 2.0, 50.0, 10.0, 0.5)
koridor_uzunluk_m = st.sidebar.number_input("Seçili Alanın Gerçek Uzunluğu (m)", 5.0, 200.0, 30.0, 1.0)

st.sidebar.caption("Görüntü üzerindeki 4 köşe noktasını (% oranında) ayarlayın:")
ust_sol_x = st.sidebar.slider("Üst Sol X (%)", 0, 100, 30)
ust_sag_x = st.sidebar.slider("Üst Sağ X (%)", 0, 100, 70)
ust_y = st.sidebar.slider("Üst Çizgi Y (%)", 0, 100, 40)
alt_sol_x = st.sidebar.slider("Alt Sol X (%)", 0, 100, 10)
alt_sag_x = st.sidebar.slider("Alt Sağ X (%)", 0, 100, 90)
alt_y = st.sidebar.slider("Alt Çizgi Y (%)", 0, 100, 90)

pencere_boyutu = st.sidebar.slider("Yumuşatma Penceresi (Kare)", 3, 30, 8)

@st.cache_resource
def model_yukle():
    return YOLO("yolo11n.pt")

model = model_yukle()

# WebRTC STUN Sunucu Yapılandırması (Bulut bağlantısı için gerekli)
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

# Takip geçmişi sözlüğü
gecmis_veriler = defaultdict(lambda: deque(maxlen=30))

def kusbakisi_radar_olustur(noktalar_dict, genislik_m, uzunluk_m, radar_w=180, radar_h=280):
    """Kuşbakışı 2D haritayı oluşturur (Video üstüne PiP olarak eklenmek üzere)."""
    radar = np.zeros((radar_h, radar_w, 3), dtype=np.uint8)
    radar[:] = (25, 25, 25)

    for i in range(1, 5):
        y_cizgi = int(radar_h * (i / 5))
        cv2.line(radar, (0, y_cizgi), (radar_w, y_cizgi), (70, 70, 70), 1)
        mesafe_etiket = f"{uzunluk_m * (1 - i/5):.0f}m"
        cv2.putText(radar, mesafe_etiket, (5, y_cizgi - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)

    cv2.rectangle(radar, (1, 1), (radar_w - 2, radar_h - 2), (0, 200, 255), 2)

    for track_id, (xm, ym, hiz) in noktalar_dict.items():
        px = int(np.clip((xm / genislik_m) * radar_w, 8, radar_w - 8))
        py = int(np.clip((ym / uzunluk_m) * radar_h, 8, radar_h - 8))
        cv2.circle(radar, (px, py), 6, (0, 255, 0), -1)
        cv2.putText(radar, f"#{track_id} {hiz:.0f}km/s", (max(px - 25, 2), max(py - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    return radar

def kare_isle(frame, fps_tahmini=30.0):
    """Tek bir video karesi üzerinde YOLO, Homography ve Hız hesabını çalıştırır."""
    h, w = frame.shape[:2]
    zaman = time.time()

    src_pts = np.float32([
        [w * ust_sol_x / 100, h * ust_y / 100],
        [w * ust_sag_x / 100, h * ust_y / 100],
        [w * alt_sag_x / 100, h * alt_y / 100],
        [w * alt_sol_x / 100, h * alt_y / 100]
    ])
    dst_pts = np.float32([
        [0, 0],
        [koridor_genislik_m, 0],
        [koridor_genislik_m, koridor_uzunluk_m],
        [0, koridor_uzunluk_m]
    ])
    H_matris = cv2.getPerspectiveTransform(src_pts, dst_pts)

    poligon = src_pts.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [poligon], isClosed=True, color=(0, 200, 255), thickness=2)

    sonuclar = model.track(frame, persist=True, classes=hedef_siniflar, verbose=False)
    radar_hedefleri = {}

    if sonuclar[0].boxes.id is not None:
        kutular = sonuclar[0].boxes.xyxy.cpu().numpy()
        idler = sonuclar[0].boxes.id.int().cpu().numpy()

        for kutu, track_id in zip(kutular, idler):
            x1, y1, x2, y2 = map(int, kutu)
            piksel_boy = max(y2 - y1, 1)

            ayak_x = (x1 + x2) / 2.0
            ayak_y = float(y2)

            optik_mesafe_m = (gercek_yukseklik * odak_uzakligi) / piksel_boy

            nokta_piksel = np.array([[[ayak_x, ayak_y]]], dtype=np.float32)
            nokta_metre = cv2.perspectiveTransform(nokta_piksel, H_matris)[0][0]
            xm, ym = float(nokta_metre[0]), float(nokta_metre[1])

            iz_gecmisi = gecmis_veriler[track_id]
            iz_gecmisi.append((zaman, optik_mesafe_m, xm, ym))
            while len(iz_gecmisi) > pencere_boyutu:
                iz_gecmisi.popleft()

            hiz_kmh = 0.0
            if len(iz_gecmisi) >= 2:
                t0, d0, x0, y0 = iz_gecmisi[0]
                t1, d1, x1_m, y1_m = iz_gecmisi[-1]
                dt = t1 - t0 if (t1 - t0) > 0.001 else (len(iz_gecmisi) / fps_tahmini)

                yer_degistirme_homography = np.sqrt((x1_m - x0)**2 + (y1_m - y0)**2)
                hiz_homography = (yer_degistirme_homography / dt) * 3.6
                hiz_optik = (abs(d1 - d0) / dt) * 3.6

                if olcum_modu == "Sadece Homography (Kuşbakışı)":
                    hiz_kmh = hiz_homography
                elif olcum_modu == "Sadece Optik Boyut (Pinhole)":
                    hiz_kmh = hiz_optik
                else:
                    hiz_kmh = (0.7 * hiz_homography) + (0.3 * hiz_optik)

            radar_hedefleri[track_id] = (xm, ym, hiz_kmh)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (int(ayak_x), int(ayak_y)), 5, (0, 0, 255), -1)
            etiket = f"ID:{track_id} | {optik_mesafe_m:.1f}m | {hiz_kmh:.1f} km/s"
            cv2.putText(frame, etiket, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Kuşbakışı Radarı sağ üst köşeye (PiP) yerleştir
    radar_img = kusbakisi_radar_olustur(radar_hedefleri, koridor_genislik_m, koridor_uzunluk_m)
    rh, rw = radar_img.shape[:2]
    if h > rh + 20 and w > rw + 20:
        frame[10:10+rh, w-rw-10:w-10] = radar_img

    return frame

# --- 1. MOD: IPHONE SAFARI WEBRTC CANLI AKIŞ ---
if kaynak_tipi == "iPhone Canlı Kamera (Safari WebRTC)":
    st.info("📱 iPhone Safari'de **START** butonuna basın. Kamera izni verdikten sonra sağ alt köşedeki **Select Device** menüsünden **Back Telephoto Camera (3x/5x)** veya **Back Camera** lensini seçebilirsiniz.")

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        islenmis_img = kare_isle(img)
        return av.VideoFrame.from_ndarray(islenmis_img, format="bgr24")

    webrtc_streamer(
        key="iphone-hiz-radari",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {
                "facingMode": "environment",  # Varsayılan olarak iPhone arka kamerasını açar
                "width": {"ideal": 1280},
                "height": {"ideal": 720}
            },
            "audio": False
        },
        async_processing=True
    )

# --- 2. MOD: VİDEO DOSYASI YÜKLEME ---
else:
    yuklenen = st.file_uploader("Test videosu seçin (MP4, MOV)", type=["mp4", "mov", "avi"])
    if yuklenen is not None:
        with open("gecici_pro.mp4", "wb") as f:
            f.write(yuklenen.read())

        cap = cv2.VideoCapture("gecici_pro.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        video_alani = st.empty()
        durdur = st.button("Analizi Durdur")

        while cap.isOpened() and not durdur:
            ret, frame = cap.read()
            if not ret:
                break
            islenmis = kare_isle(frame, fps_tahmini=fps)
            video_alani.image(cv2.cvtColor(islenmis, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        cap.release()