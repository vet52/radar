import streamlit as st
import cv2
import numpy as np
import time
from collections import defaultdict, deque
from ultralytics import YOLO

st.set_page_config(page_title="iPhone Pro & Homography Hız Radarı", layout="wide")
st.title("🏇/🚗 Kuşbakışı Perspektif (Homography) ve iPhone Kamera Hız Radarı")

# --- YAN MENÜ: IPHONE DONANIM VE KALİBRASYON AYARLARI ---
st.sidebar.header("1. iPhone Kamera ve Hedef")
hedef_tipi = st.sidebar.selectbox("Hedef Türü", ["At Yarışı (At)", "Yol Trafiği (Araç)"])

lens_secimi = st.sidebar.selectbox(
    "iPhone Aktif Kamera Lensi",
    ["1x Ana Kamera (24mm)", "3x Telefoto (77mm)", "5x Telefoto (120mm)"]
)

# 1080p video akışı için yaklaşık iPhone eşdeğer piksel odak uzaklıkları
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

# Gerçek dünyadaki ölçüm koridorunun metre boyutları (Örn: Pist/Yol genişliği ve uzunluğu)
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
kaynak_tipi = st.sidebar.radio("Görüntü Kaynağı", ["Video Yükle", "iPhone IP/RTSP Kamera"])

@st.cache_resource
def model_yukle():
    return YOLO("yolo11n.pt")

model = model_yukle()

def kusbakisi_radar_ciz(noktalar_dict, genislik_m, uzunluk_m):
    """Hedeflerin kuşbakışı 2D harita üzerindeki anlık konumlarını çizer."""
    radar_w, radar_h = 300, 500
    radar = np.zeros((radar_h, radar_w, 3), dtype=np.uint8)
    radar[:] = (30, 30, 30)

    # Kılavuz çizgileri
    for i in range(1, 5):
        y_cizgi = int(radar_h * (i / 5))
        cv2.line(radar, (0, y_cizgi), (radar_w, y_cizgi), (70, 70, 70), 1)
        mesafe_etiket = f"{uzunluk_m * (1 - i/5):.0f}m"
        cv2.putText(radar, mesafe_etiket, (5, y_cizgi - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    cv2.rectangle(radar, (2, 2), (radar_w - 2, radar_h - 2), (0, 200, 255), 2)

    for track_id, (xm, ym, hiz) in noktalar_dict.items():
        px = int(np.clip((xm / genislik_m) * radar_w, 10, radar_w - 10))
        py = int(np.clip((ym / uzunluk_m) * radar_h, 10, radar_h - 10))
        cv2.circle(radar, (px, py), 8, (0, 255, 0), -1)
        cv2.putText(radar, f"ID:{track_id} ({hiz:.0f})", (px + 10, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return radar

def akis_isle(video_kaynagi):
    cap = cv2.VideoCapture(video_kaynagi)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    # Her ID için: (zaman, optik_mesafe_m, homography_X_m, homography_Y_m)
    gecmis = defaultdict(lambda: deque(maxlen=pencere_boyutu))

    col_video, col_radar, col_tablo = st.columns([3, 1.2, 1.5])
    video_alani = col_video.empty()
    radar_alani = col_radar.empty()
    tablo_alani = col_tablo.empty()
    durdur = st.button("Analizi Durdur")

    while cap.isOpened() and not durdur:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        zaman = time.time()

        # 1. Homography Kaynak (Piksel) ve Hedef (Metre) Noktaları
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

        # Perspektif poligonunu ekrana çiz
        poligon = src_pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [poligon], isClosed=True, color=(0, 200, 255), thickness=2)

        sonuclar = model.track(frame, persist=True, classes=hedef_siniflar, verbose=False)
        anlik_tablo = []
        radar_hedefleri = {}

        if sonuclar[0].boxes.id is not None:
            kutular = sonuclar[0].boxes.xyxy.cpu().numpy()
            idler = sonuclar[0].boxes.id.int().cpu().numpy()

            for kutu, track_id in zip(kutular, idler):
                x1, y1, x2, y2 = map(int, kutu)
                piksel_boy = max(y2 - y1, 1)

                # Nesnenin zemine temas ettiği alt-orta nokta (Ayak / Tekerlek izi)
                ayak_x = (x1 + x2) / 2.0
                ayak_y = float(y2)

                # A) Optik (Pinhole) Mesafe
                optik_mesafe_m = (gercek_yukseklik * odak_uzakligi) / piksel_boy

                # B) Homography (Kuşbakışı Zemin) Koordinatları (Metre)
                nokta_piksel = np.array([[[ayak_x, ayak_y]]], dtype=np.float32)
                nokta_metre = cv2.perspectiveTransform(nokta_piksel, H_matris)[0][0]
                xm, ym = float(nokta_metre[0]), float(nokta_metre[1])

                gecmis[track_id].append((zaman, optik_mesafe_m, xm, ym))

                hiz_kmh = 0.0
                if len(gecmis[track_id]) >= 2:
                    t0, d0, x0, y0 = gecmis[track_id][0]
                    t1, d1, x1_m, y1_m = gecmis[track_id][-1]
                    dt = t1 - t0 if (t1 - t0) > 0 else (len(gecmis[track_id]) / fps)

                    # Kuşbakışı 2D yer değiştirme hızı
                    yer_degistirme_homography = np.sqrt((x1_m - x0)**2 + (y1_m - y0)**2)
                    hiz_homography = (yer_degistirme_homography / dt) * 3.6

                    # Optik boyut değişim hızı
                    hiz_optik = (abs(d1 - d0) / dt) * 3.6

                    if olcum_modu == "Sadece Homography (Kuşbakışı)":
                        hiz_kmh = hiz_homography
                    elif olcum_modu == "Sadece Optik Boyut (Pinhole)":
                        hiz_kmh = hiz_optik
                    else:
                        # Hibrit Füzyon: Homography (%70 ağırlık) + Optik (%30 ağırlık)
                        hiz_kmh = (0.7 * hiz_homography) + (0.3 * hiz_optik)

                radar_hedefleri[track_id] = (xm, ym, hiz_kmh)
                anlik_tablo.append({
                    "ID": f"#{track_id}",
                    "Optik Uzaklık (m)": round(optik_mesafe_m, 1),
                    "Pist Y (m)": round(ym, 1),
                    "Hız (km/s)": round(hiz_kmh, 1)
                })

                # Görsel Çizimler
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (int(ayak_x), int(ayak_y)), 5, (0, 0, 255), -1)
                etiket = f"ID:{track_id} | {optik_mesafe_m:.1f}m | {hiz_kmh:.1f} km/s"
                cv2.putText(frame, etiket, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        video_alani.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        radar_goruntu = kusbakisi_radar_ciz(radar_hedefleri, koridor_genislik_m, koridor_uzunluk_m)
        radar_alani.image(cv2.cvtColor(radar_goruntu, cv2.COLOR_BGR2RGB), caption="2D Kuşbakışı Radar", use_container_width=True)

        if anlik_tablo:
            tablo_alani.dataframe(anlik_tablo, use_container_width=True)

    cap.release()

if kaynak_tipi == "Video Yükle":
    yuklenen = st.file_uploader("Test videosu seçin (MP4, MOV)", type=["mp4", "mov", "avi"])
    if yuklenen is not None:
        with open("gecici_pro.mp4", "wb") as f:
            f.write(yuklenen.read())
        akis_isle("gecici_pro.mp4")
else:
    ip_url = st.text_input("iPhone Akış URL'si", placeholder="rtsp://192.168.1.40:8554/live")
    if ip_url and st.button("Kameraya Bağlan"):
        akis_isle(ip_url)