import os
import cv2
import time
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO

# ============================================================
# 1) KONFIGURASI STREAM RTSP
# ============================================================
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
RTSP_URL = "rtsp://admin:Pemkot2024@192.168.0.10:554/Streaming/Channels/102"

# ============================================================
# 2) YOLO
# ============================================================
model = YOLO("yolov8n.pt")
class_names = model.names
DEVICE = 0 if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Running on {DEVICE}")

# Kelas kendaraan
vehicle_keywords = {"car", "motorcycle", "truck", "bicycle", "bus"}
vehicle_classes_to_count = [cid for cid, name in class_names.items()
                            if name.lower() in vehicle_keywords]

# ============================================================
# 3) PARAMETER VIDEO & DASHBOARD
# ============================================================
PROCESS_WIDTH, PROCESS_HEIGHT = 640, 360
FRAME_SKIP = 2

# --- Opsi: gunakan centerline otomatis sebagai garis hitung
USE_AUTO_CENTERLINE_FOR_COUNT = True

# --- (opsional) garis manual tambahan untuk belok kiri/kanan dll
MANUAL_TRACK_LINES = {
    # "Belok Kiri":  [(80,210), (150,250), (210,285)],     # contoh polyline manual
    # "Belok Kanan": [(580,170), (520,220), (470,260)],
}
direction_counts = {"Auto (Centerline)": 0, **{k: 0 for k in MANUAL_TRACK_LINES.keys()}}

# ============================================================
# 4) UTIL: OPEN STREAM + AUTO RECONNECT
# ============================================================
def open_stream():
    while True:
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        if cap.isOpened():
            print("[INFO] RTSP connected")
            return cap
        print("[WARNING] Retry connect RTSP in 5s...")
        time.sleep(5)

# ============================================================
# 5) DETEKSI TEPI JALAN → POLYLINE (AUTO)
# ============================================================
# buffer smoothing polylines
poly_hist_left  = deque(maxlen=5)
poly_hist_right = deque(maxlen=5)
poly_hist_center= deque(maxlen=5)

def roi_mask(img):
    """ROI sederhana: fokus area jalan bagian bawah-pertengahan frame."""
    h, w = img.shape[:2]
    poly = np.array([
        (0, int(0.35*h)), (w, int(0.35*h)), (w, h), (0, h)
    ], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask

def fit_quadratic(xs, ys, y_samples):
    """Fit x = a*y^2 + b*y + c (lebih stabil pada kamera elevasi)"""
    if len(xs) < 6:
        return None
    try:
        coeff = np.polyfit(ys, xs, 2)  # x(y)
        x_fit = np.polyval(coeff, y_samples)
        pts = np.stack([x_fit, y_samples], axis=1).astype(int)
        return pts
    except np.linalg.LinAlgError:
        return None

def smooth_polyline(new_pts, hist_deque):
    if new_pts is None:
        return None if len(hist_deque) == 0 else np.mean(np.stack(hist_deque,0), axis=0).astype(int)
    hist_deque.append(new_pts)
    return np.mean(np.stack(hist_deque,0), axis=0).astype(int)

def detect_lane_polylines(frame_bgr):
    """
    1) Canny edges + ROI
    2) HoughLinesP untuk segmen
    3) Bagi segmen kiri/kanan berdasarkan slope
    4) Fit quadratic jadi polyline halus
    """
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blur, 80, 160)
    mask = roi_mask(edges)
    edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50,
                            minLineLength=40, maxLineGap=20)
    if lines is None:
        return None, None, None, edges

    left_pts_x, left_pts_y = [], []
    right_pts_x, right_pts_y = [], []

    for l in lines[:,0]:
        x1,y1,x2,y2 = l
        if y2 == y1:
            continue
        slope = (x2-x1)/(y2-y1)  # gunakan x(y)
        # Segment “kiri” biasanya condong negatif untuk orientasi kamera ini
        if slope < 0:
            left_pts_x.extend([x1,x2]); left_pts_y.extend([y1,y2])
        else:
            right_pts_x.extend([x1,x2]); right_pts_y.extend([y1,y2])

    y_samples = np.linspace(int(0.40*h), h-1, 30)

    left_poly  = fit_quadratic(np.array(left_pts_x),  np.array(left_pts_y),  y_samples) if len(left_pts_x)>0 else None
    right_poly = fit_quadratic(np.array(right_pts_x), np.array(right_pts_y), y_samples) if len(right_pts_x)>0 else None

    # smoothing antar frame
    left_poly_s   = smooth_polyline(left_poly,  poly_hist_left)
    right_poly_s  = smooth_polyline(right_poly, poly_hist_right)

    center_poly_s = None
    if left_poly_s is not None and right_poly_s is not None:
        center_poly = ((left_poly_s + right_poly_s) / 2.0).astype(int)
        center_poly_s = smooth_polyline(center_poly, poly_hist_center)

    # clip ke frame
    def clip_pts(pts):
        if pts is None: return None
        pts[:,0] = np.clip(pts[:,0], 0, w-1)
        pts[:,1] = np.clip(pts[:,1], 0, h-1)
        return pts
    return clip_pts(left_poly_s), clip_pts(right_poly_s), clip_pts(center_poly_s), edges

def draw_polyline(frame, pts, color=(0,255,255), thickness=3):
    if pts is None: return
    cv2.polylines(frame, [pts.reshape(-1,1,2)], isClosed=False, color=color, thickness=thickness)

# ============================================================
# 6) CEK CROSSING TERHADAP POLYLINE (UMUM)
# ============================================================
def crossed_polyline(prev_pt, curr_pt, poly_pts):
    """
    True jika lintasan (prev->curr) memotong salah satu segmen polyline.
    """
    if poly_pts is None or len(poly_pts) < 2: 
        return False

    px, py = prev_pt; cx, cy = curr_pt

    def seg_intersect(a,b,c,d):
        # cross product helper
        def ccw(p1,p2,p3): 
            return (p3[1]-p1[1])(p2[0]-p1[0]) > (p2[1]-p1[1])(p3[0]-p1[0])
        return (ccw(a,c,d) != ccw(b,c,d)) and (ccw(a,b,c) != ccw(a,b,d))

    a = (px,py); b = (cx,cy)
    for i in range(len(poly_pts)-1):
        c = tuple(poly_pts[i])
        d = tuple(poly_pts[i+1])
        if seg_intersect(a,b,c,d):
            return True
    return False

# ============================================================
# 7) MAIN LOOP
# ============================================================
cap = open_stream()
track_history = {}
counted_ids = set()
total_counted_vehicles = set()
vehicle_types_count = {name: 0 for cid, name in class_names.items() if cid in vehicle_classes_to_count}

frame_count = 0
last_processed_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Stream lost, reconnecting...")
        cap.release()
        cap = open_stream()
        continue

    frame_resized = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
    frame_count += 1

    # ==== DETEKSI TEPI JALAN → POLYLINE ====
    left_poly, right_poly, center_poly, edges = detect_lane_polylines(frame_resized)

    if frame_count % FRAME_SKIP == 0:
        results = model.track(frame_resized, persist=True, tracker="botsort.yaml",
                              conf=0.35, iou=0.5, device=DEVICE)

        annotated = frame_resized.copy()

        # --- gambar polyline tepi jalan otomatis
        draw_polyline(annotated, left_poly,  (0,255,255), 3)   # kiri
        draw_polyline(annotated, right_poly, (0,255,255), 3)   # kanan
        draw_polyline(annotated, center_poly,(0,200,0),   2)   # centerline

        # --- (opsional) garis manual (polyline atau line 2 titik)
        for name, pts in MANUAL_TRACK_LINES.items():
            pts_np = np.array(pts, np.int32).reshape((-1,1,2))
            cv2.polylines(annotated, [pts_np], False, (0, 200, 255), 2)
            cv2.putText(annotated, name, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # === YOLO boxes ===
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()

            for box, tid, cid in zip(boxes, track_ids, class_ids):
                if cid not in vehicle_classes_to_count:
                    continue

                x1,y1,x2,y2 = box
                cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                cname = class_names[cid]

                hist = track_history.get(tid, deque(maxlen=30))
                hist.append((cx,cy))
                track_history[tid] = hist

                # COUNT: centerline otomatis
                if USE_AUTO_CENTERLINE_FOR_COUNT and len(hist)>1 and tid not in counted_ids:
                    if crossed_polyline(hist[-2], hist[-1], center_poly):
                        direction_counts["Auto (Centerline)"] += 1
                        vehicle_types_count[cname] += 1
                        counted_ids.add(tid)
                        total_counted_vehicles.add(tid)

                # COUNT: garis manual (jika diaktifkan)
                if len(hist)>1 and tid not in counted_ids and len(MANUAL_TRACK_LINES)>0:
                    for name, pts in MANUAL_TRACK_LINES.items():
                        pts_np = np.array(pts, np.int32)
                        if crossed_polyline(hist[-2], hist[-1], pts_np):
                            direction_counts[name] += 1
                            vehicle_types_count[cname] += 1
                            counted_ids.add(tid)
                            total_counted_vehicles.add(tid)
                            break

                # gambar box
                cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.circle(annotated, (cx,cy), 3, (255,255,255), -1)
                cv2.putText(annotated, f"ID:{tid} {cname}", (x1, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        last_processed_frame = annotated.copy()

    # ==== overlay info ====
    display = last_processed_frame if last_processed_frame is not None else frame_resized.copy()
    y = 18
    cv2.putText(display, "ANALISIS ARAH:", (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1); y+=20
    for k,v in direction_counts.items():
        cv2.putText(display, f"- {k}: {v}", (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1); y+=18
    y+=6
    cv2.putText(display, "TOTAL JENIS:", (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1); y+=20
    for k,v in vehicle_types_count.items():
        cv2.putText(display, f"- {k}: {v}", (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1); y+=18
    y+=6
    cv2.putText(display, f"TOTAL UNIK: {len(total_counted_vehicles)}", (10,y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Deteksi & Analisis Kendaraan RTSP (Auto Lane Polyline)", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
