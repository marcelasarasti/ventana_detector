import cv2
import numpy as np

# CONFIGURACIÓN
video_path = "ventanav.mp4"
brick_real_width_cm = 33
brick_real_height_cm = 23

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("No se pudo abrir el video :c .")
    exit()

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

def ordenar_puntos(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # LADRILLO
    lower_orange = np.array([5, 100, 80])
    upper_orange = np.array([25, 255, 255])
    mask_brick = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_brick = cv2.morphologyEx(mask_brick, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_brick, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    brick_candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / float(h)
        cx = x + w // 2
        if (800 < area < 6000 and 1.2 < aspect_ratio < 1.6 and
                frame.shape[1] * 0.2 < cx < frame.shape[1] * 0.8):
            brick_candidates.append((x, y, w, h))

    if not brick_candidates:
        continue

    bx, by, bw, bh = max(brick_candidates, key=lambda r: r[2] * r[3])
    scale_x = brick_real_width_cm / bw
    scale_y = brick_real_height_cm / bh

    # VENTANA
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    mask_window = cv2.inRange(hsv, lower_black, upper_black)
    dilated = cv2.dilate(mask_window, kernel, iterations=4)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    window_candidates = []
    window_contour = None

    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4 and cv2.contourArea(cnt) > 2000:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if 0.5 < aspect_ratio < 3.5 and w > 50 and h > 50:
                window_contour = approx
                xw, yw, ww, hw = x, y, w, h
                break

    if window_contour is None:
        continue

    # Cálculo de dimensiones
    window_width_m = round((ww * scale_x) / 100, 2)
    window_height_m = round((hw * scale_y) / 100, 2)
    area_m2 = round(window_width_m * window_height_m, 2)

    # HOMOGRAFÍA
    src_pts = ordenar_puntos(np.float32([p[0] for p in window_contour]))
    dst_size = 300
    dst_pts = np.float32([
        [0, 0],
        [dst_size, 0],
        [dst_size, dst_size],
        [0, dst_size]
    ])
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(frame, H, (dst_size, dst_size))

    # DIBUJAR
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
    cv2.putText(frame, "LADRILLO", (bx, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.rectangle(frame, (xw, yw), (xw + ww, yw + hw), (0, 255, 0), 2)
    cv2.putText(frame, "VENTANA DETECTADA", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)

    texto = f"Dimensiones: {window_width_m} m x {window_height_m} m | Área: {area_m2} m²"
    cv2.putText(frame, texto, (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    print(texto)

    # MOSTRAR AMBAS VENTANAS
    cv2.imshow("Detección en video", frame)
    cv2.imshow("Ventana rectificada (Homografía)", warped)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
