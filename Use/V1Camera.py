import cv2
import numpy as np
import tensorflow as tf
import threading

# ---------------------------
# CẤU HÌNH VÀ THAM SỐ
# ---------------------------
img_size = 112
window_size = 20
model_path = r'D:\CODE\DNU_AI\AI_Leaning\Tenserflow\BTL_Violent_Behavior\BTL_Violent_Behavior\models\V4cnn_data2_112_20_optimized.keras'
output_video_path = r'D:\CODE\DNU_AI\AI_Leaning\Tenserflow\BTL_Violent_Behavior\BTL_Violent_Behavior\xuat\output_camera.mp4'  # Đường dẫn lưu video đầu ra

# ---------------------------
# TẢI MÔ HÌNH
# ---------------------------
model = tf.keras.models.load_model(model_path)

# ---------------------------
# GLOBAL VARIABLE VÀ LOCK CHO DỰ ĐOÁN
# ---------------------------
latest_prob = None
lock = threading.Lock()

# ---------------------------
# HÀM TIỀN XỬ LÝ
# ---------------------------
def preprocess_frame(frame, target_size=224):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (target_size, target_size))
    frame_norm = frame_resized / 255.0
    return frame_norm.astype(np.float32)

def update_prediction(frames_window):
    global latest_prob
    input_sequence = np.array(frames_window)
    input_sequence = np.expand_dims(input_sequence, axis=0)
    pred = model.predict(input_sequence)
    with lock:
        latest_prob = pred[0][0]

# ---------------------------
# HIỂN THỊ VIDEO VỚI OVERLAY DỰ ĐOÁN TỪ CAMERA
# ---------------------------
def display_camera_with_async_prediction():
    cap = cv2.VideoCapture(3)
    if not cap.isOpened():
        print("Error: Cannot open camera!")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    delay = int(1000 / fps)

    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame from camera!")
        return
    height, width = frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    sliding_window = []
    pred_thread = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()

        processed = preprocess_frame(frame, img_size)
        sliding_window.append(processed)
        if len(sliding_window) > window_size:
            sliding_window.pop(0)

        if len(sliding_window) == window_size and (pred_thread is None or not pred_thread.is_alive()):
            window_copy = sliding_window.copy()
            pred_thread = threading.Thread(target=update_prediction, args=(window_copy,))
            pred_thread.start()

        with lock:
            prob = latest_prob

        if prob is not None:
            # overlay_text = f"Violence: {prob * 100:.2f}%"
            # cv2.putText(display_frame, overlay_text, (width - 300, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            if prob > 0.7:
                color = (0, 0, 255)  # Đỏ cho mức độ bạo lực cao
                status = "VIOLENCE"
            else:
                color = (0, 255, 0)  # Xanh cho không bạo lực
                status = "NO VIOLENCE"

            overlay_text = f"Status: {status}\nViolence: {prob * 100:.2f}%"
            cv2.putText(display_frame, overlay_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)



        cv2.imshow("Camera Prediction", display_frame)
        out.write(display_frame)  # Lưu frame vào video đầu ra

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()  # Giải phóng VideoWriter
    cv2.destroyAllWindows()

# ---------------------------
# Ví dụ sử dụng: Hiển thị camera và dự đoán
# ---------------------------
display_camera_with_async_prediction()