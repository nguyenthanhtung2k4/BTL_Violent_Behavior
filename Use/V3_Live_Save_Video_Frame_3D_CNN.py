import cv2
import numpy as np
import tensorflow as tf
import threading
import time
import os

# ---------------------------
# Duong dan video
video_path = r"..\Data\Data_Test\camera.mp4"  # Thay bằng đường dẫn video của bạn
output_video_path = r"D:\CODE\DNU_AI\AI_Leaning\Tenserflow\BTL_Violent_Behavior\BTL_Violent_Behavior\xuat\ouput.mp4"  # Đường dẫn video đầu ra
output_frame_dir = r"D:\CODE\DNU_AI\AI_Leaning\Tenserflow\BTL_Violent_Behavior\BTL_Violent_Behavior\xuat"  # Thư mục lưu các frame ổn định
# ---------------------------

# ---------------------------
# CẤU HÌNH VÀ THAM SỐ
# ---------------------------

img_size = 112 
window_size = 20
model_path = r'models/3D_CNN_data1_112_20_batch16_e15.keras'
violence_threshold = 0.70  # Ngưỡng bạo lực
frames_to_save =  15 # Số frame bạo lực cần lưu
tbTime=5 # time    tinh  trung binh  ty le  do

# Tạo thư mục lưu frame nếu chưa tồn tại
if not os.path.exists(output_frame_dir):
    os.makedirs(output_frame_dir)

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
def preprocess_frame(frame, target_size=112):
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
# HIỂN THỊ VIDEO VỚI OVERLAY DỰ ĐOÁN (THEO THỜI GIAN THỰC)
# ---------------------------
def display_video_with_async_prediction(video_path, output_video_path, output_frame_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file!")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    sliding_window = []
    pred_thread = None
    start_time = time.time()
    predictions = []
    stable_time = None
    frame_count = 0
    violence_frames = []

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
            predictions.append(prob)
            if time.time() - start_time >= tbTime:
                avg_prob = np.mean(predictions)
                if stable_time is None:
                    stable_time = avg_prob
                else:
                    if prob >= violence_threshold:
                        if abs(avg_prob - stable_time) < 0.05:
                            print("Valorant")
                            violence_frames.append((display_frame.copy(), prob)) #Lưu frame và xác xuất ngay khi ổn định.
                        stable_time = avg_prob
                start_time = time.time()
                predictions = []

            if prob >= violence_threshold:
                color = (0, 0, 255)
                status = "VIOLENCE"
            else:
                color = (0, 255, 0)
                status = "NO VIOLENCE"

            overlay_text = f"Status: {status}\nViolence: {prob * 100:.2f}%"
            cv2.putText(display_frame, overlay_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        out.write(display_frame)
        cv2.imshow("Video Prediction", display_frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
        frame_count += 1

    # Lưu các frame bạo lực
    violence_frames.sort(key=lambda x: x[1], reverse=True)
    for i in range(min(frames_to_save, len(violence_frames))):
        frame, prob = violence_frames[i]
        frame_filename = os.path.join(output_frame_dir, f"violence_frame_{i}.jpg")
        cv2.putText(frame, f"Violence: {prob * 100:.2f}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(frame_filename, frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# ---------------------------
# Ví dụ sử dụng
# ---------------------------

display_video_with_async_prediction(video_path, output_video_path, output_frame_dir)