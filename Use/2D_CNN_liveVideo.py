import cv2
import numpy as np
import tensorflow as tf
import threading

# ---------------------------
# CẤU HÌNH VÀ THAM SỐ
# ---------------------------
img_size = 112  # Kích thước mỗi frame sau khi resize
window_size = 20  # Số frame trong cửa sổ trượt để dự đoán (input của mô hình)
feature_size = 3 * 3 * 512  # Đầu ra từ VGG16
model_path = r'models/2DCNN_data2_112_20_e50_batch32.keras'  # Đường dẫn tới mô hình đã train

# ---------------------------
# TẢI MÔ HÌNH
# ---------------------------
model = tf.keras.models.load_model(model_path)

# ---------------------------
# BIẾN TOÀN CỤC VÀ LOCK
# ---------------------------
latest_prob = None
lock = threading.Lock()

# ---------------------------
# HÀM TIỀN XỬ LÝ FRAME
# ---------------------------
def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (img_size, img_size))
    frame_norm = frame_resized / 255.0
    return frame_norm.astype(np.float32)

# ---------------------------
# HÀM TRÍCH XUẤT ĐẶC TRƯNG VỚI VGG16
# ---------------------------
vgg_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))
feature_extractor = tf.keras.Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block5_pool').output)

def extract_features(frames):
    frames = np.array(frames)  # Shape: (20, 112, 112, 3)
    features = feature_extractor.predict(frames)  # Shape: (20, 3, 3, 512)
    return features.reshape(20, feature_size)  # Shape: (20, 4608)

# ---------------------------
# HÀM DỰ ĐOÁN TRONG THREAD RIÊNG
# ---------------------------
def update_prediction(frames_window):
    global latest_prob
    input_sequence = np.expand_dims(extract_features(frames_window), axis=0)  # Shape: (1, 20, 4608)
    pred = model.predict(input_sequence)
    with lock:
        latest_prob = pred[0][0]

# ---------------------------
# HIỂN THỊ VIDEO VỚI OVERLAY DỰ ĐOÁN
# ---------------------------
def display_video_with_async_prediction(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file!")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)
    
    sliding_window = []
    pred_thread = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display_frame = frame.copy()
        processed = preprocess_frame(frame)
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
            if prob > 0.7:
                color = (0, 0, 255)
                status = "VIOLENCE"
            else:
                color = (0, 255, 0)
                status = "NO VIOLENCE"
            
            overlay_text = f"Status: {status}\nViolence: {prob * 100:.2f}%"
            cv2.putText(display_frame, overlay_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
        cv2.imshow("Video Prediction", display_frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# ---------------------------
# CHẠY CHƯƠNG TRÌNH
# ---------------------------
video_path = r'Data/Data_Test/dibo.mp4'
display_video_with_async_prediction(video_path)
