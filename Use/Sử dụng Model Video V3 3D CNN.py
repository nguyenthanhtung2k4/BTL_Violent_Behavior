import cv2
import numpy as np
import tensorflow as tf
import threading



# ---------------------------
# Duong dan video 
# video_path = "./Normal_Videos/Normal_Videos_050_x264.mp4"  # Thay bằng đường dẫn video của bạn
video_path = "./data_quay/no3.mp4"  # Thay bằng đường dẫn video của bạn
# ---------------------------

# ---------------------------
# CẤU HÌNH VÀ THAM SỐ
# ---------------------------

# window_size = 10 #10                 # Số frame trong cửa sổ trượt để dự đoán (input của mô hình)
# model_path = 'fight_3dcnn.keras' # Đường dẫn tới mô hình 3D CNN đã train

# window_size = 20 #10                 # Số frame trong cửa sổ trượt để dự đoán (input của mô hình)
# model_path = '3dcnn_fight_detection.keras' # Đường dẫn tới mô hình 3D CNN đã train

img_size = 112                        # Kích thước mỗi frame sau khi resize: 112 x 112
window_size = 20  #10                 # Số frame trong cửa sổ trượt để dự đoán (input của mô hình)
# model_path = '3.1dcnn_fight_detection_224_20.keras' # Đường dẫn tới mô hình 3D CNN đã train
# model_path = 'V3dcnn_fight_detection.Data2_224_20.keras' # Đường dẫn tới mô hình 3D CNN đã train
model_path = 'D:\CODE\DNU_AI\AI_Leaning\Tenserflow\models\V4dcnn_data2_112_20_optimized.keras' # Đường dẫn tới mô hình 3D CNN đã train

# ---------------------------
# TẢI MÔ HÌNH
# ---------------------------
model = tf.keras.models.load_model(model_path)

# ---------------------------
# GLOBAL VARIABLE VÀ LOCK CHO DỰ ĐOÁN
# ---------------------------
latest_prob = None               # Biến lưu giá trị dự đoán mới nhất (xác suất bạo lực)
lock = threading.Lock()          # Lock để đảm bảo an toàn khi truy cập biến toàn cục

# ---------------------------
# HÀM TIỀN XỬ LÝ
# ---------------------------
def preprocess_frame(frame, target_size=112):
    """
    Tiền xử lý một frame:
      - Chuyển đổi màu từ BGR sang RGB
      - Resize về (target_size, target_size)
      - Chuẩn hóa pixel về [0,1]
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (target_size, target_size))
    frame_norm = frame_resized / 255.0
    return frame_norm.astype(np.float32)

def update_prediction(frames_window):
    """
    Hàm chạy trong thread riêng:
      - Nhận vào cửa sổ trượt (window) các frame đã tiền xử lý
      - Tạo thành đầu vào có shape (1, window_size, 112, 112, 3)
      - Chạy model.predict và cập nhật biến latest_prob
    """
    global latest_prob
    input_sequence = np.array(frames_window)                # Shape: (window_size, 112, 112, 3)
    input_sequence = np.expand_dims(input_sequence, axis=0)   # Shape: (1, window_size, 112, 112, 3)
    pred = model.predict(input_sequence)
    with lock:
        latest_prob = pred[0][0]

# ---------------------------
# HIỂN THỊ VIDEO VỚI OVERLAY DỰ ĐOÁN (THEO THỜI GIAN THỰC)
# ---------------------------
def display_video_with_async_prediction(video_path):
    """
    Hàm này:
      - Mở video từ file
      - Sử dụng cửa sổ trượt gồm 'window_size' frame để cập nhật dự đoán bất đồng bộ
      - Overlay kết quả dự đoán (tỷ lệ phần trăm bạo lực) lên góc phải của mỗi frame
      - Hiển thị video theo thời gian thực (theo FPS của video)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file!")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)  # Thời gian hiển thị mỗi frame (ms)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    sliding_window = []  # Cửa sổ trượt chứa các frame đã được tiền xử lý
    pred_thread = None   # Biến lưu thread dự đoán
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display_frame = frame.copy()   # Giữ lại frame gốc để hiển thị
        
        # Tiền xử lý frame và thêm vào cửa sổ trượt
        processed = preprocess_frame(frame, img_size)
        sliding_window.append(processed)
        if len(sliding_window) > window_size:
            sliding_window.pop(0)
        
        # Nếu có đủ frame trong cửa sổ và không có thread dự đoán đang chạy, bắt đầu dự đoán
        if len(sliding_window) == window_size and (pred_thread is None or not pred_thread.is_alive()):
            # Tạo bản sao của cửa sổ để không bị thay đổi khi thread chạy
            window_copy = sliding_window.copy()
            pred_thread = threading.Thread(target=update_prediction, args=(window_copy,))
            pred_thread.start()
        
        # Lấy giá trị dự đoán hiện tại (nếu có)
        with lock:
            prob = latest_prob
        
        # Kiểm tra nếu prob không phải là None trước khi so sánh
        if prob is not None:
            if prob > 0.7:
                color = (0, 0, 255)  # Đỏ cho mức độ bạo lực cao
                status = "VIOLENCE"
            
            else:
                color = (0, 255, 0)  # Xanh cho không bạo lực
                status = "NO VIOLENCE"
            # if prob > 0.8:
            #     color = (0, 0, 255)  # Đỏ cho mức độ bạo lực cao
            #     status = "HIGH VIOLENCE"
            # elif prob > 0.5:
            #     color = (0, 255, 255)  # Vàng cho mức độ bạo lực trung bình
            #     status = "MEDIUM VIOLENCE"
            # else:
            #     color = (0, 255, 0)  # Xanh cho không bạo lực
            #     status = "NO VIOLENCE"
            
            overlay_text = f"Status: {status}\nViolence: {prob * 100:.2f}%"
            cv2.putText(display_frame, overlay_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
        # Hiển thị frame
        cv2.imshow("Video Prediction", display_frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# ---------------------------
# Ví dụ sử dụng
# ---------------------------

display_video_with_async_prediction(video_path)
