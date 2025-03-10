import cv2
import numpy as np
import tensorflow as tf
import threading

# ---------------------------
# CẤU HÌNH VÀ THAM SỐ
# ---------------------------
img_size = 112                    # Kích thước mỗi frame sau khi resize: 224x224 (theo yêu cầu của mô hình)
window_size = 20             # Số frame trong cửa sổ trượt để dự đoán (input của mô hình)
model_path = 'D:\CODE\DNU_AI\AI_Leaning\Tenserflow\models\V4dcnn_data2_112_20_optimized.keras' # Đường dẫn tới mô hình 3D CNN đã được huấn luyện

# ---------------------------
# TẢI MÔ HÌNH
# ---------------------------
# Load mô hình đã được huấn luyện từ file .keras
model = tf.keras.models.load_model(model_path)

# ---------------------------
# GLOBAL VARIABLE VÀ LOCK CHO DỰ ĐOÁN
# ---------------------------
latest_prob = None               # Biến lưu giá trị dự đoán mới nhất (xác suất bạo lực)
lock = threading.Lock()          # Lock để đảm bảo an toàn khi truy cập biến toàn cục

# ---------------------------
# HÀM TIỀN XỬ LÝ
# ---------------------------
def preprocess_frame(frame, target_size=224):
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
      - Tạo thành đầu vào có shape (1, window_size, 224, 224, 3)
      - Chạy model.predict và cập nhật biến latest_prob
    """
    global latest_prob
    input_sequence = np.array(frames_window)                # Shape: (window_size, 224, 224, 3)
    input_sequence = np.expand_dims(input_sequence, axis=0)   # Shape: (1, window_size, 224, 224, 3)
    pred = model.predict(input_sequence)
    with lock:
        latest_prob = pred[0][0]

# ---------------------------
# HIỂN THỊ VIDEO VỚI OVERLAY DỰ ĐOÁN TỪ CAMERA
# ---------------------------
def display_camera_with_async_prediction():
    """
    Hàm này:
      - Mở camera (sử dụng cv2.VideoCapture(0))
      - Sử dụng cửa sổ trượt gồm 'window_size' frame để cập nhật dự đoán bất đồng bộ
      - Overlay kết quả dự đoán (tỷ lệ phần trăm bạo lực) lên góc phải của mỗi frame
      - Hiển thị video theo thời gian thực từ camera
      - Nhấn 'q' để thoát hiển thị
    """
    # Mở camera: truyền 0 để sử dụng camera mặc định
    cap = cv2.VideoCapture(3)
    if not cap.isOpened():
        print("Error: Cannot open camera!")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Nếu không lấy được FPS, mặc định 30 FPS
    delay = int(1000 / fps)
    
    # Lấy kích thước của frame từ camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame from camera!")
        return
    height, width = frame.shape[:2]
    
    sliding_window = []  # Cửa sổ trượt chứa các frame đã được tiền xử lý
    pred_thread = None   # Biến lưu thread dự đoán
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display_frame = frame.copy()   # Giữ lại frame gốc để overlay kết quả
        
        # Tiền xử lý frame và thêm vào cửa sổ trượt
        processed = preprocess_frame(frame, img_size)
        sliding_window.append(processed)
        if len(sliding_window) > window_size:
            sliding_window.pop(0)
        
        # Nếu đủ frame trong cửa sổ và không có thread dự đoán đang chạy, bắt đầu dự đoán
        if len(sliding_window) == window_size and (pred_thread is None or not pred_thread.is_alive()):
            window_copy = sliding_window.copy()  # Tạo bản sao của cửa sổ trượt
            pred_thread = threading.Thread(target=update_prediction, args=(window_copy,))
            pred_thread.start()
        
        # Lấy giá trị dự đoán hiện tại (nếu có)
        with lock:
            prob = latest_prob
        
        # Nếu có dự đoán, overlay kết quả lên góc phải của frame
        if prob is not None:
            overlay_text = f"Violence: {prob * 100:.2f}%"
            # Vị trí text: góc phải, ví dụ: (width - 300, 30)
            cv2.putText(display_frame, overlay_text, (width - 300, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Hiển thị frame
        cv2.imshow("Camera Prediction", display_frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# ---------------------------
# Ví dụ sử dụng: Hiển thị camera và dự đoán
# ---------------------------
display_camera_with_async_prediction()
