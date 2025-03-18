# import cv2
# import numpy as np
# import tensorflow as tf
# import threading

# # =============================
# # Cấu hình
# # =============================
# VIDEO_PATH = r"Data/Data_Test/camera.mp4"  # Đường dẫn video
# MODEL_PATH = r"models/CNN3D_LSTM_HockeyFight.keras"  # Model đã train
# IMG_SIZE = 112  # Kích thước ảnh input
# WINDOW_SIZE = 16  # Số frame để dự đoán
# CLASS_NAMES = ["Non-Violent", "Violent"]

# # =============================
# # Tải mô hình
# # =============================
# model = tf.keras.models.load_model(MODEL_PATH)

# # Biến toàn cục để lưu dự đoán mới nhất
# latest_prob = None
# lock = threading.Lock()

# # =============================
# # Hàm tiền xử lý ảnh
# # =============================
# def preprocess_frame(frame):
#     """ Tiền xử lý frame: đổi màu, resize, chuẩn hóa """
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
#     frame = frame / 255.0  # Chuẩn hóa
#     return frame.astype(np.float32)

# # =============================
# # Hàm dự đoán dùng CNN3D + LSTM
# # =============================
# def update_prediction(frames_window):
#     """ Chạy mô hình dự đoán bất đồng bộ """
#     global latest_prob
#     input_sequence = np.array(frames_window)  # Shape: (WINDOW_SIZE, 112, 112, 3)
#     input_sequence = np.expand_dims(input_sequence, axis=0)  # Shape: (1, WINDOW_SIZE, 112, 112, 3)
    
#     pred = model.predict(input_sequence)
    
#     with lock:
#         latest_prob = pred[0][1]  # Xác suất bạo lực

# # =============================
# # Hiển thị video với dự đoán
# # =============================
# def display_video_with_async_prediction(video_path):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("Lỗi: Không thể mở video!")
#         return
    
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     delay = int(1000 / fps)  # Thời gian chờ để video chạy mượt
#     sliding_window = []
#     pred_thread = None

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         display_frame = frame.copy()
        
#         # Tiền xử lý frame
#         processed = preprocess_frame(frame)
#         sliding_window.append(processed)
#         if len(sliding_window) > WINDOW_SIZE:
#             sliding_window.pop(0)
        
#         # Khi đủ số frame, bắt đầu dự đoán
#         if len(sliding_window) == WINDOW_SIZE and (pred_thread is None or not pred_thread.is_alive()):
#             window_copy = sliding_window.copy()
#             pred_thread = threading.Thread(target=update_prediction, args=(window_copy,))
#             pred_thread.start()
        
#         # Lấy kết quả dự đoán mới nhất
#         with lock:
#             prob = latest_prob
        
#         # Hiển thị kết quả dự đoán trên video
#         if prob is not None:
#             if prob > 0.75:
#                 color = (0, 0, 255)  # Đỏ: Bạo lực
#                 status = "VIOLENCE"
#             else:
#                 color = (0, 255, 0)  # Xanh: Không bạo lực
#                 status = "NON-VIOLENCE"

#             overlay_text = f"Status: {status} ({prob * 100:.2f}%)"
#             cv2.putText(display_frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

#         cv2.imshow("Violence Detection", display_frame)

#         if cv2.waitKey(delay) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()

# # =============================
# # Chạy hệ thống nhận diện
# # =============================
# display_video_with_async_prediction(VIDEO_PATH)


import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# ================== 1. HYPERPARAMETERS ==================
IMG_SIZE = 112
FRAMES_PER_VIDEO = 20
CHANNELS = 3
NUM_CLASSES = 2
BATCH_SIZE = 1
MODEL_PATH = r"models/CNN3D_LSTM_HockeyFight.keras"

# ================== 2. VIDEO PROCESSING FUNCTION ==================
def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < FRAMES_PER_VIDEO:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frames.append(frame)
    cap.release()
    while len(frames) < FRAMES_PER_VIDEO:
        frames.append(np.zeros((IMG_SIZE, IMG_SIZE, CHANNELS)))
    return np.array(frames) / 255.0

# ================== 3. PREDICTION FUNCTION ==================
def predict_video(model, video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)  # Delay between frames to maintain 1:1 ratio
    frame_count = 0
    sliding_window = []

    # Create VideoWriter to save the processed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Add frame to the sliding window for prediction
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = cv2.resize(processed_frame, (IMG_SIZE, IMG_SIZE))
        sliding_window.append(processed_frame)

        if len(sliding_window) > FRAMES_PER_VIDEO:
            sliding_window.pop(0)

        if len(sliding_window) == FRAMES_PER_VIDEO:
            # Predict on the current sliding window
            input_data = np.array([sliding_window]) / 255.0  # Normalize
            prediction = model.predict(input_data)
            prob = prediction[0][1]  # Probability of class '1' (violence)

            # Draw the result on the frame
            if prob > 0.75:
                color = (0, 0, 255)  # Red for violence
                label = f"Violence: {prob*100:.2f}%"
            else:
                color = (0, 255, 0)  # Green for no violence
                label = f"No Violence: {prob*100:.2f}%"

            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Write the processed frame to the output video file
        out.write(frame)

        # Display the frame
        cv2.imshow("Video Prediction", frame)

        # Wait to ensure video runs at real-time speed
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    out.release()  # Release the video writer
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load the trained model
    model = load_model(MODEL_PATH)

    # Video file to predict on
    video_path = r'Data/Data_Test/camera.mp4'

    # Output file where processed video will be saved
    output_video_path = r'xuat/camera_output.mp4'

    # Start prediction and save the result
    predict_video(model, video_path, output_video_path)
