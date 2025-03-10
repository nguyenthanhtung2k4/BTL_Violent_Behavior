import cv2
import numpy as np
import tensorflow as tf

# ========================================================================
# 1. Tham số và Hằng số (phải giống với lúc huấn luyện)
# ========================================================================
IMG_SIZE = 112
FRAMES_PER_VIDEO = 20
CHANNELS = 3
NUM_CLASSES = 2

# ========================================================================
# 2. Hàm tiền xử lý video (phải giống với lúc huấn luyện)
# ========================================================================
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

# ========================================================================
# 3. Load mô hình đã huấn luyện
# ========================================================================
model_path = "D:\CODE\DNU_AI\AI_Leaning\Tenserflow\models\V4.1dcnn_122_20_batch_16_e50.keras"  # Đường dẫn đến mô hình
model = tf.keras.models.load_model(model_path)

# ========================================================================
# 4. Hàm dự đoán
# ========================================================================
def predict_violence(video_path):
    video_frames = load_video_frames(video_path)
    video_frames = np.expand_dims(video_frames, axis=0)  # Thêm batch dimension

    prediction = model.predict(video_frames)
    class_index = np.argmax(prediction)  # Lấy chỉ số lớp có xác suất cao nhất
    confidence = prediction[0][class_index]  # Lấy xác suất của lớp dự đoán

    if class_index == 0:
        label = "No Violence"
    else:
        label = "Violence"

    return label, confidence

# ========================================================================
# 5. Sử dụng mô hình để dự đoán
# ========================================================================
video_path_to_predict = "D:\CODE\DNU_AI\AI_Leaning\Tenserflow\data_test\smartt.mp4"  # Đường dẫn đến video cần dự đoán
predicted_label, confidence = predict_violence(video_path_to_predict)

print(f"Video: {video_path_to_predict}")
print(f"Prediction: {predicted_label}")
print(f"Confidence: {confidence:.4f}")