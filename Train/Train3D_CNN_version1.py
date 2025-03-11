import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping


# ========================================================================
# 1. Tham số và Hằng số
# ========================================================================
IMG_SIZE = 224  # Giảm kích thước để tiết kiệm bộ nhớ
FRAMES_PER_VIDEO = 20
CHANNELS = 3
NUM_CLASSES = 2
BATCH_SIZE = 8  # 8
EPOCHS = 15  # 15

# ========================================================================
# 2. Tiền xử lý dữ liệu
# ========================================================================
def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < FRAMES_PER_VIDEO:
        ret, frame = cap.read()
        if not ret:
            break
        # Tiền xử lý frame: chuyển đổi màu và thay đổi kích thước
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frames.append(frame)
    cap.release()

    # Nếu video có số frame ít hơn yêu cầu, padding bằng ảnh đen
    while len(frames) < FRAMES_PER_VIDEO:
        frames.append(np.zeros((IMG_SIZE, IMG_SIZE, CHANNELS)))

    return np.array(frames) / 255.0  # Chuẩn hóa

def load_dataset(data_dir):
    video_paths = []
    labels = []

    for video_file in os.listdir(data_dir):
        if video_file.endswith('.mp4'):
            video_paths.append(os.path.join(data_dir, video_file))
            # Kiểm tra tên tệp để xác định nhãn: nếu chứa 'fi' thì là fight, ngược lại là no-fight
            if 'NV' in video_file:
                labels.append(0)  # 0: fight
            else:
                labels.append(1)  # 1: no-fight

    return video_paths, to_categorical(labels, num_classes=NUM_CLASSES)

# ========================================================================
# 3. Tạo Data Generator
# ========================================================================
class VideoDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, video_paths, labels, batch_size):
        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.video_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_paths = self.video_paths[idx*self.batch_size : (idx+1)*self.batch_size]
        batch_labels = self.labels[idx*self.batch_size : (idx+1)*self.batch_size]

        batch_videos = []
        for path in batch_paths:
            video = load_video_frames(path)
            batch_videos.append(video)

        return np.array(batch_videos), np.array(batch_labels)

# ========================================================================
# 4. Xây dựng mô hình 3D CNN
# ========================================================================
def build_3dcnn_model():
    model = Sequential([
        # Block 1
        Conv3D(32, (3, 3, 3), activation='relu',
               input_shape=(FRAMES_PER_VIDEO, IMG_SIZE, IMG_SIZE, CHANNELS)),
        MaxPooling3D((1, 2, 2)),

        # Block 2
        Conv3D(64, (3, 3, 3), activation='relu'),
        MaxPooling3D((1, 2, 2)),

        # Block 3
        Conv3D(128, (3, 3, 3), activation='relu'),
        MaxPooling3D((2, 2, 2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ========================================================================
# 5. Huấn luyện và Đánh giá
# ========================================================================
def plot_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()

    plt.show()

# ========================================================================
# 6. Thực thi chính
# ========================================================================
if __name__ == "__main__":
    # Load dataset
    data_dir = "../Dataset/DataSetCustom"  # Thay đổi đường dẫn
    video_paths, labels = load_dataset(data_dir)

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_val, y_train, y_val = train_test_split(
        video_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # Tạo generators
    train_generator = VideoDataGenerator(X_train, y_train, BATCH_SIZE)
    val_generator = VideoDataGenerator(X_val, y_val, BATCH_SIZE)

    # Xây dựng mô hình 3D CNN
    model = build_3dcnn_model()
    model.summary()



    early_stopping = EarlyStopping(
          monitor='val_loss',
          patience=5,
          restore_best_weights=True
      )


    # Huấn luyện mô hình
    history = model.fit(
          train_generator,
          validation_data=val_generator,
          epochs=EPOCHS,
          verbose=1,
          callbacks=[early_stopping]
      )





    # Đánh giá và visualize kết quả
    plot_history(history)

    # Lưu mô hình theo định dạng Keras (.keras) thay vì .h5
    model.save("../models/V3.1dcnn_data2_224_20_e15_bath8.keras", save_format="keras")
