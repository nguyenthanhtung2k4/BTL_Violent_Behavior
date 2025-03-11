import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# ========================================================================
# 1. Tham số và Hằng số
# ========================================================================
IMG_SIZE = 112
FRAMES_PER_VIDEO = 25
CHANNELS = 3
NUM_CLASSES = 2
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.5  # Điều chỉnh dropout rate

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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frames.append(frame)
    cap.release()

    while len(frames) < FRAMES_PER_VIDEO:
        frames.append(np.zeros((IMG_SIZE, IMG_SIZE, CHANNELS)))

    return np.array(frames) / 255.0

def load_dataset(data_dir):
    video_paths = []
    labels = []

    for video_file in os.listdir(data_dir):
        if video_file.endswith('.mp4'):
            video_paths.append(os.path.join(data_dir, video_file))
            if 'NV' in video_file:
                labels.append(0)  # 0: no-violence
            else:
                labels.append(1)  # 1: violence

    return video_paths, to_categorical(labels, num_classes=NUM_CLASSES)

# ========================================================================
# 3. Tạo Data Generator (với Data Augmentation)
# ========================================================================
class VideoDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, video_paths, labels, batch_size, augment=False):
        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size
        self.augment = augment

    def __len__(self):
        return int(np.ceil(len(self.video_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_paths = self.video_paths[idx*self.batch_size : (idx+1)*self.batch_size]
        batch_labels = self.labels[idx*self.batch_size : (idx+1)*self.batch_size]

        batch_videos = []
        for path in batch_paths:
            video = load_video_frames(path)
            if self.augment:
                video = self.augment_video(video)
            batch_videos.append(video)

        return np.array(batch_videos), np.array(batch_labels)

    def augment_video(self, video):
        for i in range(video.shape[0]):
            if np.random.random() < 0.5:
                video[i] = cv2.flip(video[i], 1)
            if np.random.random() < 0.3:
                video[i] = tf.image.random_brightness(video[i], max_delta=0.2)
        return video

# ========================================================================
# 4. Xây dựng mô hình 3D CNN (với BatchNormalization, L2 Regularization)
# ========================================================================
def build_3dcnn_model():
    model = Sequential([
        Conv3D(16, (3, 3, 3), activation='relu',
               input_shape=(FRAMES_PER_VIDEO, IMG_SIZE, IMG_SIZE, CHANNELS),
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling3D((1, 2, 2)),

        Conv3D(32, (3, 3, 3), activation='relu',
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling3D((1, 2, 2)),

        Conv3D(64, (3, 3, 3), activation='relu',
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling3D((2, 2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(DROPOUT_RATE),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
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
    data_dir = "../Dataset/DataSetCustom"
    video_paths, labels = load_dataset(data_dir)

    X_train, X_val, y_train, y_val = train_test_split(
        video_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    train_generator = VideoDataGenerator(X_train, y_train, BATCH_SIZE, augment=True)
    val_generator = VideoDataGenerator(X_val, y_val, BATCH_SIZE)

    model = build_3dcnn_model()
    model.summary()

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[early_stopping]
    )

    plot_history(history)

    model.save("../models/V4dcnn_data2_112_60_optimized.keras", save_format="keras")