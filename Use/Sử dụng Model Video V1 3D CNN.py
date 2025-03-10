import cv2
import numpy as np
import tensorflow as tf


# Ví dụ sử dụng: truyền vào đường dẫn video đầu vào và đường dẫn lưu video đầu ra
input_video_path = "./data_test/chem.mp4"  # Thay bằng đường dẫn video của bạn
output_video_path = "Video4_chem_v3.1dcnn_data2_224_20.mp4"   # Đường dẫn video kết quả

# Đặt các tham số
img_size = 224                    # Kích thước của mỗi khung hình để phù hợp với mô hình
_images_per_file =  20          # Số khung hình trong mỗi đoạn (segment) để đưa vào mô hình
# Giả sử mô hình của bạn đã được huấn luyện với input shape: (None, 10, 112, 112, 3)

# Đường dẫn tới mô hình đã huấn luyện (định dạng .keras)
# model_path = '3.1dcnn_fight_detection_224_20.keras'
model_path = 'D:\CODE\DNU_AI\AI_Leaning\Tenserflow\models\V3.1dcnn_data2_224_20_e15_bath8.keras'
# Load mô hình
model = tf.keras.models.load_model(model_path)

def preprocess_frame(frame):
    """
    Tiền xử lý một khung hình:
      - Chuyển đổi màu từ BGR sang RGB
      - Thay đổi kích thước về (img_size, img_size)
      - Chuẩn hóa giá trị pixel về khoảng [0,1]
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (img_size, img_size))
    frame = frame / 255.0
    return frame.astype(np.float32)

def process_video_and_overlay(video_path, output_path):
    """
    Xử lý video đầu vào:
      - Đọc video và chia thành các đoạn (segment) gồm _images_per_file khung hình
      - Với mỗi đoạn, tiền xử lý và tạo thành input có shape (1, _images_per_file, img_size, img_size, 3)
      - Dự đoán xác suất bạo lực cho đoạn đó bằng mô hình 3D CNN
      - Overlay tỷ lệ phần trăm lên góc trái của từng khung hình trong đoạn
      - Ghi các khung hình đã overlay vào video đầu ra
    """
    # Mở video đầu vào
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Lỗi: Không mở được file video.")
        return
    
    # Lấy thông số video (FPS, kích thước)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Sử dụng cùng FPS và kích thước ban đầu để ghi video đầu ra
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frames_buffer = []  # Bộ nhớ chứa các khung hình hiện tại của đoạn

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Kết thúc nếu không còn frame nào
        frames_buffer.append(frame)
        # Khi đủ _images_per_file frame, xử lý đoạn này
        if len(frames_buffer) == _images_per_file:
            # Tiền xử lý từng khung hình (chuyển về kích thước 112x112 và chuẩn hóa)
            processed_frames = [preprocess_frame(f) for f in frames_buffer]
            # Tạo input cho mô hình: shape (1, _images_per_file, img_size, img_size, 3)
            input_sequence = np.array(processed_frames)
            input_sequence = np.expand_dims(input_sequence, axis=0)
            
            
            
            # Debug shape của input
            print(f"Shape of input_sequence: {input_sequence.shape}")

            
            # Dự đoán xác suất bạo lực cho đoạn này
            prediction = model.predict(input_sequence)
            violence_prob = prediction[0][0]  # Giả sử index 0 biểu thị xác suất "bạo lực"
            
            # Overlay kết quả lên từng khung hình trong đoạn
            for f in frames_buffer:
                text = f"Violence: {violence_prob * 100:.2f}%"
                # Vị trí text: góc trái, ví dụ: (10, 30)
                cv2.putText(f, text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                out.write(f)
            frames_buffer = []  # Xóa buffer cho đoạn tiếp theo

    # Nếu còn dư các frame (không đủ 10) thì padding với frame cuối cùng
    if len(frames_buffer) > 0:
        while len(frames_buffer) < _images_per_file:
            frames_buffer.append(frames_buffer[-1].copy())
        processed_frames = [preprocess_frame(f) for f in frames_buffer]
        input_sequence = np.array(processed_frames)
        input_sequence = np.expand_dims(input_sequence, axis=0)
        prediction = model.predict(input_sequence)
        violence_prob = prediction[0][0]
        for f in frames_buffer:
            text = f"Violence: {violence_prob * 100:.2f}%"
            cv2.putText(f, text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            out.write(f)
    
    cap.release()
    out.release()
    print("Video đầu ra đã được lưu tại:", output_path)


process_video_and_overlay(input_video_path, output_video_path)

# Thêm đoạn code để xử lý video từ webcam
def process_webcam():
    """
    Xử lý video trực tiếp từ webcam:
    - Đọc frame từ webcam
    - Xử lý tương tự như với video file
    - Hiển thị kết quả trực tiếp
    """
    cap = cv2.VideoCapture(0)  # Mở webcam (0 là webcam mặc định)
    if not cap.isOpened():
        print("Lỗi: Không thể mở webcam")
        return
        
    frames_buffer = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frames_buffer.append(frame)
        
        if len(frames_buffer) == _images_per_file:
            # Xử lý đoạn frame
            processed_frames = [preprocess_frame(f) for f in frames_buffer]
            input_sequence = np.array(processed_frames)
            input_sequence = np.expand_dims(input_sequence, axis=0)
            
            # Dự đoán
            prediction = model.predict(input_sequence)
            violence_prob = prediction[0][0]
            
            # Hiển thị frame cuối cùng với kết quả
            text = f"Violence: {violence_prob * 100:.2f}%"
            cv2.putText(frame, text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('Webcam Detection', frame)
            
            frames_buffer.pop(0)  # Xóa frame đầu tiên để giữ buffer size
            
        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

## Chạy phát hiện bạo lực qua webcam
# print("Đang khởi động webcam detection...")
# process_webcam()

