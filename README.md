<h1 align="center">Nhận diện hành vi bạo lực trong trường học</h1>

<div align="center">

<p align="center">
  <img src="reluts/logoDaiNam.png" alt="DaiNam University Logo" width="200"/>
  <img src="reluts/LogoAIoTLab.png" alt="AIoTLab Logo" width="170"/>
</p>

[![Made by AIoTLab](https://img.shields.io/badge/Made%20by%20AIoTLab-blue?style=for-the-badge)](https://www.facebook.com/DNUAIoTLab)
[![Fit DNU](https://img.shields.io/badge/Fit%20DNU-green?style=for-the-badge)](https://fitdnu.net/)
[![DaiNam University](https://img.shields.io/badge/DaiNam%20University-red?style=for-the-badge)](https://dainam.edu.vn)

</div>

<h2 align="center">Mô hình phát hiện hành vi bạo lực học đường sử dụng
mạng nơ-ron tích chập CNN</h2>

<p align="left">
  
Dự án "Violent_Behavior" tập trung vào việc phát hiện và phân loại hành vi bạo lực trong video sử dụng các mô hình học sâu. Mục tiêu là xây dựng một hệ thống có khả năng nhận diện tự động các hành động bạo lực, từ đó hỗ trợ trong các ứng dụng giám sát an ninh, phân tích nội dung video, và các lĩnh vực liên quan.

</p>

---

## 🌟 Giới thiệu

- **Mô hình nhận diện** Mô hình sẽ nhận diện qua thông qua video và trả về kết quả ngay bên góc trái màn hình.
- **💡 Thông báo:** Khi nhận dạng được nếu mô hình nhận dạnh được đó là hành vi bạo lực trên 75% sẽ hiện thông báo đó là hành động bạo lực.
<!-- - **📊 Quản lý dữ liệu:** Dữ liệu điểm danh được lưu trong MongoDB, có thể xem lịch sử và xuất ra file CSV.
- **🖥️ Giao diện thân thiện:** Sử dụng Tkinter cho giao diện quản lý và Flask cho xử lý điểm danh qua web. -->
<!-- 
---
## 🏗️ HỆ THỐNG
<p align="center">
  <img src="images/Quytrinhdiemdanh.png" alt="System Architecture" width="800"/>
</p> -->

---
## 📂 Cấu trúc dự án

📦 Project  
├── 📂 Data  # Thư mục chứa Dữ liệu video<br>
    ├── 📂 Data  # Thư mục chứa Dữ liệu video<br>
├── 📂 Data  # Thư mục chứa Dữ liệu video
├── 📂 AttendanceDB  # Thư mục chứa cơ sở dữ liệu MongoDB backup  
├── 📂 AttendanceDB  # Thư mục chứa cơ sở dữ liệu MongoDB backup  
├── 📂 AttendanceDB  # Thư mục chứa cơ sở dữ liệu MongoDB backup  
├── 📂 AttendanceDB  # Thư mục chứa cơ sở dữ liệu MongoDB backup  
├── 📂 AttendanceDB  # Thư mục chứa cơ sở dữ liệu MongoDB backup  


---



## 🛠️ CÔNG NGHỆ SỬ DỤNG

<div align="center">

### 📡 Phần cứng
[![Arduino](https://img.shields.io/badge/Arduino-00979D?style=for-the-badge&logo=arduino&logoColor=white)](https://www.arduino.cc/)
[![LED](https://img.shields.io/badge/LED-green?style=for-the-badge)]()
[![Buzzer](https://img.shields.io/badge/Buzzer-red?style=for-the-badge)]()
[![WiFi](https://img.shields.io/badge/WiFi-2.4GHz-orange?style=for-the-badge)]()

### 🖥️ Phần mềm
[![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)]()
[![MongoDB](https://img.shields.io/badge/MongoDB-4.x-green?style=for-the-badge&logo=mongodb)]()
[![Flask](https://img.shields.io/badge/Flask-Framework-black?style=for-the-badge&logo=flask)]()
[![Tkinter](https://img.shields.io/badge/Tkinter-GUI-yellow?style=for-the-badge)]()
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-blue?style=for-the-badge)]()

</div>

## 🛠️ Yêu cầu hệ thống

### 🔌 Phần cứng
- **Arduino Uno** (hoặc board tương thích) với **LED (2 màu) và còi**.
- **Cáp USB** để kết nối Arduino với máy tính.
- ⚠️ **Lưu ý:** Mặc định mã nguồn Arduino trong `ThongBao.ino` sử dụng cổng `COM5`. Nếu Arduino của bạn sử dụng cổng khác, hãy thay đổi biến `SERIAL_PORT` trong `chuongTrinh.py`.

### 💻 Phần mềm
- **🐍 Python 3+**
- **🍃 MongoDB** (kết nối mặc định: `mongodb://localhost:27017/`)
- **⚡ Arduino IDE** để nạp file `ThongBao.ino` lên board Arduino.

### 📦 Các thư viện Python cần thiết
Cài đặt các thư viện bằng lệnh:

    pip install pillow qrcode pymongo tkcalendar flask pyserial gtts pygame
## 🧮 Bảng mạch

### 🔩 Kết nối phần cứng:
<img src="images/Ketnoiphancung.png" alt="System Architecture" width="800"/>

### ⛓️‍💥 Hướng dẫn cắm dây
| Thiết bị        | Chân trên thiết bị | Kết nối Arduino UNO | Ghi chú                         |
|-----------------|-------------------|---------------------|---------------------------------|
| Breadboard      | -                 | -                   | Dùng để kết nối linh kiện       |
| Đèn LED xanh    | Anode (+), Cathode (-) | Anode → Digital Pin 9, Cathode → GND | Led thông báo khi sinh viên điểm danh đúng giờ|
| Đèn LED đỏ      | Anode (+), Cathode (-) | Anode → Digital Pin 10, Cathode → GND | Led thông báo khi sinh viên điểm danh muộn|
| Buzzer         | (+), (-)            | (+) → Digital Pin 11, (-) → GND |Còi thông báo khi sinh viên điểm danh muộn|
| 7 dây điện      | -                 | -                   | Dùng để nối các linh kiện       |

## 🚀 Hướng dẫn cài đặt và chạy
1️⃣ Chuẩn bị phần cứng
- **Nạp mã Arduino**:

    1. Mở file `ThongBao.ino` bằng Arduino IDE.
    2. Kết nối board Arduino với máy tính.
    3. Nạp (upload) mã nguồn lên board.
    4. Đảm bảo Arduino xuất hiện trên cổng COM5 (hoặc thay đổi trong `chuongTrinh.py` nếu cổng khác COM5).

2️⃣ Cài đặt thư viện Python. 

Cài đặt Python 3 nếu chưa có, sau đó cài đặt các thư viện cần thiết bằng pip.

3️⃣ Cấu hình MongoDB
- Cài đặt MongoDB nếu chưa có.
- Khởi động MongoDB và đảm bảo đang hoạt động tại `mongodb://localhost:27017/`.
- Khôi phục cơ sở dữ liệu từ bản sao lưu:

        mongorestore --db AttendanceDB "đường-dẫn-đến-thư-mục-AttendanceDB"
- Ví dụ:

        mongorestore --db AttendanceDB "C:\Users\LENOVO\Documents\Demo2QR\AttendanceDB"
📌 Lưu ý:
-	Tránh trùng lặp cơ sở dữ liệu: Trước khi thực hiện restore, hãy kiểm tra xem MongoDB đã có cơ sở dữ liệu tên AttendanceDB chưa. Nếu có, bạn có thể gặp lỗi hoặc dữ liệu cũ có thể bị ghi đè.
-	Đảm bảo MongoDB đang chạy: Nếu MongoDB chưa được khởi động, lệnh mongorestore sẽ không hoạt động.

4️⃣ Chạy các chương trình

Để đảm bảo hệ thống hoạt động đúng cách, bạn cần khởi chạy `chuongTrinh.py` trước, thay vì chạy từng file con riêng lẻ. File này cung cấp giao diện chính và bao gồm logic kết nối với Arduino board. Nếu chạy trực tiếp các file con, việc kết nối với Arduino sẽ không hoạt động.

✅ Chạy ứng dụng chính (`chuongTrinh.py`):

    python chuongTrinh.py
- Ứng dụng sẽ:

    - Khởi động **LED Service** tại `localhost:6000` để điều khiển LED và còi.
    - Hiển thị giao diện chính (Tkinter) với các nút: **Tạo mã QR** và **Xem điểm danh**

✅ Chạy ứng dụng quản lý điểm danh (`Diemdanh.py`):

    python Diemdanh.py

✅ Chạy ứng dụng tạo mã QR (`TaoQR.py`):

    python TaoQR.py

## 📖 Hướng dẫn sử dụng
1️⃣ Điểm danh qua QR code

- Sinh viên nhận email chứa mã QR.
- Khi quét mã, trình duyệt sẽ gửi yêu cầu điểm danh đến Flask server.
- Hệ thống kiểm tra tính hợp lệ và cập nhật vào MongoDB, đồng thời điều khiển Arduino:
    - ✅ Điểm danh đúng hạn → LED xanh.
    - ⏳ Điểm danh trễ → LED đỏ, còi, phát thông báo.
    
2️⃣ Quản lý sinh viên & mã QR
- Qua giao diện của TaoQR.py, bạn có thể:
    - Thêm, sửa, xóa thông tin sinh viên.
    - Nhập/xuất danh sách sinh viên từ/đến file CSV.
    - Tạo QR cho sinh viên theo lớp hoặc toàn bộ sinh viên.
    - Xóa mã QR cũ một cách thủ công.

3️⃣ Xem lịch sử điểm danh
- Qua giao diện của Diemdanh.py, bạn có thể:
    - Lọc danh sách điểm danh theo ngày, lớp, trạng thái.
    - Xuất dữ liệu điểm danh ra file CSV.
    - Hệ thống tự động cập nhật và chốt các phiên điểm danh.

## ⚙️ Cấu hình & Ghi chú

1. Cổng Arduino: 
- Mặc định sử dụng COM5, có thể cập nhật trong `chuongTrinh.py`.
2. Email gửi mã QR:
- Trong `TaoQR.py`, cập nhật thông tin *sender_email* và *sender_password*.(sender email là địa chỉ email gửi, sender password là mật khẩu ứng dụng của email đó.)
3. Thời gian hiệu lực mã QR: 
- Mã QR có hiệu lực 100 phút kể từ thời điểm tạo.
4. Môi trường mạng: 
- Thiết bị quét QR cần kết nối cùng mạng với máy chủ.

## 📰 Poster
<p align="center">
  <img src="images/PosterNhom1.png" alt="System Architecture" width="800"/>
</p>

## 🤝 Đóng góp
Dự án được phát triển bởi 4 thành viên:

| Họ và Tên       | Vai trò                  |
|-----------------|--------------------------|
| Nguyễn Nam Hưng | Phát triển toàn bộ mã nguồn, thiết kế cơ sở dữ liệu, kiểm thử, triển khai dự án và thực hiện video giới thiệu.|
| Hoàng Mạnh Linh | Biên soạn tài liệu Overleaf, Poster, Powerpoint, thuyết trình, đề xuất cải tiến, và hỗ trợ bài tập lớn.|
| Đào Đức Mạnh    | Thiết kế slide PowerPoint, hỗ trợ bài tập lớn.  |
| Cao Văn Huy     | Hỗ trợ bài tập lớn       |

© 2025 NHÓM 1, CNTT16-03, TRƯỜNG ĐẠI HỌC ĐẠI NAM





