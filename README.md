# BTL4 — Phân tích và trực quan hóa cảnh giao thông

Môn học: Xử lý ảnh số và Thị giác máy tính  
Giảng viên: TS. Võ Thanh Hùng  
Trường Đại học Bách Khoa — ĐHQG TP.HCM

---

## Mô tả

Hệ thống Computer Vision phân tích video giao thông ngoài trời theo thời gian thực, gồm hai nhánh song song:

- **Phát hiện làn đường**: Canny Edge Detection + Probabilistic Hough Transform, có ROI masking và lọc theo góc/vị trí.
- **Nhận diện & theo dõi xe**: YOLOv8s (object detection) + ByteTrack (multi-object tracking), đếm xe theo hướng UP/DOWN qua vạch ảo.

Toàn bộ kết quả được trực quan hóa trên giao diện **Streamlit** với 4 thanh điều chỉnh tham số thời gian thực.

---

## Cấu trúc thư mục

```
BTL4.1/
├── app.py              # Giao diện Streamlit chính
├── lane_detector.py    # Module phát hiện làn đường (Canny + Hough)
├── detector.py         # Module nhận diện & theo dõi (YOLOv8 + ByteTrack)
├── counter.py          # Module đếm xe theo hướng
├── utils.py            # Hàm vẽ UI và kết quả
├── requirements.txt    # Danh sách thư viện
├── video.mp4           # Video đầu vào mẫu
├── models/             # Thư mục chứa file trọng số YOLOv8
│   └── yolov8s.pt
└── report/             # Báo cáo LaTeX
```

---

## Yêu cầu hệ thống

- Python 3.10+
- (Khuyến nghị) GPU với CUDA để tăng tốc YOLOv8

---

## Cài đặt

```bash
# 1. Clone repository
git clone https://github.com/<username>/<repo-name>.git
cd BTL4.1

# 2. Cài đặt thư viện
pip install -r requirements.txt
```

> **Lưu ý:** `ultralytics` sẽ tự tải `yolov8s.pt` lần đầu nếu chưa có trong thư mục `models/`.

---

## Chạy ứng dụng

```bash
streamlit run app.py
```

Trình duyệt sẽ mở tự động tại `http://localhost:8501`.

### Hướng dẫn sử dụng

1. **Upload video**: Nhấn "Browse files" ở sidebar, chọn file `.mp4` / `.avi` / `.mov`.
2. **Điều chỉnh tham số** (sidebar):
   - `line_y` [50–600]: Vị trí vạch đếm (pixel theo chiều dọc).
   - `Canny Low` [10–100]: Ngưỡng thấp Canny (mặc định 50).
   - `Canny High` [100–300]: Ngưỡng cao Canny (mặc định 150).
   - `Hough Threshold` [20–150]: Ngưỡng tích lũy Hough (mặc định 50).
3. **Bật/tắt tùy chọn**:
   - ☑ *Hiển thị làn đường (Hough Lines)*: vẽ đường vàng lên frame.
   - ☑ *Hiển thị Canny Edge*: hiển thị edge map ở cột phải (chế độ 2 cột).
4. Nhấn **"Xử lý video"** để bắt đầu. Kết quả TOTAL / UP / DOWN / CAR / TRUCK hiển thị theo thời gian thực ở góc trên trái mỗi frame.

---

## Kết quả đầu ra

| Thành phần | Mô tả |
|------------|-------|
| Bounding box xanh lá | Ô tô và bus (nhãn `Car-ID`) |
| Bounding box đỏ | Xe tải (nhãn `Truck-ID`) |
| Đường vàng | Làn đường phát hiện bởi Hough Lines |
| Vạch xanh dương | Vạch đếm ảo tại vị trí `line_y` |
| Góc trên trái | Thống kê TOTAL / UP / DOWN / CAR / TRUCK |
