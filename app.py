import streamlit as st
import cv2
import tempfile
from detector import Detector
from counter import VehicleCounter
from lane_detector import LaneDetector
from utils import draw_ui, draw_lanes

st.set_page_config(page_title="Vehicle Counter PRO", layout="wide")
st.title("Vehicle Counter PRO (YOLOv8 + ByteTrack)")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Cài đặt")
    line_y = st.slider("Vị trí counting line", 50, 600, 300)

    st.subheader("Phân tích hình học (3.2)")
    show_lanes   = st.checkbox("Hiển thị làn đường (Hough Lines)", value=True)
    show_edges   = st.checkbox("Hiển thị Canny Edge (so sánh input/output)", value=False)
    canny_low    = st.slider("Canny threshold thấp",  10, 100, 50)
    canny_high   = st.slider("Canny threshold cao",  100, 300, 150)
    hough_thresh = st.slider("Hough threshold",       20, 150,  50)

# ── Upload video ──────────────────────────────────────────────────────────────
video_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap          = cv2.VideoCapture(tfile.name)
    detector     = Detector("models/yolov8s.pt")
    counter      = VehicleCounter(line_y)
    lane_det     = LaneDetector(canny_low, canny_high, hough_thresh)

    # Layout: nếu bật edge view thì chia 2 cột
    if show_edges:
        col_left, col_right = st.columns(2)
        with col_left:
            st.caption("Output — Detection + Lane Analysis")
            frame_window = st.image([])
        with col_right:
            st.caption("Canny Edge Map (input → edge)")
            edge_window = st.image([])
    else:
        frame_window = st.image([])

    stats_text = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 3.1 Tiền xử lý: resize
        frame = cv2.resize(frame, (960, 540))

        # 3.2 Phân tích hình học: phát hiện làn đường
        lanes = lane_det.detect(frame) if show_lanes else []

        # 3.3 Object detection + tracking
        detections = detector.detect(frame)
        counter.update(detections)

        up, down      = counter.get_total()
        class_totals  = counter.get_class_totals()

        # 3.4 Trực quan hóa
        frame_draw = draw_lanes(frame.copy(), lanes)
        frame_draw = draw_ui(
            frame_draw,
            detections,
            line_y,
            up, down,
            class_totals["car"],
            class_totals["truck"],
        )

        frame_window.image(frame_draw, channels="BGR")

        # So sánh input vs output (edge map)
        if show_edges:
            edge_frame = lane_det.get_edge_frame(frame)
            edge_window.image(edge_frame, channels="BGR")

        stats_text.write(
            f"**Trạng thái:** Đang xử lý... | "
            f"**Tổng:** {up + down} xe | "
            f"**Làn phát hiện:** {len(lanes)} đường"
        )

    cap.release()
    st.success("Đã xử lý xong video!")
