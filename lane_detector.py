import cv2
import numpy as np


class LaneDetector:
    """
    Phân tích hình học cảnh giao thông (Mục 3.2).
    Pipeline: Grayscale → Gaussian Blur → Canny Edge → ROI Mask → Hough Lines → Angle Filter
    """

    def __init__(self, canny_low=50, canny_high=150, hough_threshold=80):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold

    def _to_grayscale(self, frame):
        """Chuyển đổi không gian màu BGR → Grayscale (Mục 3.1)."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _reduce_noise(self, gray):
        """Giảm nhiễu bằng Gaussian Blur (Mục 3.1)."""
        return cv2.GaussianBlur(gray, (7, 7), 0)

    def _detect_edges(self, blurred):
        """Phát hiện biên bằng Canny (Mục 3.2 - edges)."""
        return cv2.Canny(blurred, self.canny_low, self.canny_high)

    def _apply_roi(self, edges, frame_shape):
        """
        Giới hạn vùng quan tâm — phần mặt đường,
        loại bỏ bầu trời và vùng không liên quan.
        """
        h, w = frame_shape[:2]
        mask = np.zeros_like(edges)

        # ROI: cắt lề trái (hàng rào) và lề phải, chỉ giữ phần mặt đường
        # Bỏ 22% bên trái (hàng rào chắn), bỏ 8% bên phải (lề cỏ)
        roi_vertices = np.array([[
            (int(w * 0.22), h),
            (int(w * 0.22), int(h * 0.35)),
            (int(w * 0.92), int(h * 0.35)),
            (int(w * 0.92), h),
        ]], dtype=np.int32)

        cv2.fillPoly(mask, roi_vertices, 255)
        return cv2.bitwise_and(edges, mask)

    def _detect_lines(self, roi_edges):
        """Phát hiện đường thẳng bằng Hough Transform (Mục 3.2 - lines)."""
        return cv2.HoughLinesP(
            roi_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=80,
            maxLineGap=30,
        )

    def _filter_lines(self, lines, frame_w, top_n=6):
        """
        Lọc và giữ top_n đường dài nhất.
        - Loại đường gần nằm ngang (< 20°)
        - Loại đường nằm quá sát lề trái (x trung bình < 28% width) → hàng rào
        """
        scored = []
        for (x1, y1, x2, y2) in lines:
            dx = x2 - x1
            dy = y2 - y1
            length = np.hypot(dx, dy)
            angle  = abs(np.degrees(np.arctan2(dy, dx))) if dx != 0 else 90.0
            x_mid  = (x1 + x2) / 2

            if angle < 20:
                continue
            if x_mid < frame_w * 0.28:   # vẫn bị bắt hàng rào trái
                continue

            scored.append((length, (x1, y1, x2, y2)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [line for _, line in scored[:top_n]]

    def get_edge_frame(self, frame):
        """Trả về ảnh cạnh (Canny) để so sánh input vs output (Mục 3.4)."""
        gray    = self._to_grayscale(frame)
        blurred = self._reduce_noise(gray)
        edges   = self._detect_edges(blurred)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    def detect(self, frame):
        """
        Chạy toàn bộ pipeline phát hiện làn đường.
        Trả về danh sách các đường (x1, y1, x2, y2) đã lọc.
        """
        gray    = self._to_grayscale(frame)
        blurred = self._reduce_noise(gray)
        edges   = self._detect_edges(blurred)
        roi     = self._apply_roi(edges, frame.shape)
        raw     = self._detect_lines(roi)

        if raw is None:
            return []

        h, w  = frame.shape[:2]
        lines = [tuple(line[0]) for line in raw]
        return self._filter_lines(lines, frame_w=w)
