from ultralytics import YOLO

class Detector:
    def __init__(self, model_path):
        # Load model YOLOv8
        self.model = YOLO(model_path)

    def detect(self, frame):
        # Sử dụng ByteTrack tích hợp sẵn
        # persist=True: Giữ ID của vật thể qua các khung hình
        # tracker="bytetrack.yaml": Thuật toán tracking tốc độ cao
        results = self.model.track(
            frame, 
            persist=True, 
            tracker="bytetrack.yaml", 
            conf=0.4, 
            iou=0.5,
            verbose=False
        )[0]

        outputs = []
        
        # Kiểm tra nếu có vật thể được track thành công
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.int().cpu().tolist()
            ids = results.boxes.id.int().cpu().tolist()
            clss = results.boxes.cls.int().cpu().tolist()

            for box, track_id, cls_id in zip(boxes, ids, clss):
                # 2: car, 5: bus, 7: truck
                if cls_id in [2, 5, 7]:
                    outputs.append({
                        "id": track_id,
                        "bbox": box, # [x1, y1, x2, y2]
                        "cls": cls_id
                    })

        return outputs