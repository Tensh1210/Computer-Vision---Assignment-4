import cv2

COLORS = {
    2: (0, 255, 0),   # car
    5: (0, 255, 0),   # bus = car
    7: (0, 0, 255),   # truck
}

LABELS = {
    2: "Car",
    5: "Car",
    7: "Truck",
}

#def draw_ui(frame, detections, line_y, up, down):
def draw_ui(frame, detections, line_y, up, down, car_total, truck_total):
    h, w = frame.shape[:2]

    # line
    cv2.line(frame, (0, line_y), (w, line_y), (255, 0, 0), 2)

    for obj in detections:
        x1, y1, x2, y2 = obj["bbox"]
        track_id = obj["id"]
        cls_id = obj["cls"]

        color = COLORS.get(cls_id, (255,255,255))
        label = LABELS.get(cls_id, "Obj")

        cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
        cv2.putText(frame, f"{label}-{track_id}", (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    total = up + down

    cv2.putText(frame, f"TOTAL: {total}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    cv2.putText(frame, f"UP: {up}", (20,70),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    cv2.putText(frame, f"DOWN: {down}", (20,100),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    cv2.putText(frame, f"CAR: {car_total}", (20,140),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    cv2.putText(frame, f"TRUCK: {truck_total}", (20,170),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    return frame


def draw_lanes(frame, lines):
    """Vẽ các đường làn đường phát hiện được lên frame (Mục 3.2)."""
    for (x1, y1, x2, y2) in lines:
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    return frame