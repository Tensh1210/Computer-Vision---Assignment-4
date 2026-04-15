class VehicleCounter:
    def __init__(self, line_y):
        self.line_y = line_y

        self.prev_positions = {}
        self.counted_ids = set()

        self.counts = {
            "car": {"up": 0, "down": 0},
            "truck": {"up": 0, "down": 0},
        }

    def update(self, detections):
        for obj in detections:
            track_id = obj["id"]
            cls_id = obj["cls"]
            x1, y1, x2, y2 = obj["bbox"]

            cy = int((y1 + y2) / 2)

            if cls_id in [2, 5]:
                name = "car"
            elif cls_id == 7:
                name = "truck"
            else:
                continue

            if track_id in self.prev_positions:
                prev_y = self.prev_positions[track_id]

                crossed = False
                direction = None

                if prev_y < self.line_y and cy >= self.line_y:
                    crossed = True
                    direction = "down"

                elif prev_y > self.line_y and cy <= self.line_y:
                    crossed = True
                    direction = "up"

                if crossed:
                    if track_id in self.counted_ids:
                        continue

                    self.counts[name][direction] += 1
                    self.counted_ids.add(track_id)

            self.prev_positions[track_id] = cy
    
    def get_class_totals(self):
        return {
        "car": self.counts["car"]["up"] + self.counts["car"]["down"],
        "truck": self.counts["truck"]["up"] + self.counts["truck"]["down"],
    }     

    def get_total(self):
        up = sum(v["up"] for v in self.counts.values())
        down = sum(v["down"] for v in self.counts.values())
        return up, down

    def get_counts(self):
        return self.counts