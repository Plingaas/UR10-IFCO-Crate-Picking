import cupy as cp
from ultralytics import YOLO
from utils.helper import print_with_time


class YoloSegmenter:
    def __init__(self, model_path="data/yolo_crate_pallet_segmenter.pt", conf=0.9) -> None:
        super().__init__()
        self.conf = conf
        try:
            self.model = YOLO(model_path).to("cuda")
            print_with_time("YOLO", "YOLO model loaded to GPU")
        except Exception:
            self.model = YOLO(model_path)
            print("YOLO", "YOLO model loaded to CPU")

    def predict(self, image):
        results = self.model.predict(image, show=False, retina_masks=True, verbose=False, conf=self.conf, device=0)

        masks, class_ids = None, None

        if not results[0].masks:
            return None

        print_with_time("Yolo", f"Found {len(results[0].masks)} objects.")

        masks_cp = results[0].masks.data  # (N, H, W)
        class_ids_cp = results[0].boxes.cls  # (N,)
        confidences_cp = results[0].boxes.conf  # (N,)
        boxes_cp = results[0].boxes.xyxy  # (N, 4)

        # Convert to numpy
        masks = cp.asnumpy(masks_cp)
        class_ids = cp.asnumpy(class_ids_cp).astype(int)
        confidences = cp.asnumpy(confidences_cp)
        boxes = cp.asnumpy(boxes_cp)

        detections = []
        for i in range(masks.shape[0]):
            detection = {
                "bbox": boxes[i].tolist(),  # [x1, y1, x2, y2]
                "class_id": class_ids[i],  # int
                "confidence": confidences[i],  # float
                "mask": masks[i],  # np.ndarray (H, W)
            }
            detections.append(detection)

        return detections
