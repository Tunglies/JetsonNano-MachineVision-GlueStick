from src.logger import logging
from src.utils import Utils

class BasicModel:
    def __init__(self, weights_file: str, conf_threshold: float = 0.45) -> None:
        self.weights = Utils.get_data_dir().joinpath("weights").joinpath(weights_file)
        logging.info(
            "Checking Weights File: %s, Exists: %s",
            self.weights.name,
            self.weights.exists(),
        )
        self.weights = self.weights.as_posix()

        self.device = ...
        self.model = ...
        self.conf_threshold: float = conf_threshold

    def pre_process(self, frame): ...

    def predict(self, frame, rate: float = 0.45): ...

    def post_process(self, outputs, frame, threshold: float = 0.4): ...

    def extract_boxes(self, boxes): ...


class Pytorch(BasicModel):
    def __init__(self, weights_file: str = "best.pt") -> None:
        super().__init__(weights_file)
        import torch

        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        logging.debug("Checking Available Device: %s", self.device)

        logging.info("Loading Model To Device: %s", self.device)
        self.model = torch.load(self.weights, map_location=self.device)


# Yolov8
class Onnx(BasicModel):
    def __init__(
        self, weights_file: str = "best.onnx", conf_threshold: float = 0.45
    ) -> None:
        super().__init__(weights_file, conf_threshold)
        import cv2
        import onnxruntime as ort
        import numpy as np

        self.cv2 = cv2
        self.ort = ort
        self.np = np

        self.model = ort.InferenceSession(self.weights, providers=['CUDAExecutionProvider'])
        self.model_outputs = self.model.get_outputs()
        self.model_inputs = self.model.get_inputs()
        self.input_names = [input.name for input in self.model_inputs]
        self.output_names = [output.name for output in self.model_outputs]
        self.input_shape = self.model_inputs[0].shape
        self.input_height, self.input_width = self.input_shape[2], self.input_shape[3]

        self.num_classes = 1
        self.label = "GlueSticker"
        self.resize_shape = (640, 640)
        self.font_scale = 640 * 0.0009
        self.thickness = 2
        self.color = (0, 0, 255)

    def pre_process(self, frame):
        self.img_height, self.img_width = frame.shape[:2]
        frame = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        frame = self.cv2.resize(frame, self.resize_shape)
        frame = frame.transpose(2, 0, 1) / 255.0
        return frame[self.np.newaxis, :, :, :].astype(self.np.float32)

    def predict(self, frame):
        return self.model.run(self.output_names, {self.input_names[0]: frame})

    def post_process(self, outputs, frame):
        self.num_classes = 1
        predictions = self.np.squeeze(outputs).T

        scores = self.np.max(predictions[:, 4 : 4 + self.num_classes], axis=1)
        mask = scores > self.conf_threshold
        if not self.np.any(mask):
            return False

        predictions = predictions[mask]
        scores = scores[mask]
        box_predictions = predictions[..., : self.num_classes + 4]

        class_ids = self.np.argmax(box_predictions[:, 4:], axis=1)
        boxes = self.extract_boxes(box_predictions)
        indices = self.nms(boxes, scores)
        boxes = boxes[indices]

        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box.astype(int)
            self.cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, 2)

            caption = f"{self.label} {int(score * 100)}%"
            (tw, th), _ = self.cv2.getTextSize(
                text=caption,
                fontFace=self.cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=self.font_scale,
                thickness=self.thickness,
            )
            th = int(th * 1.2)

            self.cv2.rectangle(frame, (x1, y1), (x1 + tw, y1 - th), self.color, -1)
            self.cv2.putText(
                frame,
                caption,
                (x1, y1),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),
                self.thickness,
                self.cv2.LINE_AA,
            )
            
        return frame

    def xywh2xyxy(self, x):
        y = self.np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def rescale_boxes(self, boxes, input_shape, image_shape):
        input_shape = self.np.array(
            [input_shape[1], input_shape[0], input_shape[1], input_shape[0]]
        )
        boxes = (
            boxes
            / input_shape
            * self.np.array(
                [image_shape[1], image_shape[0], image_shape[1], image_shape[0]]
            )
        )
        return boxes

    def extract_boxes(self, box_predictions):
        boxes = box_predictions[:, :4]
        boxes = self.rescale_boxes(
            boxes,
            (self.input_height, self.input_width),
            (self.img_height, self.img_width),
        )
        boxes = self.xywh2xyxy(boxes)
        boxes[:, 0] = self.np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = self.np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = self.np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = self.np.clip(boxes[:, 3], 0, self.img_height)
        return boxes

    def nms(self, boxes, scores):
        sorted_indices = self.np.argsort(scores)[::-1]
        keep_boxes = []
        while sorted_indices.size > 0:
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)
            ious = self.compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
            keep_indices = self.np.where(ious < 0.45)[0]
            sorted_indices = sorted_indices[keep_indices + 1]
        return keep_boxes

    def compute_iou(self, box, boxes):
        xmin = self.np.maximum(box[0], boxes[:, 0])
        ymin = self.np.maximum(box[1], boxes[:, 1])
        xmax = self.np.minimum(box[2], boxes[:, 2])
        ymax = self.np.minimum(box[3], boxes[:, 3])

        intersection_area = self.np.maximum(0, xmax - xmin) * self.np.maximum(
            0, ymax - ymin
        )

        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area

        iou = intersection_area / union_area
        return iou


class Engine(BasicModel):
    def __init__(self, weights_file: str = "best.engine") -> None:
        super().__init__(weights_file)
