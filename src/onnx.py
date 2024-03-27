from src.model import BasicModel
import cv2
import onnxruntime as ort
import numpy as np


class Onnx(BasicModel):
    def __init__(
        self, onnx_weights_file: str = "best.onnx", conf_threshold: float = 0.45, resize_shape: tuple = (640, 640)
    ) -> None:
        super().__init__(onnx_weights_file, conf_threshold)
        self.model = ort.InferenceSession(self.weights, providers=['CUDAExecutionProvider'])
        self.model_outputs = self.model.get_outputs()
        self.model_inputs = self.model.get_inputs()
        self.input_names = [input.name for input in self.model_inputs]
        self.output_names = [output.name for output in self.model_outputs]
        self.input_shape = self.model_inputs[0].shape
        self.input_height, self.input_width = self.input_shape[2], self.input_shape[3]

        self.num_classes = 1
        self.label = "GlueSticker"
        self.resize_shape = resize_shape
        self.font_scale = self.resize_shape[0] * 0.0009
        self.thickness = 2
        self.color = (0, 0, 255)

    def load(self):
        return self

    def pre_process(self, frame):
        self.img_height, self.img_width = frame.shape[:2]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, self.resize_shape)
        frame = frame.transpose(2, 0, 1) / 255.0
        return frame[np.newaxis, :, :, :].astype(np.float32)

    def predict(self, frame):
        return self.model.run(self.output_names, {self.input_names[0]: frame})

    def post_process(self, outputs, frame):
        self.num_classes = 1
        predictions = np.squeeze(outputs).T

        scores = np.max(predictions[:, 4 : 4 + self.num_classes], axis=1)
        mask = scores > self.conf_threshold
        if not np.any(mask):
            return frame

        predictions = predictions[mask]
        scores = scores[mask]
        box_predictions = predictions[..., : self.num_classes + 4]

        class_ids = np.argmax(box_predictions[:, 4:], axis=1)
        boxes = self.extract_boxes(box_predictions)
        indices = self.nms(boxes, scores)
        boxes = boxes[indices]

        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, 2)
            caption = f"{self.label} {int(score * 100)}%"
            (tw, th), _ = cv2.getTextSize(
                text=caption,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=self.font_scale,
                thickness=self.thickness,
            )
            th = int(th * 1.2)

            cv2.rectangle(frame, (x1, y1), (x1 + tw, y1 - th), self.color, -1)
            cv2.putText(
                frame,
                caption,
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),
                self.thickness,
                cv2.LINE_AA,
            )
            
        return frame

    def xywh2xyxy(self, x):
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def rescale_boxes(self, boxes, input_shape, image_shape):
        input_shape = np.array(
            [input_shape[1], input_shape[0], input_shape[1], input_shape[0]]
        )
        boxes = (
            boxes
            / input_shape
            * np.array(
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
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)
        return boxes

    def nms(self, boxes, scores):
        sorted_indices = np.argsort(scores)[::-1]
        keep_boxes = []
        while sorted_indices.size > 0:
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)
            ious = self.compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
            keep_indices = np.where(ious < 0.45)[0]
            sorted_indices = sorted_indices[keep_indices + 1]
        return keep_boxes

    def compute_iou(self, box, boxes):
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])

        intersection_area = np.maximum(0, xmax - xmin) * np.maximum(
            0, ymax - ymin
        )

        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area

        iou = intersection_area / union_area
        return iou
    
    def detect(self, frame):
        pred = self.pre_process(frame)
        predict = self.predict(pred)
        output = self.post_process(predict, frame)
        return output