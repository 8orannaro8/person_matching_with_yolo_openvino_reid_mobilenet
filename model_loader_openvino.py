# model_loader_openvino.py

import cv2
import numpy as np
from openvino.runtime import Core

class DummyTensor:
    def __init__(self, array):
        self._array = array
    def cpu(self): return self
    def numpy(self): return self._array

class BoxWrapper:
    def __init__(self, boxes, classes):
        self.xyxy = DummyTensor(np.array(boxes, dtype=np.float32))
        self.cls  = DummyTensor(np.array(classes, dtype=np.int64))

class Result:
    def __init__(self, boxes, classes):
        self.boxes = BoxWrapper(boxes, classes)

class OpenVINOYOLOv5:
    def __init__(self, model_xml_path, device='CPU',
                 conf_threshold=0.25, iou_threshold=0.45):
        self.ie = Core()
        self.model_ir = self.ie.read_model(model=model_xml_path)
        self.compiled = self.ie.compile_model(self.model_ir, device_name=device)
        self.input_layer = self.compiled.input(0)
        self.output_layer = self.compiled.output(0)
        self.conf_thres = conf_threshold
        self.iou_thres  = iou_threshold
        _, _, self.H, self.W = self.input_layer.shape
        print(f"[INFO] OpenVINO YOLOv5 loaded: {model_xml_path}, input_shape={self.input_layer.shape}")

    def preprocess(self, frame):
        img = cv2.resize(frame, (self.W, self.H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)

    def postprocess(self, raw, orig_w, orig_h):
        # raw: (1, N, 85)
        pred = raw[0]                # (N,85)
        xywh      = pred[:, :4]      # (N,4)
        obj_conf  = pred[:, 4]       # (N,)
        cls_conf  = pred[:, 5:]      # (N,80)

        # 1) 가장 높은 클래스 인덱스
        class_ids = np.argmax(cls_conf, axis=1)  # (N,)

        # 2) 각 행에 대해 그 클래스의 confidence만 추출
        #    방법 A: fancy indexing
        class_scores = cls_conf[np.arange(cls_conf.shape[0]), class_ids]  # (N,)

        #    또는 방법 B: take_along_axis
        # class_scores = np.take_along_axis(cls_conf,
        #                                   class_ids[:, None],
        #                                   axis=1).flatten()

        # 3) 최종 confidence = objectness * class_score
        confidences = obj_conf * class_scores    # (N,)

        # 이어서 threshold 필터링, xywh→xyxy 변환, NMS 적용…
        mask = (confidences >= self.conf_thres) & (class_ids == 0)
        xywh        = xywh[mask]
        confidences = confidences[mask]
        # … 이하 동일 …

        # no need to keep other classes
        # 5) xywh -> xyxy (on 416×416 scale)
        #    then scale to original frame size
        boxes = []
        for (x, y, w, h) in xywh:
            x1 = x - w/2;  y1 = y - h/2
            x2 = x + w/2;  y2 = y + h/2
            # scale to orig image
            boxes.append([
                x1 * (orig_w / self.W),
                y1 * (orig_h / self.H),
                x2 * (orig_w / self.W),
                y2 * (orig_h / self.H)
            ])

        # 6) NMS
        # cv2.dnn.NMSBoxes expects int boxes, so convert
        int_boxes = [[int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1])] 
                     for b in boxes]
        idxs = cv2.dnn.NMSBoxes(int_boxes, confidences.tolist(),
                                self.conf_thres, self.iou_thres)
        final_boxes, final_classes = [], []
        if len(idxs) > 0:
            for i in idxs.flatten():
                final_boxes.append(boxes[i])
                final_classes.append(0)  # only person class
        return final_boxes, final_classes

    def __call__(self, frame):
        orig_h, orig_w = frame.shape[:2]
        input_tensor = self.preprocess(frame)
        raw = self.compiled([input_tensor])[self.output_layer]
        boxes, classes = self.postprocess(raw, orig_w, orig_h)
        return [Result(boxes, classes)]
