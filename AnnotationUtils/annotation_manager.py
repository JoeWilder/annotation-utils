import numpy as np
from enum import Enum
from enum import Enum
from .coco_annotations import CocoAnnotations
from .yolo_annotations import YoloAnnotations
from typing import List, Tuple
import cv2
from random import randint


class AnnotationManager:
    """Used to manipulate a list of annotations into a more useable format"""

    class Format(Enum):
        COCO = "coco"
        YOLOSEG = "yoloseg"

    def __init__(self, raw_annotation_data: List[Tuple[str, str, np.ndarray[np.ndarray[bool]]]] = None):
        self._raw_ann_data = raw_annotation_data

    def _get_annotation_class(self, format: Format):
        if format == self.Format.COCO:
            return CocoAnnotations
        elif format == self.Format.YOLOSEG:
            return YoloAnnotations

    def load(self, data_path: str, format: Format):
        ann_class = self._get_annotation_class(format)
        self._raw_ann_data = ann_class.load(data_path)

    def add_annotation(self, raw_annotation_data: list[Tuple[str, str, np.ndarray[np.ndarray[bool]]]]):
        self._raw_ann_data.extend(raw_annotation_data)

    def add_annotation(self, image_path: str, label: str, boolean_mask: np.ndarray[np.ndarray[bool]]):
        self._raw_ann_data.append([image_path, label, boolean_mask])

    def convert(self, format: Format):
        annotation_class = self._get_annotation_class(format)
        ann_instance = annotation_class(self._raw_ann_data)
        ann_instance.convert()

    def write(self, format: Format, output_path: str):
        annotation_class = self._get_annotation_class(format)
        ann_instance = annotation_class(self._raw_ann_data)
        ann_instance.write(output_path)

    def display(self, format: Format | None = None):
        if format is None:
            self.display_base_data()
        else:
            annotation_class = self._get_annotation_class(format)
            ann_instance = annotation_class(self._raw_ann_data)
            ann_instance.display()

    def display_base_data(self):
        display_image_path, _, _ = self._raw_ann_data[0]
        image = cv2.imread(display_image_path)

        for image_path, _, annotation_mask in self._raw_ann_data:
            if display_image_path != image_path:
                continue
            if image.shape[:2] != annotation_mask.shape:
                raise ValueError("The dimensions of the image and the mask do not match")

            binary_mask = annotation_mask.astype(np.uint8) * 255

            if len(binary_mask.shape) != 2:
                raise ValueError("Binary mask must be a single-channel image")

            fill_color = (randint(0, 255), randint(0, 255), randint(0, 255))
            outline_color = tuple(max(0, c - 60) for c in fill_color)

            overlay = image.copy()

            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                cv2.polylines(overlay, [contour], isClosed=True, color=outline_color, thickness=5)
                cv2.fillPoly(overlay, [contour], fill_color)

            alpha = 0.5
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        cv2.imshow("Base Annotation", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
