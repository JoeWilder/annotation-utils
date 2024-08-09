import numpy as np
import cv2
from random import randint
import os
from typing import List, Tuple

from .base_annotations import BaseAnnotations


class YolosegAnnotations(BaseAnnotations):
    def __init__(self, annotatation_data: List[Tuple[str, str, np.ndarray[np.ndarray[bool]]]]):
        super().__init__(annotatation_data)
        self._raw_ann_data = annotatation_data
        self.default_path = os.getcwd()

    @staticmethod
    def default_path() -> str:
        return os.getcwd()

    def convert(self):
        annotations = []
        category_map = {}

        for image_path, label, annotation_mask in self._raw_ann_data:

            if not label in category_map.keys():
                category_id = len(category_map)
                category_map[label] = len(category_map)
            else:
                category_id = category_map[label]

            im = cv2.imread(image_path)
            image_height, image_width, _ = im.shape

            contours, _ = cv2.findContours(annotation_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            segmentation = []
            for contour in contours:
                contour = contour.flatten().tolist()
                segmentation.append(contour)

            yolo_segmentation = [f"{(x) / image_width:.5f} {(y) / image_height:.5f}" for x, y in zip(segmentation[0][::2], segmentation[0][1::2])]
            yolo_segmentation = " ".join(yolo_segmentation)

            yolo_annotation = f"{category_id} {yolo_segmentation}"

            done = False
            for item in annotations:
                if image_path in item:
                    item[1].append(yolo_annotation)
                    done = True
                    break

            if not done:
                annotations.append((image_path, [yolo_annotation]))

        return annotations

    @staticmethod
    def load(data_path: str):
        class_path = os.path.join(data_path, "classes.txt")

        if not os.path.isfile(class_path):
            raise FileNotFoundError(
                f"No YOLO classes.txt file found. Please create a classes.txt file that contains one label per line for each annotation class in the following location: {data_path}"
            )

        category_map = {}
        with open(class_path) as file:
            i = 0
            for line in file:
                category_map[i] = line

        mask_list = []

        for annotation_file in os.listdir(data_path):
            file_ext = os.path.splitext(annotation_file)[1]
            if file_ext != ".txt":
                continue

            if annotation_file == "classes.txt":
                continue

            base_name = os.path.splitext(annotation_file)[0]

            with open(os.path.join(data_path, annotation_file), "r") as file:
                for line in file:
                    path = None
                    for image_file in os.listdir(data_path):
                        split = os.path.splitext(image_file)
                        if split[1] == ".txt":
                            continue
                        if split[0] != base_name:
                            continue

                        path = os.path.join(data_path, image_file)
                    class_id, *poly = line.split(" ")
                    label = category_map.get(int(class_id))

                    if path is None:
                        raise FileNotFoundError(f"Could not find image '{annotation_file.replace(".txt", "")}' in {data_path}")

                    img = cv2.imread(path)
                    h, w = img.shape[:2]

                    poly = np.asarray(poly, dtype=np.float16).reshape(-1, 2)
                    poly *= [w, h]

                    mask = np.zeros((h, w), dtype=np.uint8)

                    poly = np.array(poly).reshape((-1, 2)).astype(np.int32)
                    cv2.fillPoly(mask, [poly], 1)

                    mask = mask.astype(bool)

                    entry = [path, label, mask]
                    mask_list.append(entry)

        return mask_list

    def write(self, output_path: str):

        converted_annotations = self.convert()

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for entry in converted_annotations:
            file_name = f"{os.path.splitext(os.path.basename(entry[0]))[0]}.txt"
            output_file = os.path.join(output_path, file_name)
            if os.path.exists(output_file):
                raise OSError(f"The following file already exists: {output_file}")
            with open(output_file, "w") as file:
                for annotation_line in entry[1]:
                    file.write(f"{annotation_line}\n")

    def display(self):
        cv2.namedWindow("YOLO Annotation", cv2.WINDOW_NORMAL)

        converted_annotations = self.convert()
        file_name = converted_annotations[0][0]
        img = cv2.imread(file_name)
        h, w = img.shape[:2]
        for label in converted_annotations[0][1]:
            class_id, *poly = label.split(" ")

            # Convert segmentation from 1D to 2D, then unscale points
            poly = np.asarray(poly, dtype=np.float16).reshape(-1, 2)
            poly *= [w, h]

            fill_color = (randint(0, 255), randint(0, 255), randint(0, 255))

            overlay = img.copy()
            cv2.fillPoly(overlay, [poly.astype(np.int32)], fill_color)
            line_color = tuple(max(0, c - 60) for c in fill_color)
            cv2.polylines(overlay, [poly.astype(np.int32)], isClosed=True, color=line_color, thickness=2)

            alpha = 0.4
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        cv2.imshow("YOLO Annotation", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
