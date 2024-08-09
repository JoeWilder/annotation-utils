import numpy as np
import cv2
from random import randint
import os
import json
from typing import List, Tuple

from .base_annotations import BaseAnnotations


class CocoAnnotations(BaseAnnotations):
    def __init__(self, annotatation_data: List[Tuple[str, str, np.ndarray[np.ndarray[bool]]]]):
        super().__init__(annotatation_data)
        self._raw_ann_data = annotatation_data

    @staticmethod
    def default_path() -> str:
        return "coco-annotations.json"

    def convert(self):
        coco_dict = {
            "info": {"description": "", "version": "", "year": "", "contributor": "", "date_created": ""},
            "licenses": [{"id": 1, "name": "Unknown", "url": "Unknown"}],
            "images": [],
            "annotations": [],
            "categories": [],
        }

        image_map = {}
        category_map = {}
        annotation_id = 1

        for image_path, label, annotation_mask in self._raw_ann_data:

            contours, _ = cv2.findContours(annotation_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            segmentation = []
            for contour in contours:
                contour = contour.flatten().tolist()
                segmentation.append(contour)

            if not image_path in image_map.keys():
                image_id = len(image_map)
                image_map[image_path] = len(image_map)
                im = cv2.imread(image_path)
                h, w, _ = im.shape
                coco_dict["images"].append({"id": image_id, "file_name": image_path, "height": h, "width": w})
            else:
                image_id = image_map[image_path]

            label = label
            if not label in category_map.keys():
                category_id = len(category_map)
                category_map[label] = len(category_map)
                coco_dict["categories"].append({"id": category_id, "name": label, "supercategory": "object"})
            else:
                category_id = category_map[label]

            area = cv2.contourArea(np.array(segmentation).reshape(-1, 2).astype(np.float32))
            x, y, w, h = cv2.boundingRect(np.array(segmentation).reshape(-1, 2).astype(np.float32))
            area = float(area)
            bbox = [float(x), float(y), float(w), float(h)]
            coco_dict["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": segmentation,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

        return coco_dict

    @staticmethod
    def load(data_path: str):
        f = open(data_path)
        coco_data = json.load(f)
        mask_list: list[str, str, np.ndarray[bool]] = []

        image_map = {}
        for image_data in coco_data["images"]:
            image_id = image_data["id"]
            if image_id not in image_map.keys():
                im = cv2.imread(image_data["file_name"])
                image_height, image_width, _ = im.shape
                image_map[image_data["id"]] = (image_data["file_name"], (image_height, image_width))

        category_map = {}
        for category_data in coco_data["categories"]:
            category_id = category_data["id"]
            if category_id not in category_map.keys():
                category_map[category_data["id"]] = category_data["name"]

        for annotation in coco_data["annotations"]:
            path, size = image_map.get(annotation["image_id"])
            label = category_map.get(annotation["category_id"])

            height, width = size
            mask = np.zeros((height, width), dtype=np.uint8)

            for segmentation in annotation["segmentation"]:
                poly = np.array(segmentation).reshape((-1, 2)).astype(np.int32)
                cv2.fillPoly(mask, [poly], 1)

            mask = mask.astype(bool)

            entry = [path, label, mask]
            mask_list.append(entry)

        return mask_list

    def write(self, output_path: str):
        converted_annotations = self.convert()

        if os.path.exists(output_path):
            raise OSError(f"The following file already exists: {output_path}")

        with open(output_path, "w") as json_file:
            json.dump(converted_annotations, json_file, indent=4)

    def display(self):

        converted_annotations = self.convert()

        cv2.namedWindow("COCO Annotation", cv2.WINDOW_NORMAL)

        images = converted_annotations["images"]
        image_path = images[0]["file_name"]
        image_id = images[0]["id"]

        image = cv2.imread(image_path)

        for annotation in converted_annotations["annotations"]:
            if annotation["image_id"] == image_id:
                segmentation = annotation["segmentation"]
                segmentation_points = np.array(segmentation, dtype=np.int32).reshape((-1, 2))

                fill_color = (randint(0, 255), randint(0, 255), randint(0, 255))
                line_color = tuple(max(0, c - 60) for c in fill_color)

                overlay = image.copy()
                cv2.fillPoly(overlay, [segmentation_points], fill_color)
                cv2.polylines(overlay, [segmentation_points], isClosed=True, color=line_color, thickness=2)

                alpha = 0.4
                cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        cv2.imshow("COCO Annotation", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
