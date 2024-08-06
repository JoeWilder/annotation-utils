import numpy as np
import os
import cv2
from enum import Enum
import json
from annotation import ImageAnnotation
from random import randint


class AnnotationHandler:
    """Used to manipulate a list of annotations into a more useable format"""

    def __init__(self, mask_list: list[str, str, np.ndarray[bool]]):
        self._format = self.Format.MASK
        self._annotation_list: list[ImageAnnotation] = self._populate_annotation_list(mask_list)
        self._converted_annotations = None

    class Format(Enum):
        MASK = 1
        COCO = 2
        YOLOSEG = 3

    def _populate_annotation_list(self, mask_list: list[str, str, np.ndarray[np.ndarray[bool]]]) -> list[ImageAnnotation]:
        annotation_list = []
        for entry in mask_list:
            ann = ImageAnnotation(entry[2])
            ann.label = entry[1]
            ann.image = entry[0]
            annotation_list.append(ann)
        return annotation_list

    def add_annotation(self, image_path: str, label: str, mask: np.ndarray[bool]):
        ann = ImageAnnotation(mask)
        ann.label = label
        ann.image = image_path
        self._annotation_list.append(ann)
        self._format = self.Format.MASK

    def display_base(self):
        image_to_ann = self._annotation_list[0].image

        cv2.namedWindow("Base Annotation", cv2.WINDOW_NORMAL)

        image = cv2.imread(image_to_ann)

        for annotation in self._annotation_list:
            if image_to_ann != annotation.image:
                continue

            if image.shape[:2] != annotation.mask.shape:
                raise ValueError("The dimensions of the image and the mask do not match")

            binary_mask = annotation.mask.astype(np.uint8) * 255

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

    def convert_coco(self, description: str = "coco dataset", version: float = 1.0, year: int = 2024, contributor: str = "", date_created: str = ""):
        coco_dict = {
            "info": {"description": description, "version": version, "year": year, "contributor": contributor, "date_created": date_created},
            "licenses": [{"id": 1, "name": "Unknown", "url": "Unknown"}],
            "images": [],
            "annotations": [],
            "categories": [],
        }

        image_map = {}
        category_map = {}
        annotation_id = 1

        for annotation in self._annotation_list:

            contours, _ = cv2.findContours(annotation.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            segmentation = []
            for contour in contours:
                contour = contour.flatten().tolist()
                segmentation.append(contour)

            image_path = annotation.image
            if not image_path in image_map.keys():
                image_id = len(image_map)
                image_map[image_path] = len(image_map)
                im = cv2.imread(image_path)
                h, w, _ = im.shape
                coco_dict["images"].append({"id": image_id, "file_name": image_path, "height": h, "width": w})
            else:
                image_id = image_map[image_path]

            label = annotation.label
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

        self._format = self.Format.COCO
        self._converted_annotations = coco_dict

    def write_coco(self, output_path: str = "coco-annotations.json"):

        if not self._format is self.Format.COCO:
            self.convert_coco()

        if os.path.exists(output_path):
            raise OSError(f"The following file already exists: {output_path}")

        with open(output_path, "w") as json_file:
            json.dump(self._converted_annotations, json_file, indent=4)

    def display_coco(self):
        if not self._format is self.Format.COCO:
            self.convert_coco()

        cv2.namedWindow("COCO Annotation", cv2.WINDOW_NORMAL)

        coco_dict = self._converted_annotations
        images = coco_dict["images"]
        image_path = images[0]["file_name"]
        image_id = images[0]["id"]

        image = cv2.imread(image_path)

        for annotation in coco_dict["annotations"]:
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

    def convert_yolo(self):

        annotations = []
        category_map = {}

        for annotation in self._annotation_list:

            label = annotation.label
            if not label in category_map.keys():
                category_id = len(category_map)
                category_map[label] = len(category_map)
            else:
                category_id = category_map[label]

            image_filename = annotation.image
            im = cv2.imread(image_filename)
            image_height, image_width, _ = im.shape

            contours, _ = cv2.findContours(annotation.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            segmentation = []
            for contour in contours:
                contour = contour.flatten().tolist()
                segmentation.append(contour)

            yolo_segmentation = [f"{(x) / image_width:.5f} {(y) / image_height:.5f}" for x, y in zip(segmentation[0][::2], segmentation[0][1::2])]
            yolo_segmentation = " ".join(yolo_segmentation)

            yolo_annotation = f"{category_id} {yolo_segmentation}"

            done = False
            for item in annotations:
                if image_filename in item:
                    item[1].append(yolo_annotation)
                    done = True
                    break

            if not done:
                annotations.append((image_filename, [yolo_annotation]))

        self._format = self.Format.YOLOSEG
        self._converted_annotations = annotations

    def write_yolo(self, output_path: str = "labels"):
        if not self._format is self.Format.YOLOSEG:
            self.convert_yolo()

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for entry in self._converted_annotations:
            file_name = f"{os.path.splitext(os.path.basename(entry[0]))[0]}.txt"
            output_file = os.path.join(output_path, file_name)
            if os.path.exists(output_file):
                raise OSError(f"The following file already exists: {output_file}")
            with open(output_file, "w") as file:
                for annotation_line in entry[1]:
                    file.write(f"{annotation_line}\n")

    def display_yolo(self):

        cv2.namedWindow("YOLO Annotation", cv2.WINDOW_NORMAL)

        if not self._format is self.Format.YOLOSEG:
            self.convert_yolo()
        file_name = self._converted_annotations[0][0]
        img = cv2.imread(file_name)
        h, w = img.shape[:2]
        for label in self._converted_annotations[0][1]:
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

    def display_sample(self):
        if self._format is self.Format.COCO:
            self.display_coco()
        elif self._format is self.Format.YOLOSEG:
            self.display_yolo()
        else:
            self.display_base()
