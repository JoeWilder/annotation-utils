import numpy as np
import os
import cv2
from random import randint


class ImageAnnotation:
    """Represent an image mask annotation"""

    def __init__(self, mask: np.ndarray[np.ndarray[bool]]):
        self.mask = mask
        self.label = None
        self.image = None

    def get_mask(self):
        return self.mask

    def set_mask(self, mask: np.ndarray[np.ndarray[bool]]):
        self.mask = mask

    @staticmethod
    def is_valid_file(image_path: str, supported_extensions: list[str] = ["png", "jpg", "jpeg"]):
        return image_path.split(".")[-1] in supported_extensions

    def display(self):
        """Display the mask on the given image"""
        image_path = self.image
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Invalid file provided: {image_path}")
        if not self.is_valid_file(image_path):
            raise ValueError(f"The following file type is not supported: {image_path.split('.')[-1]}")

        cv2.namedWindow("Base Annotation", cv2.WINDOW_NORMAL)
        image = cv2.imread(image_path)

        if image.shape[:2] != self.mask.shape:
            raise ValueError("The dimensions of the image and the mask do not match")

        binary_mask = self.mask.astype(np.uint8) * 255

        if len(binary_mask.shape) != 2:
            raise ValueError("Binary mask must be a single-channel image")

        fill_color = (randint(0, 255), randint(0, 255), randint(0, 255))
        color_overlay = np.zeros_like(image)
        color_overlay[:] = fill_color

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            outline_color = tuple(max(0, c - 60) for c in fill_color)
            cv2.polylines(color_overlay, [contour], isClosed=True, color=outline_color, thickness=2)

        color_mask = cv2.bitwise_and(color_overlay, cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR))
        masked_image = cv2.addWeighted(image, 1, color_mask, 0.5, 0)

        cv2.imshow("Base Annotation", masked_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save(self, output_path: str):
        """Save the mask array to a npy file"""
        directory = os.path.dirname(output_path)
        if directory != "" and not os.path.exists(directory):
            raise FileNotFoundError(f"The output directory could not be found: '{directory}'")
        if not self.is_valid_file(output_path, supported_extensions=["npy"]):
            raise ValueError("File must have the .npy extension")

        np.save(output_path, self.mask)
