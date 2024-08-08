import numpy as np
from typing import List, Tuple


class BaseAnnotations:

    def __init__(self, annotatation_data: List[Tuple[str, str, np.ndarray[np.ndarray[bool]]]]):
        self._base_mask: np.ndarray[np.ndarray[bool]] = annotatation_data[2]
        self._label: str = annotatation_data[1]
        self._image: str = annotatation_data[0]

    @property
    def base_mask(self) -> np.ndarray[np.ndarray[bool]]:
        return self._base_mask

    @base_mask.setter
    def base_mask(self, base_mask: np.ndarray[np.ndarray[bool]]):
        self._base_mask = base_mask

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, label: str):
        self._label = label

    @property
    def image(self) -> str:
        return self._image

    @image.setter
    def image(self, image: str):
        self._image = image

    def convert(self):
        """Converts from the raw data to the classes specified format"""
        raise NotImplementedError("Derived classes must override the 'convert' method")

    def load(self, data_path: str):
        """Converts annotation file of specified format to universal data format"""
        raise NotImplementedError("Derived classes must override the 'load' method")

    def write(self, output_path: str):
        """Writes specified format annotations to disk"""
        raise NotImplementedError("Derived classes must override the 'write' method")

    def display(self):
        """Display annotations in the specified format"""
        raise NotImplementedError("Derived classes must override the 'draw' method")
