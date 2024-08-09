![Title](assets/annotation-utils.png?raw=true)

## Introduction

Annotation Utils is a lightweight Python package designed to streamline image annotation workflows by offering a universal annotation format and various utility functions. It allows you to easily convert labeled boolean masks into popular formats such as COCO and YOLOSEG. You can also append annotations on the fly and display annotations.

<p align="center">
  <img src="assets/combined-image.png" alt="functionality preview"/>
</p>

## Installing

To use AnnotationUtils in your project, run the following command:
```bash
pip install git+https://github.com/JoeWilder/annotation-utils.git
```

To run the project locally (The only dependencies are numpy and opencv-python)
```bash
git clone https://github.com/JoeWilder/AnnotationUtils.git
cd AnnotationUtils
pip install -r requirements.txt
python demo.py
```


## Functionality

To represent annotations across multiple formats, we need a base format. Our base representation is a list of entries, where each entry has the following format:

```[image_path, category_label, boolean_mask]```

Here are some examples of instantiating an AnnotationHandler

### Loading 
The following section will go over how to load annotation data into the project. This will convert the data into a boolean mask if necessary, which can then be converted into any other available format
#### Manually Load from numpy arrays (np array must match image dimensions and have boolean dtype) 
```python
from AnnotationUtils import AnnotationHandler

anns_list = []
mask1 = np.load(r"..\example_data\coral_mask1.npy")
mask2 = np.load(r"..\example_data\coral_mask2.npy")
mask3 = np.load(r"..\example_data\coral_mask3.npy")

anns_list.append([r"..\example_data\coral.png", "coral", mask1])
anns_list.append([r"..\example_data\coral.png", "coral", mask2])
anns_list.append([r"..\example_data\coral.png", "coral", mask3])

anns = Annotations(anns_list)
```

#### Load from COCO json file
```python
anns = Annotations()
anns.load(Annotations.Format.COCO, r"example_data\coco-annotations.json")
```

#### Load from YOLO-SEG txt files
```python
anns = Annotations()
anns.load(Annotations.Format.YOLOSEG, r"example_data")
```

### Utility Functions

Once we have loaded our data, we can convert to any format that we need. We can also save conversions to disk, and display the new annotations to make sure they loaded correctly

#### Convert and write to disk
```python
anns.write(Annotations.Format.COCO, r"example_data\coco-annotations.json")
anns.write(Annotations.Format.YOLOSEG, r"example_data")
```

#### Display annotations
```python
anns.display() # Unversal boolean mask format
anns.display(Annotations.Format.COCO) # Convert to coco if needed, and display
anns.display(Annotations.Format.YOLOSEG) # Convert to yolo if needed, and display
```

## Version History
* 1.0.1 (8/9/2024)
    * Maintainability refactor
* 1.0.0 (8/8/2024)
    * Project package
* 0.2 (8/7/2024)
    * Load annotations from file
* 0.1 (8/6/2024)
    * Initial Release

## License

This project is licensed under the MIT License - see the LICENSE.md file for details
