![Title](assets/annotation-utils.png?raw=true)

## Introduction

Annotation Utils is a lightweight Python package designed to streamline image annotation workflows by offering a universal annotation format and various utility functions. It allows you to easily convert labeled boolean masks into popular formats such as COCO and YOLOSEG, as well as append annotations and display annotations.

<p align="center">
  <img src="assets/combined-image.png" alt="functionality preview"/>
</p>

## Installing

For now, just clone the project and install any dependencies that are needed

```bash
git clone https://github.com/JoeWilder/AnnotationUtils.git
```


## Functionality

To represent annotations across multiple formats, we need a base type. Our base representation is a list of entries, where each entry has the following format:

```[image_path, category_label, boolean_mask]```

Here are some examples of instantiating an AnnotationHandler

### Loading 
The following section will go over how to load annotation data into the project. This will convert the data into a boolean mask, which can then be converted into any other available format
#### Load from numpy arrays (np array must match image dimensions and have boolean dtype) 
```python
annotation_list = []
mask1 = np.load(r"..\example_data\coral_mask1.npy")
mask2 = np.load(r"..\example_data\coral_mask2.npy")
mask3 = np.load(r"..\example_data\coral_mask3.npy")

annotation_list.append([r"..\example_data\coral.png", "coral", mask1])
annotation_list.append([r"..\example_data\coral.png", "coral", mask2])
annotation_list.append([r"..\example_data\coral.png", "coral", mask3])

annotation_utils = AnnotationHandler(annotation_list)
```

#### Load from COCO json file
```python
annotation_utils = AnnotationHandler()
annotation_utils.from_coco(r"..\example_data\coco-annotations.json")
annotation_utils.display_base()
```

#### Load from YOLO-SEG txt files
```python
annotation_utils = AnnotationHandler()
annotation_utils.from_yolo(r"..\example_data")
annotation_utils.display_base()
```

### Utility Functions

Once we have loaded our data, we can convert to any format that we need. We can also save conversions to disk, and display the new annotations to make sure they came out good

#### Convert and write to disk
```python
annotation_utils.write_coco(r"..\example_data\coco-annotations.json")
annotation_utils.write_yolo(r"..\example_data")
```

#### Display annotations
```python
annotation_utils.display_base() # Unversal boolean mask format
annotation_utils.display_coco() # Convert to coco if needed, and display
annotation_utils.display_yolo() # Convert to yolo if needed, and display
```


This project is meant to be used as a dependency for other projects. To see the functionality of this package in action, read main.py and run it from the src directory

```bash
cd src
python main.py
```


## Version History
* 1.0.0 (8/8/2024)
    * Project package
* 0.2 (8/7/2024)
    * Load annotations from file
* 0.1 (8/6/2024)
    * Initial Release

## License

This project is licensed under the MIT License - see the LICENSE.md file for details


