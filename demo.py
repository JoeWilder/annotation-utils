from AnnotationUtils.annotation_manager import AnnotationManager
import numpy as np

if __name__ == "__main__":

    # Load the first three masks from our example data
    annotation_list = []
    mask1 = np.load(r"example_data\coral_mask1.npy")
    mask2 = np.load(r"example_data\coral_mask2.npy")
    mask3 = np.load(r"example_data\coral_mask3.npy")

    # Each mask belongs to an image, and has a label describing what the mask is
    annotation_list.append((r"example_data\coral.png", "coral", mask1))
    annotation_list.append((r"example_data\coral.png", "coral", mask2))
    annotation_list.append((r"example_data\coral.png", "coral", mask3))

    annotation_utils = AnnotationManager(annotation_list)

    # Display the masks to make sure it checks out
    annotation_utils.display()

    # Convert masks to COCO format and save annotations to file
    annotation_utils.write(AnnotationManager.Format.COCO, r"example_data\coco-annotations.json")

    # Display COCO annotations to confirm conversion went smoothly
    annotation_utils.display(AnnotationManager.Format.COCO)

    # Convert masks to YOLOSEG format and save annotations to file(s)
    annotation_utils.write(AnnotationManager.Format.YOLOSEG, r"example_data")

    # Display YOLO annotations to confirm conversion went smoothly
    annotation_utils.display(AnnotationManager.Format.YOLOSEG)

    # We can also add to existing annotations
    mask4 = np.load(r"example_data\coral_mask4.npy")
    annotation_utils.add_annotation(r"example_data\coral.png", "coral", mask4)
    annotation_utils.display(AnnotationManager.Format.COCO)

    # Load annotations from existing files
    annotation_utils = AnnotationManager()
    annotation_utils.load(r"example_data\coco-annotations.json", AnnotationManager.Format.COCO)
    annotation_utils.display(AnnotationManager.Format.COCO)

    # Loading YOLO files requires the creation of a classes.txt file, where each line is a label
    # For our example data, create classes.txt in the example_data directory and add the following on line 1: "coral"
    annotation_utils = AnnotationManager()
    annotation_utils.load(r"example_data", AnnotationManager.Format.YOLOSEG)
    annotation_utils.display(AnnotationManager.Format.YOLOSEG)
