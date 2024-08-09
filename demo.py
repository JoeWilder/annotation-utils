from AnnotationUtils.annotations import Annotations
import numpy as np

if __name__ == "__main__":

    # Load the first three masks from our example data
    anns_list = []
    mask1 = np.load(r"example_data\coral_mask1.npy")
    mask2 = np.load(r"example_data\coral_mask2.npy")
    mask3 = np.load(r"example_data\coral_mask3.npy")

    # Each mask belongs to an image, and has a label describing what the mask is
    anns_list.append((r"example_data\coral.png", "coral", mask1))
    anns_list.append((r"example_data\coral.png", "coral", mask2))
    anns_list.append((r"example_data\coral.png", "coral", mask3))

    anns = Annotations(anns_list)

    # Display the masks to make sure it checks out
    anns.display()

    # Convert masks to COCO format and save annotations to file
    anns.write(Annotations.Format.COCO, r"example_data\coco-annotations.json")

    # Display COCO annotations to confirm conversion went smoothly
    anns.display(Annotations.Format.COCO)

    # Convert masks to YOLOSEG format and save annotations to file(s)
    anns.write(Annotations.Format.YOLOSEG, r"example_data")

    # Display YOLO annotations to confirm conversion went smoothly
    anns.display(Annotations.Format.YOLOSEG)

    # We can also add to existing annotations
    mask4 = np.load(r"example_data\coral_mask4.npy")
    anns.add_annotation(r"example_data\coral.png", "coral", mask4)
    anns.display(Annotations.Format.COCO)

    # Load annotations from existing files
    anns = Annotations()
    anns.load(Annotations.Format.COCO, r"example_data\coco-annotations.json")
    anns.display(Annotations.Format.COCO)

    # Loading YOLO files requires the creation of a classes.txt file, where each line is a label
    # For our example data, create classes.txt in the example_data directory and add the following on line 1: "coral"
    anns = Annotations()
    anns.load(Annotations.Format.YOLOSEG, r"example_data")
    anns.display(Annotations.Format.YOLOSEG)
