import matplotlib.pyplot as plt
import cv2
import numpy as np


def display_images_side_by_side(image_paths, titles, output_path):
    images = [cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) for image_path in image_paths]

    height, width, _ = images[0].shape
    combined_height = height * 2
    combined_width = width * 2

    combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

    combined_image[0:height, 0:width] = images[0]
    combined_image[0:height, width:combined_width] = images[1]
    combined_image[height:combined_height, 0:width] = images[2]
    combined_image[height:combined_height, width:combined_width] = images[3]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

    ax.imshow(combined_image)
    ax.axis("off")

    for idx, title in enumerate(titles):
        y = 0.25 if idx < 2 else 0.75
        x = 0.25 if idx % 2 == 0 else 0.75
        plt.text(x, y, title, ha="center", va="center", transform=ax.transAxes, fontsize=10, color="white", weight="bold")

    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


image_paths = ["base-annotations.png", "coco-annotations.png", "yolo-annotations.png", "append-coco-annotations.png"]
titles = ["YOLO-SEG Annotations", "Append Existing Annotations", "Base Annotations", "COCO Annotations"]
display_images_side_by_side(image_paths, titles, "combined-image.png")
