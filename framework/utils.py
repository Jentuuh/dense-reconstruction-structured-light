import cv2
import numpy as np
from typing import List, Optional
import xml.etree.ElementTree as ET


def load_images(path: str) -> Optional[List[np.array]]:
    try:
        tree = ET.parse(path)
        if tree is None:
            return None
        images_node = tree.find("images")
        images = []
        for path in images_node.text.split("\n"):
            if path == "":
                continue
            image = cv2.imread(path)
            if image is None or image.size == 0:
                print(f"Failed to load {path}")
            else:
                images.append(image)

        return images
    except FileNotFoundError:
        print(f"{path} file not found")
        return None
