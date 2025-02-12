import os
import cv2
import numpy as np
from typing import Dict, List, Tuple


class WiderFaceDataset:
    def __init__(self, root_dir: str, split: str = "train"):
        """
        Args:
            root_dir (str): Root directory of WIDER Face dataset
            split (str): 'train', 'val', or 'test'
        """
        self.root_dir = root_dir
        self.split = split
        self.images = []
        self.annotations = {}  # Dictionary: image_path -> annotations

        # Set up paths
        self.images_dir = os.path.join(root_dir, f"WIDER_{split}", "images")
        self.anno_file = os.path.join(
            root_dir,
            "wider_face_split",
            (
                f"wider_face_{split}_bbx_gt.txt"
                if split != "test"
                else "wider_face_test_filelist.txt"
            ),
        )

        # Load annotations
        self._load_annotations()

    def _load_annotations(self):
        """Parse annotation file and store image paths and their annotations."""
        with open(self.anno_file, "r") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]

            idx = 0
            while idx < len(lines):
                # Get image path
                image_path = lines[idx]

                if self.split != "test":
                    # Get number of faces in this image
                    num_faces = int(lines[idx + 1])

                    # Get face annotations
                    annotations = []
                    for face_idx in range(num_faces):
                        bbox_line = lines[idx + 2 + face_idx]
                        # Convert string of values to float numbers
                        bbox_values = [float(x) for x in bbox_line.split()]
                        annotations.append(
                            {
                                "bbox": bbox_values[:4],  # x, y, w, h
                                "blur": bbox_values[4],
                                "expression": bbox_values[5],
                                "illumination": bbox_values[6],
                                "invalid": bbox_values[7],
                                "occlusion": bbox_values[8],
                                "pose": bbox_values[9],
                            }
                        )

                    # Store annotations
                    self.annotations[image_path] = annotations
                    idx += 2 + num_faces
                else:
                    # For test set, we only have image paths
                    self.annotations[image_path] = None
                    idx += 1

                # Store image path
                self.images.append(image_path)

    def resize_image_and_boxes(
        self,
        image: np.ndarray,
        boxes: List[dict],
        target_size: Tuple[int, int] = (256, 256),
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Resize image and adjust bounding boxes accordingly.

        Args:
            image: Original image
            boxes: List of annotation dictionaries
            target_size: Desired output size (width, height)

        Returns:
            Tuple of (resized_image, adjusted_boxes)
        """
        orig_h, orig_w = image.shape[:2]
        target_w, target_h = target_size

        # Resize image
        resized_image = cv2.resize(image, target_size)

        if boxes is None:  # For test set
            return resized_image, None

        # Calculate scaling factors
        w_scale = target_w / orig_w
        h_scale = target_h / orig_h

        # Adjust bounding boxes
        adjusted_boxes = []
        for box in boxes:
            x, y, w, h = box["bbox"]
            adjusted_box = box.copy()
            adjusted_box["bbox"] = [
                x * w_scale,  # new x
                y * h_scale,  # new y
                w * w_scale,  # new width
                h * h_scale,  # new height
            ]
            adjusted_boxes.append(adjusted_box)

        return resized_image, adjusted_boxes

    def __getitem__(self, idx: int) -> Dict:
        """
        Load and preprocess image and annotations for given index.

        Args:
            idx: Index of the sample to load

        Returns:
            Dictionary containing:
                - image: Resized image array (256x256x3)
                - boxes: List of adjusted annotation dictionaries
                - image_path: Original image path
        """
        image_path = self.images[idx]
        full_path = os.path.join(self.images_dir, image_path)

        # Read image
        image = cv2.imread(full_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Get annotations
        boxes = self.annotations[image_path]

        # Resize image and adjust boxes
        image, boxes = self.resize_image_and_boxes(image, boxes)

        # Normalize pixel values by dividing by 255.0 so values range from 0 to 1.
        image = image / 255.0

        # Filter out invalid or extremely small faces.
        if boxes is not None:
            filtered_boxes = []
            for box in boxes:
                # Exclude faces marked as invalid.
                if box.get("invalid", 0) == 1:
                    continue
                # Exclude faces with width or height less than 20 pixels.
                _, _, w, h = box["bbox"]
                if w < 20 or h < 20:
                    continue
                filtered_boxes.append(box)
            boxes = filtered_boxes

        return {"image": image, "boxes": boxes, "image_path": image_path}

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.images)
