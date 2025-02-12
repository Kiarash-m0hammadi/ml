import os
import numpy as np
from data_loader import WiderFaceDataset
from tqdm import tqdm  # Optional: shows progress during preprocessing


def preprocess_and_save(
    root_dir: str, split: str = "train", output_file: str = "preprocessed_data.npz"
):
    dataset = WiderFaceDataset(root_dir=root_dir, split=split)
    images = []
    annotations = []

    for idx in tqdm(range(len(dataset)), desc="Preprocessing"):
        sample = dataset[idx]
        images.append(sample["image"])
        annotations.append(sample["boxes"])

    images = np.array(images)  # Shape: (num_samples, height, width, channels)
    # Save images and annotations in a single .npz file.
    np.savez(output_file, images=images, annotations=annotations)
    print(f"Saved preprocessed data to {output_file}")


if __name__ == "__main__":
    # Set the dataset root directory (adjust as needed)
    root_dir = "e:/projects/ml/WIDER_train"
    preprocess_and_save(
        root_dir=root_dir, split="train", output_file="preprocessed_train.npz"
    )
