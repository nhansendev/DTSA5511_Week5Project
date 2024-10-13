from PIL import Image
import numpy as np
import os

SCRIPT_DIR = os.getcwd()


def make_numpy_file(data_path, save_name):
    # Find all files in the target folder, assumed to be only images
    files = [os.path.join(data_path, f) for f in os.listdir(data_path)]

    images = []
    for f in files:
        # Images may be grayscale, so call convert after reading to ensure RGB format
        images.append(np.asarray(Image.open(f).convert("RGB")))

    # Convert list to array, then save as a numpy file
    images = np.array(images)
    np.save(os.path.join(SCRIPT_DIR, "Data", save_name + ".npy"), images)


if __name__ == "__main__":
    make_numpy_file(os.path.join(SCRIPT_DIR, "Data", "monet_jpg"), "monet")
