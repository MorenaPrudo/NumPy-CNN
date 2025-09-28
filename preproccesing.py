from PIL import Image
import numpy as np
import os

from matplotlib import pyplot as plt

test_path = os.path.expanduser(r"~\Documents\Datasets\CarDD\test\images")
test_label_path = os.path.expanduser(r"~\Documents\Datasets\CarDD\test\labels")
train_path = os.path.expanduser(r"~\Documents\Datasets\CarDD\train\images")
train_label_path = os.path.expanduser(r"~\Documents\Datasets\CarDD\train\labels")
val_path = os.path.expanduser(r"~\Documents\Datasets\CarDD\val\images")
val_label_path = os.path.expanduser(r"~\Documents\Datasets\CarDD\val\labels")
target_size = (224,224)

def load_image_data(path):
    image_arrays = []
    for image_name in os.listdir(path):
        image_path = os.path.join(path,image_name) #creating individual image path by combining file path and image name.
        image = Image.open(image_path).convert('RGB') #converts jpg to an Image object using the image path.
        image = image.resize(target_size)
        image_array = np.array(image) #converts image object to an array
        image_arrays.append(image_array)
    return np.stack(image_arrays).reshape(-1,224,224,3) #creates a 4D Numpy array by stacking the images

def load_label_data(path):
    label_array = []
    for txtfile in os.listdir(path):
        txtpath = os.path.join(path,txtfile)
        with open(txtpath, "r") as file:
            label_array.append(int(file.read(1)))
    return np.array(label_array)