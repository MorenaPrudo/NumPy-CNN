from PIL import Image
import numpy as np
import os
import preproccesing


X_train = preproccesing.load_image_data(preproccesing.train_path)
X_val = preproccesing.load_image_data(preproccesing.val_path)
X_test = preproccesing.load_image_data(preproccesing.test_path)
print("Images loaded!")

y_train = preproccesing.load_label_data(preproccesing.train_label_path)
y_val = preproccesing.load_label_data(preproccesing.val_label_path)
y_test = preproccesing.load_label_data(preproccesing.test_label_path)
print("Labels loaded!")

#Model.train()

