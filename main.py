import numpy as np
import preproccesing
import matplotlib.pyplot as plt
import pickle
from modelutil import Model

#test if images are in correct order
X_train = preproccesing.load_image_data(preproccesing.train_path)
X_val = preproccesing.load_image_data(preproccesing.val_path)
X_test = preproccesing.load_image_data(preproccesing.test_path)
print("Images loaded!")
X_train = np.transpose(X_train,(0,3,1,2))
X_val = np.transpose(X_val,(0,3,1,2))
X_test = np.transpose(X_test,(0,3,1,2))

y_train = preproccesing.load_label_data(preproccesing.train_label_path)
y_val = preproccesing.load_label_data(preproccesing.val_label_path)
y_test = preproccesing.load_label_data(preproccesing.test_label_path)
print("Labels loaded!")

print(X_train.shape)

'''
print(X_train.shape)
#X_train = X_train[:,::-1,:,:]
y_train = y_train[0:15]
mean = np.mean(X_train,axis=(0,2,3)).reshape(1,-1,1,1)

num_images = X_train.shape[0]
rows,cols = 3,5
fig,axes = plt.subplots(rows,cols,figsize=[10,6])
axes = axes.flatten()

for i in range(num_images):
    image_c = X_train[i]
    #image_h = np.transpose(image_c,(1,2,0))
    image_h = image_c.astype(np.uint8)
    axes[i].imshow(image_h)
    axes[i].axis('off')

plt.tight_layout()
plt.show()

print(y_train)
'''

mean = np.mean(X_train, axis =(0,2,3)).reshape(1,-1,1,1)
std = np.std(X_train, axis =(0,2,3)).reshape(1,-1,1,1)
X_train = (X_train - mean)/std
X_val = (X_val - mean)/std
X_test = (X_test - mean)/std


model = Model()
model.train(X_train,y_train,X_val,y_val,X_test,y_test)

path = "model.pkl"
with open(path, 'wb') as f:
    pickle.dump(model, f)

print("Model saved!")
