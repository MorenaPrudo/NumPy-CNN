
import preproccesing
from modelutil import Model


X_train = preproccesing.load_image_data(preproccesing.train_path)
X_val = preproccesing.load_image_data(preproccesing.val_path)
X_test = preproccesing.load_image_data(preproccesing.test_path)
print("Images loaded!")

y_train = preproccesing.load_label_data(preproccesing.train_label_path)
y_val = preproccesing.load_label_data(preproccesing.val_label_path)
y_test = preproccesing.load_label_data(preproccesing.test_label_path)
print("Labels loaded!")

print(y_train[0:10])
#model = Model()
#model.train(X_train,y_train,X_val,y_val)
