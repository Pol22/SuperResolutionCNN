from keras.models import Sequential
from keras.layers import Conv2D
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split


BATCH_SIZE = 10
IMG_SIZE = 256
NUM_CHANELS = 3
f1 = 9
n1 = 64
f2 = 1
n2 = 32
f3 = 1


model = Sequential()

model.add(Conv2D(n1, kernel_size=f1, activation='relu',
    input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANELS), padding='same'))
model.add(Conv2D(n2, kernel_size=f2, activation='relu', padding='same'))
model.add(Conv2D(NUM_CHANELS, kernel_size=f3, padding='same'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

print(model.summary())

# load 
bicubic_imgs = os.listdir('dataset/bicubic')
dataset = []
for img in bicubic_imgs:
    img_path = os.path.join('dataset/bicubic', img)
    dataset.append(np.asarray(Image.open(img_path), dtype=np.float32))

bicubic_dataset = np.asarray(dataset)

ref_imgs = os.listdir('dataset/reference')
dataset = []
for img in ref_imgs:
    img_path = os.path.join('dataset/reference', img)
    dataset.append(np.asarray(Image.open(img_path), dtype=np.float32))

ref_dataset = np.asarray(dataset)
print("Bicubic dataset shape:", bicubic_dataset.shape)
print("Reference dataset shape:", ref_dataset.shape)

X_train, X_test, y_train, y_test = train_test_split(
    bicubic_dataset, ref_dataset,
    test_size=0.2, random_state=12444, shuffle=True)

print(X_test.shape)
print(X_train.shape)

# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5)
# model.save('srcnn.h5')
from keras.models import load_model
model = load_model('srcnn.h5')
x = 232
sr_75 = model.predict(np.reshape(X_test[x,:,:,:], (1, IMG_SIZE, IMG_SIZE, NUM_CHANELS)))
sr_75 = np.asarray(sr_75, dtype=np.uint8)
sr_75 = np.squeeze(sr_75)
img_75 = Image.fromarray(sr_75)
img_75.save('res_75.jpg')
test_75 = Image.fromarray(X_test[x].astype(np.uint8))
test_75.save('test_75.jpg')
ref_75 = Image.fromarray(y_test[x].astype(np.uint8))
ref_75.save('ref_75.jpg')


