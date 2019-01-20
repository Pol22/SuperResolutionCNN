from keras.models import Sequential
from keras.layers import Conv2D
from keras import regularizers
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.measure import compare_psnr
import matplotlib.pyplot as plt


BATCH_SIZE = 10
IMG_SIZE = 127
NUM_CHANELS = 3
f1 = 9
f2 = 3
f3 = 5
n1 = 64
n2 = 32


model = Sequential()

model.add(Conv2D(n1, kernel_size=f1, activation='relu',
    input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANELS), padding='same'))
model.add(Conv2D(n2, kernel_size=f2, activation='relu', padding='same'))
model.add(Conv2D(NUM_CHANELS, kernel_size=f3, padding='same',
    activation='tanh'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

print(model.summary())

# load 
bicubic_imgs = os.listdir('dataset/bicubic')
dataset = []
for img in bicubic_imgs:
    img_path = os.path.join('dataset/bicubic', img)
    arr = np.asarray(Image.open(img_path), dtype=np.float32)
    arr = (arr - 127) / 128
    dataset.append(arr)

bicubic_dataset = np.asarray(dataset)

ref_imgs = os.listdir('dataset/reference')
dataset = []
for img in ref_imgs:
    img_path = os.path.join('dataset/reference', img)
    arr = np.asarray(Image.open(img_path), dtype=np.float32)
    arr = (arr - 127) / 128
    dataset.append(arr)

ref_dataset = np.asarray(dataset)

print("Bicubic dataset shape:", bicubic_dataset.shape)
print("Reference dataset shape:", ref_dataset.shape)

X_train, X_test, y_train, y_test = train_test_split(
    bicubic_dataset, ref_dataset,
    test_size=0.1, random_state=12444, shuffle=True)

print('Test shape:', X_test.shape)
print('Train shape:', X_train.shape)

history = model.fit(X_train, y_train, batch_size=32,
    validation_data=(X_test, y_test), epochs=100)
model.save('srcnn_v4.h5')
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


from keras.models import load_model
model = load_model('srcnn_v4.h5')
x = 1574
img = np.reshape(X_train[x,:,:,:], (1, IMG_SIZE, IMG_SIZE, NUM_CHANELS))
sr_75 = model.predict(img)
sr_75 = np.reshape(sr_75, (IMG_SIZE, IMG_SIZE, NUM_CHANELS))
sr_75 = sr_75 * 128 + 127
sr_75 = np.asarray(sr_75, dtype=np.uint8)
img_75 = Image.fromarray(sr_75)
img_75.save('res_75.jpg')
img = np.reshape(img, (IMG_SIZE, IMG_SIZE, NUM_CHANELS))
img = img * 128 + 127
img = np.asarray(img, np.uint8)
test_75 = Image.fromarray(img)
test_75.save('test_75.jpg')
ref = y_train[x]
ref = np.reshape(ref, (IMG_SIZE, IMG_SIZE, NUM_CHANELS))
ref = ref * 128 + 127
ref = np.asarray(ref, np.uint8)
ref_75 = Image.fromarray(ref)
ref_75.save('ref_75.jpg')

print('SRCNN:', compare_psnr(sr_75, ref))
print('Bicubic:', compare_psnr(img, ref))




