from keras.models import Sequential
from keras.layers import Conv2D


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

model.compile(optimizer='adam', loss='mean_squared_error')

print(model.summary())