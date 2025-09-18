import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Flatten
from keras.applications.vgg16 import VGG16

conv_base = VGG16(weights='imagenet', 
                  include_top=False, 
                  input_shape=(150, 150, 3))

print(conv_base.summary())


model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

conv_base.trainable = False

print(model.summary())

# create train and test images 

train_ds =keras.preprocessing.image_dataset_from_directory(
          directory='train',
          labels='inferred',
          label_mode='binary',
          batch_size=32,
          image_size=(150, 150)
)

test_ds =keras.preprocessing.image_dataset_from_directory(
          directory='test',
            labels='inferred',
            label_mode='binary',
            batch_size=32,
            image_size=(150, 150)
)

def process(image, label):
    image = tensorflow.cast(image/255.0, tensorflow.float32)
    return image, label

train_ds = train_ds.map(process)
test_ds = test_ds.map(process)

model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

history = model.fit(train_ds,
                    epochs=10,
                    validation_data=test_ds)

model.save('model_transfer_learning.h5')
