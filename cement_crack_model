import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers, optimizers
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input

# Training variables
num_classes = 2
epochs = 2 
image_resize = 224
batch_size = 100

dir_path = r"C:\Users\tonta\Downloads\concrete_data_week4\concrete_data_week4\valid"
print("PATH:", dir_path)

# Define image data generators
data_generator = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input
)

validation_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input)

training_dataset_dir = dir_path
image_generator = data_generator.flow_from_directory(
    training_dataset_dir,
    batch_size=batch_size,
    class_mode='categorical',
    seed=24
)

vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model = models.Sequential()
model.add(vgg16_base)
print("TEST 2")

model.add(layers.Flatten())

model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(
    optimizer=optimizers.Adam(),
    loss='categorical_crossentropy',  # Corrected typo in loss function name
    metrics=['accuracy']
)

model.summary()

history = model.fit_generator(
    image_generator,
    steps_per_epoch=image_generator.n // batch_size,
    epochs=epochs
)
