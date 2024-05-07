
import os
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers, optimizers
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
import scipy

# Training variables
num_classes = 2
epochs = 2 
image_resize = 224
batch_size = 100

# Directory path for the dataset
dir_path = r"C:\Users\tedst\Downloads\concrete_data_week4\valid"
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

# Directory paths for training dataset and positive images
training_dataset_dir = "C:\\Users\\tedst\\Downloads\\concrete_data_week4\\valid"
positive_image_generator = data_generator.flow_from_directory(
    training_dataset_dir,
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='categorical',
    seed=24
)

# Directory path for negative images
training_dataset_dir = r"C:\Users\tedst\Downloads\concrete_data_week4\valid\negative"
print(training_dataset_dir)
negative_image_generator = data_generator.flow_from_directory(
    training_dataset_dir,
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='categorical',
    seed=24
)

# Load pre-trained VGG16 model
vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Create a sequential model
model = models.Sequential()
model.add(vgg16_base)
print("TEST 2")

# Flatten the output of the VGG16 base
model.add(layers.Flatten())

# Add dense layers
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

# Freeze the VGG16 layers
model.layers[0].trainable = False

# Print model summary
model.summary()

# Compile the model
model.compile(
    optimizer=optimizers.Adam(),
    loss='categorical_crossentropy',  # Corrected typo in loss function name
    metrics=['accuracy']
)

# Calculate steps per epoch for positive and negative images
steps_per_epoch_pos = len(positive_image_generator)
steps_per_epoch_neg = len(negative_image_generator)

# Train the model
history = model.fit_generator(
    positive_image_generator,
    steps_per_epoch=positive_image_generator.n // batch_size,
    epochs=epochs,
    validation_data=negative_image_generator,
    validation_steps=steps_per_epoch_neg,
    verbose=1
)

# Save the trained model
model.save('classifier_vgg16_model.h5')

# Load ResNet and VGG16 models
resnet_model = load_model(r"C:\Users\tedst\Documents\classifier_resnet_model.h5")
vgg16_model = load_model('classifier_vgg16_model.h5')

# Create an image data generator for testing
datagent = ImageDataGenerator()

# Generate test data for negative images
test_gen = datagent.flow_from_directory("concrete_data_week4/valid/negative", target_size=(224,224), shuffle=False)

# Evaluate VGG16 model performance
print("The Performance by vgg16 model is below")
vgg = vgg16_model.evaluate_generator(test_gen)
print(vgg)
print("Loss is ", str(vgg[0]))
print("The accuracy is ", str(vgg[1]))

# Evaluate ResNet model performance
print("The Performance by resnet50 model is below")
resnet = resnet_model.evaluate_generator(test_gen)
print(resnet)
print("Loss is ", str(resnet[0]))
print("The accuracy is ", str(resnet[1]))
