import cv2
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

# ✅ Set your image directory
image_directory = r'D:\ImageDataset'  # <-- change if needed

# ✅ Build paths to the 'yes' and 'no' folders
no_tumor_path = os.path.join(image_directory, 'no')
yes_tumor_path = os.path.join(image_directory, 'yes')

# ✅ Check if the folders exist
if not os.path.exists(no_tumor_path) or not os.path.exists(yes_tumor_path):
    print("❌ ERROR: One or both image folders not found.")
    print("Expected folders:")
    print("  -", no_tumor_path)
    print("  -", yes_tumor_path)
    exit()

# ✅ Load file names
no_tumor_images = os.listdir(no_tumor_path)
yes_tumor_images = os.listdir(yes_tumor_path)

dataset = []
label = []
INPUT_SIZE = 64

# ✅ Load images from 'no' folder
for image_name in no_tumor_images:
    if image_name.lower().endswith(('.jpg', '.png')):
        image_path = os.path.join(no_tumor_path, image_name)
        img = cv2.imread(image_path)
        if img is not None:
            img = Image.fromarray(img, 'RGB').resize((INPUT_SIZE, INPUT_SIZE))
            dataset.append(np.array(img))
            label.append(0)

# ✅ Load images from 'yes' folder
for image_name in yes_tumor_images:
    if image_name.lower().endswith(('.jpg', '.png')):
        image_path = os.path.join(yes_tumor_path, image_name)
        img = cv2.imread(image_path)
        if img is not None:
            img = Image.fromarray(img, 'RGB').resize((INPUT_SIZE, INPUT_SIZE))
            dataset.append(np.array(img))
            label.append(1)

# ✅ Check if data was loaded
if len(dataset) == 0:
    print("❌ ERROR: No images loaded. Please check your dataset paths.")
    exit()
else:
    print(f"✅ Loaded {len(dataset)} images successfully.")

# ✅ Convert and preprocess data
dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=42)
x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

# ✅ Build CNN model
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# ✅ Train the model
model.fit(x_train, y_train,
          batch_size=16,
          verbose=1,
          epochs=10,
          validation_data=(x_test, y_test),
          shuffle=False)

# ✅ Save the trained model
model.save('BrainTumor10Epochs.h5')
print("✅ Model training complete and saved as BrainTumor10Epochs.h5")

import matplotlib.pyplot as plt

# Show a sample of images from the dataset
def show_images(images, labels, num_images=5):
    num_images = min(num_images, len(images))  # Ensure we don't try to display more images than we have
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i])
        plt.title('Tumor' if labels[i] == 1 else 'No Tumor')
        plt.axis('off')
    plt.show()

# Display first few images from the dataset
show_images(dataset, label, num_images=5)
