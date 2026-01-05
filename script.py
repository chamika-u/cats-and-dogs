# Save this as train.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Rescaling, RandomFlip, RandomRotation

# 1. Load Data
img_height, img_width = 150, 150
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
  'dataset/training_set',
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
  'dataset/training_set',
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

# Performance Optimization: Add caching and prefetching
# Cache dataset in memory to avoid I/O bottleneck
# Prefetch allows data loading to happen in parallel with training
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 2. Build Model
# Data augmentation improves model generalization without needing more data
# This makes training more efficient by getting better results with same dataset
model = Sequential([
  Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  RandomFlip("horizontal"),  # Randomly flip images horizontally
  RandomRotation(0.1),       # Randomly rotate images slightly
  Conv2D(16, 3, padding='same', activation='relu'),
  MaxPooling2D(),
  Conv2D(32, 3, padding='same', activation='relu'),
  MaxPooling2D(),
  Conv2D(64, 3, padding='same', activation='relu'),
  MaxPooling2D(),
  Flatten(),
  Dense(128, activation='relu'),
  Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 3. Train
print("Starting training... this might take a while.")
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=5 # Reduced to 5 for a quick test
)

# 4. Save
model.save('cat_dog_classifier.h5')
print("Model saved as cat_dog_classifier.h5!")