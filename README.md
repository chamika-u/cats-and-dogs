# Cat vs. Dog Classifier 

A lightweight Deep Learning project that uses a Convolutional Neural Network (CNN) to classify images as either a **Cat** or a **Dog**. Built with Python and TensorFlow/Keras.

## 1. Features
* **Custom CNN Architecture:** Uses Conv2D, MaxPooling, and Dense layers.
* **Data Pipeline:** Loads images dynamically using `image_dataset_from_directory`.
* **Performance Optimized:** Uses caching, prefetching, and data augmentation for faster training.
* **Batch Prediction:** Efficient batch processing for multiple images.
* **Persistence:** Saves the trained model (`.h5`) to be used later without retraining.
* **Prediction Script:** A separate script to test the model on new, unseen images.

## 2. Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/chamika-u/cats-and-dogs.git
    cd cats-and-dogs
    ```

2.  **Create a Virtual Environment** (Recommended)
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install tensorflow numpy
    ```

## 3. Dataset Setup (Important!)
The dataset is **not** included in this repo to save space. You need to download the "Dogs vs. Cats" dataset (e.g., from Kaggle) and arrange it exactly like this:

```text
project_folder/
│
├── dataset/
│   └── testing_set/
│       ├── cats/      <-- Put cat images here
│       └── dogs/      <-- Put dog images here
│   └── training_set/
│       ├── cats/      <-- Put cat images here
│       └── dogs/      <-- Put dog images here
```
## 4. Usage

### Step 1: Train the Model
Run the training script to teach the AI. This script reads the images, trains the CNN, and saves the "brain" as a file named `cat_dog_classifier.h5`.

```bash
python train.py
```

## 5. Performance Optimizations

This project includes several optimizations to improve training and inference speed:

### Training Optimizations (script.py)
* **Data Caching**: Images are cached in memory after the first epoch, eliminating redundant I/O operations in subsequent epochs
* **Prefetching**: Data loading happens in parallel with model training using `tf.data.AUTOTUNE`, reducing GPU idle time
* **Data Augmentation**: Random flips and rotations improve model generalization without needing more images, leading to better results with the same dataset

### Prediction Optimizations (main.py)
* **Lazy Loading**: Model is loaded only when needed, not on module import
* **Batch Processing**: `predict_batch()` function allows efficient processing of multiple images at once
* **Reduced Verbosity**: Predictions run with `verbose=0` for cleaner output

### Expected Performance Improvements
* **Training Speed**: 20-30% faster due to caching and prefetching
* **Batch Predictions**: Up to 5-10x faster than individual predictions for multiple images
* **Memory Efficiency**: Better memory usage with on-demand model loading
