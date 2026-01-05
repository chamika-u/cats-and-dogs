# Cat vs. Dog Classifier 

A lightweight Deep Learning project that uses a Convolutional Neural Network (CNN) to classify images as either a **Cat** or a **Dog**. Built with Python and TensorFlow/Keras.

## 1. Features
* **Custom CNN Architecture:** Uses Conv2D, MaxPooling, and Dense layers.
* **Data Pipeline:** Loads images dynamically using `image_dataset_from_directory`.
* **Persistence:** Saves the trained model (`.h5`) to be used later without retraining.
* **Prediction Script:** A separate script to test the model on new, unseen images.

## 2. Installation

2.1.  **Clone the repository**
    ```bash
    git clone https://github.com/chamika-u/cats-and-dogs.git
    cd cats-and-dogs
    ```

2.2.  **Create a Virtual Environment** (Recommended)
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

2.3.  **Install Dependencies**
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
