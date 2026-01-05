import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model

# Module-level model cache for efficiency
_model_cache = {}

def predict_image(image_path, model_path='cat_dog_classifier.h5'):
    """
    Predict whether an image is a cat or dog.
    
    Args:
        image_path: Path to the image file
        model_path: Path to the trained model (default: 'cat_dog_classifier.h5')
    
    Returns:
        str: 'DOG' or 'CAT'
    """
    # Load model from cache or disk (avoids reloading on every call)
    if model_path not in _model_cache:
        _model_cache[model_path] = load_model(model_path)
    model = _model_cache[model_path]
    
    # Load and preprocess image
    test_image = load_img(image_path, target_size=(150, 150))
    test_image_array = img_to_array(test_image)
    test_image_array = np.expand_dims(test_image_array, axis=0)  # Add batch dimension
    # Note: Normalization (dividing by 255) is handled by the Rescaling layer in the model
    
    # Predict
    result = model.predict(test_image_array, verbose=0)  # verbose=0 for cleaner output
    
    return "DOG" if result[0][0] > 0.5 else "CAT"

def predict_batch(image_paths, model_path='cat_dog_classifier.h5'):
    """
    Efficiently predict multiple images at once using batch processing.
    This is much faster than predicting images one by one.
    
    Args:
        image_paths: List of paths to image files
        model_path: Path to the trained model
    
    Returns:
        list: List of predictions ('DOG', 'CAT', or None for failed images)
              None indicates the image failed to load
    """
    # Load model from cache or disk
    if model_path not in _model_cache:
        _model_cache[model_path] = load_model(model_path)
    model = _model_cache[model_path]
    
    # Load and preprocess all images with error handling
    images = []
    valid_indices = []
    for i, path in enumerate(image_paths):
        try:
            img = load_img(path, target_size=(150, 150))
            img_array = img_to_array(img)
            images.append(img_array)
            valid_indices.append(i)
        except (OSError, ValueError) as e:
            # Handle specific expected errors during image loading
            print(f"Warning: Failed to load image {path}: {e}")
            continue
    
    if not images:
        raise ValueError("No valid images to process")
    
    # Stack images into a batch
    batch = np.array(images)
    # Note: Normalization (dividing by 255) is handled by the Rescaling layer in the model
    
    # Single prediction call for all images (much faster)
    results = model.predict(batch, verbose=0)
    
    # Convert results to labels
    predictions = ["DOG" if result[0] > 0.5 else "CAT" for result in results]
    
    # Return predictions aligned with original input
    # None indicates failed image loading
    full_predictions = [None] * len(image_paths)
    for idx, pred in zip(valid_indices, predictions):
        full_predictions[idx] = pred
    
    return full_predictions

# Example usage
if __name__ == "__main__":
    # Single image prediction
    # Replace with your actual image path
    test_image_path = 'path/to/your/test/image.jpg'
    
    try:
        prediction = predict_image(test_image_path)
        print(f"It's a {prediction}")
    except FileNotFoundError:
        print(f"Image not found at {test_image_path}")
        print("Please update the path to point to your test image.")
    except Exception as e:
        print(f"Error during prediction: {e}")
    
    # Batch prediction example (commented out)
    # image_list = ['image1.jpg', 'image2.jpg', 'image3.jpg']
    # predictions = predict_batch(image_list)
    # for img, pred in zip(image_list, predictions):
    #     if pred is None:
    #         print(f"{img}: Failed to load")
    #     else:
    #         print(f"{img}: {pred}")