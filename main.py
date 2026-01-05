import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model

# Load the trained model from file
model = load_model('cat_dog_classifier.h5')

# Load an image to test
test_image_path = 'D:\\projects\\cat or dog\\dataset\\test_set\\dogs\\dog.4013.jpg'
test_image = load_img(test_image_path, target_size=(150, 150))
test_image_array = img_to_array(test_image)
test_image_array = np.expand_dims(test_image_array, axis=0) # Add batch dimension

# Predict either it's a cat or a dog
result = model.predict(test_image_array)

if result[0][0] > 0.5:
    print("It's a DOG")
else:
    print("It's a CAT")