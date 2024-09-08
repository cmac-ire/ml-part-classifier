import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import pandas as pd

# Define a DataFrame to store the results
results_df = pd.DataFrame(columns=['Image', 'Result', 'Confidence'])

# Load the pre-trained model
model = load_model('FYP_iteration1.h5')  # Update with the correct model file

# Function to preprocess the image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = img / 255.0  # Rescale the image pixels to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Path to the folder containing test images
test_folder = r'C:\Users\cfarr\Desktop\fyp_pi_code\fyp_realtime\test'

# Get a list of all image files in the folder
test_image_files = [file for file in os.listdir(test_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Create a list to store the results
results = []

# Perform prediction on each test image
for test_image_file in test_image_files:
    image_path = os.path.join(test_folder, test_image_file)
    image = preprocess_image(image_path)
    predictions = model.predict(image)

    # Assuming binary classification with a single output node
    if predictions.shape[1] == 1:
        # If the prediction is above 0.5, classify as "Fail"
        if predictions[0][0] >= 0.5:
            result_label = "Pass"
            confidence = predictions[0][0]
        else:
            result_label = "Fail"
            confidence = 1 - predictions[0][0]

    # Append results to the list
    results.append([test_image_file, result_label, confidence])

# Create the DataFrame from the list of results
results_df = pd.DataFrame(results, columns=['Image', 'Result', 'Confidence'])

# Print the results
print("\n\n\Failed Parts:")
print(results_df[results_df['Result'] == 'Fail'])
print("\n\nPassed Parts:")
print(results_df[results_df['Result'] == 'Pass'])
