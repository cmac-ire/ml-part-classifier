import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import pandas as pd

# Define a DataFrame to store the results
results_df = pd.DataFrame(columns=['Image', 'Predicted Class', 'Confidence'])

# Load the pre-trained model
model = load_model('FYP.h5')  # Update with the correct model file

# Function to preprocess the image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = img / 255.0  # Rescale the image pixels to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Path to the folder containing test images
test_folder = r'C:\Users\cfarr\Desktop\in-progress\test-dataset'

# Get a list of all image files in the folder
test_image_files = [file for file in os.listdir(test_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Create a list to store the results
results = []

# Confidence threshold for classifying as "Unrecognized"
threshold_unrecognized = 0.53  # Adjust the threshold as needed

# Perform prediction on each test image
for test_image_file in test_image_files:
    image_path = os.path.join(test_folder, test_image_file)
    image = preprocess_image(image_path)
    predictions = model.predict(image)

    # Check the shape of predictions
    if predictions.shape[1] == 1:
        # Model output is one-dimensional, treat it as binary classification
        if predictions[0][0] >= 0.5:
            predicted_class_label = "Damaged"
            confidence = predictions[0][0]
        else:
            predicted_class_label = "Undamaged"
            confidence = 1 - predictions[0][0]
    else:
        # Model output is two-dimensional, treat it as multiclass classification
        if predictions[0][0] >= predictions[0][1]:
            predicted_class_label = "Damaged"
            confidence = predictions[0][0]
        else:
            predicted_class_label = "Undamaged"
            confidence = predictions[0][1]

    # Check if the image represents an industrial part
    is_industrial_part = predicted_class_label in ["Damaged", "Undamaged"]

    # Separate classification for Unrecognized images
    if not is_industrial_part:
        predicted_class_label = "Unrecognized"
        confidence = max(predictions[0][0], predictions[0][1])

    # Append results to the list
    results.append([test_image_file, predicted_class_label, confidence])

# Create the DataFrame from the list of results
results_df = pd.DataFrame(results, columns=['Image', 'Predicted Class', 'Confidence'])

# Print the results
print("\n\n\nDamaged Images:")
print(results_df[results_df['Predicted Class'] == 'Damaged'])
print("\n\n\nUndamaged Images:")
print(results_df[results_df['Predicted Class'] == 'Undamaged'])
print("\n\n\nUnrecognized Images:")
print(results_df[results_df['Predicted Class'] == 'Unrecognized'])
