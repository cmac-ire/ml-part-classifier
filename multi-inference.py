import tensorflow as tf
from tensorflow import keras
import os

model = keras.models.load_model('save_at_5.keras')

folder_path = r"C:\Users\cfarr\Desktop\Github Projects\fyp-ml\test-keras-images"

class_labels = ['dog', 'cat']
threshold = 0.8

predicted_dog = []
predicted_cat = []
unrecognized = []

for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        img_path = os.path.join(folder_path, filename)

        img = keras.utils.load_img(img_path, target_size=(180, 180))

        img_array = keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) 

        predictions = model.predict(img_array)
        confidence = predictions[0, 0]

        if confidence > threshold:
            predicted_dog.append((filename, confidence))
        else:
            if 1 - confidence > threshold:
                predicted_cat.append((filename, 1 - confidence))
            else:
                unrecognized.append((filename, confidence))

def display_results(category, results):
    print(f"**Predicted {category.capitalize()}:**\n")
    print("| Image      | Confidence |")
    print("|------------|------------|")
    for result in results:
        print(f"| {result[0]:<10} | {100 * result[1]:.2f}%       |")
    print("\n")

display_results("dog", predicted_dog)
display_results("cat", predicted_cat)
display_results("unrecognized", unrecognized)
