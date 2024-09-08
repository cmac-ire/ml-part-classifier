import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageFile
import os
import shutil
import time
import snap7
from snap7.util import set_bool
from snap7.type import Areas

# Load TensorFlow/Keras model
model = keras.models.load_model('FYP_iteration1.h5')

# Directories
ftp_folder = '/home/ftpuser'
input_folder = '/home/pi/Desktop/fyp_pi_code/fyp_realtime/images'
processed_folder = '/home/pi/Desktop/fyp_pi_code/fyp_realtime/scanned'

# Class labels for your model
class_labels = ['pass', 'fail']
threshold = 0.9

# List to keep track of images that need reprocessing
images_to_process = []

# Snap7 communication configuration
PLC_IP = '192.168.0.1'  # PLC's IP address
client = snap7.client.Client()

try:
    client.connect(PLC_IP, 0, 1)
    if client.get_connected():
        print("Connected to PLC")
    else:
        print("Failed to connect to PLC")
        client = None
except Exception as e:
    print(f"Failed to connect to PLC: {e}")
    client = None

# Ensure truncated images are handled correctly
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Function to check if the file is ready for processing
def is_file_ready(file_path):
    initial_size = os.path.getsize(file_path)
    time.sleep(0.5)  # Short delay to ensure file writing is complete
    return initial_size == os.path.getsize(file_path)

# Function to process a single image
def process_image(image_path):
    retries = 3  # Number of retries
    for attempt in range(retries):
        try:
            # Check if the file is ready
            if not is_file_ready(image_path):
                raise IOError("File is not ready for processing.")

            # Open and verify the image
            with open(image_path, 'rb') as f:
                img = Image.open(f)
                img.verify()  # Verify that the image is complete

            # Reopen the image for processing after verification
            img = keras.utils.load_img(image_path, target_size=(224, 224))
            img_array = keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            predictions = model.predict(img_array)
            confidence = predictions[0][0]

            # Determine pass or fail based on the confidence
            if confidence > threshold:
                return 'pass', confidence
            else:
                return 'fail', confidence
        
        except (IOError, tf.errors.InvalidArgumentError, Image.DecompressionBombError) as e:
            print(f"Error processing image {image_path}: {e}")
            if attempt < retries - 1:
                print("Retrying...")
                time.sleep(0.5)  # Short delay before retrying
            else:
                return 'error', 0
    
    return 'error', 0

# Function to send result to PLC using Snap7
def send_result_to_plc(filename, category, confidence):
    if client and client.get_connected():
        print(f"Sending '{category}' result for {filename} with confidence {confidence} to PLC")
        
        try:
            # Read the existing data from DB1 (1 byte, for example)
            db_number = 1
            start_address = 0
            size = 1
            data = client.read_area(Areas.DB, db_number, start_address, size)
            
            # Clear previous bits
            set_bool(data, 0, 0, False)  # Clear pass_detected
            set_bool(data, 0, 1, False)  # Clear fail_detected

            # Set the corresponding bit based on category
            if category == 'pass':
                set_bool(data, 0, 0, True)  # Set pass_detected
            elif category == 'fail':
                set_bool(data, 0, 1, True)  # Set fail_detected
            
            # Write the modified data back to the PLC
            client.write_area(Areas.DB, db_number, start_address, data)
        except Exception as e:
            print(f"Error sending data to PLC: {e}")
    else:
        print(f"Failed to connect to PLC. File: {filename}, Category: {category}, Confidence: {confidence}")

# Main loop to continuously monitor and process images
while True:
    found_new_image = False
    
    # Move new images from FTP folder to the input folder
    for filename in os.listdir(ftp_folder):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            ftp_image_path = os.path.join(ftp_folder, filename)
            dest_image_path = os.path.join(input_folder, filename)
            
            try:
                shutil.move(ftp_image_path, dest_image_path)
                print(f"Moved {filename} from FTP folder to processing folder")
                images_to_process.append(filename)
                found_new_image = True
            except Exception as e:
                print(f"Failed to move {filename} from FTP folder to processing folder: {e}")
    
    # Process images in the list
    for filename in images_to_process:
        img_path = os.path.join(input_folder, filename)
        
        if os.path.exists(img_path):  # Ensure image still exists in input_folder
            category, confidence = process_image(img_path)
            print(f"File: {filename}, Classified as: {category}, Confidence: {confidence}")
            send_result_to_plc(filename, category, confidence)
            
            # Move processed image to the processed folder
            shutil.move(img_path, os.path.join(processed_folder, filename))
            
            # Remove from images_to_process to avoid re-processing
            images_to_process.remove(filename)
        
    if found_new_image:
        time.sleep(0.1)  # Wait until next image appears
    else:
        time.sleep(0.1)  # Adjust sleep time if needed for faster response
