# FYP-ML: Machine Learning Part Classification System

*This project was developed with the help of AI tools, using them as a guide, assistant, and for significant code generation. All final creative decisions and deployments reflect my own work and intentions.*

## Overview

This repository contains the code and documentation for the final year project (FYP) titled **Machine Learning Part Classification System**. The project integrates machine learning techniques with industrial automation to classify parts efficiently, leveraging both software and hardware components. 

## Project Goals

The main objectives of this project include:
- **Integration of Machine Learning with Industrial Automation**: Utilizing TensorFlow and Keras to develop a part classification model.
- **Real-time Part Classification**: Deploying the model in an industrial setting using a Siemens PLC and a Raspberry Pi.
- **Data Collection and Preprocessing**: Gathering and preprocessing data from various sensors for model training and validation.
- **System Implementation**: Combining the model with a PLC-controlled system to automate the classification process.

## Key Components

### 1. **Machine Learning Model**
   - **Frameworks**: TensorFlow, Keras
   - **Task**: Part classification using image data.
   - **Performance**: Optimized for high accuracy in a real-time industrial setting.

### 2. **Industrial Automation Integration**
   - **PLC**: Siemens PLC for controlling the automation process.
   - **Raspberry Pi**: For deploying the ML model and handling communication between the PLC and the model.
   - **Communication Protocols**: Use of MQTT and Modbus for communication between devices.

### 3. **Data Pipeline**
   - **Data Collection**: Using cameras and sensors integrated into the automation line.
   - **Preprocessing**: Image processing techniques such as normalization, resizing, and augmentation.
   - **Training**: Supervised learning on labeled datasets.

## Setup Instructions

### Prerequisites
- Python
- TensorFlow and Keras
- Siemens TIA Portal (for PLC programming)
- Raspberry Pi with Raspbian OS
- MQTT Broker (e.g., Mosquitto)

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/cmac-ire/fyp-ml.git
    cd fyp-ml
    ```

2. **Install Python dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Setup Raspberry Pi**:
    - Ensure Raspbian OS is installed.
    - Install necessary libraries and setup communication protocols.

4. **PLC Programming**:
    - Use Siemens TIA Portal to program the PLC according to the provided logic.

5. **Model Deployment**:
    - Train the model using provided datasets.
    - Deploy the model to the Raspberry Pi for real-time classification.

### Usage

1. **Data Collection**:
    - Ensure the system is connected to the sensors/cameras.
    - Run the data collection script to gather images of parts.

2. **Training the Model**:
    - Preprocess the collected data.
    - Train the model using the `fyp.py` script.
    - Save the trained model for deployment.

3. **Deploy and Run**:
    - Deploy the model on the Raspberry Pi.
    - Start the system and monitor the real-time classification through the PLC interface.

## Results

The project achieved an impressive 99% accuracy in part classification.

## Future Work

- **Enhancements**: Implementing advanced ML techniques like deep learning for improved accuracy.
- **Expansion**: Extending the system to handle different types of parts and materials.
- **Optimization**: Reducing latency and increasing processing speed for higher throughput.

## License

This project is licensed under the MIT License.

## Contact

For any queries or collaboration opportunities, please contact Cormac Farrelly at cfarr311y@gmail.com

