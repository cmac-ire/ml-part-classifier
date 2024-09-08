from keras import layers, models, optimizers, callbacks
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50V2

# Prepare the Dataset
train_dir = r'C:\Users\cfarr\Desktop\fyp_pi_code1\fyp_realtime\train'
validation_dir = r'C:\Users\cfarr\Desktop\fyp_pi_code1\fyp_realtime\val'

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Load the ResNet-101 model without the top layers
base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Add new layers on top of the pre-trained model
model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dropout(0.5))  # Add dropout for regularization
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.3))  # Add dropout for regularization
model.add(layers.Dense(1, activation='sigmoid'))

# Use an adaptive learning rate optimizer like Adam
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(learning_rate=1e-3),  # Adjust learning rate
              metrics=['accuracy'])

# Add early stopping
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=4,
    restore_best_weights=True
)

history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=14,  # Increase epochs for more training
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[early_stopping],
        verbose=1
    )

# Save the model
model.save('FYP_iteration1.h5')