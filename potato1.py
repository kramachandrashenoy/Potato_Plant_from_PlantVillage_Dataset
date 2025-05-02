import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.utils.class_weight import compute_class_weight  

# Constants
BATCH_SIZE = 32
IMAGE_SIZE = 256
CHANNELS = 3
EPOCHS = 20  
NUM_CLASSES = 3
DATA_DIR = "/home/rc/fyp/FYP/PlantVillage"  # CHANGE THIS TO YOUR LOCAL PATH

# Load dataset without mapping first to get class_names
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

# Capture class_names before mapping
class_names = train_ds.class_names
print("Class names:", class_names)  # Verify: Should be ['Bacteria', 'Fungi', 'Healthy']

# Apply preprocessing with map
def preprocess_image(image, label):
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image, label

train_ds = train_ds.map(preprocess_image)
val_ds = val_ds.map(preprocess_image)

# Compute class weights
labels = np.concatenate([y.numpy() for _, y in train_ds])
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)  # Debug: Should show ~{0: 0.89, 1: 0.68, 2: 2.52}

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.3), 
    layers.RandomZoom(0.3),     
    layers.RandomContrast(0.2), 
])

# Load base model
base_model = tf.keras.applications.MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

# Build model
model = tf.keras.Sequential([
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Train model 
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop, lr_schedule],
    class_weight=class_weight_dict  
)

# Evaluate model
loss, accuracy = model.evaluate(val_ds)
print(f"Validation accuracy: {accuracy * 100:.2f}%")

# Save model
model.save("potato_classifier.h5")  # Using the name from your context
print("Model saved locally as potato_classifier.h5")

from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

# Get true labels and predictions from the validation set
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

# Compute metrics
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
val_acc = accuracy  # From earlier model.evaluate()

# Generate classification report
report = classification_report(y_true, y_pred, target_names=class_names)

# Save to file
with open("potatoresults.txt", "w") as f:
    f.write(f"Validation Accuracy: {val_acc * 100:.2f}%\n")
    f.write(f"Weighted Precision: {precision:.4f}\n")
    f.write(f"Weighted Recall: {recall:.4f}\n")
    f.write(f"Weighted F1 Score: {f1:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print("Results saved to potatoresults.txt")
