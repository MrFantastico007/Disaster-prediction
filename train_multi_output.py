# %%% CELL 1: Imports and Drive Mount %%%
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, RandomFlip, RandomRotation, RandomContrast,
    GlobalAveragePooling2D, BatchNormalization, Dense, Dropout
)
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# Assuming running locally or if on Colab:
# from google.colab import drive
# drive.mount('/content/drive')

# %%% CELL 2: Configuration & Variables %%%
# Configuration
# Assuming the data directory is named 'disaster_final' and in the same root folder
DATA_DIR = 'disaster_final' 
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE

DISASTER_TYPES = ['earthquake', 'flood', 'wildfire']
INTENSITIES = ['high', 'medium', 'low']

disaster_to_idx = {name: idx for idx, name in enumerate(DISASTER_TYPES)}
intensity_to_idx = {name: idx for idx, name in enumerate(INTENSITIES)}

# %%% CELL 3: Custom Data Loader Function %%%
def create_dataset_from_directory(base_path, batch_size=32, target_size=(224, 224)):
    """
    Crawls the directory structure to create a tf.data.Dataset for multi-output.
    Expects: base_path/[disaster_type]/[intensity]/image.jpg
    """
    image_paths = []
    disaster_labels = []
    intensity_labels = []

    search_pattern = os.path.join(base_path, '*', '*', '*.*')
    all_files = glob.glob(search_pattern)
    
    for file_path in all_files:
        parts = os.path.normpath(file_path).split(os.sep)
        
        if len(parts) >= 3:
            intensity = parts[-2].lower()
            disaster = parts[-3].lower()

            if disaster in disaster_to_idx and intensity in intensity_to_idx:
                image_paths.append(file_path)
                disaster_labels.append(disaster_to_idx[disaster])
                intensity_labels.append(intensity_to_idx[intensity])

    print(f"Found {len(image_paths)} valid images in {base_path}")

    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    label1_ds = tf.data.Dataset.from_tensor_slices(disaster_labels)
    label2_ds = tf.data.Dataset.from_tensor_slices(intensity_labels)

    labels_ds = tf.data.Dataset.zip((label1_ds, label2_ds)).map(
        lambda l1, l2: {'disaster_output': l1, 'intensity_output': l2}
    )

    def process_path(file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False) 
        img = tf.image.resize(img, target_size)
        return img

    image_ds = path_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    dataset = tf.data.Dataset.zip((image_ds, labels_ds))
    dataset = dataset.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    return dataset

if __name__ == "__main__":
    # %%% CELL 4: Load the Datasets %%%
    print("\nProcessing Datasets...")
    train_ds = create_dataset_from_directory(os.path.join(DATA_DIR, 'train'), BATCH_SIZE, IMG_SIZE)
    train_ds = train_ds.unbatch().shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

    val_ds = create_dataset_from_directory(os.path.join(DATA_DIR, 'validation'), BATCH_SIZE, IMG_SIZE)
    test_ds = create_dataset_from_directory(os.path.join(DATA_DIR, 'test'), BATCH_SIZE, IMG_SIZE)

    # %%% CELL 5: Define the Multi-Output Model %%%
    print("\nBuilding model...")
    data_augmentation = Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.3),
        RandomContrast(0.2)
    ], name="data_augmentation")

    inputs = Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)

    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=x)
    base_model.trainable = False 

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    common_features = BatchNormalization()(x)

    h_disaster = Dense(256, activation='relu')(common_features)
    h_disaster = Dropout(0.5)(h_disaster)
    output_disaster = Dense(len(DISASTER_TYPES), activation='softmax', name='disaster_output')(h_disaster)

    h_intensity = Dense(256, activation='relu')(common_features)
    h_intensity = Dropout(0.5)(h_intensity)
    output_intensity = Dense(len(INTENSITIES), activation='softmax', name='intensity_output')(h_intensity)

    model = Model(inputs=inputs, outputs=[output_disaster, output_intensity])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)
    ]

    # %%% CELL 6: Phase 1 Training (Feature Extraction) %%%
    print("\n--- PHASE 1: Feature Extraction ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            'disaster_output': 'sparse_categorical_crossentropy',
            'intensity_output': 'sparse_categorical_crossentropy'
        },
        metrics={'disaster_output': 'accuracy', 'intensity_output': 'accuracy'}
    )

    EPOCHS_PHASE_1 = 15
    history_1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE_1,
        callbacks=callbacks
    )

    # %%% CELL 7: Phase 2 Training (Fine-Tuning) %%%
    print("\n--- PHASE 2: Fine-Tuning ---")
    base_model.trainable = True

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss={
            'disaster_output': 'sparse_categorical_crossentropy',
            'intensity_output': 'sparse_categorical_crossentropy'
        },
        metrics={'disaster_output': 'accuracy', 'intensity_output': 'accuracy'}
    )

    EPOCHS_PHASE_2 = 30
    initial_epoch = len(history_1.history['loss'])
    total_epochs = initial_epoch + EPOCHS_PHASE_2

    history_2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks
    )

    # %%% CELL 8: Evaluation & Generating Plots (Confusion Matrix) %%%
    print("\n--- EVALUATION on Test Set ---")
    results = model.evaluate(test_ds)
    print(f"Overall Test Loss: {results[0]:.4f}")

    print("\nExtracting predictions for Confusion Matrices...")
    true_disaster, true_intensity = [], []
    pred_disaster, pred_intensity = [], []

    for images, labels in test_ds:
        true_disaster.extend(labels['disaster_output'].numpy())
        true_intensity.extend(labels['intensity_output'].numpy())
        
        batch_preds = model.predict(images, verbose=0)
        pred_disaster.extend(np.argmax(batch_preds[0], axis=1))
        pred_intensity.extend(np.argmax(batch_preds[1], axis=1))

    # --- Plot Confusion Matrices ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    cm_disaster = confusion_matrix(true_disaster, pred_disaster)
    sns.heatmap(cm_disaster, annot=True, fmt='d', cmap='Blues', 
                xticklabels=DISASTER_TYPES, yticklabels=DISASTER_TYPES, ax=axes[0])
    axes[0].set_title('Confusion Matrix: Disaster Type')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    cm_intensity = confusion_matrix(true_intensity, pred_intensity)
    sns.heatmap(cm_intensity, annot=True, fmt='d', cmap='Greens', 
                xticklabels=INTENSITIES, yticklabels=INTENSITIES, ax=axes[1])
    axes[1].set_title('Confusion Matrix: Intensity Level')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.show()

    # --- Plot Training History ---
    def combine_histories(h1, h2, metric):
        return h1.history[metric] + h2.history[metric]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0, 0].plot(combine_histories(history_1, history_2, 'disaster_output_accuracy'), label='Train')
    axes[0, 0].plot(combine_histories(history_1, history_2, 'val_disaster_output_accuracy'), label='Val')
    axes[0, 0].axvline(x=initial_epoch, color='r', linestyle='--', label='Fine-Tuning')
    axes[0, 0].set_title('Disaster Output - Accuracy')
    axes[0, 0].legend()

    axes[0, 1].plot(combine_histories(history_1, history_2, 'disaster_output_loss'), label='Train')
    axes[0, 1].plot(combine_histories(history_1, history_2, 'val_disaster_output_loss'), label='Val')
    axes[0, 1].axvline(x=initial_epoch, color='r', linestyle='--', label='Fine-Tuning')
    axes[0, 1].set_title('Disaster Output - Loss')
    axes[0, 1].legend()

    axes[1, 0].plot(combine_histories(history_1, history_2, 'intensity_output_accuracy'), label='Train')
    axes[1, 0].plot(combine_histories(history_1, history_2, 'val_intensity_output_accuracy'), label='Val')
    axes[1, 0].axvline(x=initial_epoch, color='r', linestyle='--', label='Fine-Tuning')
    axes[1, 0].set_title('Intensity Output - Accuracy')
    axes[1, 0].legend()

    axes[1, 1].plot(combine_histories(history_1, history_2, 'intensity_output_loss'), label='Train')
    axes[1, 1].plot(combine_histories(history_1, history_2, 'val_intensity_output_loss'), label='Val')
    axes[1, 1].axvline(x=initial_epoch, color='r', linestyle='--', label='Fine-Tuning')
    axes[1, 1].set_title('Intensity Output - Loss')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
