import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

print("🚀 Starting Real Plant Disease Model Training")
print("TensorFlow version:", tf.__version__)
print("=" * 60)

# ---------- CONFIG ----------
DATASET_PATH = "plantvillage dataset/color"  # Update this path if needed

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
MODEL_NAME = "plant_disease_model.h5"
# ----------------------------

def check_dataset(dataset_path):
    """Verify dataset structure"""
    print(f"🔍 Checking dataset: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        print("💡 Please update DATASET_PATH in train.py with the correct path")
        print("💡 Current working directory:", os.getcwd())
        return False, []
    
    classes = [d for d in os.listdir(dataset_path) 
               if os.path.isdir(os.path.join(dataset_path, d))]
    
    if not classes:
        print("❌ No class folders found")
        return False, []
    
    print(f"✅ Found {len(classes)} plant classes")
    
    # Show class distribution
    print("\n📊 Class distribution (first 10):")
    for i, class_name in enumerate(classes[:10]):
        class_path = os.path.join(dataset_path, class_name)
        num_images = len([f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"   {i+1:2d}. {class_name}: {num_images} images")
    
    if len(classes) > 10:
        print(f"   ... and {len(classes) - 10} more classes")
    
    return True, classes

def create_model(num_classes):
    """Create a CNN model for plant disease classification"""
    print(f"🔄 Creating model for {num_classes} classes...")
    
    model = Sequential([
        # First Conv Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(*IMG_SIZE, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Second Conv Block  
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Third Conv Block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Fourth Conv Block (Added for better feature extraction)
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Classifier
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    return model

def plot_training(history):
    """Plot training history"""
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Training plot saved as 'training_results.png'")

def calculate_class_weights(generator):
    """Calculate class weights for imbalanced datasets"""
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(generator.classes),
        y=generator.classes
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    print("📊 Class weights for imbalanced data:")
    for class_idx, weight in class_weight_dict.items():
        class_name = list(generator.class_indices.keys())[class_idx]
        print(f"   {class_name}: {weight:.2f}")
    
    return class_weight_dict

def main():
    # Check dataset
    dataset_ok, classes = check_dataset(DATASET_PATH)
    if not dataset_ok:
        print("\n❌ Dataset check failed. Please:")
        print("   1. Check if the dataset path is correct")
        print("   2. Download the PlantVillage dataset if you don't have it")
        print("   3. Ensure the folder structure is: plantvillage dataset/color/class_name/images/")
        return
    
    print("\n📊 Preparing data generators...")
    
    # Data augmentation and preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        validation_split=0.2  # 80% train, 20% validation
    )
    
    # Validation data generator (only rescaling, no augmentation)
    val_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        validation_split=0.2
    )
    
    # Create data generators
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    print(f"✅ Training images: {train_generator.samples}")
    print(f"✅ Validation images: {val_generator.samples}")
    print(f"✅ Classes: {list(train_generator.class_indices.keys())}")
    
    # Calculate class weights for imbalanced data
    class_weight_dict = calculate_class_weights(train_generator)
    
    # Save class information
    class_info = {
        'classes': list(train_generator.class_indices.keys()),
        'class_indices': train_generator.class_indices,
        'img_size': IMG_SIZE,
        'num_classes': len(classes),
        'training_samples': train_generator.samples,
        'validation_samples': val_generator.samples
    }
    
    with open('class_names.json', 'w') as f:
        json.dump(class_info, f, indent=2)
    print("✅ Class names saved to 'class_names.json'")
    
    # Create model
    model = create_model(len(classes))
    
    print("\n📋 Model Summary:")
    model.summary()
    
    # Enhanced callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_NAME,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=4,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print("\n🎯 Starting training...")
    print("⏰ This may take 15-45 minutes depending on your hardware...")
    print("📊 You can monitor progress below:\n")
    
    # Train model with class weights
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        class_weight=class_weight_dict,  # Handle imbalanced data
        verbose=1
    )
    
    # Save final model
    model.save(MODEL_NAME)
    print(f"✅ Model saved as '{MODEL_NAME}'")
    
    # Evaluate model
    print("\n📈 Evaluating model...")
    train_loss, train_acc = model.evaluate(train_generator, verbose=0)
    val_loss, val_acc = model.evaluate(val_generator, verbose=0)
    
    print(f"🎯 Final Training Accuracy: {train_acc:.4f} ({train_acc:.2%})")
    print(f"🎯 Final Validation Accuracy: {val_acc:.4f} ({val_acc:.2%})")
    print(f"📉 Final Training Loss: {train_loss:.4f}")
    print(f"📉 Final Validation Loss: {val_loss:.4f}")
    
    # Plot results
    plot_training(history)
    
    # Save training history
    history_dict = {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']]
    }
    
    with open('training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)
    print("✅ Training history saved to 'training_history.json'")
    
    print("\n🎉 Training completed successfully!")
    print("📁 Generated files:")
    print(f"   - {MODEL_NAME} (Trained model)")
    print(f"   - class_names.json (Class information)")
    print(f"   - training_results.png (Training history plot)")
    print(f"   - training_history.json (Training metrics)")
    
    # Final recommendations
    if val_acc > 0.85:
        print("\n✅ Excellent! Model performance is good.")
    elif val_acc > 0.75:
        print("\n⚠️  Good performance. Consider training for more epochs or tuning hyperparameters.")
    else:
        print("\n❌ Model performance needs improvement. Consider:")
        print("   - Adding more data")
        print("   - Using transfer learning")
        print("   - Increasing model complexity")
        print("   - Training for more epochs")

if __name__ == "__main__":
    main()