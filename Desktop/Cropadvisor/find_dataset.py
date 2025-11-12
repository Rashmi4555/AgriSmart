import os

print("🔍 Finding your PlantVillage dataset...")
print("=" * 50)

# Expanded list of possible paths
possible_paths = [
    "plantvillage dataset",
    "plantvillage dataset/color", 
    "PlantVillage",
    "PlantVillage/color",
    "dataset",
    "plant_disease_dataset",
    "plant_village",
    "Plant",
    "plant",
    "color",  # Sometimes it's just in a folder called 'color'
    "PlantVillageDataset",
    "/kaggle/input/plantvillage-dataset/plantvillage dataset/Color",  # Common Kaggle path
    "/kaggle/input/plantvillage-dataset/plantvillage dataset/color",
    "data",
    "images"
]

found_paths = []

for path in possible_paths:
    if os.path.exists(path):
        try:
            # Get all subdirectories (classes)
            classes = [d for d in os.listdir(path) 
                      if os.path.isdir(os.path.join(path, d))]
            
            if classes:
                # Count images in first few classes to verify it's the dataset
                total_images = 0
                for class_name in classes[:3]:  # Check first 3 classes
                    class_path = os.path.join(path, class_name)
                    if os.path.exists(class_path):
                        images = [f for f in os.listdir(class_path) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                        total_images += len(images)
                
                found_paths.append({
                    'path': path, 
                    'num_classes': len(classes),
                    'sample_classes': classes[:5],
                    'approx_images': total_images * len(classes) // 3 if total_images > 0 else 0
                })
        except PermissionError:
            print(f"⚠️  Permission denied: {path}")
        except Exception as e:
            print(f"⚠️  Error reading {path}: {e}")

if found_paths:
    print("✅ Found datasets:")
    for dataset in found_paths:
        print(f"📁 {dataset['path']}")
        print(f"   Classes: {dataset['num_classes']}")
        print(f"   Sample: {dataset['sample_classes']}")
        print(f"   Estimated total images: ~{dataset['approx_images']}")
        print()
else:
    print("❌ No dataset found in common locations")
    print("\n💡 Try these solutions:")
    print("1. Check if the dataset is downloaded and extracted")
    print("2. Run this in your terminal to see all folders:")
    print("   import os; print([d for d in os.listdir('.') if os.path.isdir(d)])")
    print("3. The dataset might be in a parent directory")