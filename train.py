import os
import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import time
import config

def augment_image(img_array):
    augmented = []
    
    augmented.append(img_array)
    
    angle = np.random.randint(-15, 15)
    h, w = img_array.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    rotated = cv2.warpAffine(img_array, M, (w, h), borderValue=255)
    augmented.append(rotated)
    
    scale = np.random.uniform(0.8, 1.2)
    scaled = cv2.resize(img_array, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    if scaled.shape[0] > h or scaled.shape[1] > w:
        y_start = (scaled.shape[0] - h) // 2
        x_start = (scaled.shape[1] - w) // 2
        scaled = scaled[y_start:y_start+h, x_start:x_start+w]
    else:
        canvas = np.ones((h, w), dtype=np.uint8) * 255
        y_start = (h - scaled.shape[0]) // 2
        x_start = (w - scaled.shape[1]) // 2
        canvas[y_start:y_start+scaled.shape[0], x_start:x_start+scaled.shape[1]] = scaled
        scaled = canvas
    augmented.append(scaled)
    
    kernel_size = np.random.choice([3, 5])
    blurred = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0)
    augmented.append(blurred)
    
    noise = np.random.normal(0, 10, img_array.shape).astype(np.uint8)
    noisy = cv2.add(img_array, noise)
    augmented.append(noisy)
    
    kernel = np.ones((2,2), np.uint8)
    eroded = cv2.erode(img_array, kernel, iterations=1)
    augmented.append(eroded)
    
    dilated = cv2.dilate(img_array, kernel, iterations=1)
    augmented.append(dilated)
    
    brightness = np.random.randint(-30, 30)
    bright = cv2.convertScaleAbs(img_array, alpha=1, beta=brightness)
    augmented.append(bright)
    
    inverted = 255 - img_array
    augmented.append(inverted)
    
    return augmented

def load_images_from_folder(folder, use_augmentation=True):
    images = []
    labels = []
    
    print(f"Loading images from: {folder}")
    
    for digit in range(10):
        digit_folder = os.path.join(folder, str(digit))
        if not os.path.exists(digit_folder):
            continue
            
        files = [f for f in os.listdir(digit_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"  Digit {digit}: {len(files)} images", end='')
        
        original_count = len(files)
        augmented_count = 0
        
        for filename in files:
            filepath = os.path.join(digit_folder, filename)
            try:
                img = Image.open(filepath)
                
                if img.mode == 'RGBA':
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background
                
                img_array = np.array(img)
                
                if len(img_array.shape) == 3:
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img_array
                
                resized = cv2.resize(gray, config.IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                
                if config.AUTO_INVERT and np.mean(resized) > 127:
                    resized = 255 - resized
                
                if use_augmentation:
                    augmented_imgs = augment_image(resized)
                    
                    for aug_img in augmented_imgs:
                        if config.NORMALIZE:
                            aug_img_normalized = aug_img.astype(np.float32) / 255.0
                        else:
                            aug_img_normalized = aug_img
                        
                        flattened = aug_img_normalized.flatten()
                        images.append(flattened)
                        labels.append(digit)
                        augmented_count += 1
                else:
                    if config.NORMALIZE:
                        resized = resized.astype(np.float32) / 255.0
                    
                    flattened = resized.flatten()
                    images.append(flattened)
                    labels.append(digit)
                
            except Exception as e:
                print(f"\n    Error loading {filename}: {e}")
        
        if use_augmentation:
            print(f" -> {augmented_count} total (augmented)")
        else:
            print()
    
    return np.array(images), np.array(labels)

def train_model():
    print("\n" + "="*60)
    print("DIGIT RECOGNITION - TRAINING WITH AUGMENTATION")
    print("="*60 + "\n")
    
    if not os.path.exists(config.TRAIN_DIR):
        print(f"Error: Training directory not found: {config.TRAIN_DIR}")
        return
    
    print("[1/5] Loading dataset with augmentation...")
    X, y = load_images_from_folder(config.TRAIN_DIR, use_augmentation=True)
    
    if len(X) == 0:
        print("Error: No training data found!")
        return
    
    print(f"\nTotal images after augmentation: {len(X)}")
    print(f"Image shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    print(f"\n[2/5] Splitting dataset (test size: {config.TEST_SIZE*100:.0f}%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples: {len(X_test)}")
    
    print("\n[3/5] Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=config.N_ESTIMATORS,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        max_depth=30,
        min_samples_split=5
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"  Training completed in {training_time:.2f} seconds")
    
    print("\n[4/5] Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n  Overall Accuracy: {accuracy*100:.2f}%\n")
    
    print("Accuracy per digit:")
    for digit in range(10):
        mask = y_test == digit
        if mask.sum() > 0:
            digit_accuracy = (y_pred[mask] == digit).sum() / mask.sum()
            print(f"  Digit {digit}: {digit_accuracy*100:.2f}%")
    
    print("\n[5/5] Saving model...")
    with open(config.MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    file_size = os.path.getsize(config.MODEL_PATH) / (1024*1024)
    print(f"  Model saved: {config.MODEL_PATH}")
    print(f"  File size: {file_size:.2f} MB")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nAugmented samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Training time: {training_time:.2f} seconds")
    print("\nModel now handles:")
    print("  - Rotated images")
    print("  - Scaled images")
    print("  - Blurred images")
    print("  - Noisy images")
    print("  - Thin lines (eroded)")
    print("  - Thick lines (dilated)")
    print("  - Brightness variations")
    print("  - Inverted colors")
    print("\nRun: python gui.py")
    print("="*60 + "\n")

if __name__ == "__main__":
    train_model()