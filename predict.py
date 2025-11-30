import os
import numpy as np
import cv2
from PIL import Image
import pickle
import config

def load_model():
    if not os.path.exists(config.MODEL_PATH):
        print(f"Error: Model not found at {config.MODEL_PATH}")
        print("Please train the model first: python train.py")
        return None
    
    with open(config.MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model

def preprocess_image(image):
    if isinstance(image, str):
        img = Image.open(image)
    else:
        img = image
    
    original_size = img.size
    
    if img.mode == 'RGBA':
        alpha = np.array(img.split()[3])
        if np.mean(alpha) < 250:
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        else:
            img = img.convert('RGB')
    elif img.mode == 'P':
        img = img.convert('RGB')
    elif img.mode == 'L':
        img = Image.fromarray(np.array(img))
    elif img.mode not in ['RGB', 'L']:
        img = img.convert('RGB')
    
    img_array = np.array(img)
    
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    gray_copy = gray.copy()
    
    _, binary = cv2.threshold(gray_copy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        margin = int(max(w, h) * 0.1)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(gray.shape[1] - x, w + 2 * margin)
        h = min(gray.shape[0] - y, h + 2 * margin)
        
        gray = gray[y:y+h, x:x+w]
    
    mean_intensity = np.mean(gray)
    
    if mean_intensity > 127:
        gray = 255 - gray
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    gray = clahe.apply(gray)
    
    max_dim = max(gray.shape)
    square = np.ones((max_dim, max_dim), dtype=np.uint8) * 255
    y_offset = (max_dim - gray.shape[0]) // 2
    x_offset = (max_dim - gray.shape[1]) // 2
    square[y_offset:y_offset+gray.shape[0], x_offset:x_offset+gray.shape[1]] = gray
    
    resized = cv2.resize(square, config.IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    
    kernel = np.ones((2,2), np.uint8)
    resized = cv2.morphologyEx(resized, cv2.MORPH_CLOSE, kernel)
    
    if config.NORMALIZE:
        resized = resized.astype(np.float32) / 255.0
    
    flattened = resized.flatten()
    
    return flattened.reshape(1, -1)

def predict_digit(model, image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0]
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(processed)[0]
        confidence = probabilities[prediction]
    else:
        confidence = 1.0
    
    return int(prediction), float(confidence)

def predict_from_path(image_path):
    print(f"\nPredicting: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return None, None
    
    model = load_model()
    if model is None:
        return None, None
    
    prediction, confidence = predict_digit(model, image_path)
    
    word = config.NUMBER_TO_WORD[prediction]
    
    print(f"\nResult:")
    print(f"  Digit: {prediction}")
    print(f"  Word: {word.upper()}")
    print(f"  Confidence: {confidence*100:.1f}%\n")
    
    return prediction, confidence

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    
    predict_from_path(sys.argv[1])