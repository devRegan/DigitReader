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
    
    gray = cv2.medianBlur(gray, 3)
    
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    mean_intensity = np.mean(gray)
    if mean_intensity > 127:
        gray = 255 - gray
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    kernel = np.ones((2,2), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        margin = int(max(w, h) * 0.15)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(binary.shape[1] - x, w + 2 * margin)
        h = min(binary.shape[0] - y, h + 2 * margin)
        binary = binary[y:y+h, x:x+w]
    
    max_dim = max(binary.shape)
    square = np.zeros((max_dim, max_dim), dtype=np.uint8)
    y_offset = (max_dim - binary.shape[0]) // 2
    x_offset = (max_dim - binary.shape[1]) // 2
    square[y_offset:y_offset+binary.shape[0], x_offset:x_offset+binary.shape[1]] = binary
    
    resized = cv2.resize(square, config.IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    
    kernel_dilate = np.ones((2,2), np.uint8)
    resized = cv2.dilate(resized, kernel_dilate, iterations=1)
    
    if config.NORMALIZE:
        resized = resized.astype(np.float32) / 255.0
    
    return resized.flatten().reshape(1, -1)

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