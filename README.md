# DigitReader

## Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv
```

### 2. Activate Virtual Environment

**On Linux/macOS:**
```bash
source .venv/bin/activate
```

**On Windows:**
```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Deactivate Virtual Environment (when done)

```bash
deactivate
```

## Usage

### Train the Model

Before running predictions, you need to train the model first:

```bash
python train.py
```

### Run Predictions

#### Option 1: GUI Interface

```bash
python gui.py
```

#### Option 2: Command Line

```bash
python predict.py <path_to_image>
```

**Example:**
```bash
python predict.py test_image.png
```

## Notes

- Make sure to train the model before attempting to make predictions
- The virtual environment must be activated whenever you work with this project
- Supported image formats depend on your implementation (typically PNG, JPG, JPEG)