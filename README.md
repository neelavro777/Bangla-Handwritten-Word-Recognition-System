# Bangla Handwritten Word Recognition System

This repository implements a complete, containerized system for recognizing handwritten Bangla words. It combines classic computer vision techniques for horizontal character segmentation with a deep Convolutional Neural Network (CNN) for character classification.

The system is built with three primary technological pillars:

- **PyTorch** for deep learning model design, training, and GPU acceleration.
- **MLflow** for robust experiment tracking, hyperparameter logging, metric visualization, and model registry.
- **Streamlit** for an interactive, canvas-based web application that displays Unicode text recognition.
- **Docker** for containerizing the application to ensure repeatable and portable deployments.

---

## Problem Statement

Handwritten character recognition for the Bangla script is highly challenging due to:

- A complex alphabet comprising vowels, consonants, numerals, and an extensive set of compound conjunct characters.
- High structural variability in individual handwriting styles.
- Cursive and continuous strokes that connect adjacent characters.

This system approaches the problem by dividing handwritten word recognition into two key stages:

1. **Horizontal Word Segmentation:** A handwritten word drawn on a digital canvas is processed using OpenCV to identify individual character contours. Bounding boxes are filtered to remove noise, and sorted left-to-right to isolate the individual characters in sequence.
2. **Deep Learning Classification:** Each isolated character is preprocessed, resized to 32x32 pixels, and passed through a deep CNN built in PyTorch to classify it into one of the 84 target classes. The predicted class is then mapped to its corresponding Bangla Unicode character to reconstruct the written word sequence.

---

## Dataset Description

The system is trained on the **BanglaLekha-Isolated** dataset, a large-scale, public dataset of isolated handwritten Bangla characters.

### Dataset Statistics

- **Total Samples:** 166,105 handwritten character images.
- **Target Classes:** 84 distinct character classes.
- **Class Structure:**
  - **Classes 1 to 11:** Vowels (অ to ঔ)
  - **Classes 12 to 50:** Consonants (ক to ঁ)
  - **Classes 51 to 60:** Bangla Numerals (০ to ৯)
  - **Classes 61 to 84:** Compound Conjuncts (ক্ত to স্ত)

### Image Naming Convention

Each image file in the dataset follows a specific metadata-encoding naming convention:
`[District]_[Institution]_[Gender]_[Age]_[Date]_[FormID]_[ClassID].png`

For example, in `01_0001_0_08_0916_1990_1.png`:

- **First 2 digits (01):** District identifier from which the sample was collected.
- **Next 4 digits (0001):** Institution identifier.
- **Next 1 digit (0):** Gender of the subject (0 for male, 1 for female).
- **Next 2 digits (08):** Age of the subject.
- **Next 4 digits (0916):** Date of collection (September 16, 2016).
- **Next 4 digits (1990):** Serial number of the collection form.
- **Last 1 digit (1):** Character class of the sample (matches the directory name, 1 to 84).

---

## Installation & Local Setup

The project uses the modern Python package manager **uv** for fast and reproducible environment management.

### 1. Synchronize the Environment

Clone this repository and run the following command in the project root to create a virtual environment and synchronize all dependencies:

```bash
uv sync
```

Alternatively, if you prefer standard pip workflows:

```bash
uv venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Dataset Placement

Since the dataset is large, it should not be committed to Git. Create the target directory and place your extracted dataset there:

1. Extract your dataset zip file.
2. Place the folders numbered `1` through `84` inside the directory:
   `data/raw/Images/`
3. Verify that the path `data/raw/Images/1/` contains the `.png` images for class 1.

---

## Model Training & MLflow Experiment Tracking

The training script leverages GPU acceleration if CUDA is available and tracks parameters and metrics to an MLflow SQLite database.

### 1. Run the Training Script

Run the training pipeline using uv:

```bash
uv run python train.py
```

During execution, you will see batch-level logs printed every 500 batches, followed by epoch summaries showing train/validation loss and accuracy. The trained weights will be saved to `models/model.pt` and the model dictionary mapping to `labels.json`.

### 2. Start the MLflow UI

To view the logged runs, hyperparameter charts, and registered models, start the MLflow server:

```bash
uv run mlflow ui --host 127.0.0.1 --port 5000 &
```

Open your browser and navigate to `http://127.0.0.1:5000`. Switch the selected experiment from **Default** to **bangla-ocr** in the left sidebar to visualize and compare your training runs.

---

## Streamlit Interactive Web Application

Launch the Streamlit app to interactively test the model:

```bash
uv run streamlit run app.py
```

This launches a browser window where you can:

1. Draw a handwritten word (composed of multiple characters) on the digital canvas.
2. Press **Predict** to trigger the pipeline.
3. Observe the OpenCV segmentation bounding boxes and character metrics (including confidence percentages) displayed in order.
4. Read the reconstructed Unicode sequence.

---

## Docker Containerization

To package the application and its dependencies into a portable, isolated environment:

### 1. Build the Docker Image

```bash
sudo docker build -t bangla-ocr-app:0.1 .
```

### 2. Run the Container

```bash
sudo docker run -p 8501:8501 bangla-ocr-app:0.1
```

Open your browser and go to `http://localhost:8501` to use the application.

_Note: If port 8501 is already in use by a local Streamlit server on your host machine, you can map the container to a different host port (such as 8502) instead:_

```bash
sudo docker run -p 8502:8501 bangla-ocr-app:0.1
```

_Access this version in your browser at `http://localhost:8502`._

---

## System Architecture & Approach

- **Preprocessing**: Input images from the Mendeley dataset are resized to 32x32 pixels and normalized. The canvas drawing from Streamlit is captured as RGBA, converted to Grayscale, and thresholded.
- **Segmentation**: OpenCV (`cv2.findContours`) segments the white strokes on the black background. Contours are filtered and sorted left-to-right to form a word sequence.
- **Model**: A CNN built with PyTorch extracts spatial features using `Conv2d` and `MaxPool2d` layers, followed by `Linear` layers for multi-class classification.
- **MLflow Tracking**: During training, hyperparameters (epochs, batch_size, img_size) and metrics (accuracy, loss) are tracked. The final model is saved alongside a mapped `labels.json` dictionary. The trained model is logged using MLflow's native PyTorch flavor.

## Limitations & Future Improvements

- **Segmentation Strategy**: The current segmentation uses basic bounding box contouring. This works well for cleanly separated characters but may fail for cursive or overlapping characters. Advanced techniques like CTC (Connectionist Temporal Classification) models would provide more robust sequence recognition without explicit segmentation.
- **Training Time**: The current implementation restricts image sizes for quicker execution. Deeper architectures like MobileNet or ResNet could improve accuracy at the cost of training time.
