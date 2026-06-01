import json
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ---- Define the PyTorch model (needed for loading weights)
class BanglaOCRModel(nn.Module):
    def __init__(self, num_classes):
        super(BanglaOCRModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---- IMPORTANT — Re-declare normalize exactly as it was in the training notebook.
def normalize(images: np.ndarray) -> np.ndarray:
    return images.astype("float32") / 255.0

# Mapping of directory names (1-84) in BanglaLekha-Isolated dataset to actual Bangla Unicode characters
BANGLA_CHAR_MAP = {
    "1": "অ", "2": "আ", "3": "ই", "4": "ঈ", "5": "উ", "6": "ঊ", "7": "ঋ", "8": "এ", "9": "ঐ", "10": "ও",
    "11": "ঔ", "12": "ক", "13": "খ", "14": "গ", "15": "ঘ", "16": "ঙ", "17": "চ", "18": "ছ", "19": "জ", "20": "ঝ",
    "21": "ঞ", "22": "ট", "23": "ঠ", "24": "ড", "25": "ঢ", "26": "ণ", "27": "ত", "28": "থ", "29": "দ", "30": "ধ",
    "31": "ন", "32": "প", "33": "ফ", "34": "ব", "35": "ভ", "36": "ম", "37": "য", "38": "র", "39": "ল", "40": "শ",
    "41": "ষ", "42": "স", "43": "হ", "44": "ড়", "45": "ঢ়", "46": "য়", "47": "ৎ", "48": "ং", "49": "ঃ", "50": "ঁ",
    "51": "০", "52": "১", "53": "২", "54": "৩", "55": "৪", "56": "৫", "57": "৬", "58": "৭", "59": "৮", "60": "৯",
    "61": "ক্ত", "62": "ক্ষ", "63": "গ্ধ", "64": "ঙ্ক", "65": "ঙ্গ", "66": "চ্ছ", "67": "জ্জ", "68": "জ্ঞ", "69": "ট্ট", "70": "দ্ধ",
    "71": "ন্ত", "72": "ন্দ", "73": "ন্ন", "74": "প্ট", "75": "প্প", "76": "ফ্র", "77": "ব্দ", "78": "ব্ধ", "79": "ভ্ৰ", "80": "ম্প",
    "81": "ল্ট", "82": "ল্ক", "83": "শ্চ", "84": "স্ত"
}

# ---- paths (relative to this file for seamless Docker execution) ----
ROOT          = Path(__file__).parent
MODEL_PATH    = ROOT / "models" / "model.pt"
LABELS_PATH   = ROOT / "labels.json"

# ---- page config ----
st.set_page_config(
    page_title="Bangla OCR System",
    layout="centered",
)

# ---- cached loader: runs once per server startup ----
@st.cache_resource(show_spinner="Loading Bangla OCR model…")
def load_artifacts():
    with open(LABELS_PATH, "r") as f:
        class_labels_dict = json.load(f)
    num_classes = len(class_labels_dict)
    model = BanglaOCRModel(num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return class_labels_dict, model

try:
    class_labels_dict, model = load_artifacts()
    # Ensure keys are sorted if we need to map model output indices
    class_labels = [class_labels_dict[str(i)] for i in range(len(class_labels_dict))]
    # Assume 32x32 based on our training script
    input_shape = (32, 32)
except Exception as e:
    st.error(f"Failed to load model or labels. Please run training first. Error: {e}")
    st.stop()

def segment_characters(rgba: np.ndarray):
    """
    Takes an RGBA image from the canvas, finds contours, 
    and segments into individual character bounding boxes, sorted left-to-right.
    """
    # Convert RGBA to grayscale
    img = Image.fromarray(rgba.astype("uint8")).convert("L")
    arr = np.array(img)

    # Find contours on the black background with white strokes
    # Thresholding to get binary image
    _, binary = cv2.threshold(arr, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out very small contours (noise)
    min_area = 20
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    # Get bounding boxes
    boxes = [cv2.boundingRect(c) for c in valid_contours]
    
    # Sort boxes from left to right (by x-coordinate)
    boxes = sorted(boxes, key=lambda b: b[0])
    
    character_images = []
    for (x, y, w, h) in boxes:
        # Extract Region of Interest
        roi = arr[y:y+h, x:x+w]
        
        # Make it a square by padding
        side = max(w, h)
        padded = np.zeros((side, side), dtype=np.uint8)
        
        # Center the roi inside the padded square
        x_offset = (side - w) // 2
        y_offset = (side - h) // 2
        padded[y_offset:y_offset+h, x_offset:x_offset+w] = roi

        # Resize to input_shape
        char_img = Image.fromarray(padded).resize((input_shape[1], input_shape[0]), Image.LANCZOS)
        char_arr = np.asarray(char_img, dtype="float32") / 255.0
        character_images.append(char_arr)

    return character_images, boxes

# ============================== UI Layout ==============================
st.title("Bangla Handwritten Word Recognizer")
st.caption(
    "Draw a Bangla word (multiple characters) in the black box. "
    "The system will segment and recognize each character from left to right."
)

col_draw, col_pred = st.columns([1.05, 1])

with col_draw:
    st.subheader("1. Draw")
    canvas = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=10,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=500, # Wider to accommodate a word
        drawing_mode="freedraw",
        key="bangla-canvas",
    )
    predict_clicked = st.button("Predict", type="primary", use_container_width=True)

with col_pred:
    st.subheader("2. Prediction")
    if predict_clicked and canvas.image_data is not None:
        if canvas.image_data.sum() < 1e-3:
            st.warning("The canvas looks empty — draw a word and click Predict.")
        else:
            char_images, boxes = segment_characters(canvas.image_data)
            
            if not char_images:
                st.warning("Could not detect any valid characters.")
            else:
                predicted_word_labels = []
                st.write(f"Detected {len(char_images)} character(s):")
                
                for idx, char_arr in enumerate(char_images):
                    # PyTorch expects shape (1, 1, 32, 32) (batch_size, channels, H, W)
                    x_input = char_arr[np.newaxis, np.newaxis, ...]
                    x_tensor = torch.tensor(x_input, dtype=torch.float32)
                    
                    # Predict probabilities using PyTorch model
                    with torch.no_grad():
                        logits = model(x_tensor)
                        probs = torch.softmax(logits, dim=1).numpy()[0]
                    
                    top = int(np.argmax(probs))
                    class_folder_name = class_labels[top]
                    predicted_label = BANGLA_CHAR_MAP.get(class_folder_name, class_folder_name)
                    predicted_word_labels.append(predicted_label)
                    
                    # Display each char metric
                    st.metric(label=f"Char {idx+1}",
                               value=str(predicted_label),
                               delta=f"{probs[top]*100:.1f}% confident")
                               
                final_word = " ".join(predicted_word_labels)
                st.success(f"**Predicted Word Sequence:** {final_word}")
    else:
        st.info("Draw something on the left, then press **Predict**.")

# ---- sidebar diagnostics ----
with st.sidebar:
    st.header("Model card")
    st.write("**Architecture:** Conv2d -> MaxPool -> Conv2d -> MaxPool -> Linear(128) -> Linear(84)")
    st.write(f"**PyTorch:** `{torch.__version__}`")
    st.write(f"**Classes Trained:** `{len(class_labels)}`")
