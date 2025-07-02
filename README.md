# Lazy Landmark Finder

Lazy Landmark Finder is a deep learning project that recognizes Egyptian landmarks from images, even when the images are low-quality, blurry, or taken from unusual angles. The project demonstrates a full machine learning workflow: dataset preparation, model training, inference, and web deployment.

---

## 🚩 Overview
This project aims to make landmark recognition robust to "lazy" (imperfect) photos. It uses a pre-trained ResNet18 model, fine-tuned on a curated dataset of Egyptian landmarks, and provides:
- Automated dataset selection (top 20 most-photographed landmarks)
- Data augmentation to simulate real-world, low-effort photography
- Model training with validation
- Inference for single images
- A Streamlit web app for easy, interactive deployment

---

## ✨ Features
- **Automated dataset preparation:** Selects and copies the top 20 Egyptian landmark folders for training.
- **Data augmentation:** Simulates blurry, rotated, and low-light images to improve model robustness.
- **Transfer learning:** Fine-tunes a ResNet18 model for landmark classification.
- **Validation and best model saving:** Tracks and saves the best model during training.
- **Single image inference:** Predicts the landmark in any user-supplied image.
- **Web app:** User-friendly interface for uploading images and getting predictions.

---

## 📁 Project Structure
```
lazy_landmark_finder/
├── src/                   # All source code (Python scripts)
│   ├── gldv2_subset.py    # Dataset selection/copy script
│   ├── train.py           # Model training script
│   ├── infer.py           # Single image inference script
│   └── app.py             # Streamlit web app
├── requirements.txt       # Python dependencies
├── README.md              # Project overview and instructions
├── .gitignore             # Files/folders to ignore in git
└── LICENSE                # Open-source license
```

**Note:**
- The `data/`, `dataset/`, `models/`, and `notebooks/` directories are not included in the repository due to size constraints. Please follow the instructions in the README to download or generate these files locally.

---

## ⚙️ How it Works

### 1. **Dataset Preparation (`src/gldv2_subset.py`)**
- Scans the raw Egyptian landmarks dataset.
- Selects the top 20 landmarks with the most images.
- Copies these folders into the `dataset/` directory for training.

### 2. **Model Training (`src/train.py`)**
- Loads the processed dataset.
- Applies data augmentation (blur, color jitter, rotation, flips).
- Splits data into training and validation sets.
- Fine-tunes a ResNet18 model for multi-class classification.
- Tracks validation accuracy and saves the best model to `models/lazy_landmark_model_best.pth`.

### 3. **Inference (`src/infer.py`)**
- Loads the trained model and class names.
- Accepts a user-supplied image path.
- Preprocesses the image and predicts the landmark class.

### 4. **Web App (`src/app.py`)**
- Provides a Streamlit interface for uploading images.
- Displays the uploaded image and the predicted landmark.
- Runs entirely in the browser for easy sharing and demo.

---

## 🚀 Setup
1. **Clone the repo and navigate to the project directory.**
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Prepare the dataset:**
   - Place your raw Egyptian landmarks data in `data/egypt_landmarks/images/`.
   - Run:
     ```sh
     python src/gldv2_subset.py
     ```
4. **Train the model:**
   ```sh
   python src/train.py
   ```
5. **Run inference:**
   ```sh
   python src/infer.py path_to_image.jpg
   ```
6. **Launch the web app:**
   ```sh
   streamlit run src/app.py
   ```

---

## 📝 Usage
- **Dataset Preparation:**
  - Selects and copies the top 20 landmark folders for training.
- **Training:**
  - Fine-tunes a ResNet18 model with data augmentation and validation.
- **Inference:**
  - Predicts the landmark in any user-supplied image.
- **Web App:**
  - Upload an image and get a prediction in your browser.

---

## 🙏 Credits
- Egyptian landmarks dataset: [kaggle egyptian landmark dataset]
- Model: PyTorch ResNet18
- Web app: Streamlit

---

## ⬇️ Download Data and Models

To keep this repository lightweight, data and model files are not included. Please download them separately:

- **Egyptian Landmarks Dataset:**
  - Download from [Kaggle/egypt landmarks] or your preferred source.
  - Place the images in `data/egypt_landmarks/images/`.

- **Trained Model Weights:**
 
  - Place the file in the `models/` directory as `lazy_landmark_model_best.pth`.

If you train your own model, the weights will be saved automatically in the `models/` directory.


