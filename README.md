# 🥋 TKD-AI-Coach

> AI-powered Taekwondo technique analysis using computer vision, biomechanical feature engineering, and deep learning.

## 🎯 Overview

TKD-AI-Coach is a computer-vision and deep-learning system designed to analyze Taekwondo kicking techniques from video and identify predefined technical errors.

The system converts video into human pose landmarks, extracts biomechanically meaningful features, segments the kick sequence, and uses temporal deep-learning models to classify the execution.

### Supported Kicks

- Front Kick
- Roundhouse Kick
- Side Kick

### Detected Technical Errors

- Dropped Guard
- Leaning Back
- No Chamber
- No Retraction
- Lazy Chamber
- No Pivot
- Foot Pointing Up
- Off Balance

## 🧠 System Architecture

```text
Video / Webcam
      ↓
MediaPipe Pose Landmarker
      ↓
1-Euro Noise Filtering
      ↓
Biomechanical Feature Extraction
      ↓
Kick Segmentation
      ↓
Sequence Cleaning & Normalization
      ↓
LSTM / GRU / ST-GCN
      ↓
Target-Specific Classification
      ↓
Technical Error Prediction
      ↓
Streamlit Interface
```

## 🔬 Key Features

- Real-time webcam inference
- Offline video analysis
- MediaPipe-based pose estimation
- Biomechanical feature engineering
- 1-Euro temporal filtering
- Automatic kick segmentation
- LSTM, GRU, and ST-GCN models
- Performance-level data splitting to reduce data leakage
- Training-time data augmentation
- Macro-F1 based evaluation
- Interactive Streamlit interface

## 📐 Biomechanical Features

The current system uses eight engineered features:

| Feature | Description |
|---|---|
| `base_knee_angle` | Knee angle of the supporting leg |
| `base_pivot_angle` | Pivot orientation of the supporting leg |
| `striking_chamber_angle` | Chamber position of the kicking leg |
| `striking_foot_angle` | Orientation of the striking foot |
| `spine_angle` | Trunk/spine posture |
| `striking_ankle_y` | Vertical position of the striking ankle |
| `base_ankle_y` | Vertical position of the supporting ankle |
| `guard_dropped` | Guard position indicator |

These features provide a compact biomechanical representation of the movement instead of training directly on raw video frames.

## 🤖 Models

The project evaluates multiple temporal deep-learning architectures:

### LSTM

Models temporal dependencies in the kick sequence using Long Short-Term Memory networks.

### GRU

Provides a lighter recurrent alternative to LSTM for temporal sequence modelling.

### ST-GCN

Uses a custom spatial-temporal graph representation to model relationships between groups of biomechanical features over time.

### Custom Kinematic Graph

```text
Node 0 → Base Knee + Base Pivot
Node 1 → Striking Chamber + Striking Foot
Node 2 → Spine + Guard
Node 3 → Striking Ankle Y + Base Ankle Y
```

## 🦵 Kick Segmentation

The system automatically identifies the active kick sequence using a state-based segmentation process.

```text
REST
  ↓
KICK DETECTION
  ↓
ACTIVE KICK
  ↓
PEAK / EXTENSION
  ↓
RETRACTION
  ↓
LANDING
  ↓
SEQUENCE COMPLETE
```

## 📊 Dataset

A custom Taekwondo dataset was created using multiple camera perspectives and different execution/error classes.

The dataset includes:

- Front view
- Side view
- 45-degree view

Multiple camera views of the same physical kick are treated as a single performance during dataset splitting to reduce data leakage.

The current dataset contains **142 source recordings** across the supported kicks and error categories.

The raw videos are converted into feature-based CSV sequences for model training.

## 🔐 Data Leakage Control

Data leakage is controlled at the **physical-performance level**.

For example:

```text
One Physical Kick
 ├── Front View
 ├── Side View
 └── 45° View
```

These recordings represent one physical performance rather than three independent samples.

Therefore, all views and augmented variants belonging to the same performance remain within the same dataset partition.

<img width="1175" height="435" alt="image" src="https://github.com/user-attachments/assets/39648f9b-3776-4a9d-bebb-161ca8b17e73" />


## 🔄 Data Augmentation

Training data can be augmented using:

- Angular noise
- Temporal warping
- Controlled perturbations

Augmentation is applied only to the training data.

## ⚙️ Training Pipeline

```text
Raw Videos
     ↓
Pose Extraction
     ↓
Biomechanical Feature Extraction
     ↓
1-Euro Filtering
     ↓
Kick Segmentation
     ↓
Performance-Level Data Split
     ↓
Training Augmentation
     ↓
Normalization
     ↓
LSTM / GRU / ST-GCN
     ↓
Validation
     ↓
Best Model Checkpoint
```

The training pipeline includes:

- Adam optimizer
- Weight decay
- Gradient clipping
- Early stopping
- Class-weighted loss
- Sequence normalization
- Macro-F1 based evaluation

## 🛠️ Tech Stack

- **Python**
- **PyTorch**
- **MediaPipe**
- **OpenCV**
- **NumPy**
- **Pandas**
- **Scikit-learn**
- **Streamlit**
- **Matplotlib**
- **Jupyter Notebook**

## 📁 Project Structure

```text
TKD-AI-Coach/
│
├── main.py
├── extract.py
├── requirements.txt
├── README.md
│
├── notebooks/
│   ├── preprocessing/
│   ├── analysis/
│   └── training/
│
├── models/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── augmented/
│
├── utils/
│
└── assets/
```

## 🚀 Installation

### Clone the Repository

```bash
git clone https://github.com/<YOUR-USERNAME>/TKD-AI-Coach.git
cd TKD-AI-Coach
```

### Create a Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run main.py
```

The application supports:

- Video-based analysis
- Live webcam inference
- Automatic kick detection
- Technical error classification
- Model prediction and visualization

## 📈 Evaluation

The system evaluates model performance using:

- Accuracy
- Precision
- Recall
- Macro-F1
- Per-class F1
- Confusion Matrix

Deployment-related considerations also include:

- Inference latency
- Model size
- Computational requirements
- Training/inference preprocessing consistency

<img width="715" height="403" alt="image" src="https://github.com/user-attachments/assets/59d2044b-e56c-444f-a960-f6d82b5e79dd" />


## ⚠️ Limitations

- The current dataset contains a limited number of independent practitioners.
- Some error classes require additional genuine human performances.
- The current system primarily evaluates a single practitioner.
- Camera viewpoint and recording conditions can affect pose estimation.
- The current implementation is a research prototype rather than a universal Taekwondo assessment system.
- Model confidence scores should not automatically be interpreted as calibrated probabilities.

## 🔮 Future Work

- Larger and more diverse athlete dataset
- Relational biomechanical features
- Improved kick segmentation
- TCN + Temporal Attention architecture
- Multi-view pose fusion
- Multi-person tracking
- Visual coaching feedback
- Voice-based coaching feedback
- Athlete progress tracking
- Edge/mobile deployment
- Confidence calibration
- Uncertainty estimation

## 🧪 Planned Next Architecture

A lightweight temporal architecture is planned as a future experiment:

```text
Video / Webcam
      ↓
MediaPipe Pose
      ↓
1-Euro Filter
      ↓
Core + Relational + Motion Features
      ↓
Kick Segmentation
      ↓
Normalization
      ↓
Lightweight TCN
      ↓
Temporal Attention
      ↓
Global Pooling
      ↓
Target-Specific Classifier
      ↓
Error + Confidence
      ↓
Coaching Feedback
```

The new architecture will be evaluated against the existing LSTM, GRU, and ST-GCN baselines using the same leakage-controlled evaluation protocol.

## 📌 Project Status

**Research Prototype — Active Development**

The current implementation provides an end-to-end pipeline from video input to automated Taekwondo technique classification and Streamlit-based inference.

## ⚠️ Disclaimer

TKD-AI-Coach is an experimental AI-based sports technique analysis system.

Its predictions are intended to assist athletes, coaches, and researchers and should not be treated as authoritative coaching, medical assessment, competition judging, or medical advice.

