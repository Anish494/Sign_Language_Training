### Context Aware Nepali Sign Language Understanding System
### Tribhuvan University | Thapathali Campus | 2026

A real-time Nepali Sign Language (NSL) recognition system that combines MediaPipe landmark extraction, a Transformer-based gesture classifier, and CLIP scene understanding to recognize 52 NSL word signs and build continuous sentences from live webcam input.

---

## Project Status: In Progress

---

## What Has Been Built

### 1. Dataset Construction
- Multi-source dataset from three sources:
  - **National Federation of Deaf Nepal (NFDN)** — primary collection with real deaf community members
  - **Kaggle NSL Dataset** — 22 word-level signs, 25 frames per video
  - **Self-recorded iPhone videos** — additional 60 signs
- **52 NSL word signs**, approximately **14,000 processed samples**
- Pre-organized Train / Validation / Test splits

### 2. MediaPipe Landmark Extraction
- Uses **MediaPipe Tasks API**
- Extracts **291-dimensional feature vector** per frame:
  - Pose: 33 × 5 = 165 (x, y, z, visibility, presence)
  - Left hand: 21 × 3 = 63 (x, y, z)
  - Right hand: 21 × 3 = 63 (x, y, z)
- Feature order: `pose → left_hand → right_hand`
- Pose detection rate: **100%**
- At least one hand detection: **~90%**

### 3. Transformer Gesture Classifier
- Built **from scratch** in PyTorch — no pretrained backbone
- Input: variable-length sequences padded/truncated to **(100, 291)**
- Architecture:
  - Linear projection: 291 → 160 dimensions
  - Sinusoidal positional encoding
  - 4 × Encoder blocks (4 attention heads, FFN 320-dim)
  - Padding mask — ignores zero-padded frames
  - Masked mean pooling → classification head → 52 classes
  - **~804K parameters**
- Training configuration:
  - Optimizer: AdamW (lr = 0.0001, weight decay = 0.001)
  - Loss: Cross Entropy with label smoothing (0.1)
  - Max epochs: 180, early stopping patience: 15
  - On-the-fly augmentation: Gaussian noise, random scaling, temporal shift
  - Device: NVIDIA RTX 4060 Laptop GPU
- **Best Validation Accuracy: 77%**
- ~40× improvement over random baseline (1.9%)

### 4. Dynamic Sliding Window
- Replaces fixed-frame triggering with confidence-driven variable window
- Min frames: 20, Max frames: 80
- Confidence threshold relaxes linearly from 0.90 → 0.55 as window grows
- Word committed when:
  - Confidence exceeds dynamic threshold, or
  - Motion pause detected, or
  - Window reaches maximum limit
- Buffer clears after each committed word

### 5. Motion-Based Sign Segmentation
- Hand landmark velocity computed between consecutive frames
- Stillness for **0.4 seconds** → word boundary detected
- Stillness for **1.0 second** → sentence boundary, screen clears
- No manual input required between signs

### 6. Sentence Builder
- Accumulates unique consecutive recognized words
- Suppresses duplicate consecutive words
- Maximum 5 words displayed at once
- Completed sentences logged automatically to `detected_sentences.txt`
- Example output: `ma → ghar → janu`

### 7. CLIP Scene Understanding
- Model: **OpenAI CLIP ViT-B/32** — pretrained and frozen
- Zero-shot classification across **12 scene categories**:
  - Hospital, Classroom, Home, Office, Kitchen, Shop,
    Library, Pharmacy, Waiting Room, Restaurant,
    Outdoor, Community Center
- Text embeddings computed **once at startup**
- Runs every **30 frames** during inference
- Live test result: **Library detected at 94% confidence**

### 8. Real-Time Inference Pipeline
- Three models running simultaneously on GPU
- On-screen display:
  - **Top left** — predicted sign, confidence score, top-3 predictions
  - **Top right** — detected scene and CLIP confidence
  - **Bottom** — running sentence of unique recognized words
  - **Skeleton overlay** — hand and pose landmarks drawn on frame
- Controls: `SPACE` to reset buffer and sentence, `Q` to quit

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 77% |
| Random Baseline | 1.9% |
| Improvement over Random | ~40× |
| Best Sign F1 | Nurse — 1.00 |
| Weakest Sign F1 | Boat — 0.39 |
| Pose Detection Rate | 100% |
| Hand Detection Rate | ~90% |
| CLIP Scene Confidence | 94% (library) |
| Model Parameters | ~804K |

---

## Project Structure

```
NSL_Project/
├── model2.py                 ← Transformer architecture with masking
├── train3.py                 ← Training pipeline with augmentation
├── predict2.py               ← Real-time inference (MediaPipe + Transformer + CLIP)
├── evaluate2.ipynb           ← Evaluation metrics and figures
├── explore.ipynb             ← Dataset exploration
├── class_mapping.json        ← Sign name to index mapping
├── best_model2.pth           ← Trained model weights
├── hand_landmarker.task      ← MediaPipe hand landmark model
├── pose_landmarker.task      ← MediaPipe pose landmark model
├── ViT-B-32.pt               ← CLIP ViT-B/32 model weights
└── detected_sentences.txt    ← Auto-logged recognized sentences
```

---

## Setup and Installation

### Requirements
```
Python        3.11.9
torch         2.5.1+cu121
mediapipe     0.10.35
opencv        4.9.0.80
numpy         1.26.4
clip          OpenAI
scikit-learn
Pillow
```

### Installation
```bash
# Create and activate virtual environment
python -m venv myenv
myenv\Scripts\activate.bat        # Windows
source myenv/bin/activate         # Mac/Linux

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other packages
pip install mediapipe==0.10.35
pip install opencv-contrib-python==4.9.0.80
pip install numpy==1.26.4
pip install scikit-learn Pillow ftfy regex

# Install CLIP
pip install git+https://github.com/openai/CLIP.git

# Install Jupyter (optional)
pip install jupyter ipykernel
```

### Download Model Files
```bash
# MediaPipe hand landmark model
wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

# MediaPipe pose landmark model
wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
```

### Run Real-Time Inference
```bash
python predict2.py
```

### Train the Model
```bash
python train3.py
```

---

## Remaining Work

| Module | Status |
|--------|--------|
| MediaPipe Landmark Extraction | Completed |
| Transformer Gesture Classifier | Completed |
| Dynamic Sliding Window | Completed |
| Motion-Based Segmentation | Completed |
| Sentence Builder | Completed |
| CLIP Scene Understanding | Completed |
| Confidence-Based Fusion Module | In Progress |
| Intent Reasoning Module | Remaining |
| Context Disambiguation | Remaining |
| Response Generation (English + Nepali) | Remaining |
| Text-to-Speech (gTTS) | Remaining |
| End-to-End Integration and Evaluation | Remaining |

---

## Known Issues

- Training used legacy `mp.solutions.holistic` API while inference uses new MediaPipe Tasks API — subtle feature distribution mismatch exists and re-extraction is planned
- Hand detection dropout in fast or occluded frames introduces zero-padding noise into input sequences
- Signer-independent evaluation not yet conducted
- Dataset from Er. Dinesh Bhusal (Pulchowk Campus) awaited — expected to improve accuracy further

---

## Team

| Name | Roll Number |
|------|-------------|
| Anish Kumar Singh | THA079BCT004 |
| Jeewan Bhatt | THA079BCT015 |
| Ruby Kumari Sah | THA079BCT034 |

**Supervisor:** Associate Professor Er. Shanta Maharjan  
**Department:** Electronics and Computer Engineering  
**Institution:** Thapathali Campus, Tribhuvan University, Nepal  
**Date:** August 2026  

---

## References

- Boháček, M. and Hrúz, M. (2023). *Interpreting Sign Language Recognition using Transformers and MediaPipe Landmarks.* ACM ICMI.
- Radford, A. et al. (2021). *Learning Transferable Visual Models from Natural Language Supervision (CLIP).* OpenAI.
- Bhusal, D. and Baral, D.S. (2024). *Sentence-Level Nepali Sign Language Recognition.* Pulchowk Campus, Tribhuvan University.
- Srivastava, S. et al. (2024). *Continuous Sign Language Recognition System using Deep Learning with MediaPipe Holistic.* arXiv:2411.04517.
- Kothadiya, A. et al. (2023). *SIGNFORMER: DeepVision Transformer for Sign Language Recognition.* IEEE Access, 11, 4730–4739.