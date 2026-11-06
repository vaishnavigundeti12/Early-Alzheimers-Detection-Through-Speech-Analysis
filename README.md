# 🧠 MediSquad — Early Alzheimer's Detection Through Speech Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?style=flat-square)
![ML](https://img.shields.io/badge/ML-XGBoost%20|%20LightGBM%20|%20RF-green?style=flat-square)
![License](https://img.shields.io/badge/License-Research%20Only-yellow?style=flat-square)

**AI-powered cognitive impairment detection using speech & language biomarkers**

</div>

---

## 📋 Project Overview

MediSquad analyzes speech transcripts from three clinical tasks to classify patients into:
- ✅ **HC** — Healthy Control (normal cognition)
- ⚠️ **MCI** — Mild Cognitive Impairment
- 🔴 **Dementia** — Alzheimer's Disease

### 🗂️ Dataset
- **157 participants** with audio recordings and transcripts
- **DementiaBank-style corpus** (Cookie Theft Picture Description)
- Three test paradigms: CTD, PFT, SFT
- Labels: HC, MCI, Dementia + MMSE scores

---

## 🚀 Quick Start

### Option 1: Automated Setup (Windows)
```bat
# Double-click setup.bat
# Then double-click run_app.bat
```

### Option 2: Manual Setup
```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 5. Train models (runs on actual dataset)
python train_models.py

# 6. Launch the app
streamlit run app.py
```

---

## 🤖 Machine Learning Models

| Model | Description |
|-------|-------------|
| Random Forest | 200 trees, balanced class weights |
| XGBoost | Gradient boosted trees, 200 estimators |
| LightGBM | Fast gradient boosting, 200 estimators |
| SVM (RBF) | Support Vector Machine with RBF kernel |
| Gradient Boosting | Sklearn GBM, 150 estimators |
| Logistic Regression | Multinomial, balanced |
| **Ensemble** | **Voting (RF + XGB + LGB)** |

---

## 🔬 Speech Features Extracted (40+)

### Cookie Theft Description (CTD)
- Word count, unique words, Type-Token Ratio
- Filler word density (um, uh, er)
- Pause count and total pause duration
- Repetition rate, content word ratio
- Sentence count, average sentence length

### Semantic Fluency Test (SFT — Animal Naming)
- Total animals named, unique animals
- Repetition rate in animal list

### Phonemic Fluency Test (PFT — P-words)
- Total P-words, unique P-words
- Fluency rate (TTR)

### Pre-computed Linguistic Features
- Brunét's Index, Moving-Average TTR
- Content density, total recording seconds
- MMSE score, age, gender

---

## 📁 Project Structure

```
Medisquad/
│
├── app.py                         # 🌐 Main Streamlit application
├── train_models.py                # 🤖 Model training script
├── requirements.txt               # 📦 Python dependencies
├── setup.bat                      # ⚙️ Automated setup script (Windows)
├── run_app.bat                    # ▶️ Run app script (Windows)
├── README.md                      # 📖 This file
│
├── Data_AUG_13.11.2024_output.csv # 📊 Main augmented dataset
├── df_train.csv                   # 🏋️ Training transcripts
├── df_test.csv                    # 🧪 Test transcripts
├── Training_No.csv                # 📋 Additional training data
│
├── Archive/                       # 🎵 Audio recordings (WAV files)
│   ├── Process-rec-001/
│   │   ├── Process-rec-001__CTD.wav
│   │   ├── Process-rec-001__PFT.wav
│   │   └── Process-rec-001__SFT.wav
│   └── ... (157 patient records)
│
├── models/                        # 💾 Trained model artifacts (auto-created)
│   ├── Random_Forest.pkl
│   ├── XGBoost.pkl
│   ├── LightGBM.pkl
│   ├── SVM.pkl
│   ├── scaler.pkl
│   ├── imputer.pkl
│   ├── label_encoder.pkl
│   ├── feature_names.pkl
│   ├── results_summary.pkl
│   └── feature_importances.pkl
│
└── venv/                          # 🐍 Virtual environment
```

---

## 🖥️ Application Pages

1. **🏠 Home & Predict** — Enter transcripts or upload audio → get classification
2. **📊 Model Performance** — Compare all models with metrics & confusion matrices
3. **📈 Dataset Explorer** — Interactive charts on the DementiaBank corpus
4. **🔬 Feature Analysis** — Feature importance & speech biomarker descriptions
5. **ℹ️ About** — How the system works & clinical background

---

## ⚕️ Medical Disclaimer

> This tool is a **research prototype** and is **NOT** a clinical diagnostic device.
> It should not be used to make medical decisions. Always consult qualified
> healthcare professionals (neurologist, geriatrician) for proper cognitive assessment.

---

## 📚 References

- DementiaBank: Pitt Corpus (https://dementia.talkbank.org/)
- Luz et al., ADReSS Challenge 2020
- Fraser et al., "Linguistic features identify Alzheimer's disease in narrative speech" (2016)
- Balagopalan et al., "The effect of heterogeneous data for Alzheimer's disease detection" (2021)
