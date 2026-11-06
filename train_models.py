import os
import re
import shutil
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score
import nltk
from nltk.tokenize import word_tokenize
import warnings

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# ─── 1. FFMPEG SETUP (Kept here so it doesn't break) ───
def setup_ffmpeg():
    try:
        import imageio_ffmpeg
        src = Path(imageio_ffmpeg.get_ffmpeg_exe())
        bin_dir = Path("ffmpeg_bin")
        bin_dir.mkdir(exist_ok=True)
        dst = bin_dir / "ffmpeg.exe"
        if not dst.exists():
            shutil.copy(str(src), str(dst))
        os.environ["PATH"] = str(bin_dir.resolve()) + os.pathsep + os.environ["PATH"]
    except: pass
setup_ffmpeg()

for pkg in ['punkt', 'stopwords', 'punkt_tab']:
    nltk.download(pkg, quiet=True)

# ─── PATHS ───
DATA_DIR = Path("Dataset").resolve()
MASTER_CSV = DATA_DIR / "Data_AUG_13.11.2024_output.csv"
TRAIN_FOLDER = DATA_DIR / "Archive" 

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def clean_numeric(val, default):
    if pd.isna(val): return default
    try: return float(str(val).replace('*', '').strip())
    except: return default

def extract_audio_features(wav_path):
    features = {'audio_duration': 0.0, 'speech_rate': 0.0, 'avg_pause_duration': 0.0, 'pause_ratio': 0.0, 'pitch_mean': 0.0, 'pitch_std': 0.0, 'energy_mean': 0.0, 'energy_std': 0.0}
    if wav_path is None or not wav_path.exists(): return features
    try:
        y, sr = librosa.load(str(wav_path.resolve()), sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)
        features['audio_duration'] = duration
        if duration < 1.0: return features
        rms = librosa.feature.rms(y=y)[0]
        features['energy_mean'] = float(np.mean(rms))
        features['energy_std'] = float(np.std(rms))
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_clean = f0[~np.isnan(f0)]
        if len(f0_clean) > 0:
            features['pitch_mean'] = float(np.mean(f0_clean))
            features['pitch_std'] = float(np.std(f0_clean))
        non_silent = librosa.effects.split(y, top_db=30)
        pause_durations = []
        if len(non_silent) > 1:
            for i in range(1, len(non_silent)):
                pause_len = (non_silent[i][0] - non_silent[i-1][1]) / sr
                if pause_len > 0.5: pause_durations.append(pause_len)
        if pause_durations:
            features['avg_pause_duration'] = float(np.mean(pause_durations))
            features['pause_ratio'] = float(sum(pause_durations) / duration)
        return features
    except: return features

def extract_text_features(ctd, sft, pft, duration):
    clean = re.sub(r'\([^)]*\)', ' ', ctd or "").lower()
    try: words = word_tokenize(clean)
    except: words = clean.split()
    words = [w for w in words if w.isalpha()]
    unique = set(words)
    ttr = len(unique) / len(words) if words else 0
    fillers = sum(clean.count(f) for f in ['um', 'uh', 'er', 'hmm'])
    return {
        'ctd_word_count': len(words), 'ctd_ttr': ttr, 'ctd_filler_density': fillers / len(words) if words else 0,
        'speech_rate': len(words) / duration if duration > 0 else 0,
        'sft_unique_animals': len(set([w.strip().lower() for w in (sft or "").split(',') if w.strip()])), 
        'pft_word_count': len([w for w in (pft or "").split(',') if w.strip()]), 
        'unique_words': len(unique), 'ttr_num': ttr
    }

def find_file(folder, suffix, task):
    if not folder.exists(): return None
    for f in folder.iterdir():
        if f.suffix.lower() == suffix and task in f.name.upper():
            return f
    return None

def main():
    print("="*70)
    print("  MEDISQUAD TRAINING - FINAL 80/20 SPLIT")
    print("="*70)
    
    if not MASTER_CSV.exists():
        print(f"❌ Master CSV missing at {MASTER_CSV}")
        return
        
    master_df = pd.read_csv(MASTER_CSV, low_memory=False)
    master_df['Record-ID'] = master_df['Record-ID'].astype(str).str.strip()
    
    records = []
    print(f"\n📂 Processing Archive folders for {len(master_df)} records...")
    for _, row in master_df.iterrows():
        rid = row['Record-ID']
        folder = TRAIN_FOLDER / rid
        if not folder.exists(): continue
        
        ctd_txt = find_file(folder, '.txt', 'CTD')
        sft_txt = find_file(folder, '.txt', 'SFT')
        pft_txt = find_file(folder, '.txt', 'PFT')
        ctd_wav = find_file(folder, '.wav', 'CTD')
        
        ctd = ctd_txt.read_text(encoding='utf-8') if ctd_txt else ""
        sft = sft_txt.read_text(encoding='utf-8') if sft_txt else ""
        pft = pft_txt.read_text(encoding='utf-8') if pft_txt else ""
            
        audio_feats = extract_audio_features(ctd_wav)
        text_feats = extract_text_features(ctd, sft, pft, audio_feats['audio_duration'])
        
        record = {
            'Record-ID': rid, 'Class': row['Class'],
            'age_num': clean_numeric(row.get('Age'), 65.0),
            'gender_num': 1.0 if str(row.get('Gender')).lower() in ['male', 'm', '1'] else 0.0,
            'mmse_num': clean_numeric(row.get('Converted-MMSE'), 25.0)
        }
        record.update(text_feats)
        record.update(audio_feats)
        records.append(record)
        print(f"  [{rid}] Processed.")
        
    df = pd.DataFrame(records)
    print(f"\n✅ Loaded {len(df)} labeled records.")
    
    # 80/20 Stratified Split
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Class'])
    print(f"✅ Train: {len(df_train)} | Test: {len(df_test)}")

    exclude_cols = ['Record-ID', 'Class']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X_train, X_test = df_train[feature_cols], df_test[feature_cols]
    le = LabelEncoder()
    y_train, y_test = le.fit_transform(df_train['Class']), le.transform(df_test['Class'])
    
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    
    print("\n🤖 Training models...")
    svm = CalibratedClassifierCV(SVC(), ensemble=False)
    models = {
        'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient_Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': svm,
        'Ensemble': VotingClassifier(estimators=[('rf', RandomForestClassifier(n_estimators=100, random_state=42)), ('gb', GradientBoostingClassifier(random_state=42)), ('svm', CalibratedClassifierCV(SVC(), ensemble=False))], voting='soft')
    }
    
    best_model, best_f1, best_name = None, 0, ""
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
        acc = accuracy_score(y_test, preds)
        print(f"  {name:20s} | Accuracy: {acc:.3f} | F1: {f1:.3f}")
        if f1 >= best_f1:
            best_f1, best_model, best_name = f1, model, name

    print(f"\n🏆 Best Model: {best_name} (F1: {best_f1:.3f})")
    
    for f in MODELS_DIR.glob("*.pkl"):
        try: f.unlink()
        except: pass
        
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(imputer, MODELS_DIR / "imputer.pkl")
    joblib.dump(le, MODELS_DIR / "label_encoder.pkl")
    joblib.dump(feature_cols, MODELS_DIR / "feature_names.pkl")
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    joblib.dump(best_name, MODELS_DIR / "best_model_name.pkl")
    
    print("✅ Done! Run `streamlit run app.py`")

if __name__ == "__main__":
    main()