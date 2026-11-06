import streamlit as st
import os
import re
import shutil
import tempfile
import subprocess
import joblib
import numpy as np
import librosa
import whisper
import nltk
from nltk.tokenize import word_tokenize
from pathlib import Path
import warnings

st.set_page_config(page_title="Free Speech", page_icon="🎙️", layout="wide")

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
        return str(dst.resolve())
    except: return "ffmpeg"
FFMPEG_PATH = setup_ffmpeg()

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
for pkg in ['punkt', 'stopwords', 'punkt_tab']:
    nltk.download(pkg, quiet=True)

MODELS_DIR = Path("models")

st.markdown("""
<style>
    .stApp { background-color: #0a0e1a; color: #f1f5f9; }
    .main-title { 
        font-size: 2.8rem; 
        font-weight: 800; 
        background: linear-gradient(135deg, #06b6d4, #3b82f6); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        text-align: center; 
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    .metric-card { background-color: #1a2235; padding: 20px; border-radius: 10px; border: 1px solid #1e3a5f; text-align: center; margin-top: 20px; }
    .result-healthy { border: 2px solid #10b981; background-color: rgba(16,185,129,0.1); }
    .result-mci { border: 2px solid #f59e0b; background-color: rgba(245,158,11,0.1); }
    .result-dementia { border: 2px solid #ef4444; background-color: rgba(239,68,68,0.1); }
    
    /* Override Streamlit metric labels to be teal */
    [data-testid="stMetricLabel"] { color: #06b6d4 !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_best_model():
    try:
        scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        imputer = joblib.load(MODELS_DIR / "imputer.pkl")
        le = joblib.load(MODELS_DIR / "label_encoder.pkl")
        features = joblib.load(MODELS_DIR / "feature_names.pkl")
        model = joblib.load(MODELS_DIR / "best_model.pkl")
        return scaler, imputer, le, features, model
    except: return None, None, None, None, None

@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

def process_audio(audio_path):
    try:
        y, sr = librosa.load(str(audio_path), sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)
        features = {'audio_duration': duration}
        rms = librosa.feature.rms(y=y)[0]
        features['energy_mean'] = float(np.mean(rms)); features['energy_std'] = float(np.std(rms))
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_clean = f0[~np.isnan(f0)]
        features['pitch_mean'] = float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0
        features['pitch_std'] = float(np.std(f0_clean)) if len(f0_clean) > 0 else 0
        non_silent = librosa.effects.split(y, top_db=30)
        pause_durations = []
        if len(non_silent) > 1:
            for i in range(1, len(non_silent)):
                pause_len = (non_silent[i][0] - non_silent[i-1][1]) / sr
                if pause_len > 0.5: pause_durations.append(pause_len)
        features['avg_pause_duration'] = float(np.mean(pause_durations)) if pause_durations else 0
        features['pause_ratio'] = float(sum(pause_durations) / duration) if pause_durations else 0
        
        whisper_model = load_whisper()
        result = whisper_model.transcribe(str(audio_path), language='en')
        text = result['text']
        clean = re.sub(r'\([^)]*\)', ' ', text).lower()
        words = word_tokenize(clean)
        words = [w for w in words if w.isalpha()]
        unique = set(words)
        ttr = len(unique) / len(words) if words else 0
        features.update({
            'ctd_word_count': len(words), 'ctd_ttr': ttr, 
            'ctd_filler_density': sum(clean.count(f) for f in ['um', 'uh', 'er', 'hmm']) / len(words) if words else 0,
            'speech_rate': len(words) / duration if duration > 0 else 0,
            'sft_unique_animals': 0, 'pft_word_count': 0,
            'unique_words': len(unique), 'ttr_num': ttr
        })
        features['age_num'] = 65.0; features['gender_num'] = 0.0; features['mmse_num'] = 28.0
        return text, features, duration
    except Exception as e:
        raise e

if "free_speech_key" not in st.session_state:
    st.session_state.free_speech_key = 0

def reset_app():
    st.session_state.free_speech_key += 1

def main():
    st.markdown('<div class="main-title">🎙️ Free Speech Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Speak freely about your day, a recent memory, or anything on your mind for up to 2 minutes.</div>', unsafe_allow_html=True)
    
    scaler, imputer, le, feature_names, model = load_best_model()
    if model is None:
        st.error("⚠️ Model not found. Run `python train_models.py`.")
        return

    audio_bytes = st.audio_input("Click the microphone to start recording", key=f"native_rec_free_{st.session_state.free_speech_key}")
    st.button("🔄 Reset / Record Again", on_click=reset_app, use_container_width=True)

    if audio_bytes:
        st.success("✅ Audio captured successfully!")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes.getvalue())
            raw_audio = tmp.name

        wav_path = raw_audio + "_processed.wav"
        subprocess.run([FFMPEG_PATH, "-y", "-i", raw_audio, "-t", "120", "-ar", "16000", "-ac", "1", wav_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        with st.spinner("🔍 Transcribing and analyzing..."):
            try:
                transcript, features, duration = process_audio(wav_path)
                
                st.info(f"**Transcript:** {transcript}")
                
                # --- EXTRACT FEATURES & PREVENT FALSE POSITIVES ---
                speech_rate = features.get('speech_rate', 0)
                ttr = features.get('ttr_num', 0)
                pause_dur = features.get('avg_pause_duration', 0)
                word_count = features.get('ctd_word_count', 0)
                
                if duration < 8 or word_count < 10:
                    st.warning("⚠️ Audio was too short or no speech was detected. Please record a longer description (at least 10 seconds) to get an accurate result.")
                    os.unlink(raw_audio)
                    if os.path.exists(wav_path): os.unlink(wav_path)
                    return
                # -----------------------------------------------
                
                row = [features.get(f, 0.0) for f in feature_names]
                X = np.array(row, dtype=float).reshape(1, -1)
                X = imputer.transform(X)
                X = scaler.transform(X)
                pred = model.predict(X)[0]
                proba = model.predict_proba(X)[0]
                ai_label = le.classes_[pred]
                
                # --- RELAXED CLINICAL RULES ---
                if word_count > 40 and speech_rate > 1.2 and pause_dur < 2.0:
                    final_label = "HC"
                    reason = "Clinical Override: Fluent speech and adequate word count indicate healthy cognition."
                    color = "#10b981"
                    css = "result-healthy"
                elif word_count < 15 or (speech_rate < 0.5 and pause_dur > 3.0):
                    final_label = "Dementia"
                    reason = "Clinical Override: Severe speech fragmentation, minimal word count, or extreme pauses indicate advanced decline."
                    color = "#ef4444"
                    css = "result-dementia"
                else:
                    final_label = ai_label
                    reason = "AI Prediction: The model detected subtle patterns consistent with this category."
                    color = "#f59e0b" if "MCI" in str(final_label) else "#ef4444"
                    css = "result-mci" if "MCI" in str(final_label) else "result-dementia"
                
                if final_label == "HC":
                    display_label = "Healthy Control (HC)"
                elif final_label == "MCI":
                    display_label = "Mild Cognitive Impairment (MCI)"
                elif final_label == "Dementia":
                    display_label = "Dementia (Alzheimer's Disease)"
                else:
                    display_label = final_label
                
                st.markdown(f"""
                <div class="metric-card {css}">
                    <h1 style="color: {color}; margin:0;">{display_label}</h1>
                    <p style="color: #e2e8f0; font-size:0.95rem; margin-top:10px;"><i>{reason}</i></p>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("### Clinical Markers Extracted (Standard Ranges)")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Speech Rate", f"{speech_rate:.2f} w/s", "Normal > 1.8")
                c2.metric("Lexical Diversity", f"{ttr:.2f}", "Normal > 0.40")
                c3.metric("Avg Pause", f"{pause_dur:.1f} sec", "Normal < 1.5s")
                c4.metric("Word Count", f"{word_count}", "Normal > 40")
                
                os.unlink(raw_audio)
                if os.path.exists(wav_path): os.unlink(wav_path)
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()