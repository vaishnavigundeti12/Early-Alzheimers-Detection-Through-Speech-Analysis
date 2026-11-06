"""
Quick verification script - tests all imports and basic functionality
Run: venv\Scripts\python verify_setup.py
"""
import sys
import os
import pandas as pd
from pathlib import Path

print("=" * 60)
print("  MediSquad Setup Verification")
print("=" * 60)

errors = []

# Test imports
imports = [
    ('streamlit', 'streamlit'),
    ('numpy', 'numpy'),
    ('pandas', 'pandas'),
    ('sklearn', 'scikit-learn'),
    ('xgboost', 'xgboost'),
    ('lightgbm', 'lightgbm'),
    ('joblib', 'joblib'),
    ('nltk', 'nltk'),
    ('matplotlib', 'matplotlib'),
    ('seaborn', 'seaborn'),
    ('plotly', 'plotly'),
    ('librosa', 'librosa'),
    ('soundfile', 'soundfile'),
]

print("\n✔ Checking imports:")
for mod, pkg in imports:
    try:
        __import__(mod)
        print(f"  ✅ {pkg}")
    except ImportError as e:
        print(f"  ❌ {pkg}: {e}")
        errors.append(pkg)

# Test data files (Updated to look in Dataset folder)
print("\n✔ Checking data files:")
files = [
    os.path.join('Dataset', 'Data_AUG_13.11.2024_output.csv'),
    'train_models.py',
    'app.py',
    'requirements.txt',
]
for f in files:
    if os.path.exists(f):
        size = os.path.getsize(f) / 1024
        print(f"  ✅ {f} ({size:.1f} KB)")
    else:
        print(f"  ❌ {f} - NOT FOUND")
        errors.append(f)

# Test dataset loading (Updated path)
print("\n✔ Testing dataset load:")
try:
    df = pd.read_csv(os.path.join('Dataset', 'Data_AUG_13.11.2024_output.csv'), low_memory=False)
    print(f"  ✅ Dataset: {len(df)} records, {len(df.columns)} columns")
    if 'Class' in df.columns:
        dist = df['Class'].value_counts()
        for cls, cnt in dist.items():
            print(f"     → {cls}: {cnt}")
except Exception as e:
    print(f"  ❌ Dataset load failed: {e}")
    errors.append("dataset")

# Check folders
print("\n✔ Checking raw data folders:")
train_folder = Path('Dataset') / 'Archive'
test_folder = Path('Dataset') / 'test_Set' / 'test_Set'

if train_folder.exists():
    recs = list(train_folder.glob('Process-rec-*'))
    print(f"  ✅ Archive folder found ({len(recs)} records)")
else:
    print(f"  ❌ Archive folder NOT FOUND at {train_folder}")
    errors.append("Archive folder")

if test_folder.exists():
    recs = list(test_folder.glob('Process-test-*'))
    print(f"  ✅ test_Set folder found ({len(recs)} records)")
else:
    print(f"  ❌ test_Set folder NOT FOUND at {test_folder}")
    errors.append("test_Set folder")

# Check models
print("\n✔ Checking trained models:")
models_dir = Path("models")
if models_dir.exists():
    model_files = list(models_dir.glob("*.pkl"))
    print(f"  ✅ {len(model_files)} model files found")
    for f in model_files:
        print(f"     → {f.name} ({f.stat().st_size/1024:.1f} KB)")
else:
    print("  ⚠️ Models not trained yet. Run: python train_models.py")

print("\n" + "=" * 60)
if errors:
    print(f"❌ {len(errors)} issue(s) found: {errors}")
    print("   Fix and rerun.")
else:
    print("✅ All checks passed! Run: streamlit run app.py")
print("=" * 60)