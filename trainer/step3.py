# @title ## 📦 Step 3: Download Data & Setup Environment { display-mode: "form" }
# @markdown This downloads all required data and installs dependencies.
# @markdown
# @markdown **Time:** ~15-20 minutes (mostly downloading)
# @markdown
# @markdown **What gets downloaded:**
# @markdown - Pre-computed audio features (~17 GB) - for negative examples
# @markdown - Validation features (~176 MB) - for false positive testing
# @markdown - Room impulse responses - for reverb augmentation
# @markdown - Background audio (music/noise) - for augmentation
# @markdown
# @markdown ⚠️ **License Note:** Data has mixed licenses. Models trained here are for **non-commercial personal use only**.

import locale
def getpreferredencoding(do_setlocale=True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

import os
import sys
from pathlib import Path

print("="*60)
print("📦 STEP 3: ENVIRONMENT SETUP & DATA DOWNLOAD")
print("="*60)

# ============================================================
# 3.1 INSTALL DEPENDENCIES (must happen before scipy/numpy imports!)
# ============================================================
print("\n🔧 Installing dependencies...")

# Unload any cached numpy/scipy modules to avoid version conflicts
mods_to_remove = [k for k in sys.modules.keys() if k.startswith(('numpy', 'scipy'))]
for mod in mods_to_remove:
    del sys.modules[mod]

# Fix numpy/scipy compatibility FIRST
!pip install -q --force-reinstall 'numpy==1.26.4' 'scipy==1.13.1'

!git clone -q https://github.com/dscripka/openwakeword 2>/dev/null || echo "openwakeword already cloned"
!pip install -q -e ./openwakeword --no-deps

# Core dependencies
!pip install -q mutagen==1.47.0
!pip install -q torchinfo==1.8.0
!pip install -q torchmetrics==1.2.0
!pip install -q speechbrain==0.5.14
!pip install -q audiomentations==0.33.0
!pip install -q torch-audiomentations==0.11.0
!pip install -q acoustics==0.2.6
!pip install -q onnxruntime==1.22.1 ai_edge_litert==1.4.0 onnxsim
!pip install -q onnx onnx_graphsurgeon sng4onnx
!pip install -q onnx_tf tensorflow 2>/dev/null || true  # Prevents train.py crash
!pip install -q pronouncing==0.2.0
!pip install -q datasets==2.14.6
!pip install -q deep-phonemizer==0.0.19

print("✅ Dependencies installed")

# ============================================================
# NOW import scipy/numpy after installation (fresh import)
# ============================================================
import numpy as np
import scipy.io.wavfile
from tqdm.auto import tqdm
import datasets

print(f"   numpy version: {np.__version__}")
print(f"   scipy version: {scipy.__version__}")

# ============================================================
# 3.2 DOWNLOAD REQUIRED MODELS
# ============================================================
print("\n📥 Downloading openWakeWord model files...")

model_dir = "./openwakeword/openwakeword/resources/models"
os.makedirs(model_dir, exist_ok=True)

model_files = [
    ("embedding_model.onnx", "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx"),
    ("embedding_model.tflite", "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite"),
    ("melspectrogram.onnx", "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx"),
    ("melspectrogram.tflite", "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.tflite"),
]

for filename, url in model_files:
    filepath = os.path.join(model_dir, filename)
    if not os.path.exists(filepath):
        !wget -q -O {filepath} {url}
        print(f"   ✅ {filename}")
    else:
        print(f"   ⏭️ {filename} (already exists)")

# ============================================================
# 3.3 DOWNLOAD ROOM IMPULSE RESPONSES
# ============================================================
print("\n📥 Downloading room impulse responses...")

rir_dir = "./mit_rirs"
if not os.path.exists(rir_dir) or len(os.listdir(rir_dir)) < 100:
    os.makedirs(rir_dir, exist_ok=True)

    # Install git-lfs
    !git lfs install

    if not os.path.exists("MIT_environmental_impulse_responses"):
        !git clone -q https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses

    # Process the RIR files
    wav_files = list(Path("./MIT_environmental_impulse_responses/16khz").glob("*.wav"))
    if wav_files:
        rir_dataset = datasets.Dataset.from_dict({
            "audio": [str(i) for i in wav_files]
        }).cast_column("audio", datasets.Audio())

        for row in tqdm(rir_dataset, desc="Processing RIRs"):
            name = row['audio']['path'].split('/')[-1]
            scipy.io.wavfile.write(
                os.path.join(rir_dir, name),
                16000,
                (row['audio']['array'] * 32767).astype(np.int16)
            )
        print(f"   ✅ {len(os.listdir(rir_dir))} RIR files")
    else:
        print("   ⚠️ No RIR files found in cloned repo")
else:
    print(f"   ⏭️ RIRs already downloaded ({len(os.listdir(rir_dir))} files)")

# ============================================================
# 3.4 DOWNLOAD BACKGROUND AUDIO
# ============================================================
print("\n📥 Downloading background audio...")

# AudioSet - currently unavailable due to dataset restructuring
audioset_dir = "./audioset_16k"

if not os.path.exists(audioset_dir) or len([f for f in os.listdir(audioset_dir) if f.endswith('.wav')]) < 50:
    os.makedirs(audioset_dir, exist_ok=True)

    print("   ⏭️ Skipping AudioSet (dataset recently restructured)")
    print("   Using FMA + pre-computed features for background audio instead.")
else:
    count = len([f for f in os.listdir(audioset_dir) if f.endswith('.wav')])
    print(f"   ⏭️ AudioSet already downloaded ({count} files)")

# FMA (Free Music Archive)
fma_dir = "./fma"
if not os.path.exists(fma_dir) or len([f for f in os.listdir(fma_dir) if f.endswith('.wav')]) < 50:
    os.makedirs(fma_dir, exist_ok=True)
    print("   Loading FMA dataset (streaming)...")

    try:
        fma_dataset = datasets.load_dataset("rudraml/fma", name="small", split="train", streaming=True)
        fma_dataset = iter(fma_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000)))

        n_hours = 3  # 3 hours of music clips for better variety
        n_clips = n_hours * 3600 // 30  # FMA clips are 30 seconds each

        for i in tqdm(range(n_clips), desc="Processing FMA"):
            try:
                row = next(fma_dataset)
                name = row['audio']['path'].split('/')[-1].replace(".mp3", ".wav")
                scipy.io.wavfile.write(
                    os.path.join(fma_dir, name),
                    16000,
                    (row['audio']['array'] * 32767).astype(np.int16)
                )
            except StopIteration:
                break
            except Exception as e:
                continue  # Skip problematic files
        print(f"   ✅ {len([f for f in os.listdir(fma_dir) if f.endswith('.wav')])} FMA files")
    except Exception as e:
        print(f"   ⚠️ FMA download failed: {e}")
else:
    count = len([f for f in os.listdir(fma_dir) if f.endswith('.wav')])
    print(f"   ⏭️ FMA already downloaded ({count} files)")

# ============================================================
# 3.5 DOWNLOAD PRE-COMPUTED FEATURES
# ============================================================
print("\n📥 Downloading pre-computed features (this is the big download)...")

# Training features (~17GB)
features_file = "./openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
if not os.path.exists(features_file):
    print("   ⬇️ Downloading training features (~17 GB)...")
    print("   This may take 10-30 minutes depending on connection speed.")
    !wget -q --show-progress https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy
    print("   ✅ Training features downloaded")
else:
    size_gb = os.path.getsize(features_file) / 1024 / 1024 / 1024
    print(f"   ⏭️ Training features already downloaded ({size_gb:.1f} GB)")

# Validation features (~176MB)
val_file = "validation_set_features.npy"
if not os.path.exists(val_file):
    print("   ⬇️ Downloading validation features (~176 MB)...")
    !wget -q --show-progress https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy
    print("   ✅ Validation features downloaded")
else:
    size_mb = os.path.getsize(val_file) / 1024 / 1024
    print(f"   ⏭️ Validation features already downloaded ({size_mb:.0f} MB)")

# ============================================================
# VERIFICATION
# ============================================================
print("\n" + "="*60)
print("📊 VERIFICATION")
print("="*60)

def count_wav_files(directory):
    if os.path.isdir(directory):
        return len([f for f in os.listdir(directory) if f.endswith('.wav')])
    return 0

rir_count = count_wav_files(rir_dir)
audioset_count = count_wav_files(audioset_dir)
fma_count = count_wav_files(fma_dir)
total_bg = audioset_count + fma_count

checks = [
    ("RIRs", rir_count, 100),
    ("AudioSet", audioset_count, 0),  # Optional - 0 minimum
    ("FMA", fma_count, 50),
]

all_ok = True
for name, actual_count, min_count in checks:
    if min_count > 0:
        status = "✅" if actual_count >= min_count else "⚠️"
        print(f"   {status} {name}: {actual_count} files (need {min_count}+)")
        if actual_count < min_count:
            all_ok = False
    else:
        status = "✅" if actual_count > 0 else "⏭️"
        print(f"   {status} {name}: {actual_count} files (optional)")

# Check feature files
for name, path in [("Training features", features_file), ("Validation features", val_file)]:
    if os.path.exists(path):
        size = os.path.getsize(path)
        size_str = f"{size/1024/1024/1024:.1f} GB" if size > 1024*1024*1024 else f"{size/1024/1024:.0f} MB"
        print(f"   ✅ {name} ({size_str})")
    else:
        print(f"   ❌ {name} (missing)")
        all_ok = False

print(f"\n   Total background audio: {total_bg} files (need 50+)")

if total_bg < 50:
    all_ok = False

if all_ok:
    print("\n" + "="*60)
    print("✅ STEP 3 COMPLETE - All data downloaded!")
    print("="*60)
    print("\n👉 Proceed to Step 4 to train your model.")
else:
    print("\n" + "="*60)
    print("⚠️ Some downloads may have failed.")
    print("="*60)
    print("\nMinimum requirements:")
    print("   • 50+ total background audio files (AudioSet + FMA)")
    print("   • Training features file (17 GB)")
    print("   • Validation features file (176 MB)")
    print("\nRe-run this cell to retry, or proceed if you have enough background audio.")
