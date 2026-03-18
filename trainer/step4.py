# @title ## 🚀 Step 4: Train Models { display-mode: "form" }
# @markdown This trains a model for each wake word you specified.
# @markdown
# @markdown **Time:** ~30-90 minutes per model (depends on settings and hardware)
# @markdown
# @markdown **Training phases:**
# @markdown 1. Generate synthetic speech clips
# @markdown 2. Augment clips with noise/reverb
# @markdown 3. Train neural network
# @markdown 4. Export to ONNX format
# @markdown
# @markdown ---
# @markdown ### 📁 Google Drive Settings (Recommended!)
# @markdown Colab's browser download can be unreliable. Google Drive ensures your models are saved safely.

enable_google_drive = True # @param {type:"boolean"}
# @markdown ↑ Enable to save models directly to Google Drive as soon as they finish.

drive_folder_name = "OpenWakeWord_Models" # @param {type:"string"}
# @markdown ↑ Folder name in your Google Drive (created automatically if it doesn't exist).

import yaml
import sys
import os
import re
import shutil

print("="*60)
print("🚀 STEP 4: MODEL TRAINING")
print("="*60)

# ============================================================
# GOOGLE DRIVE SETUP (if enabled)
# ============================================================
gdrive_enabled = False
gdrive_path = None

if enable_google_drive:
    print("\n☁️  Setting up Google Drive...")
    try:
        from google.colab import drive

        # Check if already mounted
        if not os.path.ismount('/content/drive'):
            print("   (You may be prompted to authorize access)\n")
            drive.mount('/content/drive')
        else:
            print("   Drive already mounted.")

        # Create the output folder
        drive_base = '/content/drive/MyDrive'
        drive_output_path = os.path.join(drive_base, drive_folder_name)

        if not os.path.exists(drive_output_path):
            os.makedirs(drive_output_path)
            print(f"   📂 Created folder: Google Drive/{drive_folder_name}/")
        else:
            print(f"   📂 Using folder: Google Drive/{drive_folder_name}/")

        # Verify write access
        test_file = os.path.join(drive_output_path, '.test_write')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)

        gdrive_enabled = True
        gdrive_path = drive_output_path
        print(f"   ✅ Google Drive connected! Models will be saved there.")

    except ImportError:
        print("   ⚠️ Google Drive only available in Colab. Using local storage.")
    except Exception as e:
        print(f"   ⚠️ Drive setup failed: {e}")
        print("   Models will be downloaded via browser instead.")
else:
    print("\n💾 Google Drive: DISABLED")
    print("   Models will be downloaded via browser after training.")
    print("   ⚠️ Note: Browser downloads can be unreliable in Colab.")

def sanitize_name(name):
    """Convert wake word to valid filename."""
    return re.sub(r'[^a-zA-Z0-9]+', '_', name).strip('_')

def save_to_drive(onnx_path, model_name):
    """Copy model to Google Drive. Returns True if successful."""
    if not gdrive_enabled or not gdrive_path:
        return False

    try:
        dest_path = os.path.join(gdrive_path, f"{model_name}.onnx")
        shutil.copy2(onnx_path, dest_path)

        # Verify the copy
        if os.path.exists(dest_path):
            size_kb = os.path.getsize(dest_path) / 1024
            print(f"\n☁️  SAVED TO GOOGLE DRIVE: {model_name}.onnx ({size_kb:.1f} KB)")
            print(f"   Location: Google Drive/{drive_folder_name}/{model_name}.onnx")
            return True
        else:
            print(f"\n⚠️  Drive copy verification failed for {model_name}")
            return False
    except Exception as e:
        print(f"\n⚠️  Failed to save to Drive: {e}")
        return False

def queue_download(onnx_path, model_name):
    """Queue a model file for browser download (Colab only). Non-blocking."""
    try:
        from google.colab import files
        import threading

        def trigger_download():
            try:
                files.download(onnx_path)
            except:
                pass  # Ignore errors in background thread

        print(f"\n⬇️  Queued {model_name}.onnx for download")
        thread = threading.Thread(target=trigger_download)
        thread.daemon = True
        thread.start()

        import time
        time.sleep(1)
        return True
    except ImportError:
        print(f"\n📁 Not running in Colab - find your model at: {onnx_path}")
        return False
    except Exception as e:
        print(f"\n⚠️  Auto-download skipped: {e}")
        return False

# ============================================================
# LOAD CONFIG AND START TRAINING
# ============================================================
print("\n" + "="*60)
print("🎯 STARTING TRAINING")
print("="*60)

# Patch pronouncing to fix pkg_resources error in subprocesses
import re

# Patch pronouncing and webrtcvad to fix pkg_resources error in subprocesses
pronouncing_init = "/usr/local/lib/python3.12/dist-packages/pronouncing/__init__.py"
with open(pronouncing_init, 'r') as f:
    content = f.read()

if 'from pkg_resources import resource_stream' in content:
    content = content.replace(
        'from pkg_resources import resource_stream',
        'from importlib.resources import open_binary as resource_stream'
    )
    with open(pronouncing_init, 'w') as f:
        f.write(content)
    print("✅ Patched pronouncing/__init__.py")
else:
    print("⏭️ pronouncing already patched")

# Patch webrtcvad
webrtcvad_file = "/usr/local/lib/python3.12/dist-packages/webrtcvad.py"
with open(webrtcvad_file, 'r') as f:
    content = f.read()

if 'pkg_resources' in content:
    content = content.replace(
        "__version__ = pkg_resources.get_distribution('webrtcvad').version",
        "__version__ = '2.0.10'"  # hardcode version, it's only used for display
    ).replace(
        'import pkg_resources',
        ''
    )
    with open(webrtcvad_file, 'w') as f:
        f.write(content)
    print("✅ Patched webrtcvad.py")
else:
    print("⏭️ webrtcvad already patched")

base_config = yaml.load(
    open("openwakeword/examples/custom_model.yml", 'r').read(),
    yaml.Loader
)

output_dir = "./my_custom_model"
os.makedirs(output_dir, exist_ok=True)

successful_models = []
failed_models = []
models_saved_to_drive = []
models_pending_download = []

for i, word in enumerate(wake_word_list):
    model_name = sanitize_name(word)

    print(f"\n{'='*60}")
    print(f"🎯 TRAINING MODEL {i+1}/{len(wake_word_list)}: '{word}'")
    print(f"   Model name: {model_name}")
    print(f"{'='*60}")

    # Create config for this word
    config = base_config.copy()
    config["target_phrase"] = [word]
    config["model_name"] = model_name
    config["n_samples"] = number_of_examples
    config["n_samples_val"] = max(500, number_of_examples // 10)
    config["steps"] = number_of_training_steps
    config["target_accuracy"] = target_accuracy
    config["target_recall"] = target_recall
    config["target_false_positives_per_hour"] = target_false_positives_per_hour
    config["output_dir"] = output_dir
    config["max_negative_weight"] = false_activation_penalty
    config["layer_size"] = layer_size
    config["background_paths"] = ['./audioset_16k', './fma']
    config["false_positive_validation_data_path"] = "validation_set_features.npy"
    config["feature_data_files"] = {"ACAV100M_sample": "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"}

    config_file = f'{model_name}_config.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config, f)

    try:
        # Phase 1: Generate clips
        print(f"\n📝 Phase 1/3: Generating {number_of_examples:,} synthetic speech clips...")
        !{sys.executable} openwakeword/openwakeword/train.py --training_config {config_file} --generate_clips

        # Phase 2: Augment clips
        print(f"\n🔊 Phase 2/3: Augmenting clips with noise and reverb...")
        !{sys.executable} openwakeword/openwakeword/train.py --training_config {config_file} --augment_clips

        # Phase 3: Train model
        print(f"\n🧠 Phase 3/3: Training neural network ({number_of_training_steps:,} steps)...")
        !{sys.executable} openwakeword/openwakeword/train.py --training_config {config_file} --train_model

        # Check if ONNX was created
        onnx_path = f"{output_dir}/{model_name}.onnx"
        if os.path.exists(onnx_path):
            size_kb = os.path.getsize(onnx_path) / 1024
            print(f"\n✅ SUCCESS: {onnx_path} ({size_kb:.1f} KB)")

            model_info = {
                'word': word,
                'model_name': model_name,
                'onnx_path': onnx_path
            }
            successful_models.append(model_info)

            # Try to save to Google Drive first
            if gdrive_enabled and save_to_drive(onnx_path, model_name):
                models_saved_to_drive.append(model_info)
            else:
                # Fall back to queuing download (if not using Drive)
                models_pending_download.append(model_info)
        else:
            print(f"\n❌ ONNX model not found at {onnx_path}")
            failed_models.append({'word': word, 'error': 'ONNX not created'})

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        failed_models.append({'word': word, 'error': str(e)})

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*60}")
print(f"📊 TRAINING SUMMARY")
print(f"{'='*60}")
print(f"\n✅ Successful: {len(successful_models)}")
for m in successful_models:
    print(f"   • {m['word']} → {m['onnx_path']}")

if failed_models:
    print(f"\n❌ Failed: {len(failed_models)}")
    for m in failed_models:
        print(f"   • {m['word']}: {m['error']}")

# Report on saving method
if models_saved_to_drive:
    print(f"\n☁️  SAVED TO GOOGLE DRIVE: {len(models_saved_to_drive)} model(s)")
    print(f"   Location: Google Drive/{drive_folder_name}/")
    for m in models_saved_to_drive:
        print(f"   • {m['model_name']}.onnx")
    print(f"\n✨ Your models are safely stored in Google Drive!")
    print(f"   You can access them anytime, even after this session ends.")

if models_pending_download:
    print(f"\n⬇️  PENDING DOWNLOAD: {len(models_pending_download)} model(s)")
    print(f"   Run Step 5 to download these models.")

if not successful_models:
    print(f"\n⚠️ No models were trained successfully. Check the errors above.")