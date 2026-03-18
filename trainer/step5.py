# @title ## ⬇️ Step 5: Download Your Models { display-mode: "form" }
# @markdown Downloads all generated model files via browser.
# @markdown
# @markdown **Note:** If you enabled Google Drive in Step 4, your models are already saved there!
# @markdown This step is mainly for backup or if you disabled Google Drive.

import os
import shutil
from datetime import datetime

print("="*60)
print("⬇️ STEP 5: DOWNLOAD MODELS")
print("="*60)

# Check for variables from Step 4
try:
    _gdrive_enabled = gdrive_enabled
    _gdrive_path = gdrive_path
    _drive_folder = drive_folder_name
except NameError:
    _gdrive_enabled = False
    _gdrive_path = None
    _drive_folder = "OpenWakeWord_Models"

try:
    models_to_download = successful_models
except NameError:
    models_to_download = []

if not models_to_download:
    print("\n⚠️ No models to download. Run Step 4 first.")
else:
    # Show Google Drive status
    if _gdrive_enabled and _gdrive_path:
        print(f"\n☁️  Google Drive Status: CONNECTED")
        print(f"   Your models are already saved to:")
        print(f"   Google Drive/{_drive_folder}/")
        print(f"\n   The download below is a backup copy.")

    print(f"\n📦 Generated Models:\n")
    print(f"{'Model':<35} {'Size':<15} {'Drive Status'}")
    print(f"{'-'*65}")

    download_files = []
    output_dir = "./my_custom_model"

    for m in models_to_download:
        model_name = m['model_name']
        onnx_path = m.get('onnx_path', f"{output_dir}/{model_name}.onnx")

        # Check local file
        if os.path.exists(onnx_path):
            size_kb = os.path.getsize(onnx_path) / 1024

            # Check if it's in Drive
            drive_status = "—"
            if _gdrive_enabled and _gdrive_path:
                drive_file = os.path.join(_gdrive_path, f"{model_name}.onnx")
                if os.path.exists(drive_file):
                    drive_status = "✅ Saved"
                else:
                    drive_status = "❌ Missing"

            print(f"{model_name}.onnx{' '*(30-len(model_name))} {size_kb:.1f} KB{' '*(10-len(f'{size_kb:.1f}'))} {drive_status}")
            download_files.append(onnx_path)
        else:
            print(f"{model_name}.onnx{' '*(30-len(model_name))} ❌ not found")

    # Offer to save missing files to Drive
    if _gdrive_enabled and _gdrive_path:
        missing_from_drive = []
        for m in models_to_download:
            model_name = m['model_name']
            onnx_path = m.get('onnx_path', f"{output_dir}/{model_name}.onnx")
            drive_file = os.path.join(_gdrive_path, f"{model_name}.onnx")
            if os.path.exists(onnx_path) and not os.path.exists(drive_file):
                missing_from_drive.append((onnx_path, model_name))

        if missing_from_drive:
            print(f"\n📤 Copying {len(missing_from_drive)} missing model(s) to Google Drive...")
            for onnx_path, model_name in missing_from_drive:
                try:
                    dest_path = os.path.join(_gdrive_path, f"{model_name}.onnx")
                    shutil.copy2(onnx_path, dest_path)
                    print(f"   ✅ {model_name}.onnx")
                except Exception as e:
                    print(f"   ❌ {model_name}.onnx - {e}")

    # Create zip archive and trigger download
    if download_files:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        zip_name = f'openwakeword_models_{timestamp}'

        # Copy files to temp directory for zipping
        os.makedirs(zip_name, exist_ok=True)
        for f in download_files:
            shutil.copy(f, zip_name)

        shutil.make_archive(zip_name, 'zip', zip_name)
        zip_path = f'{zip_name}.zip'

        print(f"\n📁 Created archive: {zip_path}")
        print(f"   Size: {os.path.getsize(zip_path) / 1024:.1f} KB")

        # Auto-download in Colab
        try:
            from google.colab import files
            print("\n⬇️ Starting download...")
            print("   (If download doesn't start, check your browser's download folder)")

            # Download zip first
            files.download(zip_path)

            # Also offer individual files
            print("\n📥 Individual file downloads:")
            for f in download_files:
                try:
                    files.download(f)
                    print(f"   ✅ {os.path.basename(f)}")
                except:
                    print(f"   ⚠️ {os.path.basename(f)} - download may have failed")

        except ImportError:
            print(f"\n📥 Download manually from the file browser on the left.")

        # Cleanup temp dir
        shutil.rmtree(zip_name, ignore_errors=True)

        if _gdrive_enabled:
            print(f"\n💡 Remember: Your models are also in Google Drive!")
            print(f"   Google Drive/{_drive_folder}/")
    else:
        print("\n⚠️ No model files found to download.")