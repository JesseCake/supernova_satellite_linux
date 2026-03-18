# @title ## 🧪 Step 6 (Optional): Test Your Models { display-mode: "form" }
# @markdown Quick sanity check that your models load and run.

import numpy as np
import os

print("="*60)
print("🧪 STEP 6: MODEL TESTING")
print("="*60)

try:
    models_to_test = successful_models
except NameError:
    models_to_test = []

output_dir = "./my_custom_model"

if not models_to_test:
    print("\n⚠️ No models to test. Run Step 4 first.")
else:
    try:
        import openwakeword
        from openwakeword.model import Model

        for m in models_to_test:
            model_name = m['model_name']
            word = m['word']
            onnx_path = m.get('onnx_path', f"{output_dir}/{model_name}.onnx")

            print(f"\n📊 Testing: {word} ({model_name})")

            if os.path.exists(onnx_path):
                try:
                    model = Model(
                        wakeword_models=[onnx_path],
                        inference_framework='onnx'
                    )

                    # Test with silence (should not trigger)
                    test_audio = np.zeros(16000, dtype=np.int16)
                    prediction = model.predict(test_audio)

                    print(f"   ✅ Model loaded successfully")
                    print(f"   Prediction on silence: {prediction}")
                    print(f"   (Should be close to 0.0 - no wake word in silence)")

                except Exception as e:
                    print(f"   ❌ Error testing model: {e}")
            else:
                print(f"   ⚠️ Model file not found: {onnx_path}")

    except ImportError as e:
        print(f"\n⚠️ Could not import openwakeword: {e}")
        print("   Models were still created - you can test them in your own environment.")