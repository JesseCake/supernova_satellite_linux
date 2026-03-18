# @title ## ⚙️ Step 2: Training Configuration { display-mode: "form" }
# @markdown ### Wake Words
# @markdown Enter one or more wake words, separated by commas.
# @markdown Use the exact spelling that sounded best in Step 1!
# @markdown
# @markdown **Examples:** `hey_jar_viss` or `hey_jar_viss, oh_kay_computer`

wake_words = "hey_jar_viss" # @param {type:"string"}

# @markdown ---
# @markdown ### Quick Test Mode
# @markdown Enable for faster training (~30 min) with lower quality. Good for testing!
quick_test_mode = False # @param {type:"boolean"}

# @markdown ---
# @markdown ### Training Parameters
# @markdown These are ignored if Quick Test Mode is enabled.
# @markdown
# @markdown | Parameter | Low | Default | High | Effect |
# @markdown |-----------|-----|---------|------|--------|
# @markdown | Examples | 5,000 | 25,000 | 50,000 | More = better quality, longer training |
# @markdown | Steps | 5,000 | 25,000 | 50,000 | More = better convergence, longer training |
# @markdown | False Activation Penalty | 500 | 1,500 | 3,000 | Higher = fewer false triggers, may miss quiet speech |

_number_of_examples = 25000 # @param {type:"slider", min:1000, max:50000, step:1000}
_number_of_training_steps = 25000 # @param {type:"slider", min:1000, max:50000, step:1000}
_false_activation_penalty = 1500 # @param {type:"slider", min:100, max:5000, step:100}

# @markdown ---
# @markdown ### Advanced Options
# @markdown
# @markdown **target_false_positives_per_hour** - How often model incorrectly triggers
# @markdown - `0.1` = ~1 false trigger every 10 hours (very strict)
# @markdown - `0.5` = ~1 false trigger every 2 hours (stricter)
# @markdown - `1.0` = ~1 false trigger per hour (permissive)

target_false_positives_per_hour = 0.7 # @param {type:"number"}

# @markdown **target_recall** - Percentage of real wake words detected (at evaluation threshold)
# @markdown - `0.5` = 50% detected (conservative, fewer false positives)
# @markdown - `0.7` = 70% detected (balanced)
# @markdown - `0.9` = 90% detected (sensitive, more false positives)

target_recall = 0.7 # @param {type:"number"}

# @markdown **layer_size** - Neural network hidden layer size (affects model size and accuracy)
# @markdown - `32` = ~15 KB model, fastest inference, good for simple single words
# @markdown - `64` = ~30 KB model, fast, balanced
# @markdown - `96` = ~50 KB model, better accuracy for multi-word phrases
# @markdown - `128` = ~75 KB model, best accuracy, slower inference

layer_size = 96 # @param [32, 64, 96, 128] {type:"raw"}

# Hidden defaults (not exposed in UI)
target_accuracy = 0.7  # Not configurable - doesn't significantly affect training

# ============================================================
# APPLY SETTINGS
# ============================================================

# Parse wake words
wake_word_list = [w.strip() for w in wake_words.split(',') if w.strip()]

if not wake_word_list:
    raise ValueError("❌ No wake words specified! Enter at least one wake word above.")

# Apply quick test mode
if quick_test_mode:
    number_of_examples = 5000
    number_of_training_steps = 5000
    false_activation_penalty = 500
    print("⚡ QUICK TEST MODE ENABLED")
    print("   Using reduced settings for faster training (~30 min)")
    print("   Model quality will be lower - for testing only!")
else:
    number_of_examples = _number_of_examples
    number_of_training_steps = _number_of_training_steps
    false_activation_penalty = _false_activation_penalty

print(f"\n{'='*50}")
print(f"📋 TRAINING CONFIGURATION")
print(f"{'='*50}")
print(f"\n🎯 Wake words to train: {wake_word_list}")
print(f"\n📊 Training parameters:")
print(f"   • Examples per word: {number_of_examples:,}")
print(f"   • Training steps: {number_of_training_steps:,}")
print(f"   • False activation penalty: {false_activation_penalty}")
print(f"\n📈 Evaluation targets:")
print(f"   • Target FP/hour: {target_false_positives_per_hour}")
print(f"   • Target recall: {target_recall*100:.0f}%")
print(f"\n🧠 Model architecture:")
print(f"   • Layer size: {layer_size} neurons")
print(f"\n✅ Configuration saved. Proceed to Step 3.")