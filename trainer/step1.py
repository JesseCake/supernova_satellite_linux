# @title ## 🎧 Step 1: Test Wake Word Pronunciation { display-mode: "form" }
# @markdown **Test how your wake word will sound before training!**
# @markdown
# @markdown First run takes ~1-2 minutes to setup. Subsequent runs are fast.
# @markdown
# @markdown ### Pronunciation Tips
# @markdown - Use underscores for syllable breaks: `computer` → `khum_puter`
# @markdown - Spell phonetically: `jarvis` → `jar_viss`
# @markdown - Multi-word: `hey jarvis` → `hey_jar_viss`
# @markdown - Spell out numbers: `2` → `two`
# @markdown - Avoid punctuation except `?` and `!`

# Block the broken system pkg_resources entirely
import sys
import types

# Inject a fake pkg_resources module before anything tries to import the real one (problems with latest python)
fake_pkg_resources = types.ModuleType('pkg_resources')
fake_pkg_resources.__version__ = '999.0.0'
fake_pkg_resources.require = lambda *a, **kw: None
fake_pkg_resources.resource_filename = lambda *a, **kw: ''

# Add a fake Distribution class for get_distribution calls
class _FakeDist:
    def __init__(self, name):
        self.version = '0.0.0'
        self.project_name = name

fake_pkg_resources.get_distribution = lambda name: _FakeDist(name)
fake_pkg_resources.DistributionNotFound = Exception

sys.modules['pkg_resources'] = fake_pkg_resources


target_word = 'supernova' # @param {type:"string"}

import os
import sys
from IPython.display import Audio, display

# Setup TTS on first run
if not os.path.exists("./piper-sample-generator"):
    print("🔧 First run - setting up TTS engine (~1-2 minutes)...")
    !git clone https://github.com/rhasspy/piper-sample-generator
    !wget -q -O piper-sample-generator/models/en_US-libritts_r-medium.pt 'https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt'
    !cd piper-sample-generator && git checkout 213d4d5
    !pip install -q --upgrade setuptools # Added this line to fix pkgutil.ImpImporter error
    !pip install -q piper-tts piper-phonemize-cross
    !pip install -q webrtcvad
    !pip install -q --force-reinstall 'torch==2.4.0' 'torchaudio==2.4.0' torchvision
    print("✅ TTS setup complete!\n")

if "piper-sample-generator/" not in sys.path:
    sys.path.append("piper-sample-generator/")

# Check for torch/torchaudio compatibility
try:
    import torchaudio
    torchaudio.load  # Test that it works
except (OSError, ImportError) as e:
    if "undefined symbol" in str(e) or "libtorchaudio" in str(e):
        print("⚠️ Torch/torchaudio version mismatch detected!")
        print("   This happens if Step 3 was run before Step 1.")
        print("   Please: Runtime → Restart session, then run Step 1 first.")
        raise RuntimeError("Please restart runtime and run Step 1 before Step 3")

# ============================================================
# FIX: Patch torch.load ONLY ONCE to avoid recursion error
# ============================================================
import torch

# Check if we've already patched torch.load (prevents RecursionError on re-run)
if not getattr(torch.load, '_oww_patched', False):
    _original_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return _original_torch_load(*args, **kwargs)

    # Mark it as patched so we don't do it again
    _patched_torch_load._oww_patched = True
    torch.load = _patched_torch_load

from generate_samples import generate_samples

def preview_wake_word(text):
    """Generate and play a sample of the wake word."""
    print(f"🎤 Generating audio for: '{text}'")
    generate_samples(
        text=text,
        max_samples=1,
        length_scales=[1.1],
        noise_scales=[0.7],
        noise_scale_ws=[0.7],
        output_dir='./',
        batch_size=1,
        auto_reduce_batch_size=True,
        file_names=["test_generation.wav"]
    )
    return Audio("test_generation.wav", autoplay=True)

print("\n▶️ Listen to your wake word:")
display(preview_wake_word(target_word))
print("\n💡 If it doesn't sound right, change the spelling above and run again!")
print("   Once satisfied, proceed to Step 2.")