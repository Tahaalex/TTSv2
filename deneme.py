from TTS.api import TTS
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

torch.serialization.add_safe_globals([
    XttsConfig,
    XttsAudioConfig,
    BaseDatasetConfig,
    XttsArgs
])

tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

# Speaker listesine erişim
print(tts.synthesizer.tts_model.speaker_manager.speakers.keys())

# Dil listesine erişim
print(tts.languages)