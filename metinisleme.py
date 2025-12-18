from TTS.api import TTS
from pydub import AudioSegment
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


tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True, gpu=False)


with open("kitap.txt", "r", encoding="utf-8") as f:
    text = f.read()


tts.tts_to_file(text=text, file_path="kitap_sesli.wav",speaker="Asya Anara", language="tr")


sound = AudioSegment.from_wav("kitap_sesli.wav")
sound.export("kitap_sesli.mp3", format="mp3", bitrate="192k")

print("✅ Sesli kitap hazır: kitap_sesli.mp3")

#pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121     nvidia ekran kartı için yapay zeka kütüphanesi cuda ile işler
