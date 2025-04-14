import os
import tempfile
from playsound import playsound
from TTS.api import TTS

tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
DEFAULT_SPEAKER = "voices/Ayumi_English_Voice_mixdown.wav"

def ayumi_speak(text, speaker_wav=DEFAULT_SPEAKER):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp_path = tmp.name

        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language="en",
            file_path=tmp_path
        )

        print("[DEBUG] Playing audio with playsound...")
        playsound(tmp_path)
        print("[DEBUG] Playback complete.")

        os.remove(tmp_path)

    except Exception as e:
        print(f"[DEBUG] TTS playback error: {e}")


if __name__ == "__main__":
    ayumi_speak("Hello I am Ayumi. Your virtual assistant ready to help you with anything")
