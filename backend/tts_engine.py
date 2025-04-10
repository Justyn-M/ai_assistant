import simpleaudio as sa
import tempfile
import os
from TTS.api import TTS

tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
DEFAULT_SPEAKER = "voices/Ayumi_English_Voice_mixdown.wav"

def ayumi_speak(text, speaker_wav=DEFAULT_SPEAKER):
    print(f"A.Y.U.M.I: {text}")
    
    # Create temporary .wav file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_path = tmp.name

    # Generate fully processed audio and save to temp file
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language="en",
        file_path=tmp_path
    )

    # Play it
    wave_obj = sa.WaveObject.from_wave_file(tmp_path)
    play_obj = wave_obj.play()
    play_obj.wait_done()

    # Clean up
    os.remove(tmp_path)


if __name__ == "__main__":
    ayumi_speak("Hello I am Ayumi. Your virtual assistant ready to help you with anything")
