from TTS.api import TTS
import simpleaudio as sa

tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
DEFAULT_SPEAKER = "voices/Ayumi_English_Voice_mixdown.wav"

def ayumi_speak(text, output_path="ayumi_response.wav", speaker_wav=DEFAULT_SPEAKER):
    print(f"A.Y.U.M.I: {text}")
    tts.tts_to_file(
        text=text,
        file_path=output_path,
        speaker_wav=speaker_wav,
        language="en"
    )
    wave_obj = sa.WaveObject.from_wave_file(output_path)
    play_obj = wave_obj.play()
    play_obj.wait_done()

if __name__ == "__main__":
    ayumi_speak("Hello I am Ayumi. Your virtual assistant ready to help you with anything")