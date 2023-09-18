import sounddevice as sd


def playback_audio(audio_array, sample_rate, sound_output_device=None, verbose=False):
    if verbose:
        print(
            f"Starting playback on device {sd.query_devices()[sound_output_device]['name']}, {sd.query_hostapis()[sd.query_devices()[sound_output_device]['hostapi']]['name']}"
        )
    sd.play(data=audio_array, samplerate=sample_rate, device=sound_output_device)
    sd.wait()  # Block until the current audio finished playing
