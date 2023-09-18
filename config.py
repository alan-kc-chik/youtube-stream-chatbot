POLLING_TIME = 1.0

# Special triggers
TRIGGER_TOKEN = "@Halcy"
CLEAR_HISTORY_TRIGGER = TRIGGER_TOKEN + " Doesn't look like anything to me."


# Text generation parameters
TEXT_GENERATION_MODEL_PATH = (
    "models/TheBloke_Llama-2-7b-Chat-GPTQ_gptq-4bit-64g-actorder_True"
)
TEXT_GENERATION_PROMPT = """[INST] <<SYS>>
You are a young streamer on YouTube. You are currently streaming and reading messages from the viewers.
<</SYS>>
Write a single response within 50 words to the following question from a live stream viewer:
{prompt} [/INST]
"""

# Bark TTS parameters
BARK_SPEAKER_PROMPT = "bark_infinity/assets/prompts/en_british.npz"


# RVC parameters
RVC_CHECKPOINT_PATH = "rvc/weights/my_voice.pth"
RVC_PARAMS = {
    "f0_up_key": -2,  # pitch shift in semitones
}

# Audio playback parameters
AUDIO_PLAYBACK_DEVICE = 10  # Run python query_sound_devices.py to list the available sound devices

# YouTube Authorization
# To authorize and create credentials, run python youtube_api.py --authorize SAVE_CREDENTIALS_HERE.json
STREAM_OWNER_CREDENTIALS_FILE = "secrets/owner_credentials.json"
CHATBOT_CREDENTIALS_FILE = "secrets/bot_credentials.json"


# Model cache location
XDG_CACHE_HOME = ".cache"
