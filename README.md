## Description
A work-in-progress chatbot that is powered by:
- [Llama-2 Chat](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GPTQ)
- [Bark](https://github.com/JonathanFly/bark)
- [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

## Setup
Tested on Python 3.10.
For Python 3.11, there is an issue with the latest `fairseq` release. Please try installing a `fairseq` fork following this [issue comment](https://github.com/facebookresearch/fairseq/issues/5012#issuecomment-1675400618).

### Clone exllama and install dependencies
- Install [CUDA](https://developer.nvidia.com/cuda-11-8-0-download-archive)
- Install [MSVC 2022](https://visualstudio.microsoft.com/downloads/), just the `Build Tools for Visual Studio 2022` package is sufficient (make sure `Desktop
development with C++` is ticked in the installer).
- Clone [exllama](https://github.com/turboderp/exllama) to root directory:

```
git clone https://github.com/turboderp/exllama
```

### Create a virtual environment
```
python -m venv env
```
### Activate the virtual environment
```
env\scripts\activate
```
Install dependencies
```
pip install -r torch-requirements.txt
pip install -r requirements.txt
```
Run download_models.py to download models.
```
python download_models.py
```
Authorize by signing in with the owner's channel.
```
python youtube_api.py --authorize "secrets/owner_credentials.json"
```
Authorize by signing in with the chatbot's channel.
```
python youtube_api.py --authorize "secrets/bot_credentials.json"
```
Setup the audio playback device.
```
python query_sound_devices.py
```

## Run
Run the chatbot.
```
python main.py
```