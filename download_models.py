import os
from config import XDG_CACHE_HOME

os.environ["HF_HOME"] = os.path.join(XDG_CACHE_HOME, "huggingface")
os.environ["HF_DATASETS_CACHE"] = os.path.join(XDG_CACHE_HOME, "huggingface", "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(XDG_CACHE_HOME, "huggingface", "models")
os.environ["XDG_CACHE_HOME"] = XDG_CACHE_HOME

from huggingface_hub import hf_hub_download, snapshot_download


if __name__ == "__main__":

    if not os.path.exists("rvc/models"):
        os.makedirs("rvc/models")
    hf_hub_download(repo_id="lj1995/VoiceConversionWebUI", revision="6b3b855d20267d8652eeb3af1fface82b9576d67", filename="rmvpe.pt", local_dir="rvc/models")
    hf_hub_download(repo_id="lj1995/VoiceConversionWebUI", revision="6b3b855d20267d8652eeb3af1fface82b9576d67", filename="hubert_base.pt", local_dir="rvc")
    hf_hub_download(repo_id="lj1995/VoiceConversionWebUI", revision="6b3b855d20267d8652eeb3af1fface82b9576d67", filename="ffmpeg.exe", local_dir="rvc")
    hf_hub_download(repo_id="lj1995/VoiceConversionWebUI", revision="6b3b855d20267d8652eeb3af1fface82b9576d67", filename="ffprobe.exe", local_dir="rvc")

    if not os.path.exists("models/TheBloke_Llama-2-7b-Chat-GPTQ_gptq-4bit-64g-actorder_True"):
        os.makedirs("models/TheBloke_Llama-2-7b-Chat-GPTQ_gptq-4bit-64g-actorder_True")
    snapshot_download(repo_id="TheBloke/Llama-2-7b-Chat-GPTQ", revision="e5813beb42988201e723ed05a9d3c2c6845607fc", local_dir="models/TheBloke_Llama-2-7b-Chat-GPTQ_gptq-4bit-64g-actorder_True")
    print("Download complete.")