import sys

sys.path.append("exllama")

# If stuck, delete:
# %localappdata%\torch_extensions\torch_extensions\Cache\py3[x]_cu[y]\exllama_ext\
import torch
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.tokenizer import ExLlamaTokenizer
from exllama.generator import ExLlamaGenerator
import os, glob
import argparse


def generate_text(prompt, model_directory, verbose=0):
    torch.set_grad_enabled(False)
    # torch.cuda._lazy_init()

    # Locate files we need within that directory
    tokenizer_path = os.path.join(model_directory, "tokenizer.model")
    model_config_path = os.path.join(model_directory, "config.json")
    st_pattern = os.path.join(model_directory, "*.safetensors")
    model_path = glob.glob(st_pattern)[0]

    # Create config, model, tokenizer and generator

    config = ExLlamaConfig(model_config_path)  # create config from config.json
    config.model_path = model_path  # supply path to model weights file

    model = ExLlama(config)  # create ExLlama instance and load the weights
    tokenizer = ExLlamaTokenizer(
        tokenizer_path
    )  # create tokenizer from tokenizer model file

    cache = ExLlamaCache(model)  # create cache for inference
    generator = ExLlamaGenerator(model, tokenizer, cache)  # create generator

    # Configure generator
    generator.settings.token_repetition_penalty_max = 1.18
    generator.settings.temperature = 1.2
    generator.settings.top_p = 0.37
    generator.settings.top_k = 100
    generator.settings.typical = 0.5

    # Produce a simple generation
    if verbose > 0:
        print("-" * 15)
        print("Prompt:")
        print(prompt)
    output = generator.generate_simple(prompt, max_new_tokens=128)
    response = output[len(prompt) :]

    if verbose > 0:
        print("Response:")
        print(response)
        print("-" * 15)

    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--prompt", type=str, help="Prompt")
    parser.add_argument("-d", "--model_dir", type=str, help="Path to model directory")

    args = parser.parse_args()

    result = generate_text(prompt=args.prompt, model_directory=args.model_dir, verbose=0)
    print(result)
