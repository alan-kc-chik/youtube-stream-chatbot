import contextlib
import gc
import os
import re

import random
from encodec import EncodecModel
import funcy
import numpy as np
from scipy.special import softmax
import torch

import math


import torch.distributions as torch_distributions

import torch.nn.functional as F
import tqdm
from transformers import BertTokenizer
from huggingface_hub import hf_hub_download

from .model import GPTConfig, GPT
from .model_fine import FineGPT, FineGPTConfig

import traceback
import sys
import time

import math

from rich.pretty import pprint

from .config import logger, load_all_defaults

from huggingface_hub import hf_hub_url
from collections import Counter

from devtools import debug
from collections import defaultdict


def _cast_bool_env_var(s):
    return s.lower() in ("true", "1", "t")


def get_SUNO_USE_DIRECTML():
    if _cast_bool_env_var(os.environ.get("SUNO_USE_DIRECTML", "False")):
        return True

    kwargs = {}
    defaults = load_all_defaults(*kwargs)
    if defaults["SUNO_USE_DIRECTML"] is True:
        return True
    else:
        return False


SUNO_USE_DIRECTML = get_SUNO_USE_DIRECTML()

dml = None
if SUNO_USE_DIRECTML is True:
    print(f"   --->> Experimental AMD DirectML support enabled.")
    import torch_directml

    torch.cuda.is_available = lambda: False

    dml = torch_directml.device()


if (
    torch.cuda.is_available()
    and hasattr(torch.cuda, "amp")
    and hasattr(torch.cuda.amp, "autocast")
    and hasattr(torch.cuda, "is_bf16_supported")
    and torch.cuda.is_bf16_supported()
):
    # print(f"   --->> Experimental NVIDIA BF16 support enabled.")
    autocast = funcy.partial(torch.cuda.amp.autocast, dtype=torch.bfloat16)
else:

    @contextlib.contextmanager
    def autocast():
        yield


# hold models in global scope to lazy load
global models
models = {}

global models_devices
models_devices = {}


CONTEXT_WINDOW_SIZE = 1024

SEMANTIC_RATE_HZ = 49.9
SEMANTIC_VOCAB_SIZE = 10_000

CODEBOOK_SIZE = 1024
N_COARSE_CODEBOOKS = 2
N_FINE_CODEBOOKS = 8
COARSE_RATE_HZ = 75

SAMPLE_RATE = 24_000


SUPPORTED_LANGS = [
    ("English", "en"),
    ("German", "de"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("Hindi", "hi"),
    ("Italian", "it"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Polish", "pl"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
    ("Turkish", "tr"),
    ("Chinese", "zh"),
]

ALLOWED_PROMPTS = {"announcer"}
for _, lang in SUPPORTED_LANGS:
    for prefix in ("", f"v2{os.path.sep}"):
        for n in range(10):
            ALLOWED_PROMPTS.add(f"{prefix}{lang}_speaker_{n}")


SUPPORTED_LANGS = [
    ("English", "en"),
    ("German", "de"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("Hindi", "hi"),
    ("Italian", "it"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Polish", "pl"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
    ("Turkish", "tr"),
    ("Chinese", "zh"),
]

ALLOWED_PROMPTS = {"announcer"}
for _, lang in SUPPORTED_LANGS:
    for prefix in ("", f"v2{os.path.sep}"):
        for n in range(10):
            ALLOWED_PROMPTS.add(f"{prefix}{lang}_speaker_{n}")


CUR_PATH = os.path.dirname(os.path.abspath(__file__))


default_cache_dir = os.path.join(os.path.expanduser("~"), ".cache")
CACHE_DIR = os.path.join(os.getenv("XDG_CACHE_HOME", default_cache_dir), "suno", "bark_v0")


USE_SMALL_MODELS = _cast_bool_env_var(os.environ.get("SUNO_USE_SMALL_MODELS", "False"))
GLOBAL_ENABLE_MPS = _cast_bool_env_var(os.environ.get("SUNO_ENABLE_MPS", "False"))
OFFLOAD_CPU = _cast_bool_env_var(os.environ.get("SUNO_OFFLOAD_CPU", "False"))

# Slower, possibly lower quality, but more memory efficient
SUNO_HALF_PRECISION = _cast_bool_env_var(os.environ.get("SUNO_HALF_PRECISION", "False"))

# Slower, possibly lower quality, but more memory efficient
SUNO_HALF_BFLOAT16 = _cast_bool_env_var(os.environ.get("SUNO_HALF_BFLOAT16", "False"))

SUNO_DISABLE_COMPILE = _cast_bool_env_var(os.environ.get("SUNO_DISABLE_COMPILE", "False"))

if sys.platform == "win32":
    SUNO_DISABLE_COMPILE = True


if SUNO_USE_DIRECTML is True:
    OFFLOAD_CPU = False

OFFLOAD_CPU = False

REMOTE_MODEL_PATHS = {
    "text_small": {
        "repo_id": "suno/bark",
        "file_name": "text.pt",
    },
    "coarse_small": {
        "repo_id": "suno/bark",
        "file_name": "coarse.pt",
    },
    "fine_small": {
        "repo_id": "suno/bark",
        "file_name": "fine.pt",
    },
    "text": {
        "repo_id": "suno/bark",
        "file_name": "text_2.pt",
    },
    "coarse": {
        "repo_id": "suno/bark",
        "file_name": "coarse_2.pt",
    },
    "fine": {
        "repo_id": "suno/bark",
        "file_name": "fine_2.pt",
    },
}

if not hasattr(torch.nn.functional, "scaled_dot_product_attention") and torch.cuda.is_available():
    logger.warning(
        "torch version does not support flash attention. You will get faster"
        + " inference speed by upgrade torch to newest nightly version."
    )


def _grab_best_device(use_gpu=True):
    if torch.cuda.device_count() > 0 and use_gpu:
        device = "cuda"
    elif torch.backends.mps.is_available() and use_gpu and GLOBAL_ENABLE_MPS:
        device = "mps"
    else:
        device = "cpu"

    return device


def _get_ckpt_path(model_type, use_small=False):
    key = model_type
    if use_small or USE_SMALL_MODELS:
        key += "_small"
    return os.path.join(CACHE_DIR, REMOTE_MODEL_PATHS[key]["file_name"])


def _download(from_hf_path, file_name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    hf_hub_download(repo_id=from_hf_path, filename=file_name, local_dir=CACHE_DIR)


class InferenceContext:
    def __init__(self, benchmark=False):
        # we can't expect inputs to be the same length, so disable benchmarking by default
        self._chosen_cudnn_benchmark = benchmark
        self._cudnn_benchmark = None

    def __enter__(self):
        self._cudnn_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = self._chosen_cudnn_benchmark

    def __exit__(self, exc_type, exc_value, exc_traceback):
        torch.backends.cudnn.benchmark = self._cudnn_benchmark


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


@contextlib.contextmanager
def _inference_mode():
    if SUNO_USE_DIRECTML is True:
        with InferenceContext(), torch.inference_mode(mode=False), torch.no_grad(), autocast():
            yield
    else:
        with InferenceContext(), torch.inference_mode(), torch.no_grad(), autocast():
            yield


def _clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def clean_models(model_key=None):
    global models
    model_keys = [model_key] if model_key is not None else list(models.keys())
    for k in model_keys:
        if k in models:
            del models[k]

    _clear_cuda_cache()
    gc.collect()


def _load_codec_model(device):
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    model.eval()

    print_loading_info("codec", "EncodecModelPath", device)

    if SUNO_USE_DIRECTML is True:
        model.to(dml)
    else:
        model.to(device)

    if callable(getattr(torch, "compile")) and not SUNO_DISABLE_COMPILE:
        logger.info("torch.compile available, compiling codec model.")
        model = torch.compile(model)
    else:
        logger.info(
            "torch.compile *not* available, you will get better performance if you use pytorch >= 2.0."
        )

    _clear_cuda_cache()
    return model


def load_codec_model(use_gpu=True, force_reload=False):
    global models
    global models_devices
    device = _grab_best_device(use_gpu=use_gpu)
    if device == "mps":
        # encodec doesn't support mps
        device = "cpu"
    model_key = "codec"
    if OFFLOAD_CPU:
        models_devices[model_key] = device
        device = "cpu"
    if model_key not in models or force_reload:
        clean_models(model_key=model_key)

        model = _load_codec_model(device)
        models[model_key] = model

    if SUNO_USE_DIRECTML is True:
        models[model_key].to(dml)
    else:
        models[model_key].to(device)

    return models[model_key]


####
# Generation Functionality
####


def _tokenize(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def _detokenize(tokenizer, enc_text):
    return tokenizer.decode(enc_text)


def _normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


TEXT_ENCODING_OFFSET = 10_048
SEMANTIC_PAD_TOKEN = 10_000
TEXT_PAD_TOKEN = 129_595
SEMANTIC_INFER_TOKEN = 129_599


def _load_history_prompt(history_prompt_input):
    if isinstance(history_prompt_input, str) and history_prompt_input.endswith(".npz"):
        history_prompt = np.load(history_prompt_input)
    elif isinstance(history_prompt_input, str):
        # make sure this works on non-ubuntu
        history_prompt_input = os.path.join(*history_prompt_input.split("/"))
        if history_prompt_input not in ALLOWED_PROMPTS:
            raise ValueError("history prompt not found")
        history_prompt = np.load(
            os.path.join(CUR_PATH, "assets", "prompts", f"{history_prompt_input}.npz")
        )
    elif isinstance(history_prompt_input, dict):
        assert "semantic_prompt" in history_prompt_input
        assert "coarse_prompt" in history_prompt_input
        assert "fine_prompt" in history_prompt_input
        history_prompt = history_prompt_input
    else:
        raise ValueError("history prompt format unrecognized")
    return history_prompt


def compute_log_probs(token_list, smoothing_factor, scaling_factor):
    # Count the frequency of each token.
    token_freq = Counter(token_list)

    # Add a smoothing factor.
    smoothed_token_freq = {token: freq + smoothing_factor for token, freq in token_freq.items()}

    # Normalize to create a probability distribution.
    total_tokens = len(token_list) + smoothing_factor * len(smoothed_token_freq)
    token_probs = {token: freq / total_tokens for token, freq in smoothed_token_freq.items()}

    # Transform into scaled log-probabilities.
    log_probs = {token: scaling_factor * np.log(prob) for token, prob in token_probs.items()}

    return log_probs


def estimate_s_this_seems_wrong_so_many_math_crashes(prob):
    epsilon = 1e-10
    num = 0
    den = 0
    for i in range(
        min(len(prob), 10000)
    ):  # apparently any number is fine here but they paper was on natural language so maybe not for us?
        # for i in range(768):
        b = prob[i] / (prob[i + 1] + epsilon)
        t = (i + 2) / (i + 1)
        if b > 0 and t > 0:
            num += math.log(b) * math.log(t)
            den += math.log(t) ** 2
    return num / den if den != 0 else 0


def estimate_s(prob):
    epsilon = 1e-10
    num = 0
    den = 0
    # for i in range(3000):
    # in the paper they say 100 is as good as any higher number? But it's not slow so maybe leave it higher?
    # also in the paper they don't have catch divide by 0s though...
    # also the paper was on natural language so maybe not for us. Let's just max it out
    for i in range(min(len(prob), 10000)):
        b = prob[i] / (prob[i + 1] + epsilon)
        t = (i + 2) / (i + 1)
        if b > 0 and t > 0:
            num += math.log(b if b > 0 else 1) * math.log(t if t > 0 else 1)
            # den += math.log(t)**2
            den += math.log(t if t > 0 else 1) ** 2
            # ok NOW this should never be zero and feels more right
            return num / den
    # return num / den if den != 0 else 0 # or should this be float("inf") ? doesn't seem right.


def compute_k_original_paper(n, s, tau):
    print(f"n: {n}, s: {s}, tau: {tau}")
    eps = s - 1
    k = ((eps * (2 ** (tau))) / (1 - n ** (-eps))) ** (1 / s)
    k = round(k)
    return k


def compute_k(n, s, tau, max_k):
    try:
        eps = s - 1
        n_eps = n ** (-eps)
        if s <= 0:
            return 0
        tau_s = tau ** (1 / s)
        k = (eps * 2 * tau_s / (1 - n_eps)) ** (1 / s)
        if isinstance(k, complex):
            return 0
        k = round(k)
        if k > max_k:
            return max_k
        return k
    except OverflowError:
        # Return maximum possible k
        return max_k


def compute_k_orig(n, s, tau):
    print(f"n: {n}, s: {s}, tau: {tau}")
    eps = s - 1
    k = ((eps * (2 ** (tau))) / (1 - n ** (-eps))) ** (1 / s)
    k = round(k)
    return k


def compute_k_not_right(n, s, tau, max_k):
    print(f"n: {n}, s: {s}, tau: {tau}")
    try:
        eps = s - 1
        n_eps = n ** (-eps)
        if s <= 0:
            return max_k
        tau_s = tau ** (1 / s)
        k = (eps * 2 * tau_s / (1 - n_eps)) ** (1 / s)
        k = round(k)
        return k
    except OverflowError:
        # Return maximum possible k
        return max_k


def compute_k_log(n, s, tau):
    print(f"n: {n}, s: {s}, tau: {tau}")
    eps = s - 1
    try:
        log_k = (math.log(eps) + tau * math.log(2) - math.log(1 - n ** (-eps))) / s
        k = round(math.exp(log_k))
    except OverflowError:
        k = float("inf")
    return k


# https://github.com/basusourya/mirostat/blob/master/mirostat.py


# try adjusting target tau dynamically based on just length even? Could you shape the "energy" of the clip?
def mirostat_sampling_v1(
    logits=None,
    tau=5.0,
    learning_rate=1.0,
    max_surprise=None,
    vocab_size=SEMANTIC_VOCAB_SIZE,
    indices_surprise_history=[],
    running_tot_surprise=0,
    generated=[],
):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    prob_original = torch.softmax(sorted_logits, dim=-1).tolist()

    s = estimate_s(prob_original)

    max_k = len(sorted_logits) - 1

    k = compute_k(vocab_size, s, max_surprise, max_k) + 1

    print(f"\n\nK: {k} s: {s} tau: {max_surprise}")

    sorted_logits = sorted_logits[0:k]
    sorted_indices = sorted_indices[0:k]

    prob_topk = torch.softmax(sorted_logits, dim=0)

    prev_i = torch.multinomial(prob_topk, num_samples=1, replacement=True)
    index_surprise = math.log2(1 / prob_original[prev_i])
    print(f"index_surprise: {index_surprise}")
    indices_surprise_history.append(index_surprise)

    running_tot_surprise += index_surprise
    prev = sorted_indices[prev_i]
    generated += prev.tolist()

    error_surprise = index_surprise - tau
    max_surprise -= learning_rate * error_surprise

    # full_probs = torch.zeros_like(logits) # 0? or -inf?
    full_probs = torch.empty_like(logits).fill_(-float("inf"))
    full_probs[sorted_indices] = prob_topk.to(full_probs.dtype)

    return (
        sorted_indices[prev_i],
        max_surprise,
        full_probs,
        indices_surprise_history,
        running_tot_surprise,
        generated,
    )


def mirostat_sampling_meh(
    logits=None,
    tau=5.0,
    learning_rate=1.0,
    max_surprise=None,
    vocab_size=SEMANTIC_VOCAB_SIZE,
    indices_surprise_history=[],
    running_tot_surprise=0,
    generated=[],
):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    prob_original = torch.softmax(sorted_logits, dim=-1).tolist()

    s = estimate_s(prob_original)

    max_k = len(sorted_logits) - 1

    k = compute_k(vocab_size, s, max_surprise, max_k) + 1

    print(f"\n\nK: {k} s: {s} tau: {max_surprise}")

    sorted_logits = sorted_logits[0:k]
    sorted_indices = sorted_indices[0:k]

    prob_topk = torch.softmax(sorted_logits, dim=0)

    prev_i = torch.multinomial(prob_topk, num_samples=1, replacement=True)

    index_surprise = math.log2(1 / prob_original[sorted_indices[prev_i].item()])
    print(f"index_surprise: {index_surprise}")
    indices_surprise_history.append(index_surprise)

    running_tot_surprise += index_surprise
    prev = sorted_indices[prev_i]
    generated += prev.tolist()
    error_surprise = index_surprise - tau
    max_surprise -= learning_rate * error_surprise

    full_probs = torch.empty_like(logits).fill_(-float("inf"))
    full_probs[sorted_indices] = prob_topk.to(full_probs.dtype)

    item_next = sorted_indices[prev_i]

    return (
        item_next,
        max_surprise,
        full_probs,
        indices_surprise_history,
        running_tot_surprise,
        generated,
    )


def mirostat_sampling_least(
    logits=None,
    tau=5.0,
    learning_rate=1.0,
    max_surprise=None,
    vocab_size=SEMANTIC_VOCAB_SIZE,
    indices_surprise_history=[],
    running_tot_surprise=0,
    generated=[],
):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    prob_original = torch.softmax(sorted_logits, dim=-1).tolist()

    s = estimate_s(prob_original)

    max_k = len(sorted_logits) - 1

    k = compute_k(vocab_size, s, max_surprise, max_k) + 1

    print(f"\n\nK: {k} s: {s} tau: {max_surprise}")

    sorted_logits = sorted_logits[0:k]
    sorted_indices = sorted_indices[0:k]

    prob_topk = torch.softmax(sorted_logits, dim=0)

    prev_i = torch.argmin(prob_topk).unsqueeze(0)

    index_surprise = math.log2(1 / prob_original[sorted_indices[prev_i].item()])
    print(f"index_surprise: {index_surprise}")
    indices_surprise_history.append(index_surprise)

    running_tot_surprise += index_surprise
    prev = sorted_indices[prev_i]
    generated += prev.tolist()

    error_surprise = index_surprise - tau
    max_surprise -= learning_rate * error_surprise

    full_probs = torch.empty_like(logits).fill_(-float("inf"))
    full_probs[sorted_indices] = prob_topk.to(full_probs.dtype)

    # Return least likely token and reverse generated logits
    # return sorted_indices[prev_i], max_surprise, torch.flip(full_probs, dims=[0]), indices_surprise_history, running_tot_surprise, generated
    return (
        sorted_indices[prev_i],
        max_surprise,
        full_probs,
        indices_surprise_history,
        running_tot_surprise,
        generated,
    )


def sine_wave_temperature(current_token, max_token):
    return 3.0 + 2.1 * (math.sin(2 * math.pi * (current_token / 150)) / 2.1 + 0.2)


def sine_wave_temperature(current_token, max_token, period=100, phase_shift=0):
    return 0.5 + 2.0 * (math.sin(2 * math.pi * (current_token / period) + phase_shift) / 2 + 0.5)


def sine_wave_temperature(current_token, token_period, start_phase, temp_min, temp_max):
    phase = 2 * math.pi * ((current_token + start_phase) / token_period)
    temp_range = temp_max - temp_min
    return temp_min + temp_range * ((math.sin(phase) / 2) + 0.5)


def mirostat_sampling(
    logits=None,
    tau=5.0,
    learning_rate=1.0,
    max_surprise=None,
    vocab_size=SEMANTIC_VOCAB_SIZE,
    indices_surprise_history=[],
    running_tot_surprise=0,
    generated=[],
    temperature_fn=None,
):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    prob_original = torch.softmax(sorted_logits, dim=-1).tolist()

    s = estimate_s(prob_original)

    max_k = len(sorted_logits) - 1

    k = compute_k(vocab_size, s, max_surprise, max_k) + 1

    sorted_logits = sorted_logits[0:k]
    sorted_indices = sorted_indices[0:k]

    # Current location in the segment
    current_token = len(generated)
    max_token = 768  # Maximum sample length

    if temperature_fn is not None:
        temp = temperature_fn(current_token, max_token)
        sorted_logits = torch.clamp(sorted_logits, -10000, 10000)
        # Apply to logits before softmax
        prob_topk = torch.softmax(sorted_logits / temp, dim=0)
        prob_topk = torch.clamp(prob_topk, 1e-9, 1 - 1e-9)  # Ensures probabilities are valid
    else:
        prob_topk = torch.softmax(sorted_logits, dim=0)

    prev_i = torch.multinomial(prob_topk, num_samples=1, replacement=True)

    epsilon = 1e-10
    index_surprise = math.log2(1 / (prob_original[sorted_indices[prev_i].item()] + epsilon))

    indices_surprise_history.append(index_surprise)

    running_tot_surprise += index_surprise
    prev = sorted_indices[prev_i]
    generated += prev.tolist()

    error_surprise = index_surprise - tau
    max_surprise -= learning_rate * error_surprise

    full_probs = torch.empty_like(logits).fill_(-float("inf"))
    full_probs[sorted_indices] = prob_topk.to(full_probs.dtype)

    if current_token % 25 == 0 and False:
        print(f"Temperature: {temp}")
        print(f"index_surprise: {index_surprise}")
        print(f"\n\nK: {k} s: {s} tau: {max_surprise}")

    return (
        sorted_indices[prev_i],
        max_surprise,
        full_probs,
        indices_surprise_history,
        running_tot_surprise,
        generated,
    )


def compute_negative_influence(negative_logits, n, window_size, negative_scale):
    # Check if negative_logits is empty
    if len(negative_logits) == 0:
        return 0

    # Ensure n is within range
    n = min(max(n, 0), len(negative_logits) - 1)

    # Adjust window_size if it's larger than negative_logits length
    window_size = min(window_size, len(negative_logits))

    # Get the start and end of the window
    start = max(0, n - window_size)
    end = min(len(negative_logits), n + window_size + 1)

    # Generate a Gaussian distribution for the weights and normalize them
    weights = np.exp(-((np.arange(start, end) - n) ** 2) / (2.0 * window_size**2))
    weights /= weights.sum()

    # Compute a weighted average of negative_logits within the window
    negative_influence = np.average(negative_logits[start:end], weights=weights, axis=0)

    # Adjust the influence by the negative_scale
    negative_influence *= min(max(negative_scale, 0), 1)  # Ensure negative_scale is between 0 and 1

    return negative_influence


def generate_text_semantic(
    text,
    history_prompt=None,
    temp=0.7,
    top_k=None,
    top_p=None,
    silent=False,
    min_eos_p=0.2,
    max_gen_duration_s=None,
    allow_early_stop=True,
    use_kv_caching=True,
    use_mirostat_sampling=False,
    # tau = 31100.0,
    tau=5.0,
    miro_learning_rate=1.0,
    token_repeat_penalty=0.0,
    inverted_p=None,
    bottom_k=None,
    return_logits=False,
    negative_tokens=None,
    negative_logits=None,
    negative_text_prompt_logits_scale=None,
    negative_text_prompt_logits_scale_window_size=64,
    negative_text_prompt_divergence_scale=None,
):
    """Generate semantic tokens from text."""

    if return_logits:
        all_logits = []

    if temp == 0:
        temp = 0.001
    # debug(locals())
    logger.debug(locals())
    assert isinstance(text, str)
    text = _normalize_whitespace(text)
    # assert len(text.strip()) > 0

    if history_prompt is not None:
        history_prompt = _load_history_prompt(history_prompt)
        semantic_history = history_prompt["semantic_prompt"]
        assert (
            isinstance(semantic_history, np.ndarray)
            and len(semantic_history.shape) == 1
            and len(semantic_history) > 0
            and semantic_history.min() >= 0
            and semantic_history.max() <= SEMANTIC_VOCAB_SIZE - 1
        )
    else:
        semantic_history = None
    # load models if not yet exist
    global models
    global models_devices
    if "text" not in models:
        if SUNO_USE_DIRECTML is True:
            preload_models(load_one_model_type="text")
        else:
            preload_models()
    model_container = models["text"]
    model = model_container["model"]
    tokenizer = model_container["tokenizer"]
    encoded_text = np.array(_tokenize(tokenizer, text)) + TEXT_ENCODING_OFFSET
    if OFFLOAD_CPU:
        if GLOBAL_ENABLE_MPS:
            device = _grab_best_device(use_gpu=False)
            models_devices["text"] = device
        model.to(models_devices["text"])
    device = next(model.parameters()).device
    if len(encoded_text) > 256:
        p = round((len(encoded_text) - 256) / len(encoded_text) * 100, 1)
        logger.warning(f"warning, text too long, lopping of last {p}%")
        encoded_text = encoded_text[:256]
    encoded_text = np.pad(
        encoded_text,
        (0, 256 - len(encoded_text)),
        constant_values=TEXT_PAD_TOKEN,
        mode="constant",
    )
    if semantic_history is not None:
        semantic_history = semantic_history.astype(np.int64)
        # print(f"Actual length of semantic input: {len(semantic_history)}")
        # lop off if history is too long, pad if needed
        semantic_history = semantic_history[-256:]
        semantic_history = np.pad(
            semantic_history,
            (0, 256 - len(semantic_history)),
            constant_values=SEMANTIC_PAD_TOKEN,
            mode="constant",
        )
    else:
        semantic_history = np.array([SEMANTIC_PAD_TOKEN] * 256)
    x = torch.from_numpy(
        np.hstack([encoded_text, semantic_history, np.array([SEMANTIC_INFER_TOKEN])]).astype(
            np.int64
        )
    )[None]
    assert x.shape[1] == 256 + 256 + 1
    with _inference_mode():
        if SUNO_USE_DIRECTML is True:
            device = dml
        x = x.to(device)
        n_tot_steps = 768

        # preallocate tensor
        x_initial = x.shape[1]
        x = torch.hstack([x, torch.empty([1, n_tot_steps], dtype=torch.int32, device=device)])

        # custom tqdm updates since we don't know when eos will occur
        pbar = tqdm.tqdm(disable=silent, total=n_tot_steps)
        pbar_state = 0
        tot_generated_duration_s = 0
        kv_cache = None

        # mirostat
        prev = None
        max_surprise = 2 * tau
        indices_surprise_history = []
        running_tot_surprise = 0
        miro_generated = []  # debug

        token_counts = defaultdict(int)
        for n in range(n_tot_steps):
            # if use_kv_caching and kv_cache is not None:
            #    x_input = x[:, [-1]]
            # else:
            #    x_input = x

            x_input = (
                x[:, [x_initial + n - 1]]
                if use_kv_caching and kv_cache is not None
                else x[:, : x_initial + n]
            )
            logits, kv_cache = model(
                x_input, merge_context=True, use_cache=use_kv_caching, past_kv=kv_cache
            )
            relevant_logits = logits[0, 0, :SEMANTIC_VOCAB_SIZE]
            if allow_early_stop:
                relevant_logits = torch.hstack(
                    (relevant_logits, logits[0, 0, [SEMANTIC_PAD_TOKEN]])  # eos
                )

            # Detach and convert to numpy for faster calculations
            original_device = relevant_logits.device
            relevant_logits = relevant_logits.detach().cpu().type(torch.float32).numpy()

            # Jon doing some silly ideas here, but inverted_p seems genuinely useful
            if top_p is not None or inverted_p is not None:
                if inverted_p is not None:
                    sorted_indices = np.argsort(relevant_logits)
                    cumulative_limit = inverted_p
                elif top_p is not None:
                    sorted_indices = np.argsort(relevant_logits)[::-1]
                    cumulative_limit = top_p
                sorted_logits = relevant_logits[sorted_indices]
                cumulative_probs = np.cumsum(softmax(sorted_logits))
                sorted_indices_to_remove = cumulative_probs > cumulative_limit
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                sorted_indices_to_remove[0] = False
                relevant_logits[sorted_indices[sorted_indices_to_remove]] = -np.inf

            relevant_logits = torch.from_numpy(relevant_logits)
            relevant_logits = relevant_logits.to(original_device)

            if top_k is not None or bottom_k is not None:
                if bottom_k is not None:
                    v, _ = torch.topk(
                        relevant_logits, max(bottom_k, relevant_logits.size(-1)), largest=False
                    )
                    relevant_logits[relevant_logits > v[-1]] = -float("Inf")
                elif top_k is not None:
                    v, _ = torch.topk(relevant_logits, min(top_k, relevant_logits.size(-1)))
                    relevant_logits[relevant_logits < v[-1]] = -float("Inf")

            if use_mirostat_sampling:
                logits_for_miro = relevant_logits / temp
                (
                    item_next,
                    max_surprise,
                    probs,
                    indices_surprise_history,
                    running_tot_surprise,
                    miro_generated,
                ) = mirostat_sampling(
                    logits=logits_for_miro,
                    max_surprise=max_surprise,
                    tau=tau,
                    learning_rate=miro_learning_rate,
                    vocab_size=SEMANTIC_VOCAB_SIZE,
                    indices_surprise_history=indices_surprise_history,
                    running_tot_surprise=running_tot_surprise,
                    generated=miro_generated,
                    temperature_fn=None,
                )
                # item_next = item_next.to(torch.int32)

            else:
                if token_repeat_penalty != 0.0 and token_repeat_penalty != 1.0:
                    for token, count in token_counts.items():
                        relevant_logits[token] += math.log(token_repeat_penalty) * count

                if return_logits:
                    all_logits.append(relevant_logits)

                probs = F.softmax(relevant_logits / temp, dim=-1)
                item_next = torch.multinomial(probs, num_samples=1).to(torch.int32)

            if allow_early_stop and (
                item_next == SEMANTIC_VOCAB_SIZE
                or (min_eos_p is not None and probs[-1] >= min_eos_p)
            ):
                n -= 1  # backtrack 1
                # eos found, so break
                pbar.total = n
                pbar.update(n - pbar_state)

                break
            # x = torch.cat((x, item_next[None]), dim=1)
            if token_repeat_penalty != 0.0 and token_repeat_penalty != 1.0:
                token_counts[int(item_next)] += 1

            x[0][x_initial + n] = item_next
            tot_generated_duration_s += 1 / SEMANTIC_RATE_HZ
            if max_gen_duration_s is not None and tot_generated_duration_s > max_gen_duration_s:
                pbar.total = n
                pbar.update(n - pbar_state)
                break
            if n == n_tot_steps - 1:
                pbar.total = n
                pbar.update(n - pbar_state)
                break
            del logits, relevant_logits, probs, item_next
            if n > pbar_state:
                if n > pbar.total:
                    pbar.total = n
                pbar.update(n - pbar_state)
            pbar_state = n
        pbar.total = n
        pbar.refresh()

        pbar.close()
        # out = x.detach().cpu().numpy().squeeze()[256 + 256 + 1 :]
        out = x.detach().cpu().numpy().squeeze()[x_initial : x_initial + n + 1]
        if use_mirostat_sampling and False:
            print(f"Target tau: {tau}")
            print("Total surprise value:", sum(indices_surprise_history))
            print("Average surprise value:", sum(indices_surprise_history) / len(out))
            print(f"Generated Miro: {miro_generated}")
            print(f"out: {out}")
    if OFFLOAD_CPU:
        model.to("cpu")
    assert all(0 <= out) and all(out < SEMANTIC_VOCAB_SIZE)
    _clear_cuda_cache()

    if SUNO_USE_DIRECTML is True:
        clean_models()

    if return_logits:
        return out, all_logits
    else:
        return out


def generate_text_semantic_branching_not_batching(
    text,
    history_prompt=None,
    temp=0.7,
    top_k=None,
    top_p=None,
    silent=False,
    min_eos_p=0.2,
    max_gen_duration_s=None,
    allow_early_stop=True,
    use_kv_caching=True,
    num_sample_per_step=2,
):
    """Generate semantic tokens from text."""
    assert isinstance(text, str)
    text = _normalize_whitespace(text)
    assert len(text.strip()) > 0
    if history_prompt is not None:
        history_prompt = _load_history_prompt(history_prompt)
        semantic_history = history_prompt["semantic_prompt"]
        assert (
            isinstance(semantic_history, np.ndarray)
            and len(semantic_history.shape) == 1
            and len(semantic_history) > 0
            and semantic_history.min() >= 0
            and semantic_history.max() <= SEMANTIC_VOCAB_SIZE - 1
        )
    else:
        semantic_history = None
    # load models if not yet exist
    global models
    global models_devices
    if "text" not in models:
        if SUNO_USE_DIRECTML is True:
            preload_models(load_one_model_type="text")
        else:
            preload_models()
    model_container = models["text"]
    model = model_container["model"]
    tokenizer = model_container["tokenizer"]
    encoded_text = np.array(_tokenize(tokenizer, text)) + TEXT_ENCODING_OFFSET
    if OFFLOAD_CPU:
        model.to(models_devices["text"])
    device = next(model.parameters()).device
    if len(encoded_text) > 256:
        p = round((len(encoded_text) - 256) / len(encoded_text) * 100, 1)
        logger.warning(f"warning, text too long, lopping of last {p}%")
        encoded_text = encoded_text[:256]
    encoded_text = np.pad(
        encoded_text,
        (0, 256 - len(encoded_text)),
        constant_values=TEXT_PAD_TOKEN,
        mode="constant",
    )
    if semantic_history is not None:
        semantic_history = semantic_history.astype(np.int64)
        # lop off if history is too long, pad if needed
        semantic_history = semantic_history[-256:]
        semantic_history = np.pad(
            semantic_history,
            (0, 256 - len(semantic_history)),
            constant_values=SEMANTIC_PAD_TOKEN,
            mode="constant",
        )
    else:
        semantic_history = np.array([SEMANTIC_PAD_TOKEN] * 256)
    # x = torch.from_numpy(
    #     np.hstack([
    #         encoded_text, semantic_history, np.array([SEMANTIC_INFER_TOKEN])
    #     ]).astype(np.int64)
    # )[None]

    x = torch.from_numpy(
        np.hstack([encoded_text, semantic_history, np.array([SEMANTIC_INFER_TOKEN])]).astype(
            np.int64
        )
    ).repeat(num_sample_per_step, 1)

    assert x.shape[1] == 256 + 256 + 1
    with _inference_mode():
        x = x.to(device)
        n_tot_steps = 768
        # custom tqdm updates since we don't know when eos will occur
        pbar = tqdm.tqdm(disable=silent, total=n_tot_steps)
        pbar_state = 0
        tot_generated_duration_s = 0
        kv_cache = None
        for n in range(n_tot_steps):
            if use_kv_caching and kv_cache is not None:
                x_input = x[:, [-1]]
            else:
                x_input = x
            logits, kv_cache = model(
                x_input, merge_context=True, use_cache=use_kv_caching, past_kv=kv_cache
            )
            relevant_logits = logits[0, 0, :SEMANTIC_VOCAB_SIZE]
            if allow_early_stop:
                relevant_logits = torch.hstack(
                    (relevant_logits, logits[0, 0, [SEMANTIC_PAD_TOKEN]])  # eos
                )
            if top_p is not None:
                # faster to convert to numpy
                original_device = relevant_logits.device
                relevant_logits = relevant_logits.detach().cpu().type(torch.float32).numpy()
                sorted_indices = np.argsort(relevant_logits)[::-1]
                sorted_logits = relevant_logits[sorted_indices]
                cumulative_probs = np.cumsum(softmax(sorted_logits))
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                sorted_indices_to_remove[0] = False
                relevant_logits[sorted_indices[sorted_indices_to_remove]] = -np.inf
                relevant_logits = torch.from_numpy(relevant_logits)
                relevant_logits = relevant_logits.to(original_device)
            if top_k is not None:
                v, _ = torch.topk(relevant_logits, min(top_k, relevant_logits.size(-1)))
                relevant_logits[relevant_logits < v[-1]] = -float("Inf")
            # probs = F.softmax(relevant_logits / temp, dim=-1)
            # item_next = torch.multinomial(probs, num_samples=1).to(torch.int32)

            probs = F.softmax(relevant_logits / temp, dim=-1)
            item_next = torch.multinomial(probs, num_samples=num_sample_per_step).to(torch.int32)
            if allow_early_stop and (
                item_next == SEMANTIC_VOCAB_SIZE
                or (min_eos_p is not None and probs[-1] >= min_eos_p)
            ):
                # eos found, so break
                pbar.update(n - pbar_state)
                break
            # x = torch.cat((x, item_next[None]), dim=1)
            for i in range(num_sample_per_step):
                x[i] = torch.cat((x[i], item_next[i][None]), dim=0)
            tot_generated_duration_s += 1 / SEMANTIC_RATE_HZ
            if max_gen_duration_s is not None and tot_generated_duration_s > max_gen_duration_s:
                pbar.update(n - pbar_state)
                break
            if n == n_tot_steps - 1:
                pbar.update(n - pbar_state)
                break
            del logits, relevant_logits, probs, item_next
            if n > pbar_state:
                if n > pbar.total:
                    pbar.total = n
                pbar.update(n - pbar_state)
            pbar_state = n
        pbar.total = n
        pbar.refresh()
        pbar.close()
        out = x.detach().cpu().numpy().squeeze()[256 + 256 + 1 :]
    if OFFLOAD_CPU:
        model.to("cpu")
    assert all(0 <= out) and all(out < SEMANTIC_VOCAB_SIZE)
    _clear_cuda_cache()
    return out


def generate_coarse(
    x_semantic,
    history_prompt=None,
    temp=0.7,
    top_k=None,
    top_p=None,
    silent=False,
    max_coarse_history=630,  # min 60 (faster), max 630 (more context)
    sliding_window_len=60,
    use_kv_caching=True,
    x_coarse_history_alignment_hack=-2,
):
    """Generate coarse audio codes from semantic tokens."""

    logger.debug(locals())
    assert (
        isinstance(x_semantic, np.ndarray)
        and len(x_semantic.shape) == 1
        and len(x_semantic) > 0
        and x_semantic.min() >= 0
        and x_semantic.max() <= SEMANTIC_VOCAB_SIZE - 1
    )
    assert 60 <= max_coarse_history <= 630
    assert max_coarse_history + sliding_window_len <= 1024 - 256
    semantic_to_coarse_ratio = COARSE_RATE_HZ / SEMANTIC_RATE_HZ * N_COARSE_CODEBOOKS

    max_semantic_history = int(np.floor(max_coarse_history / semantic_to_coarse_ratio))
    if history_prompt is not None:
        history_prompt = _load_history_prompt(history_prompt)
        x_semantic_history = history_prompt["semantic_prompt"]
        x_coarse_history = history_prompt["coarse_prompt"]

        # print(f"Pre Trim sem coars: {x_semantic_history.shape} {x_coarse_history.shape}")
        assert (
            isinstance(x_semantic_history, np.ndarray)
            and len(x_semantic_history.shape) == 1
            and len(x_semantic_history) > 0
            and x_semantic_history.min() >= 0
            and x_semantic_history.max() <= SEMANTIC_VOCAB_SIZE - 1
            and isinstance(x_coarse_history, np.ndarray)
            and len(x_coarse_history.shape) == 2
            and x_coarse_history.shape[0] == N_COARSE_CODEBOOKS
            and x_coarse_history.shape[-1] >= 0
            and x_coarse_history.min() >= 0
            and x_coarse_history.max() <= CODEBOOK_SIZE - 1
            and (
                round(x_coarse_history.shape[-1] / len(x_semantic_history), 1)
                == round(semantic_to_coarse_ratio / N_COARSE_CODEBOOKS, 1)
            )
        )

        x_coarse_history = _flatten_codebooks(x_coarse_history) + SEMANTIC_VOCAB_SIZE
        # trim histories correctly
        n_semantic_hist_provided = np.min(
            [
                max_semantic_history,
                len(x_semantic_history) - len(x_semantic_history) % 2,
                int(np.floor(len(x_coarse_history) / semantic_to_coarse_ratio)),
            ]
        )
        n_coarse_hist_provided = int(round(n_semantic_hist_provided * semantic_to_coarse_ratio))
        x_semantic_history = x_semantic_history[-n_semantic_hist_provided:].astype(np.int32)
        x_coarse_history = x_coarse_history[-n_coarse_hist_provided:].astype(np.int32)
        # TODO: bit of a hack for time alignment (sounds better)
        # x_coarse_history = x_coarse_history[:-2]
        x_coarse_history = x_coarse_history[:x_coarse_history_alignment_hack]

    else:
        x_semantic_history = np.array([], dtype=np.int32)
        x_coarse_history = np.array([], dtype=np.int32)

    # print(f"actual lengths we're using, x_semantic_history: {len(x_semantic_history)} x_coarse_history: {len(x_coarse_history)}")

    # load models if not yet exist
    global models
    global models_devices
    if "coarse" not in models:
        if SUNO_USE_DIRECTML is True:
            preload_models(load_one_model_type="coarse")
        else:
            preload_models()
    model = models["coarse"]
    if OFFLOAD_CPU:
        if GLOBAL_ENABLE_MPS:
            device = _grab_best_device(use_gpu=False)
            models_devices["coarse"] = device
        model.to(models_devices["coarse"])

    device = next(model.parameters()).device
    # start loop
    n_steps = int(
        round(
            np.floor(len(x_semantic) * semantic_to_coarse_ratio / N_COARSE_CODEBOOKS)
            * N_COARSE_CODEBOOKS
        )
    )
    assert n_steps > 0 and n_steps % N_COARSE_CODEBOOKS == 0

    # reminder to try filling up some of the COARSE_INFER_TOKEN with history to get better short clips
    x_semantic = np.hstack([x_semantic_history, x_semantic]).astype(np.int32)
    x_coarse = x_coarse_history.astype(np.int32)
    base_semantic_idx = len(x_semantic_history)
    with _inference_mode():
        if SUNO_USE_DIRECTML is True:
            device = dml
        x_semantic_in = torch.from_numpy(x_semantic)[None].to(device)
        x_coarse_in = torch.from_numpy(x_coarse)[None].to(device)
        n_window_steps = int(np.ceil(n_steps / sliding_window_len))
        n_step = 0
        for _ in tqdm.tqdm(range(n_window_steps), total=n_window_steps, disable=silent):
            semantic_idx = base_semantic_idx + int(round(n_step / semantic_to_coarse_ratio))
            # pad from right side
            x_in = x_semantic_in[:, np.max([0, semantic_idx - max_semantic_history]) :]
            x_in = x_in[:, :256]
            x_in = F.pad(
                x_in,
                (0, 256 - x_in.shape[-1]),
                "constant",
                COARSE_SEMANTIC_PAD_TOKEN,
            )

            x_in = torch.hstack(
                [
                    x_in,
                    torch.tensor([COARSE_INFER_TOKEN])[None].to(device),
                    x_coarse_in[:, -max_coarse_history:],
                ]
            )
            kv_cache = None
            for _ in range(sliding_window_len):
                if n_step >= n_steps:
                    continue
                is_major_step = n_step % N_COARSE_CODEBOOKS == 0

                if use_kv_caching and kv_cache is not None:
                    x_input = x_in[:, [-1]]
                else:
                    x_input = x_in

                logits, kv_cache = model(x_input, use_cache=use_kv_caching, past_kv=kv_cache)
                logit_start_idx = SEMANTIC_VOCAB_SIZE + (1 - int(is_major_step)) * CODEBOOK_SIZE
                logit_end_idx = SEMANTIC_VOCAB_SIZE + (2 - int(is_major_step)) * CODEBOOK_SIZE
                relevant_logits = logits[0, 0, logit_start_idx:logit_end_idx]
                if top_p is not None:
                    # faster to convert to numpy
                    logits_device = relevant_logits.device
                    logits_dtype = relevant_logits.type()
                    relevant_logits = relevant_logits.detach().cpu().type(torch.float32).numpy()
                    sorted_indices = np.argsort(relevant_logits)[::-1]
                    sorted_logits = relevant_logits[sorted_indices]
                    cumulative_probs = np.cumsum(softmax(sorted_logits))
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                    sorted_indices_to_remove[0] = False
                    relevant_logits[sorted_indices[sorted_indices_to_remove]] = -np.inf
                    relevant_logits = torch.from_numpy(relevant_logits)
                    relevant_logits = relevant_logits.to(logits_device).type(logits_dtype)
                if top_k is not None:
                    v, _ = torch.topk(relevant_logits, min(top_k, relevant_logits.size(-1)))
                    relevant_logits[relevant_logits < v[-1]] = -float("Inf")
                probs = F.softmax(relevant_logits / temp, dim=-1)
                # multinomial bugged on mps: shuttle to cpu if necessary
                inf_device = probs.device
                if probs.device.type == "mps":
                    probs = probs.to("cpu")
                item_next = torch.multinomial(probs, num_samples=1)
                probs = probs.to(inf_device)
                item_next = item_next.to(inf_device)
                item_next += logit_start_idx
                x_coarse_in = torch.cat((x_coarse_in, item_next[None]), dim=1)
                x_in = torch.cat((x_in, item_next[None]), dim=1)
                del logits, relevant_logits, probs, item_next
                n_step += 1
            del x_in
        del x_semantic_in
    if OFFLOAD_CPU:
        model.to("cpu")
    gen_coarse_arr = x_coarse_in.detach().cpu().numpy().squeeze()[len(x_coarse_history) :]
    del x_coarse_in
    assert len(gen_coarse_arr) == n_steps
    gen_coarse_audio_arr = gen_coarse_arr.reshape(-1, N_COARSE_CODEBOOKS).T - SEMANTIC_VOCAB_SIZE
    for n in range(1, N_COARSE_CODEBOOKS):
        gen_coarse_audio_arr[n, :] -= n * CODEBOOK_SIZE
    _clear_cuda_cache()
    if SUNO_USE_DIRECTML is True:
        clean_models()
    return gen_coarse_audio_arr


def generate_coarse_amd_directml(
    x_semantic,
    history_prompt=None,
    temp=0.7,
    top_k=None,
    top_p=None,
    silent=False,
    max_coarse_history=630,  # min 60 (faster), max 630 (more context)
    sliding_window_len=60,
    use_kv_caching=True,
    x_coarse_history_alignment_hack=-2,
):
    """Generate coarse audio codes from semantic tokens."""

    logger.debug(locals())

    assert (
        isinstance(x_semantic, np.ndarray)
        and len(x_semantic.shape) == 1
        and len(x_semantic) > 0
        and x_semantic.min() >= 0
        and x_semantic.max() <= SEMANTIC_VOCAB_SIZE - 1
    )
    assert 60 <= max_coarse_history <= 630
    assert max_coarse_history + sliding_window_len <= 1024 - 256
    semantic_to_coarse_ratio = COARSE_RATE_HZ / SEMANTIC_RATE_HZ * N_COARSE_CODEBOOKS
    max_semantic_history = int(np.floor(max_coarse_history / semantic_to_coarse_ratio))
    if history_prompt is not None:
        history_prompt = _load_history_prompt(history_prompt)
        x_semantic_history = history_prompt["semantic_prompt"]
        x_coarse_history = history_prompt["coarse_prompt"]
        assert (
            isinstance(x_semantic_history, np.ndarray)
            and len(x_semantic_history.shape) == 1
            and len(x_semantic_history) > 0
            and x_semantic_history.min() >= 0
            and x_semantic_history.max() <= SEMANTIC_VOCAB_SIZE - 1
            and isinstance(x_coarse_history, np.ndarray)
            and len(x_coarse_history.shape) == 2
            and x_coarse_history.shape[0] == N_COARSE_CODEBOOKS
            and x_coarse_history.shape[-1] >= 0
            and x_coarse_history.min() >= 0
            and x_coarse_history.max() <= CODEBOOK_SIZE - 1
            and (
                round(x_coarse_history.shape[-1] / len(x_semantic_history), 1)
                == round(semantic_to_coarse_ratio / N_COARSE_CODEBOOKS, 1)
            )
        )
        x_coarse_history = _flatten_codebooks(x_coarse_history) + SEMANTIC_VOCAB_SIZE
        # trim histories correctly
        n_semantic_hist_provided = np.min(
            [
                max_semantic_history,
                len(x_semantic_history) - len(x_semantic_history) % 2,
                int(np.floor(len(x_coarse_history) / semantic_to_coarse_ratio)),
            ]
        )
        n_coarse_hist_provided = int(round(n_semantic_hist_provided * semantic_to_coarse_ratio))
        x_semantic_history = x_semantic_history[-n_semantic_hist_provided:].astype(np.int32)
        x_coarse_history = x_coarse_history[-n_coarse_hist_provided:].astype(np.int32)
        # TODO: bit of a hack for time alignment (sounds better)
        x_coarse_history = x_coarse_history[:-2]
    else:
        x_semantic_history = np.array([], dtype=np.int32)
        x_coarse_history = np.array([], dtype=np.int32)
    # load models if not yet exist
    global models
    global models_devices
    if "coarse" not in models:
        if SUNO_USE_DIRECTML is True:
            preload_models(load_one_model_type="coarse")
        else:
            preload_models()
    model = models["coarse"]
    if OFFLOAD_CPU:
        if GLOBAL_ENABLE_MPS:
            device = _grab_best_device(use_gpu=False)
            models_devices["coarse"] = device
        model.to(models_devices["coarse"])
    # device = next(model.parameters()).device

    # start loop
    n_steps = int(
        round(
            np.floor(len(x_semantic) * semantic_to_coarse_ratio / N_COARSE_CODEBOOKS)
            * N_COARSE_CODEBOOKS
        )
    )
    assert n_steps > 0 and n_steps % N_COARSE_CODEBOOKS == 0
    x_semantic = np.hstack([x_semantic_history, x_semantic]).astype(np.int32)
    x_coarse = x_coarse_history.astype(np.int32)
    base_semantic_idx = len(x_semantic_history)
    cumulative_time = 0
    with _inference_mode():
        try:
            # x_semantic_in = torch.from_numpy(x_semantic)[None].to(dml)
            x_semantic_in_np = x_semantic[None]
            # x_coarse_in = torch.from_numpy(x_coarse)[None].to(dml)
            x_coarse_in_np = x_coarse[None]
            n_window_steps = int(np.ceil(n_steps / sliding_window_len))
            n_step = 0
            for _ in tqdm.tqdm(range(n_window_steps), total=n_window_steps, disable=silent):
                semantic_idx = base_semantic_idx + int(round(n_step / semantic_to_coarse_ratio))
                # pad from right side
                x_in_np = x_semantic_in_np[:, np.max([0, semantic_idx - max_semantic_history]) :]
                x_in_np = x_in_np[:, :256]
                """
                x_in_np = F.pad(
                    x_in_np,
                    (0, 256 - x_in_np.shape[-1]),
                    "constant",
                    COARSE_SEMANTIC_PAD_TOKEN,
                )
                """
                np_pad_size = ((0, 0), (0, 256 - x_in_np.shape[-1]))
                x_in_np = np.pad(
                    x_in_np,
                    np_pad_size,
                    constant_values=COARSE_SEMANTIC_PAD_TOKEN,
                    mode="constant",
                )

                """
                x_in = torch.hstack(
                    [
                        x_in,
                        torch.tensor([COARSE_INFER_TOKEN])[None].to(dml),
                        x_coarse_in[:, -max_coarse_history:],
                    ]
                )
                """

                coarse_infer_token_np = np.array([COARSE_INFER_TOKEN])[None]

                x_in_np = np.hstack(
                    [
                        x_in_np,
                        coarse_infer_token_np,
                        x_coarse_in_np[:, -max_coarse_history:],
                    ]
                )

                kv_cache = None
                for _ in range(sliding_window_len):
                    if n_step >= n_steps:
                        continue
                    is_major_step = n_step % N_COARSE_CODEBOOKS == 0

                    if use_kv_caching and kv_cache is not None:
                        x_input = x_in_np[:, [-1]]
                    else:
                        x_input = x_in_np

                    x_input_tensor = torch.from_numpy(x_input).to(dml)

                    logits, kv_cache = model(
                        x_input_tensor, use_cache=use_kv_caching, past_kv=kv_cache
                    )

                    logit_start_idx = SEMANTIC_VOCAB_SIZE + (1 - int(is_major_step)) * CODEBOOK_SIZE
                    logit_end_idx = SEMANTIC_VOCAB_SIZE + (2 - int(is_major_step)) * CODEBOOK_SIZE
                    relevant_logits = logits[0, 0, logit_start_idx:logit_end_idx]

                    if top_p is not None:
                        # faster to convert to numpy
                        # original_device = relevant_logits.device
                        relevant_logits = relevant_logits.detach().cpu().type(torch.float32).numpy()
                        sorted_indices = np.argsort(relevant_logits)[::-1]
                        sorted_logits = relevant_logits[sorted_indices]
                        cumulative_probs = np.cumsum(softmax(sorted_logits))
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                        sorted_indices_to_remove[0] = False
                        relevant_logits[sorted_indices[sorted_indices_to_remove]] = -np.inf
                        relevant_logits = torch.from_numpy(relevant_logits)
                        # relevant_logits = relevant_logits.to(original_device)
                        # stay as numpy, since we converted for directml anyway...
                    if top_k is not None:
                        v, _ = torch.topk(
                            relevant_logits.to(dml),
                            min(top_k, relevant_logits.to(dml).size(-1)),
                        )
                        relevant_logits[relevant_logits < v[-1]] = -float("Inf")

                    # probs = F.softmax(relevant_logits.to(dml) / temp, dim=-1)

                    start_time = time.time()

                    # item_next = torch.multinomial(probs, num_samples=1).to(torch.int32)

                    probs_np = (
                        F.softmax(relevant_logits.to(dml) / temp, dim=-1)
                        .cpu()
                        .type(torch.float32)
                        .numpy()
                    )

                    item_next_np = np.random.choice(
                        np.arange(probs_np.shape[-1]), size=1, p=probs_np.flatten()
                    )

                    # item_next = torch.from_numpy(item_next_np).to(torch.int32).to(dml)

                    # doing in raw numpy same speed with AMD directML, but maybe faster if you setup MKL correctly?
                    # actually tha wasn't quite righ anyway...
                    end_time = time.time()
                    cumulative_time = cumulative_time + (end_time - start_time)

                    # amd_multinomial = torch_distributions.Categorical(probs)
                    # action = amd_multinomial.sample((1,))
                    # item_next = amd_multinomial.log_prob(action).to(torch.int32)

                    # multinomial bugged on mps: shuttle to cpu if necessary
                    # inf_device = probs.device
                    # if probs.device.type == "mps" or True:
                    #    probs = probs.to("cpu")
                    #    # print(f"Here in coarse: {probs.device}")
                    # item_next = torch.multinomial(probs, num_samples=1)
                    # probs = probs.to(inf_device)
                    # item_next = item_next.to(inf_device)

                    item_next_np += logit_start_idx

                    x_coarse_in_np = np.hstack((x_coarse_in_np, item_next_np[None]))

                    # x_coarse_in = torch.from_numpy(x_coarse_in_np).to(dml)
                    # x_in = torch.cat((x_in_np.to(dml), item_next_np[None]), dim=1)

                    x_in_np = np.hstack((x_in_np, item_next_np[None]))
                    del logits, relevant_logits, probs_np, item_next_np
                    n_step += 1
                del x_in_np
            del x_semantic_in_np
        except RuntimeError as e:
            print(f"RuntimeError: {e}")
            # show all possble details and traceback, print to output
            print(f"Traceback: {traceback.format_exc()}")  # and print(sys.exc_info()[2])
            print(f"Exception: {sys.exc_info()[2]}")

    if OFFLOAD_CPU:
        model.to("cpu")
    gen_coarse_arr = x_coarse_in_np.squeeze()[len(x_coarse_history) :]
    del x_coarse_in_np
    assert len(gen_coarse_arr) == n_steps
    gen_coarse_audio_arr = gen_coarse_arr.reshape(-1, N_COARSE_CODEBOOKS).T - SEMANTIC_VOCAB_SIZE
    for n in range(1, N_COARSE_CODEBOOKS):
        gen_coarse_audio_arr[n, :] -= n * CODEBOOK_SIZE
    _clear_cuda_cache()
    if SUNO_USE_DIRECTML is True:
        clean_models()
    return gen_coarse_audio_arr


def generate_fine(
    x_coarse_gen,
    history_prompt=None,
    temp=0.5,
    silent=True,
):
    if temp == 0:
        temp = 0.001

    """Generate full audio codes from coarse audio codes."""
    assert (
        isinstance(x_coarse_gen, np.ndarray)
        and len(x_coarse_gen.shape) == 2
        and 1 <= x_coarse_gen.shape[0] <= N_FINE_CODEBOOKS - 1
        and x_coarse_gen.shape[1] > 0
        and x_coarse_gen.min() >= 0
        and x_coarse_gen.max() <= CODEBOOK_SIZE - 1
    )
    if history_prompt is not None:
        history_prompt = _load_history_prompt(history_prompt)
        x_fine_history = history_prompt["fine_prompt"]
        assert (
            isinstance(x_fine_history, np.ndarray)
            and len(x_fine_history.shape) == 2
            and x_fine_history.shape[0] == N_FINE_CODEBOOKS
            and x_fine_history.shape[1] >= 0
            and x_fine_history.min() >= 0
            and x_fine_history.max() <= CODEBOOK_SIZE - 1
        )
    else:
        x_fine_history = None
    n_coarse = x_coarse_gen.shape[0]
    # load models if not yet exist
    global models
    global models_devices
    if "fine" not in models:
        if SUNO_USE_DIRECTML is True:
            preload_models(load_one_model_type="fine")
        else:
            preload_models()
    model = models["fine"]
    if OFFLOAD_CPU:
        if GLOBAL_ENABLE_MPS:
            device = _grab_best_device(use_gpu=False)
            models_devices["fine"] = device
        model.to(models_devices["fine"])
    device = next(model.parameters()).device
    # make input arr
    in_arr = np.vstack(
        [
            x_coarse_gen,
            np.zeros((N_FINE_CODEBOOKS - n_coarse, x_coarse_gen.shape[1]))
            + CODEBOOK_SIZE,  # padding
        ]
    ).astype(np.int32)
    # prepend history if available (max 512)
    if x_fine_history is not None:
        x_fine_history = x_fine_history.astype(np.int32)
        in_arr = np.hstack(
            [
                x_fine_history[:, -512:].astype(np.int32),
                in_arr,
            ]
        )
        n_history = x_fine_history[:, -512:].shape[1]
    else:
        n_history = 0
    n_remove_from_end = 0
    # need to pad if too short (since non-causal model)
    if in_arr.shape[1] < 1024:
        n_remove_from_end = 1024 - in_arr.shape[1]
        in_arr = np.hstack(
            [
                in_arr,
                np.zeros((N_FINE_CODEBOOKS, n_remove_from_end), dtype=np.int32) + CODEBOOK_SIZE,
            ]
        )
    # we can be lazy about fractional loop and just keep overwriting codebooks
    n_loops = np.max([0, int(np.ceil((x_coarse_gen.shape[1] - (1024 - n_history)) / 512))]) + 1
    with _inference_mode():
        if SUNO_USE_DIRECTML is True:
            device = dml
        in_arr = torch.tensor(in_arr.T).to(device)
        for n in tqdm.tqdm(range(n_loops), disable=silent):
            start_idx = np.min([n * 512, in_arr.shape[0] - 1024])
            start_fill_idx = np.min([n_history + n * 512, in_arr.shape[0] - 512])
            rel_start_fill_idx = start_fill_idx - start_idx
            in_buffer = in_arr[start_idx : start_idx + 1024, :][None]
            for nn in range(n_coarse, N_FINE_CODEBOOKS):
                logits = model(nn, in_buffer)
                if temp is None:
                    relevant_logits = logits[0, rel_start_fill_idx:, :CODEBOOK_SIZE]
                    codebook_preds = torch.argmax(relevant_logits, -1)
                else:
                    relevant_logits = logits[0, :, :CODEBOOK_SIZE] / temp
                    probs = F.softmax(relevant_logits, dim=-1)
                    codebook_preds = torch.multinomial(
                        probs[rel_start_fill_idx:1024], num_samples=1
                    ).reshape(-1)
                codebook_preds = codebook_preds.to(torch.int32)
                in_buffer[0, rel_start_fill_idx:, nn] = codebook_preds
                del logits, codebook_preds
            # transfer over info into model_in and convert to numpy
            for nn in range(n_coarse, N_FINE_CODEBOOKS):
                in_arr[
                    start_fill_idx : start_fill_idx + (1024 - rel_start_fill_idx), nn
                ] = in_buffer[0, rel_start_fill_idx:, nn]
            del in_buffer
        gen_fine_arr = in_arr.detach().cpu().numpy().squeeze().T
        del in_arr
    if OFFLOAD_CPU:
        model.to("cpu")
    gen_fine_arr = gen_fine_arr[:, n_history:]
    if n_remove_from_end > 0:
        gen_fine_arr = gen_fine_arr[:, :-n_remove_from_end]
    assert gen_fine_arr.shape[-1] == x_coarse_gen.shape[-1]
    _clear_cuda_cache()
    if SUNO_USE_DIRECTML is True:
        clean_models()
    return gen_fine_arr


def _flatten_codebooks(arr, offset_size=CODEBOOK_SIZE):
    assert len(arr.shape) == 2
    arr = arr.copy()
    if offset_size is not None:
        for n in range(1, arr.shape[0]):
            arr[n, :] += offset_size * n
    flat_arr = arr.ravel("F")
    return flat_arr


COARSE_SEMANTIC_PAD_TOKEN = 12_048
COARSE_INFER_TOKEN = 12_050


def codec_decode(fine_tokens):
    """Turn quantized audio codes into audio array using encodec."""
    # load models if not yet exist
    global models
    global models_devices
    if "codec" not in models:
        if SUNO_USE_DIRECTML is True:
            preload_models(load_one_model_type="codec")
        else:
            preload_models()
    model = models["codec"]
    if OFFLOAD_CPU:
        if GLOBAL_ENABLE_MPS:
            device = _grab_best_device(use_gpu=False)
            models_devices["codec"] = device
        model.to(models_devices["codec"])
    device = next(model.parameters()).device
    arr = torch.from_numpy(fine_tokens)[None]
    if SUNO_USE_DIRECTML is True:
        arr = arr.to(dml)
    else:
        arr = arr.to(device)
    arr = arr.transpose(0, 1)
    emb = model.quantizer.decode(arr)
    out = model.decoder(emb)
    audio_arr = out.detach().cpu().numpy().squeeze()
    del arr, emb, out
    if OFFLOAD_CPU:
        model.to("cpu")
    if SUNO_USE_DIRECTML is True:
        clean_models()
    return audio_arr


## Added:


# Just overriding this because somehow I keep loading the wrong models?
def load_model(use_gpu=True, use_small=False, force_reload=False, model_type="text"):
    logger.debug(locals())

    _load_model_f = funcy.partial(_load_model, model_type=model_type, use_small=use_small)
    if model_type not in ("text", "coarse", "fine"):
        raise NotImplementedError()
    global models
    global models_devices
    device = _grab_best_device(use_gpu=use_gpu)
    model_key = f"{model_type}"
    if OFFLOAD_CPU:
        models_devices[model_key] = device
        device = "cpu"
    if model_key not in models or force_reload:
        ckpt_path = _get_ckpt_path(model_type, use_small=use_small)
        clean_models(model_key=model_key)
        model = _load_model_f(ckpt_path, device)
        models[model_key] = model
    if model_type == "text":
        if SUNO_USE_DIRECTML is True:
            models[model_key]["model"].to(dml)
        else:
            models[model_key]["model"].to(device)
    else:
        if SUNO_USE_DIRECTML is True:
            models[model_key].to(dml)
        else:
            models[model_key].to(device)
    logger.debug(f"Loaded {model_key} onto {device}.")
    return models[model_key]


def print_loading_info(model_key, ckpt_path, device):
    device_str = str(device)
    if SUNO_USE_DIRECTML is True:
        device_str = "directml (partial AMD GPU support)"
    if GLOBAL_ENABLE_MPS:
        device_str = "cpu/mps: Partial Apple Support"
    if OFFLOAD_CPU:
        device_str = "cpu/gpu: Offloading, cpu until needed, then gpu"

    print(f"--Loading {model_key} model from {ckpt_path} to {device_str}")


def _load_model(ckpt_path, device, use_small=False, model_type="text"):
    if model_type == "text":
        ConfigClass = GPTConfig
        ModelClass = GPT
    elif model_type == "coarse":
        ConfigClass = GPTConfig
        ModelClass = GPT
    elif model_type == "fine":
        ConfigClass = FineGPTConfig
        ModelClass = FineGPT
    else:
        raise NotImplementedError()
    model_key = f"{model_type}_small" if use_small or USE_SMALL_MODELS else model_type
    model_info = REMOTE_MODEL_PATHS[model_key]
    if not os.path.exists(ckpt_path):
        logger.info(f"{model_type} model not found, downloading into `{CACHE_DIR}`.")

        remote_filename = hf_hub_url(model_info["repo_id"], model_info["file_name"])
        print(
            f"Downloading {model_key} {model_info['repo_id']} remote model file {remote_filename} {model_info['file_name']} to {CACHE_DIR}"
        )  # added
        _download(model_info["repo_id"], model_info["file_name"])

    print_loading_info(model_key, ckpt_path, device)

    # If I try to load straight to DML, I get a strange error. So doing in two steps.
    checkpoint = torch.load(ckpt_path, map_location=device)

    # this is a hack
    model_args = checkpoint["model_args"]
    if "input_vocab_size" not in model_args:
        model_args["input_vocab_size"] = model_args["vocab_size"]
        model_args["output_vocab_size"] = model_args["vocab_size"]
        del model_args["vocab_size"]
    gptconf = ConfigClass(**checkpoint["model_args"])
    model = ModelClass(gptconf)

    if SUNO_HALF_PRECISION:
        model = model.half()
    elif SUNO_HALF_BFLOAT16:
        model.bfloat16()

    state_dict = checkpoint["model"]
    # fixup checkpoint
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    extra_keys = set(state_dict.keys()) - set(model.state_dict().keys())
    extra_keys = set([k for k in extra_keys if not k.endswith(".attn.bias")])
    missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
    missing_keys = set([k for k in missing_keys if not k.endswith(".attn.bias")])
    if len(extra_keys) != 0:
        raise ValueError(f"extra keys found: {extra_keys}")
    if len(missing_keys) != 0:
        raise ValueError(f"missing keys: {missing_keys}")
    model.load_state_dict(state_dict, strict=False)
    n_params = model.get_num_params()
    val_loss = checkpoint["best_val_loss"].item()
    logger.info(f"model loaded: {round(n_params/1e6,1)}M params, {round(val_loss,3)} loss")
    model.eval()
    if SUNO_USE_DIRECTML is True:
        model.to(dml)
    else:
        model.to(device)
    # del checkpoint, state_dict
    del checkpoint, state_dict, model_args, val_loss
    _clear_cuda_cache()
    if model_type == "text":
        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

        return {
            "model": model,
            "tokenizer": tokenizer,
        }
    return model


def preload_models(
    text_use_gpu=True,
    text_use_small=False,
    coarse_use_gpu=True,
    coarse_use_small=False,
    fine_use_gpu=True,
    fine_use_small=False,
    codec_use_gpu=True,
    force_reload=False,
    load_one_model_type=None,
):
    """Load all the necessary models for the pipeline."""

    if SUNO_USE_DIRECTML is True:
        text_use_gpu = False
        coarse_use_gpu = False
        fine_use_gpu = False

    # What is going on here
    logger.debug(
        f"USE_SMALL_MODELS = {USE_SMALL_MODELS} GLOBAL_ENABLE_MPS = {GLOBAL_ENABLE_MPS}, OFFLOAD_CPU = {OFFLOAD_CPU}"
    )
    logger.debug(
        f"text_use_gpu = {text_use_gpu}, text_use_small = {text_use_small}, coarse_use_gpu = {coarse_use_gpu}, coarse_use_small = {coarse_use_small}, fine_use_gpu = {fine_use_gpu}, fine_use_small = {fine_use_small}, codec_use_gpu = {codec_use_gpu}, force_reload = {force_reload}"
    )

    if USE_SMALL_MODELS:
        text_use_small = True
        coarse_use_small = True
        fine_use_small = True

    if _grab_best_device() == "cpu" and (
        text_use_gpu or coarse_use_gpu or fine_use_gpu or codec_use_gpu
    ):
        warning_string = " -->No GPU being used. Careful, inference might be very slow!"

        if SUNO_USE_DIRECTML is True:
            warning_string = "-->GPU using DirectML (partial AMD GPU support)"
        if GLOBAL_ENABLE_MPS:
            warning_string = "-->cpu/mps: Partial Apple Support"

        # logger.warning(warning_string)
        print(f"{warning_string}")

    if load_one_model_type is not None:
        if load_one_model_type == "text":
            _ = load_model(
                model_type="text",
                use_gpu=text_use_gpu,
                use_small=text_use_small,
                force_reload=force_reload,
            )
        elif load_one_model_type == "coarse":
            _ = load_model(
                model_type="coarse",
                use_gpu=coarse_use_gpu,
                use_small=coarse_use_small,
                force_reload=force_reload,
            )
        elif load_one_model_type == "fine":
            _ = load_model(
                model_type="fine",
                use_gpu=fine_use_gpu,
                use_small=fine_use_small,
                force_reload=force_reload,
            )
        elif load_one_model_type == "codec":
            _ = load_codec_model(use_gpu=codec_use_gpu, force_reload=force_reload)
    else:
        _ = load_model(
            model_type="text",
            use_gpu=text_use_gpu,
            use_small=text_use_small,
            force_reload=force_reload,
        )
        _ = load_model(
            model_type="coarse",
            use_gpu=coarse_use_gpu,
            use_small=coarse_use_small,
            force_reload=force_reload,
        )
        _ = load_model(
            model_type="fine",
            use_gpu=fine_use_gpu,
            use_small=fine_use_small,
            force_reload=force_reload,
        )
        _ = load_codec_model(use_gpu=codec_use_gpu, force_reload=force_reload)
