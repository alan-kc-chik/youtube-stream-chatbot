import os
import sys
import resampy

sys.path.append("rvc")

import traceback
import ffmpeg
import numpy as np
from rvc.configs.config import Config
from rvc.vc_infer_pipeline import VC
from fairseq import checkpoint_utils
import torch
from rvc.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from scipy.io.wavfile import write as write_wav

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000


def load_audio(file, sr):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # 防止小白拷路径头尾带了空格和"和回车
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.float32).flatten()


def load_hubert(path_to_hubert_pt):
    global hubert_model
    config = Config()
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [path_to_hubert_pt],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()


def get_vc(sid, to_return_protect0, to_return_protect1):
    global n_spk, tgt_sr, net_g, vc, cpt, version
    config = Config()
    if sid == "" or sid == []:
        global hubert_model
        if hubert_model is not None:  # 考虑到轮询, 需要加个判断看是否 sid 是由有模型切换到无模型的
            print("clean_empty_cache")
            del net_g, n_spk, vc, hubert_model, tgt_sr  # ,cpt
            hubert_model = net_g = n_spk = vc = hubert_model = tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            ###楼下不这么折腾清理不干净
            if_f0 = cpt.get("f0", 1)
            version = cpt.get("version", "v1")
            if version == "v1":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs256NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            elif version == "v2":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs768NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
            del net_g, cpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cpt = None
        return {"visible": False, "__type__": "update"}
    person = sid
    print("loading %s" % person)
    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    if if_f0 == 0:
        to_return_protect0 = to_return_protect1 = {
            "visible": False,
            "value": 0.5,
            "__type__": "update",
        }
    else:
        to_return_protect0 = {
            "visible": True,
            "value": to_return_protect0,
            "__type__": "update",
        }
        to_return_protect1 = {
            "visible": True,
            "value": to_return_protect1,
            "__type__": "update",
        }
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]
    return (
        {"visible": True, "maximum": n_spk, "__type__": "update"},
        to_return_protect0,
        to_return_protect1,
    )


def vc_single(
    input_audio,
    input_sample_rate,
    format="numpy",
    sid=0,
    f0_up_key=0,
    f0_file=None,
    f0_method="rmvpe",
    file_index="",
    file_index2="",
    index_rate=0.75,
    filter_radius=3,
    resample_sr=0,
    rms_mix_rate=0.25,
    protect=0.33,
):  # spk_item, input_audio0, vc_transform0,f0_file,f0method0
    global tgt_sr, net_g, vc, hubert_model, version
    if format != "numpy":
        input_audio_path = input_audio
        if input_audio_path is None:
            return "You need to upload an audio", None
        audio = load_audio(input_audio_path, SAMPLE_RATE)
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max
    else:
        audio = resampy.resample(
            input_audio, sr_orig=input_sample_rate, sr_new=SAMPLE_RATE
        )
    f0_up_key = int(f0_up_key)
    try:
        times = [0, 0, 0]
        if not hubert_model:
            load_hubert()
        if_f0 = cpt.get("f0", 1)
        file_index = (
            (
                file_index.strip(" ")
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip(" ")
                .replace("trained", "added")
            )
            if file_index != ""
            else file_index2
        )
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            input_audio,
            times,
            f0_up_key,
            f0_method,
            file_index,
            # file_big_npy,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            f0_file=f0_file,
        )
        if tgt_sr != resample_sr >= SAMPLE_RATE:
            tgt_sr = resample_sr
        
        return audio_opt, tgt_sr
    except:
        info = traceback.format_exc()
        print(info)
        return info, (None, None)


def voice_conversion(input_audio, input_sample_rate, checkpoint_path, **kwargs):
    load_hubert("rvc/hubert_base.pt")
    get_vc(sid=checkpoint_path, to_return_protect0=0.33, to_return_protect1=0.33)
    audio_arr, sample_rate = vc_single(
        input_audio=input_audio, input_sample_rate=input_sample_rate, **kwargs
    )
    return audio_arr, sample_rate


if __name__ == "__main__":
    args = {
        "input_audio": "tts_output.wav",
        "input_sample_rate": None,
        "format": "wav",
    }
    from scipy.io.wavfile import read as read_wav
    sample_rate, audio_arr = read_wav("tts_output.wav")
    audio_arr, sample_rate = voice_conversion(audio_arr, sample_rate, format="numpy")

    # audio_arr, sample_rate = voice_conversion(**args)
    write_wav("vc_output.wav", sample_rate, audio_arr)
