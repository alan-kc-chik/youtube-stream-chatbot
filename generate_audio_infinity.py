import numpy as np
import copy

from rich import print

from bark_infinity import config
from bark_infinity.api import (
    process_history_prompt,
    load_npz,
    chunk_up_text,
    estimate_spoken_time,
    generate_audio_barki,
)
from bark_infinity.generation import SAMPLE_RATE

logger = config.logger

from bark_infinity.config import (
    logger,
    console,
    VALID_HISTORY_PROMPT_DIRS,
)

from bark_infinity import generation
import time
import torch


def generate_audio_long(**kwargs):
    history_prompt = None
    history_prompt = kwargs.get("history_prompt", None)
    kwargs["history_prompt"] = None

    print(f"history_prompt: {history_prompt}")

    silent = kwargs.get("silent", None)

    full_generation_segments = []
    audio_arr_segments = []

    stable_mode_interval = kwargs.get("stable_mode_interval", None)
    if stable_mode_interval is None:
        stable_mode_interval = 1

    if stable_mode_interval < 0:
        stable_mode_interval = 0

    stable_mode_interval_counter = None

    if stable_mode_interval >= 2:
        stable_mode_interval_counter = stable_mode_interval

    history_prompt_for_next_segment = None
    base_history = None
    if history_prompt is not None:
        history_prompt_string = history_prompt
        history_prompt = process_history_prompt(history_prompt)
        if history_prompt is not None:
            base_history = load_npz(history_prompt)

            base_history = {key: base_history[key] for key in base_history.keys()}
            kwargs["history_prompt_string"] = history_prompt_string
            kwargs["previous_segment_type"] = "base_history"
            history_prompt_for_next_segment = copy.deepcopy(
                base_history
            )  # just start from a dict for consistency
        else:
            logger.error(
                f"Speaker {history_prompt} could not be found, looking in{VALID_HISTORY_PROMPT_DIRS}"
            )

            return None, None, None, None

    audio_segments = chunk_up_text(**kwargs)
    full_generation, audio_arr = (None, None)

    kwargs["output_full"] = True
    kwargs["total_segments"] = len(audio_segments)

    show_generation_times = kwargs.get("show_generation_times", None)
    all_segments_start_time = time.time()

    history_prompt_flipper = False
    if len(audio_segments) < 1:
        audio_segments.append("")

    for i, segment_text in enumerate(audio_segments):
        estimated_time = estimate_spoken_time(segment_text)
        print(f"segment_text: {segment_text}")

        prompt_text_prefix = kwargs.get("prompt_text_prefix", None)
        if prompt_text_prefix is not None:
            segment_text = f"{prompt_text_prefix} {segment_text}"

        prompt_text_suffix = kwargs.get("prompt_text_suffix", None)
        if prompt_text_suffix is not None:
            segment_text = f"{segment_text} {prompt_text_suffix}"

        kwargs["text_prompt"] = segment_text
        timeest = f"{estimated_time:.2f}"
        if estimated_time > 14 or estimated_time < 3:
            timeest = f"[bold red]{estimated_time:.2f}[/bold red]"

        current_iteration = (
            str(kwargs["current_iteration"]) if "current_iteration" in kwargs else ""
        )

        output_iterations = kwargs.get("output_iterations", "")
        iteration_text = ""
        if len(audio_segments) == 1:
            iteration_text = f"{current_iteration} of {output_iterations} iterations"

        segment_number = i + 1
        console.print(
            f"--Segment {segment_number}/{len(audio_segments)}: est. {timeest}s ({iteration_text})"
        )

        if not silent:
            print(f"{segment_text}")
        kwargs["segment_number"] = segment_number

        separate_prompts = kwargs.get("separate_prompts", False)
        separate_prompts_flipper = kwargs.get("separate_prompts_flipper", False)

        if separate_prompts_flipper is True:
            if separate_prompts is True:
                # nice to get actual generation from each speaker
                if history_prompt_flipper is True:
                    kwargs["history_prompt"] = None
                    history_prompt_for_next_segment = None
                    history_prompt_flipper = False
                    print(" <History prompt disabled for next segment.>")
                else:
                    kwargs["history_prompt"] = history_prompt_for_next_segment
                    history_prompt_flipper = True
            else:
                kwargs["history_prompt"] = history_prompt_for_next_segment

        else:
            if separate_prompts is True:
                history_prompt_for_next_segment = None
                print(" <History prompt disabled for next segment.>")
            else:
                kwargs["history_prompt"] = history_prompt_for_next_segment

        this_segment_start_time = time.time()

        full_generation, audio_arr = generate_audio_barki(text=segment_text, **kwargs)

        if full_generation is None or audio_arr is None:
            # Hmn, cancelling and restarting seems to be a bit buggy
            # let's try clearing out stuff
            kwargs = {}
            history_prompt_for_next_segment = None
            base_history = None
            full_generation = None
            print(" -----Bark Infinity Cancelled.>")
            return None, None, None, None

        if show_generation_times:
            this_segment_end_time = time.time()
            elapsed_time = this_segment_end_time - this_segment_start_time

            time_finished = f"Segment Finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(this_segment_end_time))}"
            time_taken = f"in {elapsed_time:.2f} seconds"
            print(f"  -->{time_finished} {time_taken}")

        if base_history is None:
            base_history = copy.deepcopy(full_generation)

        logger.debug(
            f"stable_mode_interval: {stable_mode_interval_counter} of {stable_mode_interval}"
        )

        if stable_mode_interval == 0:
            kwargs["previous_segment_type"] = "full_generation"
            history_prompt_for_next_segment = copy.deepcopy(full_generation)

        elif stable_mode_interval == 1:
            kwargs["previous_segment_type"] = "base_history"
            history_prompt_for_next_segment = copy.deepcopy(base_history)

        elif stable_mode_interval >= 2:
            if stable_mode_interval_counter == 1:
                # reset to base history
                stable_mode_interval_counter = stable_mode_interval
                kwargs["previous_segment_type"] = "base_history"
                history_prompt_for_next_segment = copy.deepcopy(base_history)
                logger.info(
                    f"resetting to base history_prompt, again in {stable_mode_interval} chunks"
                )
            else:
                stable_mode_interval_counter -= 1
                kwargs["previous_segment_type"] = "full_generation"
                history_prompt_for_next_segment = copy.deepcopy(full_generation)
        else:
            logger.error(
                f"stable_mode_interval is {stable_mode_interval} and something has gone wrong."
            )

            return None, None, None, None

        add_silence_between_segments = kwargs.get("add_silence_between_segments", 0.0)
        if add_silence_between_segments > 0.0:
            print(
                f"Adding {add_silence_between_segments} seconds of silence between segments."
            )
            # silence = np.zeros(int(add_silence_between_segments * SAMPLE_RATE))
            silence = np.zeros(
                int(add_silence_between_segments * SAMPLE_RATE), dtype=np.int16
            )
            audio_arr_segments.append(silence)
            yield np.concatenate([audio_arr, silence])
        else:
            yield audio_arr

        full_generation_segments.append(full_generation)
        audio_arr_segments.append(audio_arr)

    if show_generation_times or True:
        all_segments_end_time = time.time()
        elapsed_time = all_segments_end_time - all_segments_start_time

        time_finished = f"All Audio Sections Finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(all_segments_end_time))}"
        time_taken = f"in {elapsed_time:.2f} seconds"
        print(f"  -->{time_finished} {time_taken}")


def generate_audio(**kwargs):
    default_args = {
        "text_prompt": "I don't know what to say!",
        "negative_text_prompt": "",
        "history_prompt": None,
        "split_character_goal_length": 165,
        "split_character_max_length": 205,
        "split_character_jitter": 0.0,
        "in_groups_of_size": 1,
        "stable_mode_interval": 1,
        "text_splits_only": False,
        "separate_prompts": False,
        "separate_prompts_flipper": False,
        "semantic_top_k": 100,
        "semantic_top_p": 0.95,
        "coarse_top_k": 100,
        "coarse_top_p": 0.95,
        "semantic_use_mirostat_sampling": False,
        "semantic_mirostat_tau": 40.0,
        "semantic_mirostat_learning_rate": 0.75,
        "output_dir": "tts_output",
        "output_format": "wav",
        "text_temp": 0.7,
        "waveform_temp": 0.5,
        "semantic_min_eos_p": 0.2,
        "add_silence_between_segments": 0.0,
        "output_iterations": 1,
        "current_iteration": 1,
        "list_speakers": None,
        "audio_file_as_history_prompt": None,
        "prompt_file": None,
        "split_input_into_separate_prompts_by": None,
        "split_input_into_separate_prompts_by_value": None,
        "bark_speaker_as_the_prompt": None,
        "always_save_speaker": True,
        "output_filename": None,
        "extra_stats": False,
        "show_generation_times": False,
        "output_format_ffmpeg_parameters": None,
        "text_use_gpu": True,
        "text_use_small": False,
        "coarse_use_gpu": True,
        "coarse_use_small": True,
        "fine_use_gpu": True,
        "fine_use_small": False,
        "codec_use_gpu": True,
        "force_reload": False,
        "GLOBAL_ENABLE_MPS": None,
        "USE_SMALL_MODELS": None,
        "SUNO_USE_DIRECTML": False,
        "OFFLOAD_CPU": True,
        "silent": False,
        "seed": None,
        "single_starting_seed": None,
        "process_text_by_each": None,
        "group_text_by_counting": None,
        "split_type_string": None,
        "prompt_text_prefix": None,
        "prompt_text_suffix": None,
        "extra_confused_travolta_mode": None,
        "use_smaller_models": False,
        "semantic_temp": 0.5,
        "semantic_max_gen_duration_s": None,
        "semantic_allow_early_stop": True,
        "semantic_use_kv_caching": True,
        "semantic_seed": None,
        "semantic_history_oversize_limit": None,
        "coarse_temp": 0.5,
        "coarse_max_coarse_history": 630,
        "coarse_sliding_window_len": 60,
        "coarse_kv_caching": True,
        "coarse_seed": None,
        "x_coarse_history_alignment_hack": -2,
        "fine_temp": 0.3,
        "fine_seed": None,
        "render_npz_samples": False,
        "absolute_semantic_history_only": False,
        "absolute_semantic_history_only_every_x": None,
        "history_prompt_string": None,
        "previous_segment_type": "base_history",
        "output_full": True,
        "total_segments": 1,
        "segment_number": "final",
        "loglevel": "WARNING",
    }

    for k, v in default_args.items():
        if k not in kwargs:
            kwargs[k] = v

    print("Loading Bark models...")
    generation.OFFLOAD_CPU = kwargs.get("OFFLOAD_CPU", True)
    generation.USE_SMALL_MODELS = kwargs.get("USE_SMALL_MODELS", None)
    generation.preload_models(
        kwargs["text_use_gpu"],
        kwargs["text_use_small"],
        kwargs["coarse_use_gpu"],
        kwargs["coarse_use_small"],
        kwargs["fine_use_gpu"],
        kwargs["fine_use_small"],
        kwargs["codec_use_gpu"],
        kwargs["force_reload"],
    )

    # for audio_arr in generate_audio_long(**kwargs):
    #     yield audio_arr, SAMPLE_RATE

    full_audio_arr = np.concatenate([arr for arr in generate_audio_long(**kwargs)])
    return full_audio_arr, SAMPLE_RATE
