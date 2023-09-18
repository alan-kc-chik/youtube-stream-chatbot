from config import (
    TRIGGER_TOKEN,
    TEXT_GENERATION_MODEL_PATH,
    POLLING_TIME,
    AUDIO_PLAYBACK_DEVICE,
    RVC_CHECKPOINT_PATH,
    RVC_PARAMS,
    BARK_SPEAKER_PROMPT,
    TEXT_GENERATION_PROMPT,
    STREAM_OWNER_CREDENTIALS_FILE,
    CHATBOT_CREDENTIALS_FILE,
    CLEAR_HISTORY_TRIGGER,
    XDG_CACHE_HOME,
)

import os
os.environ["XDG_CACHE_HOME"] = XDG_CACHE_HOME

import time
import argparse

import torch

# Text generation
from generate_text import generate_text
import demoji
import html
from util import remove_italics, split_long_string

# Interact with YouTube
import get_chat
import youtube_api

# Text to speech
import generate_audio_infinity

# Voice conversion
import rvc_inference

# Audio playback
import threading
import queue
import audio_playback


audio_playback_queue = queue.Queue()


def audio_playback_thread_function():
    # https://stackoverflow.com/questions/61200617/how-to-play-different-sound-consecutively-and-asynchronously
    while True:
        sound = audio_playback_queue.get()
        if sound is None:
            break
        audio_playback.playback_audio(
            **sound
        )  # This blocks the thread until the audio is finished playing
        audio_playback_queue.task_done()  # This must be called, otherwise the queue never unblocks when queue.join() is called


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="YouTube Chatbot",
        description="YouTube Chatbot",
    )

    parser.add_argument(
        "-u",
        "--url",
        help="Optional: URL of the live stream. Required if owner credentials are not found in the location specified in config.py file.",
    )
    args = parser.parse_args()

    live_chat_id = None
    if not os.path.exists(STREAM_OWNER_CREDENTIALS_FILE):
        if not args.url:
            url = args.url
        else:
            raise ValueError(
                "Must specify --url STREAM_URL if owner credentials are not provided."
            )
    else:
        broadcast_title, live_chat_id, url = youtube_api.get_live_broadcast(
            STREAM_OWNER_CREDENTIALS_FILE, verbose=0
        )
        print(f"Broadcast title: {broadcast_title}")
        print(f"Live chat ID: {live_chat_id}")
        print(f"Broadcast URL: {url}")
        if input("Continue with script? (y/n)") != "y":
            exit(0)

    print(f"Start reading chat messages.")
    start_time = time.monotonic()
    timestamp = 0
    audio_playback_thread = threading.Thread(
        target=audio_playback_thread_function, daemon=True
    )
    audio_playback_thread.start()

    try:
        while True:
            # timestamp updates after every call
            messages, timestamp = get_chat.get_chat_messages(
                url, from_timestamp=timestamp
            )

            if len(messages) > 0:
                # Trigger
                clear_trigger = [
                    x for x in messages if x["message"] == CLEAR_HISTORY_TRIGGER
                ]
                if len(clear_trigger) > 0:
                    print("Restarting...")
                    # post_message.post_message("[BOT] OK, let's start again.")
                    continue

                candidate_messages = [
                    x for x in messages if x["message"].startswith(TRIGGER_TOKEN)
                ]

                if len(candidate_messages) > 0:
                    chosen_input_author, chosen_input_text = get_chat.choose_message(
                        candidate_messages
                    )
                    chosen_input_text = chosen_input_text.replace(
                        TRIGGER_TOKEN, ""
                    ).strip()

                    print(f"Chosen input: {chosen_input_text}")

                    # Construct prompt
                    prompt = TEXT_GENERATION_PROMPT
                    prompt = prompt.replace("{prompt}", chosen_input_text)

                    # Start text generation process
                    text_generation_start_time = time.time()

                    result_message = generate_text(prompt, TEXT_GENERATION_MODEL_PATH)
                    cleaned_message = result_message.strip()
                    cleaned_message = demoji.replace(cleaned_message, "")
                    cleaned_message = remove_italics(cleaned_message)
                    cleaned_message = html.unescape(cleaned_message)

                    # Free up most of the GPU memory
                    del result_message
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    print(f"Received response: {cleaned_message}")
                    print(
                        f"Text generation time taken: {time.time() - text_generation_start_time: .2f} s"
                    )

                    # Start speech generation process
                    speech_generation_start_time = time.time()
                    audio_array, sample_rate = generate_audio_infinity.generate_audio(
                        text_prompt=cleaned_message,
                        history_prompt=BARK_SPEAKER_PROMPT,
                        add_silence_between_segments=0.2,
                        coarse_temp=0.5,
                    )
                    print(
                        f"Speech generation time taken: {time.time() - speech_generation_start_time: .2f} s"
                    )

                    # Voice conversion process
                    voice_conversion_start_time = time.time()
                    audio_array, sample_rate = rvc_inference.voice_conversion(
                        input_audio=audio_array,
                        input_sample_rate=sample_rate,
                        checkpoint_path=RVC_CHECKPOINT_PATH,
                        format="numpy",
                        rms_mix_rate=0.75,  # Adjust the volume envelope scaling
                        **RVC_PARAMS,
                    )
                    print(
                        f"Voice conversion time taken: {time.time() - voice_conversion_start_time: .2f} s"
                    )

                    audio = {
                        "audio_array": audio_array,
                        "sample_rate": sample_rate,
                        "sound_output_device": AUDIO_PLAYBACK_DEVICE,
                        "verbose": True,
                    }

                    # Wait for queue to finish before processing more messages
                    # otherwise the audio will lag too far behind the current messages
                    audio_playback_queue.join()
                    # Queue the audio for playback and use this time to continue processing the next batch of messages
                    audio_playback_queue.put(audio)

                    # Post text message to YouTube Chat
                    if live_chat_id is not None:
                        if not os.path.exists(CHATBOT_CREDENTIALS_FILE):
                            raise ValueError(
                                f"{CHATBOT_CREDENTIALS_FILE} not found. Credentials required to post messages to live chat. Run python youtube_api.py --authorize to create credentials."
                            )
                        else:
                            live_chat_message_to_insert = (
                                f"[BOT] @{chosen_input_author} {cleaned_message}"
                            )
                            message_blocks = split_long_string(
                                live_chat_message_to_insert, chunk_length=200
                            )
                            for message in message_blocks:
                                youtube_api.insert_message_to_live_chat(
                                    message,
                                    CHATBOT_CREDENTIALS_FILE,
                                    live_chat_id=live_chat_id,
                                    verbose=1,
                                )
                                time.sleep(1.0)

            time.sleep(POLLING_TIME - ((time.monotonic() - start_time) % POLLING_TIME))

    except KeyboardInterrupt:
        print("stopped!")
        for _ in range(audio_playback_queue.qsize()):
            audio_playback_queue.task_done()
        audio_playback_queue.join()
        audio_playback_thread.join(timeout=1)
