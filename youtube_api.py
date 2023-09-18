# -*- coding: utf-8 -*-

# Sample Python code for youtube.liveBroadcasts.list
# See instructions for running these code samples locally:
# https://developers.google.com/explorer-help/code-samples#python

import os
import argparse

import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
from google.oauth2.credentials import Credentials

API_SERVICE_NAME = "youtube"
API_VERSION = "v3"

SCOPES = [
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.force-ssl",
]

# Disable OAuthlib's HTTPS verification when running locally.
# *DO NOT* leave this option enabled in production.
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"


def authorize_and_save_credentials(client_secrets_file, credentials_save_path):
    if not os.path.exists(client_secrets_file):
        raise ValueError(
            f"Client secret file not found at {client_secrets_file}. Please download the OAuth 2.0 Client IDs JSON on https://console.cloud.google.com/apis/credentials."
        )
    # Get credentials
    flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
        client_secrets_file, SCOPES
    )
    credentials = flow.run_console()
    with open(credentials_save_path, "w") as f:
        f.write(credentials.to_json())


def read_credentials(authorized_user_file):
    if not os.path.exists(authorized_user_file):
        raise ValueError(
            f"The credentials for the stream owner cannot be found at {authorized_user_file}. Please authorize first."
        )
    return Credentials.from_authorized_user_file(authorized_user_file)


def get_live_broadcast(credentials, verbose=0):
    # Create an API client for owner
    owner_youtube = googleapiclient.discovery.build(
        API_SERVICE_NAME, API_VERSION, credentials=read_credentials(credentials)
    )

    request = owner_youtube.liveBroadcasts().list(part="snippet", mine=True)
    response = request.execute()

    latest_chat = response["items"][0]
    broadcast_title = latest_chat["snippet"]["title"]
    live_chat_id = latest_chat["snippet"]["liveChatId"]
    broadcast_url = f"https://www.youtube.com/watch?v={latest_chat['id']}"

    if latest_chat and live_chat_id:
        if verbose > 0:
            print(f"Stream Found: {broadcast_title}")
            print(f"Chat ID Found: {live_chat_id}")
            print(f"Stream URL: {broadcast_url}")
        return broadcast_title, live_chat_id, broadcast_url
    else:
        raise ValueError("Cannot find any live streams.")


def insert_message_to_live_chat(message_text, credentials, live_chat_id, verbose=1):
    # Create an API client for bot
    bot_youtube = googleapiclient.discovery.build(
        API_SERVICE_NAME, API_VERSION, credentials=read_credentials(credentials)
    )
    request = bot_youtube.liveChatMessages().insert(
        part="snippet",
        alt="json",
        body={
            "snippet": {
                "liveChatId": live_chat_id,
                "type": "textMessageEvent",
                "textMessageDetails": {"messageText": message_text},
            }
        },
    )
    response = request.execute()
    if verbose > 0:
        print(f"Successfully inserted message: {message_text}")


def test():
    broadcast_title, live_chat_id, broadcast_url = get_live_broadcast(
        "secrets/owner_credentials.json", verbose=1
    )
    insert_message_to_live_chat(
        "This is a test message!",
        "secrets/bot_credentials.json",
        live_chat_id,
        verbose=1,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="Sends a test message to the current live stream. Uses secrets/owner_credentials.json and secrets/bot_credentials.json as owner and bot credential files.",
    )
    parser.add_argument(
        "-a",
        "--authorize",
        type=str,
        default="secrets/saved_credentials.json",
        help="Authorize an account and save credentials to the specified location. Default: secrets/saved_credentials.json",
    )
    args = parser.parse_args()

    if args.authorize:
        authorize_and_save_credentials("secrets/client_secret.json", args.authorize)
    if args.test:
        test()
