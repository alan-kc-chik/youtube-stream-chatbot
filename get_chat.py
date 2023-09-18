from chat_downloader import ChatDownloader
import random

def get_chat_messages(url, from_timestamp=0):
    chat = ChatDownloader().get_chat(url, inactivity_timeout=1)
    message_list = [message for message in chat]
    message_list_sorted = sorted(
        message_list, key=lambda x: x['timestamp'], reverse=True)
    messages = [
        x for x in message_list_sorted if x['timestamp'] > from_timestamp]

    if messages == None:
        messages = []
        return messages, from_timestamp

    return messages, message_list_sorted[0]['timestamp']


def choose_message(messages):
    candidate_author_ids = [x['author']['id'] for x in messages]

    chosen_input_author_id = random.choice(candidate_author_ids)
    messages_from_chosen_author = [
        x for x in messages if x['author']['id'] == chosen_input_author_id]
    chosen_input = random.choice(messages_from_chosen_author)

    chosen_input_text = chosen_input['message']
    chosen_input_author = chosen_input['author']['name']

    return chosen_input_author, chosen_input_text
