import re
import nltk


def split_long_string(input_string, chunk_length):
    words = input_string.split()  # Split the input string into words
    chunks = []
    current_chunk = ""

    for word in words:
        if len(current_chunk) + len(word) <= chunk_length:
            current_chunk += word + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = word + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def split_text_into_chunks(text, max_chunk_words=25):
    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)

    chunks = list()
    word_count = list()

    chunks.append(sentences[0])
    word_count.append(len(sentences[0].split()))

    if len(sentences) > 1:
        for sentence in sentences[1:]:
            new_word_count = len(sentence.split())

            if word_count[-1] + new_word_count <= max_chunk_words:
                chunks[-1] = chunks[-1] + " " + sentence
                word_count[-1] += new_word_count
            else:
                chunks.append(sentence)
                word_count.append(new_word_count)

    return chunks


def remove_italics(text):
    # Regular expression pattern to match italicized text
    pattern = r"\*{1,2}(.*?)\*{1,2}"

    # Replace italicized text with an empty string
    result = re.sub(pattern, r"", text)

    # Trim double whitespaces
    result = re.sub(r"\s+", " ", result)

    return result
