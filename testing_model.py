from preprocesing_data import load_preprocessed_data
from training_model import create_tokenizer, get_max_seq_length, encode_sequences
from keras.models import load_model
import numpy as np
import pandas as pd


def get_word_for_id(integer, tokenizer):
    """
    Maps an integer into a word.
    """

    reversed_dict = {index: word for word, index in tokenizer.word_index.items()}
    if integer in reversed_dict:
        return reversed_dict[integer]


def predict_sequence(model, tokenizer, source):
    """
    Performs the mapping for each integer in the translation
    and returns the result as a string of words.
    """
    translation = model.predict(source, verbose=0)[0]
    integers = [np.argmax(vector) for vector in translation]

    target = list()
    for i in integers:
        word = get_word_for_id(i, tokenizer)
        if word is None:
            break
        target.append(word)

    return ' '.join(target)


if __name__ == '__main__':
    # Load datasets
    dataset = load_preprocessed_data("data/english-belarusian-both.pkl")
    train = load_preprocessed_data("data/english-belarusian-train.pkl")
    test = load_preprocessed_data("data/english-belarusian-test.pkl")

    eng_docs = dataset[:, 0]
    by_docs = dataset[:, 1]

    # Generate a tokenizer for English
    eng_tokenizer = create_tokenizer(eng_docs)
    eng_vocab_size = len(eng_tokenizer.word_index) + 1
    max_eng_seq_length = get_max_seq_length(eng_docs)

    # Generate a tokenizer for Belarusian
    by_tokenizer = create_tokenizer(by_docs)
    by_vocab_size = len(by_tokenizer.word_index) + 1
    max_by_seq_length = get_max_seq_length(by_docs)

    # Prepare source data
    train_X = encode_sequences(eng_tokenizer, max_eng_seq_length, train[:, 0])
    test_X = encode_sequences(eng_tokenizer, max_eng_seq_length, test[:, 0])

    # Load model
    model = load_model("model.h5")

    # Generate predictions
    for i, source in enumerate(test_X):
        source = source.reshape((1, source.shape[0]))
        translation = predict_sequence(model, by_tokenizer, source)
        print(translation)