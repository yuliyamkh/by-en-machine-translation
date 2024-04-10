from preprocesing_data import load_preprocessed_data
import numpy as np
import keras.src.preprocessing.text
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, RepeatVector
from keras.utils import to_categorical, plot_model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences


def create_tokenizer(lines: np.ndarray) -> keras.src.preprocessing.text.Tokenizer:
    """
    Translate tokens into integers, i.e., unique indices.
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)

    return tokenizer


def get_max_seq_length(lines):
    """
    Get maximum sequence length.
    """
    return max(len(line.split()) for line in lines)


def encode_sequences(tokenizer: keras.src.preprocessing.
                     text.Tokenizer, length: int, lines: np.ndarray):
    """
    Encodes and pads sequences.
    """

    # Encode text to sequences of integers
    seqs = tokenizer.texts_to_sequences(lines)
    # Pad sequences with 0 values
    seqs = pad_sequences(seqs, maxlen=length, padding='post')

    return seqs


def encode_output(sequences, vocab_size):
    """
    One-hot encodes target sequence.
    """
    y_list = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        y_list.append(encoded)

    y = np.array(y_list)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y


# Build Neural Machine Translation Model
def define_model(input_vocab, output_vocab,
                 in_length, out_length,
                 units):

    model = Sequential()
    model.add(Embedding(input_vocab, units, input_length=in_length, mask_zero=True))
    model.add(LSTM(units))
    model.add(RepeatVector(out_length))
    model.add(LSTM(units, return_sequences=True))
    model.add(Dense(output_vocab, activation='softmax'))

    return model


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
    print(f"English Vocabulary Size: {eng_vocab_size}")
    print(f"English Maximum Sequence Length: {max_eng_seq_length}")

    # Generate a tokenizer for Belarusian
    by_tokenizer = create_tokenizer(by_docs)
    by_vocab_size = len(by_tokenizer.word_index) + 1
    max_by_seq_length = get_max_seq_length(by_docs)
    print(f"Belarusian Vocabulary Size: {by_vocab_size}")
    print(f"Belarusian Maximum Sequence Length: {max_by_seq_length}")

    # Prepare training data
    train_X = encode_sequences(eng_tokenizer, max_eng_seq_length, train[:, 0])
    train_y = encode_sequences(by_tokenizer, max_by_seq_length, train[:, 1])
    train_y = encode_output(train_y, by_vocab_size)

    # Prepare validation data
    test_X = encode_sequences(eng_tokenizer, max_eng_seq_length, test[:, 0])
    test_y = encode_sequences(by_tokenizer, max_by_seq_length, test[:, 1])
    test_y = encode_output(test_y, by_vocab_size)

    # Compile model
    model = define_model(eng_vocab_size, by_vocab_size, max_eng_seq_length, max_by_seq_length, units=25)

    model.compile(optimizer='adam', loss='categorical_crossentropy')

    # Summarize defined model
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)

    # Fit model
    filename = 'model.h5'
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    model.fit(train_X, train_y, epochs=1000, batch_size=3, validation_split=0.1, validation_data=(test_X, test_y), callbacks=[checkpoint], verbose=2, shuffle=True)
