from typing import Tuple
import pickle
import os
import string
import numpy as np


def read_text(data_path: str):
    """
    Read data.
    """
    output = []

    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
        for line in lines:
            input_doc, target_doc = line.split('\t')[:2]
            output.append([input_doc, target_doc])

    return output


def preprocess_data(dat) -> np.ndarray:
    """
    Preprocess data.
    """
    # Translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)

    for pair in dat:

        pair[0] = pair[0].lower()
        pair[0] = pair[0].translate(table)
        pair[0] = [word for word in pair[0].split() if word.isalpha()]
        pair[0] = ' '.join(pair[0])

        pair[1] = pair[1].lower()
        pair[1] = pair[1].translate(table)
        pair[1] = [word for word in pair[1].split() if word.isalpha()]
        pair[1] = ' '.join(pair[1])

    return np.array(dat)


def save_preprocessed_data(preprocessed_dat: np.ndarray, filename: str, directory='data') -> None:
    """
    Save cleaned data into a file.
    """
    filepath = os.path.join(directory, filename)

    # Serialize and save the data to the specified filepath
    pickle.dump(preprocessed_dat, open(filepath, 'wb'))
    print(f"Preprocessed data saved in {filename}")


def load_preprocessed_data(filepath: str) -> np.ndarray:
    """
    Load data from the pickle file
    """
    return pickle.load(open(filepath, 'rb'))


def train_test_split(dat: np.ndarray, test_ratio=0.2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split data into the training set and the testing set.
    """

    split_index = int(len(dat) * (1 - test_ratio))
    train_data, test_data = dat[:split_index], dat[split_index:]

    return train_data, test_data


if __name__ == '__main__':
    data = "data/bel.txt"
    docs = read_text(data)

    preprocess_docs = preprocess_data(docs)
    print(f"Number of phrase pairs: {len(preprocess_docs)}")

    # Save preprocessed dataset
    save_preprocessed_data(preprocess_docs, 'english-belarusian.pkl')

    # Load preprocessed dataset
    raw_dataset = load_preprocessed_data("data/english-belarusian.pkl")

    # Reduce dataset size
    n_pairs = 3000
    dataset = raw_dataset[:n_pairs, :]
    print(f"Number of phrase pairs in the reduced dataset: {len(dataset)}")

    # Shuffle the dataset
    np.random.seed(42)
    np.random.shuffle(dataset)
    print(f"Shuffled dataset: {dataset}")

    # Split the shuffled dataset into train and test sets
    train, test = train_test_split(dataset, test_ratio=0.1)
    save_preprocessed_data(dataset, 'english-belarusian-both.pkl')
    save_preprocessed_data(train, 'english-belarusian-train.pkl')
    save_preprocessed_data(test, 'english-belarusian-test.pkl')
