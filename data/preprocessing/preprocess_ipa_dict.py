import pandas as pd
from data.preprocessing.tokeniser import tokenise
from data.embeddings.embedding import Embedding


def preprocess(data='data/datasets/raw_data/ipa_dict.txt'):
    """
    Preprocess the IPA dictionary dataset and return a preprocessed pandas dataframe of pronunciations and embeddings 
    for the words and IPA pronunciations.
    """
    
    # Read in the pronunciation data
    df = pd.read_csv(data, sep="\t", header=None, names=['word', 'ipa'], keep_default_na=False, na_values=[])

    # Make word column lower case - as contains I, which is not the same as i.
    df.word = df.word.str.lower()

    # Remove start / (first character) and end / (last character) from ipa
    df.ipa = df.ipa.str[1:-1]

    # Remove entries with an '
    df = df[~df.word.apply(lambda x: "'" in x)]

    # Remove zero width joiners and update the g and dʒ to g and ʤ.
    df.ipa = df.ipa.apply(lambda x: x.replace('\u200d', ''))
    df.ipa = df.ipa.apply(lambda x: x.replace('ɡ', 'g'))
    df.ipa = df.ipa.apply(lambda x: x.replace('dʒ', 'ʤ'))

    # Remove entries with an "ʲ", " ", "x", "'̃", "ɬ", "r" - as they are all infrequent < 20 occurences in dataset.
    matches = ['ʲ', ' ', 'x', '̃', 'ɬ', 'r']
    df = df[~df.ipa.apply(lambda x: any(match in x for match in matches))]

    # Get unique word tokens
    df['word_tokens'] = df.word.apply(lambda x: [char for char in x])
    unique_chars = sorted(list(set(df.word.sum())))

    # Get unique ipa tokens by first tokenizing and then getting all unique items
    df['ipa_tokens'] = df.ipa.apply(lambda x: tokenise(x))
    all_tokens = df.ipa_tokens.tolist()
    flat_all_tokens = [item for token in all_tokens for item in token]
    unique_tokens = sorted(list(set(flat_all_tokens)))

    # Create embeddings for words and ipa representations
    word_embedding = Embedding(unique_chars)
    ipa_embedding = Embedding(unique_tokens)

    # Calculate word embeddings and ipa embeddings for each entry in the dataset
    df['word_embedding'] = df.word_tokens.apply(lambda x: word_embedding.encode(x))
    df['ipa_embedding'] = df.ipa_tokens.apply(lambda x: ipa_embedding.encode(x))

    return df, word_embedding, ipa_embedding


if __name__ == '__main__':
    df, chars, ipa_tokens = preprocess()
    print(df.head())
