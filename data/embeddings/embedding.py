import numpy as np
import tensorflow as tf
from ipatok import tokenise
import pickle


def default_tokenizer(input):
    return [el for el in input]


def all_tokens(tokens):
    tokens.insert(0, 'PAD')
    tokens.insert(1, 'START')
    tokens.insert(2, 'END')
    return tokens


def load_embedding(path, tokenizer=None):
    with open(path, "rb") as fp:
       tokens = pickle.load(fp)
    return Embedding(tokens, tokenizer=tokenizer)


class Embedding:
    def __init__(self, tokens, tokenizer=None):
        if 'PAD' in tokens and 'START' in tokens and 'END' in tokens:
            self.all_tokens = tokens
        else:
            self.all_tokens = all_tokens(tokens)
        
        self.size = len(self.all_tokens) - 1

        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = default_tokenizer

    def tokenize(self, sequence):
        tokenized_sequence = self.tokenizer(sequence)

        # Add start and end tokens to start and end of sequence
        tokenized_sequence.insert(0, 'START')
        tokenized_sequence.append('END')

        return tokenized_sequence

    def detokenize(self, tokens):
        # Find first occurrence of START token and start from this.
        start_index = tokens.index('START') + 1

        # Find first occurrence of END token and cut to this.
        end_index = tokens.index('END')
        
        return ''.join(tokens[start_index:end_index])

    def encode(self, sequence):
        if isinstance(sequence, tf.Tensor):
            # Split sequence
            is_tensor = True
            sequence = sequence.numpy().decode("utf-8")
        else:
            is_tensor = False

        # Tokenize sequence
        sequence_tokenized = self.tokenize(sequence)
        encoded_sequence = []
        for token in sequence_tokenized:
            encoded_sequence.append(self.all_tokens.index(token))

        # Convert back to tensor
        if is_tensor:
            return tf.convert_to_tensor(encoded_sequence)
        else:
            return np.asarray(encoded_sequence) 

    def decode(self, encoded_sequence, human_readable=False):
        if isinstance(encoded_sequence, tf.Tensor):
            # Split sequence
            is_tensor = True
            encoded_sequence = encoded_sequence.numpy()
        else:
            is_tensor = False

        decoded = []
        for char in encoded_sequence:
            decoded.append(self.all_tokens[char])
        
        detokenized = self.detokenize(decoded)

        # Convert back to tensor
        if is_tensor and not human_readable:
            return tf.convert_to_tensor(detokenized)
        else:        
            return detokenized

    def batch_encode(self, batch_sequence):
        batch_encoded = []
        max_len = 0
        for sequence in batch_sequence:
            encoded = self.encode(sequence)
            encoded_len = len(encoded)
            batch_encoded.append(encoded)
            if encoded_len > max_len:
                max_len = encoded_len

        # Pad sequences of different lengths to ensure each array is the same length
        batch_encoded_padded = []
        pad_token = self.all_tokens.index('PAD')
        for sequence in batch_encoded:
            seq_len = len(sequence)
            if seq_len < max_len:
                sequence = np.pad(sequence, (0, max_len - seq_len), mode='constant', constant_values=pad_token)
            batch_encoded_padded.append(sequence)

        return np.asarray(batch_encoded_padded)

    def batch_decode(self, batch_encoded_sequence, human_readable=False):
        batch_decoded = []
        for sequence in batch_encoded_sequence:
            batch_decoded.append(self.decode(sequence, human_readable=human_readable))
        return batch_decoded

    def save(self, path):
        with open(path, "wb") as fp:
            pickle.dump(self.all_tokens, fp)


if __name__ == '__main__':
    embedding = Embedding(['a', 'b', 'c', 'd'], tokenizer=tokenise)
    word = tf.constant('bad')

    # Embedding
    word_embedding = embedding.encode(word)
    print('')
    print('Embedding for bad:')
    print(word_embedding)
    print('')
    print('Decoded embedding for bad:')
    print(embedding.decode(word_embedding))
