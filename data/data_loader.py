from data.embeddings.embedding import load_embedding
from data.dataset_utils import create_dataset, split_dataset, save_dataset, make_batches
from data.preprocessing.preprocess_ipa_dict import preprocess
from os.path import exists
import tensorflow as tf


def load_data(ds_path="data/datasets/processed_data/", embedding_path="data/embeddings/saved_embeddings/", batch_size=1000):
    # Load or preprocess data if data exists
    if exists(ds_path + "ipa_dict_processed_train"):
        train_ds = tf.data.experimental.load(ds_path + "ipa_dict_processed_train")
        val_ds = tf.data.experimental.load(ds_path + "ipa_dict_processed_val")
        test_ds = tf.data.experimental.load(ds_path + "ipa_dict_processed_test")
        word_embedding = load_embedding(embedding_path + "word_embedding")
        ipa_embedding = load_embedding(embedding_path + "ipa_embedding")
    else:
        # Get data
        df, word_embedding, ipa_embedding = preprocess()
        ds = create_dataset(df)

        # Train/test split
        train_ds, val_ds, test_ds = split_dataset(ds)

        # Save for future use
        save_dataset(train_ds, path=ds_path + "ipa_dict_processed_train")
        save_dataset(val_ds, path=ds_path + "ipa_dict_processed_val")
        save_dataset(test_ds, path=ds_path + "ipa_dict_processed_test")
        word_embedding.save(embedding_path + "word_embedding")
        ipa_embedding.save(embedding_path + "ipa_embedding")

    train_batches = make_batches(train_ds, batch_size, 64)
    val_batches = make_batches(val_ds, batch_size, 64)
    test_batches = make_batches(test_ds, batch_size, 64)

    return (train_batches, val_batches, test_batches, word_embedding, ipa_embedding)