import tensorflow as tf


def create_dataset(df):
    word_embedding = tf.ragged.stack(list(df['word_embedding']))
    ipa_embedding = tf.ragged.stack(list(df['ipa_embedding']))
    dataset = tf.data.Dataset.from_tensor_slices((word_embedding, ipa_embedding))
    return dataset


def split_dataset(ds, train_size=0.8, validation_size=0.1, test_size=0.1):
    dataset_size = ds.cardinality().numpy()
    train_size = int(train_size * dataset_size)
    validation_size = int(validation_size * dataset_size)
    test_size = int(test_size * dataset_size)

    ds = ds.shuffle(dataset_size)   # Needs to perfectly shuffle all the data - hence needs to be dataset_size
    train_ds = ds.take(train_size)
    test_ds = ds.skip(train_size)
    validation_ds = test_ds.skip(validation_size)
    test_ds = test_ds.take(test_size)

    return train_ds, validation_ds, test_ds


def to_tensor(encoded_word, encoded_ipa):
    return encoded_word.to_tensor(), encoded_ipa.to_tensor()


# Create batches
def make_batches(ds, buffer_size, batch_size):
    return (
        ds
        .cache()
        .shuffle(buffer_size)
        .batch(batch_size)
        .map(to_tensor, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )


def save_dataset(ds, path="ff_teamname_generator/pronunciation_model/data/ipa_dict_processed"):
    tf.data.experimental.save(ds, path)


def load_dataset(ds_path="ff_teamname_generator/pronunciation_model/data/", batch_size=1000):
    ds = tf.data.experimental.load(ds_path + "ipa_dict_processed_test")
    return make_batches(ds, batch_size, 64)
