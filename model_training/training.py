# from data.data_loader import load_data
from data.data_loader import load_data
from model.transformer import Transformer
from model.pronouncer import Pronouncer, SavePronouncer, SavePronouncerBeam
from model.training_utils import CustomSchedule, loss_function, accuracy_function
from model_evaluation.test import evaluate_model
import tensorflow as tf
import time


def train(hyperparameters={"epochs": 10, "batch_size": 1000, "num_layers": 4, "d_model": 128, "num_heads": 8, "dff": 512, 
        "pe_input": 1000, "pe_target": 1000, "position_encoding_denominator": 10000, "dropout_rate": 0.1}, 
        ds="data/processed_data/", 
        save_path="saved_model_outputs/saved_models/pronouncer", 
        checkpoint_name="pronounce",
        checkpoint_load=True,
        checkpoint_save=True,
        evaluation=True
        ):

    """
    Trains a transformer model using the dataset and embeddings inputted and saves the model.
    """
    
    if isinstance(ds, str):
        (train_batches, val_batches, _, word_embedding, ipa_embedding) = load_data(ds_path=ds, batch_size=hyperparameters["batch_size"])
    else:
        (train_batches, val_batches, _, word_embedding, ipa_embedding) = ds

    # Hyperparameters
    epochs = hyperparameters["epochs"]  # Epochs to run
    num_layers = hyperparameters["num_layers"]  # Number of layers in encoder/decoder
    d_model = hyperparameters["d_model"]    # Dimension of model
    num_heads = hyperparameters["num_heads"]    # Number of heads to transformer
    dff = hyperparameters["dff"]    # Dimension of hidden layer in feedforward network
    input_vocab_size = word_embedding.size  # Size of the max vocabulary input - i.e., largest integer in input
    target_vocab_size = ipa_embedding.size   # Size of the max vocabulary target - i.e., largest integer in target
    pe_input = hyperparameters["pe_input"]  # Positional encoding of input
    pe_target = hyperparameters["pe_target"]    # Positional encoding of target
    position_encoding_denominator = hyperparameters["position_encoding_denominator"]    # Positional encoding denominator
    dropout_rate = hyperparameters["dropout_rate"]  # Dropout rate
    beta_1 = hyperparameters["beta_1"]  # Adam optimisation, beta 1
    beta_2 = hyperparameters["beta_2"]  # Adam optimisation, beta 2
    epsilon = hyperparameters["epsilon"]    # Adam optimisation, epsilon

    # Custom learning rate
    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

    # Calculate loss - need to apply padding as sequences are padded so don't want to factor these into the loss calculation
    # Changed from_logits to False - since we now have a softmax layer to output probabilities.
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

    # Define training loss and accuracy
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    # Create transformer
    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        pe_input=pe_input,
        pe_target=pe_target,
        position_encoding_denominator=position_encoding_denominator,
        rate=dropout_rate)

    # Saving checkpoints
    if checkpoint_save or checkpoint_load:
        checkpoint_path = "saved_model_outputs/checkpoints/" + checkpoint_name + "/train"

        ckpt = tf.train.Checkpoint(transformer=transformer,
                                optimizer=optimizer)

        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # Loading checkpoints - if a checkpoint exists, restore the latest checkpoint.
    if checkpoint_load:
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

    # Training step
    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        # Get target input and real for loss 
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        with tf.GradientTape() as tape:
            # Get predictions and calculate loss
            predictions, _ = transformer([inp, tar_inp],
                                        training = True)
            loss = loss_function(tar_real, predictions, loss_object)

        # Update model weights
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        # Save loss and accuracy metrics
        train_loss(loss)
        train_accuracy(accuracy_function(tar_real, predictions))

    # Training
    print('')
    print("================ Training ================")
    train_start_time = time.time()
    for epoch in range(epochs):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> word, tar -> ipa
        for (batch, (inp, tar)) in enumerate(train_batches):
            train_step(inp, tar)
        
            # Print training update every 100 batches
            if batch % 100 == 0:
                print('Epoch {0} Batch {1} Loss {2} Accuracy {3}'.format(epoch + 1, batch, train_loss.result(), train_accuracy.result()))
            
        # Save checkpoint every 5th epoch
        if (epoch + 1) % 5 == 0:
            if checkpoint_save:
                ckpt_save_path = ckpt_manager.save()
                print('Saving checkpoint for epoch {0} at {1}'.format(epoch + 1, ckpt_save_path))

        # Print training results for epoch
        print('Epoch {0} Loss {1} Accuracy {2}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))

        # Print time taken for epoch
        print(f'Time taken for epoch {epoch + 1}: {time.time() - start:.2f} secs\n')

    total_train_time = time.time() - train_start_time

    print("================ Training Summary ================")
    print(f'Total training time: {total_train_time:.2f} secs')
    print('Training accuracy: {0}'.format(train_accuracy.result().numpy()))
    print('')

    if save_path:
        save_transformer(transformer, word_embedding, ipa_embedding, save_path=save_path)

    if evaluation:
        pronouncer = Pronouncer(word_embedding, ipa_embedding, transformer)
        val_accuracy = evaluate_model(pronouncer, val_batches)
    else:
        val_accuracy = None

    return total_train_time, train_accuracy.result().numpy(), val_accuracy


def save_transformer(transformer, word_embedding, ipa_embedding, save_path="saved_model_outputs/saved_models/pronouncer"):
    pronouncer = Pronouncer(word_embedding, ipa_embedding, transformer)

    # Save a standard model
    export = SavePronouncer(pronouncer)
    tf.saved_model.save(export, export_dir=save_path + "/standard/")

    # Save a beam search model
    export_beam = SavePronouncerBeam(pronouncer)
    tf.saved_model.save(export_beam, export_dir=save_path + "/beam/")


if __name__ == "__main__":
    ds = load_data()

    # Set hyperparameters
    hyperparams = {"epochs": 10, "batch_size": 1000, "num_layers": 2, "d_model": 256, "num_heads": 8, "dff": 1024, 
        "pe_input": 500, "pe_target": 2000, "position_encoding_denominator": 1000, "dropout_rate": 0.05, "beta_1": 0.9,
        "beta_2": 0.98, "epsilon": 1e-9}

    # Train model
    train(ds=ds, hyperparameters=hyperparams, save_path="saved_model_outputs/saved_models/pronouncer", checkpoint_name="pronounce")
