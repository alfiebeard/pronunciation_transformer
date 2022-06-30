import tensorflow as tf
import time
from model.load_model import load_model
from model.training_utils import accuracy_function, full_sequence_accuracy_function
from data.data_loader import load_data


def evaluate_model(transformer, test_batches):
    """
    Evaluate the transformer on a val/test set, obtaining the accuracies.
    """

    print('')
    print("================ Evaluating Validation Set ================")

    # If transformer is a string, i.e., a file path, load it.
    if isinstance(transformer, str):
        transformer = load_model(save_path=transformer)
        is_transformer_path = True
    else:
        is_transformer_path = False

    # Define accuracies
    test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')
    full_sequence_test_accuracy = tf.keras.metrics.Mean(name='full_sequence_test_accuracy')

    test_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

    # Test step
    @tf.function(input_signature=test_step_signature)
    def test_step(inp, tar):
        # Get target input and real for loss 
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        
        # Get predictions from trained transformer
        if is_transformer_path:
            predictions, _ = transformer.model.pronouncer.transformer([inp, tar_inp], training=False)
        else:
            predictions, _ = transformer.transformer([inp, tar_inp], training=False)

        # Log the accuracies
        test_accuracy(accuracy_function(tar_real, predictions))
        full_sequence_test_accuracy(full_sequence_accuracy_function(tar_real, predictions))

    # Test the transformer model
    start = time.time()
    test_accuracy.reset_states()
    full_sequence_test_accuracy.reset_states()

    # inp -> word, tar -> ipa
    for (batch, (inp, tar)) in enumerate(test_batches):
        test_step(inp, tar)

        # Print test results every 100 batches
        if batch % 100 == 0:
            print('Batch {0} Accuracy {1}'.format(batch, test_accuracy.result()))
            print('Batch {0} Complete Match Accuracy {1}'.format(batch, full_sequence_test_accuracy.result()))

    # Print all test results
    print('')
    print("================ Validation Summary ================")
    print('Validation Set Accuracy: {0}'.format(test_accuracy.result()))
    print('Validation Set Complete Match Accuracy {0}'.format(full_sequence_test_accuracy.result()))

    print(f'Time taken: {time.time() - start:.2f} secs\n')

    return test_accuracy.result().numpy()


if __name__ == "__main__":
    (_, _, test_data, _, _) = load_data()
    print('================ Standard Model ================')
    _ = evaluate_model("saved_model_outputs/saved_models/pronouncer/standard", test_data)
    print('================ Beam Model ================')
    _ = evaluate_model("saved_model_outputs/saved_models/pronouncer/beam", test_data)
