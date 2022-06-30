import tensorflow as tf
import numpy as np


def print_pronunciation(sequence, prediction, ground_truth=None):
    print(f'{"Input:":15s}: {sequence}')
    print(f'{"Prediction":15s}: {prediction}')
    if ground_truth:
        print(f'{"Ground truth":15s}: {ground_truth}')


def get_column_indices(num_examples, column_number, dtype):
    start_indices = tf.range(num_examples, dtype=dtype)     # Create range from 0-num_examples
    start_indices = tf.concat([start_indices, tf.ones(num_examples, dtype=dtype) * tf.cast(column_number, dtype=dtype)], 0)    # Add column number in
    start_indices = tf.transpose(tf.reshape(start_indices, (2, num_examples)))    # Reshape to get [[0, col_num], [1, col_num], ..., [num_examples, col_num]]
    return start_indices


# Translating
class Pronouncer(tf.Module):
    def __init__(self, input_embedding, output_embedding, transformer):
        self.input_embedding = input_embedding
        self.output_embedding = output_embedding
        self.transformer = transformer

    def __call__(self, sequence, max_length=20):
        # Encode the sequence
        encoded_sequence = self.input_embedding.batch_encode(sequence)
        prediction, attention_weights = self.pronounce(encoded_sequence, max_length=max_length)
        return self.output_embedding.batch_decode(prediction), attention_weights

    def pronounce(self, sequence, max_length=20):
        """
        Pronounce a sequence using the transformer model.
        """

        # Convert to tensor
        input = tf.convert_to_tensor(sequence)

        num_examples = tf.shape(input)[0]

        # Define start, end token for target
        start = self.output_embedding.all_tokens.index('START')
        pad = self.output_embedding.all_tokens.index('PAD')
        end = self.output_embedding.all_tokens.index('END')

        # `tf.TensorArray` is required here (instead of a python list) so that the
        # dynamic-loop can be traced by `tf.function`.
        # output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        # output_array = output_array.write(0, tf.constant(start, dtype=tf.int64)[tf.newaxis])

        # Trialling concat for multiple inputs
        # output = tf.ones([tf.shape(input)[0], 1], input.dtype) * start

        # Start from all start tokens
        output = tf.ones([num_examples, max_length], dtype=input.dtype) * pad
        start_indices = get_column_indices(num_examples, 0, input.dtype)
        start_values = tf.ones(num_examples, dtype=input.dtype) * start
        output = tf.tensor_scatter_nd_add(output, indices=start_indices, updates=start_values)

        # Predict pronunciations
        for i in tf.range(max_length):
            # Predict the next IPA character in the sequence and store it
            predictions, _ = self.transformer([input, output], training=False)

            # select the last token from the seq_len dimension
            predictions = predictions[:, i, :]

            # Get the top prediction
            predicted_id = tf.argmax(predictions, axis=-1)

            # Store the new predictions
            indices = get_column_indices(num_examples, i + 1, input.dtype)
            output = tf.tensor_scatter_nd_add(output, indices=indices, updates=predicted_id)

            # Count entries equal to end token in output
            count_ends_output = tf.math.reduce_sum(tf.cast(tf.math.equal(output, end), input.dtype), 1)
            
            # Does each generated sequence have an end token - check each row and then check one exists for each.
            all_output_ends = tf.reduce_all(count_ends_output > 0)

            if all_output_ends:
                break

        # `tf.function` prevents us from using the attention_weights that were
        # calculated on the last iteration of the loop. So recalculate them outside
        # the loop.
        _, attention_weights = self.transformer([input, output[:,:-1]], training=False)

        return output, attention_weights


    def pronounce_beam(self, sequence, max_length=20, search_width=10):
        """
        Pronounce a sequence using the transformer model but with beam search.
        """
        
        # Convert to tensor
        input = tf.convert_to_tensor(sequence)

        num_examples = tf.shape(input)[0]

        # Define start, end token for target
        start = self.output_embedding.all_tokens.index('START')
        pad = self.output_embedding.all_tokens.index('PAD')
        end = self.output_embedding.all_tokens.index('END')

        # `tf.TensorArray` is required here (instead of a python list) so that the
        # dynamic-loop can be traced by `tf.function`.
        # output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        # output_array = output_array.write(0, tf.constant(start, dtype=tf.int64)[tf.newaxis])

        # Preallocate output tensor of size search_width*N x max_length - as we are storing the search_width candidates per example
        output = tf.ones([num_examples * search_width, max_length], dtype=input.dtype) * pad
        start_indices = get_column_indices(num_examples * search_width, 0, input.dtype)
        start_values = tf.ones(num_examples * search_width, dtype=input.dtype) * start
        output = tf.tensor_scatter_nd_add(output, indices=start_indices, updates=start_values)

        # Get top search_width initial predictions and probabilities for each example
        # This is to move from an input of size (N, max_length) to input of size (3*N, max_length)
        initial_output = output[:num_examples]
        predictions, _ = self.transformer([input, initial_output], training=False)
        predictions = predictions[:, 0, :]
        predicted_probs, predicted_ids = tf.math.top_k(predictions, k=search_width)
        probs = tf.reshape(predicted_probs, (num_examples*search_width, 1))
        predicted_ids = tf.reshape(predicted_ids, (num_examples*search_width, 1))

        # Update output 
        indices = get_column_indices(num_examples*search_width, 1, input.dtype)
        output = tf.tensor_scatter_nd_add(output, indices=indices, updates=tf.cast(tf.squeeze(predicted_ids), input.dtype))

        # Extend input to match output size, by repeating inputs
        input = tf.repeat(input, repeats=search_width, axis=0)

        for i in tf.range(1, max_length):
            predictions, _ = self.transformer([input, output], training=False)

            # select the i'th token from the seq_len dimension
            predictions = predictions[:, i, :]

            # Get top search_width probabilities and id's
            predicted_probs, predicted_ids = tf.math.top_k(predictions, k=search_width)

            # Combine with previous probabilities
            combined_probs = probs * predicted_probs

            # Get search_width largest probs from combined probs
            combined_probs = tf.reshape(combined_probs, (num_examples, search_width ** 2))
            combined_probs, combined_ids = tf.math.top_k(combined_probs, k=search_width)
            combined_probs = tf.reshape(combined_probs, (search_width * num_examples, 1))

            # Now get the predicted ids for these largest probabilities
            row_index = tf.reshape(tf.repeat(tf.range(num_examples), search_width), (num_examples, search_width)) * search_width
            first_indices = (combined_ids // search_width) + row_index
            second_indices = combined_ids % search_width
            full_combined_indices = tf.reshape(tf.concat([first_indices[...,tf.newaxis], second_indices[...,tf.newaxis]], axis=-1), [tf.shape(first_indices)[0],-1])
            full_combined_indices = tf.reshape(full_combined_indices, (num_examples * search_width, 2))

            # Get predicted id's as list and update probs
            predicted_ids = tf.reshape(tf.gather_nd(indices=full_combined_indices, params=predicted_ids), (num_examples*search_width, 1))
            probs = tf.reshape(tf.gather_nd(indices=full_combined_indices, params=predicted_probs), (num_examples*search_width, 1))

            # Update output to show candidates we are keeping - overwriting old ones
            output_ids = tf.reshape(first_indices, (num_examples * search_width, 1))
            # Have to do reshape below, since it doesn't like gather, as shape is unknown - so just reshape to output shape.
            output = tf.reshape(tf.gather(output, tf.squeeze(output_ids)), (num_examples * search_width, max_length))
            # Now add new predicted id's
            indices = get_column_indices(num_examples*search_width, i + 1, input.dtype)
            output = tf.tensor_scatter_nd_add(output, indices=indices, updates=tf.cast(tf.squeeze(predicted_ids), input.dtype))

            # Count entries equal to end token in output
            count_ends_output = tf.math.reduce_sum(tf.cast(tf.math.equal(output, end), input.dtype), 1)
            # Does each generated sequence have an end token - check each row and then check one exists for each.
            all_output_ends = tf.reduce_all(count_ends_output > 0)

            if all_output_ends:
                break

        # Get best candidate for each example
        probs_reshaped = tf.reshape(probs, (num_examples, search_width))
        probs_argmax = tf.argmax(probs_reshaped, axis=1)
        top_output = tf.gather(tf.reshape(output, (num_examples, search_width, max_length)), probs_argmax, axis=1, batch_dims=1)

        # `tf.function` prevents us from using the attention_weights that were
        # calculated on the last iteration of the loop. So recalculate them outside
        # the loop.
        _, attention_weights = self.transformer([input, output[:,:-1]], training=False)

        return top_output, output, probs, attention_weights


class SavePronouncer(tf.Module):
    def __init__(self, pronouncer):
        self.pronouncer = pronouncer
        self.model_type = "standard"

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int64)])
    def __call__(self, sequence):
        (result, _) = self.pronouncer.pronounce(sequence, max_length=100)
        return result


class SavePronouncerBeam(tf.Module):
    def __init__(self, pronouncer):
        self.pronouncer = pronouncer
        self.model_type = "beam"

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int64)])
    def __call__(self, sequence):
        (top_result, result, probs, _) = self.pronouncer.pronounce_beam(sequence, max_length=100)
        return top_result, result, probs


class LoadPronouncer:
    def __init__(self, model_name, input_embedding, output_embedding, model_type):
        self.model_name = model_name
        self.input_embedding = input_embedding
        self.output_embedding = output_embedding
        self.model = tf.saved_model.load(model_name)
        self.model_type = model_type

    def __call__(self, sequence):
        # Encode the sequence
        encoded_sequence = self.input_embedding.batch_encode(sequence)
        prediction = self.model(encoded_sequence)

        if self.model_type == "beam":
            # If beam search model, then get predictions, as also outputs search results and probabilities.
            prediction = prediction[0]

        return self.output_embedding.batch_decode(prediction, human_readable=True)
