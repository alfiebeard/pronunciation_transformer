import tensorflow as tf


# Define a custom learning rate which increases linearly then decays over time
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred, loss_object):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


def full_sequence_accuracy_function(real, pred):
    # This checks for equality between the sequences and returns T/F in each position
    predicted_sequence = tf.argmax(pred, axis=2)
    accuracies = tf.equal(real, predicted_sequence)

    # This creates the mask on the real data and applies to accuracies, i.e., F where there are no values 
    # or not an accurate prediction.
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    # Cast to floats from boolean so can sum up
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    # If the sum of the accuracies and mask is the same, then we have an exact match. Otherwise, we have less accuracies
    # than mask - which says that we have positions which are not 1 and are thus inaccurate.
    differences = tf.reduce_sum(accuracies, axis=1) - tf.reduce_sum(mask, axis=1)
    differences = differences == 0
    differences = tf.cast(differences, dtype=tf.int32)

    return tf.reduce_sum(differences) / tf.shape(differences, out_type=tf.int32)[0]
    