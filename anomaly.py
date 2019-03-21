"""
Functions to generate anomalies
"""
__author__ = "Daniel Nova"

import random
import numpy as np

# Constant field value to be used when setting the fields to a random constant
CONSTANT_FIELD_VALUE = None


def generate_random_field_value(field):
    """
    Generates a random binary value for the given field, picking a number between 0 and the maximum possible value.
    The maximum is determined by the field length, e.g. if a field has length = 4, it's maximum value is '1111'
    :param Field field: Target field
    :return: Returns a random binary string in the range [0 , field.max] with leading 0s if needed
    """
    length = field.length
    max_value = int('1' * (length + 1), 2)  # maximum possible value for that field size
    random_value = random.randint(0, max_value)

    return "{0:b}".format(random_value).zfill(length + 1)


def create_interleave_sequences(sequences):
    """
    Creates a sequence of interleaved messages from the given sequences.

    The sequences are partitioned by half.
    Where x are the sequences of the first part, and y of the second part.

    The function interleaves the sequences every word. As in:
    seq = [x_1, y_1, x_2, y_2, ..., x_n-1, y_n-1, x_n, y_n]

    This creates the discontinuity we are trying to achieve.

    :param np.array sequences: 3D array of the test sequences
    :return: np.array Interleaved array of sequences with the same shape as the input.
    """

    # both parts are turned into 2d matrices to make interleaving easier
    # there might be a vectorized way to do this without the intermediate reshaping, but I could't think of one
    reshaped_seq = np.reshape(sequences, (sequences.shape[0] * sequences.shape[1], sequences.shape[2]))

    # turn the sequences from 3d matrices to 2d
    data_anom_length = int(len(reshaped_seq) / 2)  # partitioned in half
    part_1 = reshaped_seq[:data_anom_length]
    part_2 = reshaped_seq[-data_anom_length:]

    interleaved_sequence = np.empty((len(part_1) * 2,
                                     part_1.shape[1]),
                                    dtype=np.byte)  # in 2d

    # the actual interleaving
    interleaved_sequence[0::2] = part_1
    interleaved_sequence[1::2] = part_2

    return np.reshape(interleaved_sequence, sequences.shape), "interleave"  # back to 3d


def create_discontinuity_sequences(sequences):
    """
    Creates a discontinuity by changing the second half of every sequence with data from another point in time,
    subsequence is still valid traffic, but with a sudden change in context.
    Partitions the sequences into two disjoint parts, the first half of each sequence is left untouched,
    the second part is taken from another sequence.

    The second part is taken from the second half of the given sequences.

    :param np.array sequences: Input 3D sequences
    :return: Sequences with discontinuities, same shape as input
    """
    discontinuity_sequence = np.copy(sequences)
    sequences_len = sequences.shape[0]
    sequence_len = sequences.shape[1] // 2

    seq_idx = sequences.shape[0] // 2 - 1

    for dsequence in discontinuity_sequence:
        if seq_idx >= sequences_len:
            seq_idx = 0
        dsequence[sequence_len:] = sequences[seq_idx][sequence_len:]
        seq_idx += 1

    return discontinuity_sequence, "discontinuity"


def create_reverse_sequences(sequences):
    """
    The order of words in a sequence is reversed.
    :param np.array sequences: Input 3D sequences
    :return: Reversed sequences, same shape as input
    """
    n = sequences.shape[0]
    p = sequences.shape[1]
    dlc = sequences.shape[2]

    reversed_seq = np.reshape(sequences, (n * p, dlc))    # shape into 2d
    reversed_seq = reversed_seq[::-1]                     # we use a view instead of building an array from scratch
    reversed_seq = np.reshape(reversed_seq, (n, p, dlc))  # reshape back into 3d
    return reversed_seq[::-1], "reverse"


def create_drop_sequences(sequence, length=10):
    """
    Remove words around the middle of the sequence,
    returns a new sequence without such words.

    Arguments:
    :param np.array sequence: Input 3D sequences
    :param length: amount of words to remove around the middle, default set to 10
    :return: Dropped sequences, same shape as input except for the 2nd axis
    """

    assert length < sequence.shape[1], "Cannot drop more words than the number of available words per sequence"

    indx_drop = int(sequence.shape[1] / 2)
    _len = int(length / 2)

    words_to_drop = [i for i in range(indx_drop - _len, indx_drop + _len)]  # indices to delete

    seqquence_with_drop = np.delete(sequence, words_to_drop, 1)  # remove them
    return seqquence_with_drop, "drop"


# DATA FIELD ANOMALIES

def set_field_to_max(field, word):
    """
    Sets a target field in a word to its maximum value (i.e. all ones)
    :param Field field: target field
    :param np.array word: Word binary array
    :return: Modified word
    """
    start = field.start_bit
    length = field.length

    for i in range((length + 1)):
        word[start + i] = 1

    return word, "max_value"


def set_field_to_min(field, word):
    """
    Sets a target field in a word to its minimum value (i.e. all zeros)
    :param Field field: target field
    :param np.array word: Word binary array
    :return: Modified word
    """
    start = field.start_bit
    length = field.length
    for i in range((length + 1)):
        word[start + i] = 0

    return word, "min_value"


def set_field_to_random_constant(field, word):
    """
    Sets the target field in a word to a constant value.
    If CONSTANT_FIELD_VALUE is None, a random value is set.
    :param Field field: target field
    :param np.array word: Word binary array
    :return: Modified word
    """
    start = field.start_bit
    length = field.length
    global CONSTANT_FIELD_VALUE
    if CONSTANT_FIELD_VALUE is None:  # generate the replacement for the sequence
        CONSTANT_FIELD_VALUE = np.random.randint(0, 2, size=(length + 2))

    constant_idx = 0
    for i in range(length):
        word[start + i] = CONSTANT_FIELD_VALUE[constant_idx]
        constant_idx += 1
    return word, "constant_value"


def set_field_to_random_value(field, word):
    """
    Sets the given field data to a random value, unlike the set_field_to_random_constant function,
    this method creates the random value for each word in the segment.
    :param Field field: Field to modify
    :param np.array word: Word binary array
    :return: New modified word
    """
    start = field.start_bit
    length = field.length
    random_word = np.random.randint(0, 2, size=(length + 2))

    _idx = 0
    for i in range((length + 1)):
        word[start + i] = random_word[_idx]
        _idx += 1

    return word, "random_value"


def replay_field(field, word, replayed_word):
    """
    Sets the value of a given field of a word to that of a replayed word.
    :param Field field: Field to modify
    :param np.array word: Word binary array
    :param np.array replayed_word: Replayed word
    :return:
    """
    start = field.start_bit
    length = field.length

    for i in range((length + 1)):
        word[start + i] = replayed_word[start + i]
    return word, "replay_field"


def create_field_anomaly(sequences, chosen_field, num_anom_words, modification_function, verbose=0):
    """
    Creates data field anomalies.
    Its starting point is chosen at random, but it cannot be earler than 1/3rd of the sequence, but enough to
    accomodate the anomaly.
    :param np.array sequences: 3d binary data sequences
    :param Field chosen_field: Target field for the target CAN ID.
    :param int num_anom_words: number of anomalous words to create
    :param modification_function: Modifier function for the field. Can be max, min, random constant, random, raplay,
     see the functions above
    :param int verbose:
    """
    sequence_length = sequences.shape[1]
    anomalous_sequence = np.copy(sequences)

    # starting point has to be no earlier than 1/3 of the sequence, and enough to accommodate the anomaly
    start = random.randint(int(sequence_length / 3), (sequence_length-num_anom_words-1))

    anomaly_name = ""
    global CONSTANT_FIELD_VALUE
    CONSTANT_FIELD_VALUE = None

    if verbose == 1:
        print("Anomaly will start at %d, with length %d" % (start, num_anom_words))
        print('The data for the chosen field is: Start bit: %d | Length: %d ' %
              (chosen_field.start_bit, chosen_field.length))

    seq_replay_index = sequences.shape[0] // 3  # sequence from which we get the fields for replay
    for seq_idx, sequence in enumerate(sequences):
        for word_idx, word in enumerate(sequence):
            if start <= word_idx <= start + num_anom_words:  # how long should the anomaly last
                if modification_function == replay_field:    # replay field case must be handled differently
                    replayed_word = np.copy(sequences[seq_replay_index][word_idx])
                    new_word, anomaly_name = replay_field(chosen_field, np.copy(word), replayed_word)
                    seq_replay_index += 1
                    if seq_replay_index == sequences.shape[0]:
                        seq_replay_index = 0
                else:
                    new_word, anomaly_name = np.copy(modification_function(chosen_field, np.copy(word)))
                anomalous_sequence[seq_idx][word_idx] = np.copy(new_word)

    return anomalous_sequence, anomaly_name
