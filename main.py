"""
Example script on how to generate an anomalous sequence.
Binary representation of the Data field of the CAN dump file is needed (i.e. the DataBin column in the dataset)

First both the dataset and file info file are loaded, then a test sequence is built, from which we generate the
anomalies.
"""
__author__ = "Daniel Nova"

import pandas as pd
import numpy as np
import os
import pickle
import anomaly as anom
from fields import FieldVariability
from fields import get_target_field
from fields import Field

# Files
DATA_PATH = ['data']
DATA_FILE = 'alfa2.csv'
FIELD_CLASSIFICATION_FILE_NAME = 'fields_data.pkl'

# Parameters for building the test sequences.
FREQUENCY = 0.01  # In seconds
DURATION = 3  # In seconds


def main():
    # Load data
    file_path = os.sep.join(DATA_PATH + [DATA_FILE])
    data = pd.read_csv(file_path, header=0, index_col=0, dtype={"Timestamp": float, "ID": str, "DLC": int, "Data": str})

    can_id = '0DE'  # Choose a target CAN ID

    data = data[data.ID == can_id]

    # Load fields file
    with open(FIELD_CLASSIFICATION_FILE_NAME, 'rb') as inp:
        fields = pickle.load(inp)

    fields = fields[can_id]

    # Creates the numpy arrays of binary sequences
    sequences = create_test_sequences(data, frequency=FREQUENCY, duration=DURATION)

    # Building the anomalous sequences
    # You can also sample from the original sequences to create the anomalous sequences,
    # for the sake of this example we will use a copy of the entire test sequences.
    # INTERLEAVE
    sequence_interleaved, _ = anom.create_interleave_sequences(np.copy(sequences))

    # DISCONTINUITY
    sequence_discontinuity, _ = anom.create_discontinuity_sequences(np.copy(sequences))

    # REVERSE
    sequence_reverse, _ = anom.create_reverse_sequences(np.copy(sequences))

    # DROP
    sequence_drop, _ = anom.create_drop_sequences(np.copy(sequences))

    # FIELD ANOMALIES
    # First choose the duration of the anomaly
    anom_duration = 1  # in seconds, in the ADS we tested for 0.2, 0.5, 1, and 1.5 seconds

    # Variability category of the target field
    # Other options: FieldVariability.LOW_VAR, FieldVariability.MID_VAR
    # Remember that not all fields have low/mid variability fields, in such cases get_target_field returns None
    target_field_category = FieldVariability.HIGH_VAR

    # number of packets needed for the anomaly
    num_anom_packets = int(anom_duration / FREQUENCY)

    # Randomly choose a target field for the selected category
    chosen_field = get_target_field(fields, target_field_category)

    maxfield_sequence, _ = anom.create_field_anomaly(np.copy(sequences), chosen_field,
                                                     num_anom_packets, anom.set_field_to_max, verbose=1)

    minfield_sequence, _ = anom.create_field_anomaly(np.copy(sequences), chosen_field,
                                                     num_anom_packets, anom.set_field_to_min, verbose=1)

    constant_field_sequence, _ = anom.create_field_anomaly(np.copy(sequences), chosen_field,
                                                           num_anom_packets, anom.set_field_to_random_constant,
                                                           verbose=1)

    random_field_sequence, _ = anom.create_field_anomaly(np.copy(sequences), chosen_field,
                                                         num_anom_packets, anom.set_field_to_random_value,
                                                         verbose=1)

    field_replay_sequence, _ = anom.create_field_anomaly(np.copy(sequences), chosen_field,
                                                         num_anom_packets, anom.replay_field,
                                                         verbose=1)

    print('DONE')


def create_test_sequences(data, frequency=0.01, duration=3):
    """
    Creates x-second non-overlapping sequences of data. Its length depends the duration parameter.

    :param pd.DataFrame data: data for a specific CAN ID
    :param frequency: Frequency in seconds for the target CAN ID (e.g 0.01s)
    :param int duration: duration in seconds of the sequence, default to 3
    :return: 3D array of the generated non-overlapping sequences. Its shape is n𝘅m𝘅p, where n is the number of sequences,
    m is the length of each sequence (e.g. 100 for an ID with a 0.03s frequency if duration=3),
    and p is the message cardinality (<=64 bits)
    """
    num_packets = int(duration / frequency)  # Number of packets needed i

    # Use binary representation
    observations = data['DataBin']

    sequence = pd.DataFrame()
    sequence = sequence.append([[int(x) for x in i] for i in observations])  # str data into bit array
    binary_sequences = sequence.to_numpy(dtype=np.byte)

    tot_observations = len(observations)
    split = int(tot_observations / num_packets)  # not always we can fit all the observations
    n = num_packets * split

    # reshape into a (n x num_packets x cardinality) matrix
    word_length = len(observations.iloc[0])
    return np.reshape(binary_sequences[:n], (split, num_packets, word_length))


if __name__ == '__main__':
    main()

