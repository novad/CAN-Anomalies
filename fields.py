"""
Functions and class representing fields of a CAN data message.
"""
__author__ = "Daniel Nova"

import numpy as np
from enum import Enum
import random

TYPE_CONST = 'CONST'
TYPE_MULTI_VALUE = 'MULTI-VALUE'
TYPE_SENSOR = 'SENSOR'


class FieldVariability(Enum):
    """Enums for the fields categories by word variability"""
    HIGH_VAR = 1
    MID_VAR = 2
    LOW_VAR = 3


class Field:
    def __init__(self):
        self.n_values = 0      # number of unique values of the field
        self.type = None       # field type, i.e. TYPE_CONST, TYPE_MULTI_VALUE, TYPE_SENSOR
        self.start_bit = -1    # bit where the field starts
        self.length = -1       # length of the field, how many bits it uses
        self.category = None   # variability category, see FieldVariability


def find_constant_bits(base_word, sequence, dlc):
    """
    Builds the array of constant bits for the given sequence
    :param base_word:
    :param sequence:
    :param dlc:
    :return:
    """
    fields = np.ones(dlc, dtype=bool)
    for word in sequence:
        if word is not None:
            for bit_idx in reversed( range(len(word))):
                if base_word[bit_idx] != word[bit_idx]:
                    fields [bit_idx] = False
    return fields


def remove_bits(sequences, fields):
    """ removes constant bits from a sequence. Sequences must be a 3d np array (sequences x subsequence x words) """
    constants_indices = get_constant_fields(fields)  # get indices of constants for the target ID
    return np.delete(sequences, constants_indices, axis=2)


def get_field_values(data_fields, instance):
    """
    Get the integer values of the fields for a given CAN message instance
    """
    # function used for finding fields max and mins
    values = []
    for f in data_fields:
        if f.type != TYPE_CONST:
            idx_s = f.start_bit  # first element index
            length = f.length  # last element
            data_field = instance['DataBin'][idx_s:idx_s + length + 1]
            values.append(int(data_field, 2))
    return values


def get_constant_fields(fields_list):
    """
    Returns a list of the constant bit indeces
    """
    const_indices = []
    for field in fields_list:
        if field.type == TYPE_CONST:
            const_indices += list(range(field.start_bit, field.start_bit + field.length+1))
    return const_indices


def get_target_field(fields, field_category):
    """
    Picks a random field given a specific variability category.
    :param list(Field) fields: The relative class for the target CAN ID
    :param FieldCategory field_category: Category of the random field
    :return:
    """
    all_fields = []
    for field in fields:
        if field_category == field.category:
            all_fields.append(field)
    if not all_fields:
        print('No fields of the selected variability exist for this sequence')
        return None
    # pick a random field
    return random.choice(all_fields)

