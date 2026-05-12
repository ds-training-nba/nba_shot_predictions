import random

from pandas import DataFrame
import numpy as np

def fix_action_type_target_leak(df):
    """
    fix a target leak in the ACTION_TYPE column. The simple types "Dunk Shot", "Jump Shot", "Layup Shot" and "Hook Shot"
    are significantly less successful. THis function tries to fix this bias by distributing the category randomly to the
    other categories of the same MAIN_ACTION_TYPE
    :param df:
    :return: df
    """
    action_type_prefixes_to_fix = ["Dunk", "Jump", "Layup", "Hook"]
    action_types_to_fix = [(action_type + " Shot") for action_type in action_type_prefixes_to_fix]
    sub_types = {}

    for prefix in action_type_prefixes_to_fix:
        action_type_to_fix = prefix + " Shot"
        sub_types[action_type_to_fix] = df[(df['MAIN_ACTION_TYPE'] == prefix) & (df['ACTION_TYPE'] != action_type_to_fix) ]['ACTION_TYPE'].unique()

    def fix_action_type(val):
        if val in action_types_to_fix:
            possible_types = sub_types[val]
            return random.choice(possible_types)
        else:
            return val

    df['ACTION_TYPE'] = df['ACTION_TYPE'].apply(fix_action_type)
    return df





