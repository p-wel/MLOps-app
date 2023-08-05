# Requires python 3.10

import os
import pandas as pd
import json

from .constants import *


def prepare_raw_data():
    dataset = pd.DataFrame()
    for data_filename in os.listdir(BASE_DATA_DIR):
        if not data_filename.endswith('.csv'):
            continue
        dataset = dataset._append(pd.read_csv(os.path.join(BASE_DATA_DIR, data_filename)))

    # results = dataset.get('stroke')

    for column_name in UNWANTED_COLUMN_NAMES:  # +[RESULTS_COLUMN_NAME]:
        dataset = dataset.drop(column_name, axis=1)


    with open(r'persistent/settings.json', 'r') as settings_file:
        settings = json.load(settings_file)
        mapping = settings.setdefault(COLUMN_MAPPING_DICTIONARY, dict())

        for column_name in CATEGORICAL_COLUMN_NAMES:
            all_features = set(dataset[column_name])
            for feature in all_features:
                if feature not in mapping.setdefault(column_name, []):
                    mapping[column_name].append(feature)
            this_feature_mapping = dict((value, idx) for idx, value in enumerate(mapping[column_name]))
            dataset[column_name] = dataset[column_name].map(this_feature_mapping).astype('int16')

    with open(r'persistent/settings.json', 'w') as settings_file:
        json.dump(settings, settings_file, indent=1)

    for column in dataset.columns:
        dataset[column] = dataset[column].fillna(dataset[column].median())

    dataset.to_csv(FINAL_DATA_FILENAME)
    return FINAL_DATA_FILENAME
