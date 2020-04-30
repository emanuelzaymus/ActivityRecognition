from datetime import datetime
from typing import Tuple

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from src.DataArray import DataArray

ANOTHER_FEATURES_COUNT = 3  # 3 for features: SECONDS FROM MIDNIGHT, DAY OF THE WEEK, SECONDS ELAPSED


def extract_features_from_arrays(data_arrays: list, window_size: int, sensors: list = None,
                                 with_previous_class_feature: bool = False) -> np.ndarray:
    """
    :param data_arrays:
    :param window_size:
    :param sensors:
    :param with_previous_class_feature:
    :return:
    """
    result_data_array = None

    data_arr: np.ndarray
    for data_arr in data_arrays:
        data, a = extract_features(data_arr, window_size, all_samples_labeled=True, sensors=np.array(sensors))
        if with_previous_class_feature:
            data = __add_previous_class_feature(data)
        result_data_array = data if result_data_array is None else np.append(result_data_array, data, axis=0)

    result_data_array = __encode_categorical_feature(result_data_array, 1)

    if with_previous_class_feature:
        result_data_array = __encode_categorical_feature(result_data_array, -2)

    return result_data_array


def extract_features(data_array: np.ndarray, window_size: int, all_samples_labeled: bool = False,
                     sensors: np.ndarray = None, with_previous_class_feature: bool = False) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
        Extracts feature vectors from ``data_array`` based on windowing with ``window_size``.

        Features:
            - 1st - SECONDS FROM MIDNIGHT (of the first record in the window)
            - 2nd - DAY OF THE WEEK - MON..SUN => 1..7 (of the last record in the window)
            - 3rd - SECONDS ELAPSED (between the last and the first record of the window)
            - ... - SIMPLE COUNTS OF THE SENSORS
            - Last - CLASS of the feature vector - index of the activity (of the last record of the window)

        :param data_array: Numpy array in format [[datetime.datetime  SENSOR  ACTIVITY] ... ]
                -> result of data_file_handling.get_data_array(file_name)
        :param window_size: Size of the window
        :param all_samples_labeled: If all samples are labeled with its numerical value of activity.
        :param sensors: Names of sensors (optional).
        :param with_previous_class_feature:
    Returns:
        features : ndarray
            Feature vectors with class of the feature vector - last element of the vector
        activities : ndarray
            Used activities in its order (indices of the activities are theirs positions in the array)
    """
    data: np.ndarray = data_array.copy()
    activities: np.ndarray = __fill_missing_activities(data) if not all_samples_labeled else None
    if sensors is None:
        sensors = __get_sensors(data)
    n_f_vectors: int = data.shape[0] - window_size + 1  # number of rows - window_size + 1
    n_features: int = sensors.size + ANOTHER_FEATURES_COUNT + 1  # + 1 for CLASS

    if not n_f_vectors > 0:
        exit('Window size %s is too large. Number of feature vectors = %d' % (window_size, n_f_vectors))

    features: np.ndarray = np.zeros((n_f_vectors, n_features), dtype=int)

    for x in range(n_f_vectors):
        window: np.ndarray = data[x:x + window_size]
        __fill_feature_vector_using_record_window(features[x], window, sensors)

    if with_previous_class_feature:
        features = __add_previous_class_feature(features)

    return features, activities


def __fill_feature_vector_using_record_window(feature_vector: np.ndarray, record_window: np.ndarray,
                                              sensors: np.ndarray) -> np.ndarray:
    """
    Fills the whole ``feature_vector`` with collected data from ``record_window``.
    """
    datetime_first: datetime = record_window[0, DataArray.DATETIME]
    datetime_last: datetime = record_window[-1, DataArray.DATETIME]
    # fill SECONDS FROM MIDNIGHT
    feature_vector[0] = (datetime_first - datetime_first.replace(hour=0, minute=0, second=0)).total_seconds()

    # fill DAY OF THE WEEK
    feature_vector[1] = datetime_last.isoweekday()

    # fill SECONDS ELAPSED
    feature_vector[2] = (datetime_last - datetime_first).total_seconds()

    # fill SIMPLE SENSOR COUNTS
    window_sensors: np.ndarray = record_window[:, DataArray.SENSOR]
    unique, counts = np.unique(window_sensors, return_counts=True)
    for i in range(unique.size):
        index = np.where(sensors == unique[i])
        feature_vector[index[0][0] + ANOTHER_FEATURES_COUNT] = counts[i]

    # fill CLASS of the feature vector
    feature_vector[-1] = record_window[-1, DataArray.ACTIVITY]

    return feature_vector


def __get_sensors(data_array: np.ndarray) -> np.ndarray:
    return np.unique(data_array[:, DataArray.SENSOR])


def __fill_missing_activities(data_array: np.ndarray) -> np.ndarray:
    """
    Fills and replaces ACTIVITY values with NUMBER (index) OF THE ACTIVITY in the parameter ``data_array``.

    Returns:
        ndarray: Used activities in its order
    """
    all_activities: np.ndarray = __get_activities(data_array)
    last_activities_stack = []
    contains_nothing_bool = False

    for record in data_array:
        curr_act: str = record[DataArray.ACTIVITY]

        if curr_act != DataArray.NO_ACTIVITY:
            curr_act, curr_act_state = curr_act.strip().split()
            curr_act_index: int = np.where(all_activities == curr_act)[0][0]  # activity index
            record[DataArray.ACTIVITY] = curr_act_index  # set current activity index

            if curr_act_state.lower() in (DataArray.BEGIN, DataArray.START):
                last_activities_stack.append(curr_act_index)  # set index of last activity
            elif curr_act_state.lower() == DataArray.END:
                last_activities_stack.pop()
        elif len(last_activities_stack) != 0:
            record[DataArray.ACTIVITY] = last_activities_stack[-1]
        else:
            record[DataArray.ACTIVITY] = all_activities.size  # set future activity index of NOTHING
            contains_nothing_bool = True

    if contains_nothing_bool:
        all_activities = np.append(all_activities, DataArray.NOTHING)

    return all_activities


def __get_activities(data_array: np.ndarray) -> np.ndarray:
    unique_activities: np.ndarray = np.unique(data_array[:, DataArray.ACTIVITY])

    for i in range(unique_activities.size):
        unique_activities[i] = unique_activities[i].split()[0]

    unique_activities: np.ndarray = np.unique(unique_activities)
    return unique_activities[unique_activities != DataArray.NO_ACTIVITY]


def __add_previous_class_feature(features: np.ndarray) -> np.ndarray:
    feature_vectors: np.ndarray = features[:, :-1]
    classes: np.ndarray = features[:, -1]

    prev_classes = np.zeros((feature_vectors.shape[0], 1), dtype=int)
    feature_vectors = np.column_stack((feature_vectors, prev_classes))

    feature_vectors[0, -1] = classes[0]
    for i in range(1, classes.shape[0]):
        feature_vectors[i, -1] = classes[i - 1]

    features = np.column_stack((feature_vectors, classes))
    return features


def __encode_categorical_feature(data_array: np.ndarray, feature_index: int) -> np.ndarray:
    features_before: np.ndarray = data_array[:, :feature_index]
    categorical_feature: np.ndarray = data_array[:, feature_index].reshape(-1, 1)
    features_after: np.ndarray = data_array[:, feature_index + 1:]

    categorical_feature = OneHotEncoder().fit_transform(categorical_feature).toarray()

    return np.concatenate((features_before, categorical_feature, features_after), axis=1)
