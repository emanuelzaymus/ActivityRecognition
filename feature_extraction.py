import numpy as np
from DataArray import DataArray

ANOTHER_FEATURES_COUNT = 3  # 3 for features: SECONDS FROM MIDNIGHT, DAY OF THE WEEK, SECONDS ELAPSED


def extract_features(data_array, window_size):
    """
        Extracts feature vectors from ``data_array`` based on windowing with ``window_size``.

        Features:
            - 1st - SECONDS FROM MIDNIGHT (of the first record in the window)
            - 2nd - DAY OF THE WEEK - MON..SUN => 1..7 (of the last record in the window)
            - 3rd - SECONDS ELAPSED (between the last and the first record of the window)
            - ... - SIMPLE COUNTS OF THE SENSORS
            - Last - CLASS of the feature vector - index of the activity (of the last record of the window)
    Parameters:
        data_array (ndarray): Numpy array in format [[datetime.datetime  SENSOR  ACTIVITY] ... ]
                -> result of data_file_handling.get_data_array(file_name)
        window_size (int): Size of the window
    Returns
        features : ndarray
            Feature vectors with class of the feature vector - last element of the vector
        activities : ndarray
            Used activities in its order (indices of the activities are theirs positions in the array)
    """
    data = data_array.copy()
    activities = __fill_missing_activities(data)
    sensors = __get_sensors(data)
    n_f_vectors = data.shape[0] - window_size + 1  # number of rows - window_size + 1
    n_features = sensors.size + ANOTHER_FEATURES_COUNT + 1  # + 1 for CLASS

    features = np.zeros((n_f_vectors, n_features), dtype=int)

    for x in range(n_f_vectors):
        window = data[x:x + window_size]
        __fill_feature_vector_using_record_window(features[x], window, sensors)

    return features, activities


def __fill_feature_vector_using_record_window(feature_vector, record_window, sensors):
    """
    Fills the whole ``feature_vector`` with collected data from ``record_window``.
    """
    datetime_first = record_window[0, DataArray.DATETIME]
    datetime_last = record_window[-1, DataArray.DATETIME]
    # fill SECONDS FROM MIDNIGHT
    feature_vector[0] = (datetime_first - datetime_first.replace(hour=0, minute=0, second=0)).total_seconds()

    # fill DAY OF THE WEEK
    feature_vector[1] = datetime_last.isoweekday()

    # fill SECONDS ELAPSED
    feature_vector[2] = (datetime_last - datetime_first).total_seconds()

    # fill SIMPLE SENSOR COUNTS
    window_sensors = record_window[:, DataArray.SENSOR]
    unique, counts = np.unique(window_sensors, return_counts=True)
    for i in range(unique.size):
        index = np.where(sensors == unique[i])
        feature_vector[index[0][0] + ANOTHER_FEATURES_COUNT] = counts[i]

    # fill CLASS of the feature vector
    feature_vector[-1] = record_window[-1, DataArray.ACTIVITY]

    return feature_vector


def __get_sensors(data_array):
    """
    Returns:
        ndarray: Used sensors
    """
    return np.unique(data_array[:, DataArray.SENSOR])


def __fill_missing_activities(data_array):
    """
    Fills and replaces ACTIVITY values with NUMBER (index) OF THE ACTIVITY in the parameter ``data_array``.

    Returns:
        ndarray: Used activities in its order
    """
    all_activities = __get_activities(data_array)
    last_activities_stack = []
    contains_nothing_bool = False

    for record in data_array:
        curr_act = record[DataArray.ACTIVITY]

        if curr_act != DataArray.NO_ACTIVITY:
            curr_act, curr_act_state = curr_act.strip().split()
            curr_act_index = np.where(all_activities == curr_act)[0][0]  # activity index
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


def __get_activities(data_array):
    """
    Returns:
        ndarray: Used activities
    """
    unique_activities = np.unique(data_array[:, DataArray.ACTIVITY])

    for i in range(unique_activities.size):
        unique_activities[i] = unique_activities[i].split()[0]

    unique_activities = np.unique(unique_activities)
    return unique_activities[unique_activities != DataArray.NO_ACTIVITY]
