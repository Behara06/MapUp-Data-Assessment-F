from typing import List

import pandas as pd
import numpy as np


def create_task1_df() -> pd.DataFrame:
    data_setpath = r'../datasets/dataset-1.csv'
    df = pd.read_csv(data_setpath, header="infer", delimiter=',', index_col=None)
    return df


def create_task_df() -> pd.DataFrame:
    data_setpath = r'../datasets/dataset-2.csv'
    df = pd.read_csv(data_setpath, header="infer", delimiter=',', index_col=None)
    return df


def generate_car_matrix(df) -> pd.DataFrame:
    """
    Creates a DataFrame  for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    # Write your logic here
    df = df[['id_1', 'id_2', 'car', 'rv', 'bus', 'truck']].set_index('id_1').pivot(columns=['id_2'])
    df = df.fillna(0)
    return df


def get_type_count(df) -> list[str]:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    # Write your logic here
    car_type_dict = dict()
    car_type_lambda = lambda x: 'low' if x <= 15 else 'medium' if 15 < x <= 25 else 'high'
    df['car_type'] = df['car'].apply(car_type_lambda)
    df_count_dict = df[['car_type', 'car']].groupby('car_type').count()
    car_type_dict['car_type'] = df_count_dict.to_dict()['car']
    sorted_car_type_dict = sorted(car_type_dict.items(), key=lambda x: x[0], reverse=False)
    return sorted_car_type_dict


def get_bus_indexes(df) -> list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    # Write your logic here
    bus_mean_val = round(df['bus'].mean(), 2)
    df['bus_mean_val'] = bus_mean_val
    index_list = df.index[df['bus'] > 2 * df['bus_mean_val']]
    return sorted(index_list.to_list())


def filter_routes(df) -> list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    # Write your logic here
    df['avg_truck'] = round(np.average(df['truck']), 2)
    route_list = df[['route']].where(df['avg_truck'] > 7, df[['route']])
    sorted_route_list = sorted(route_list['route'].values.tolist())

    return sorted_route_list


def multiply_matrix(matrix) -> pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    # Write your logic here
    for row in matrix.index:
        for col in matrix:
            val = matrix.at[row, col]
            if val > 20:
                new_val = round(val * 0.75, 1)
            else:
                new_val = round(val * 1.25, 1)
            matrix.at[row, col] = new_val
    return matrix


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    df = df[['id', 'id_2', 'startTime']]
    required_timestamps = [pd.Timestamp(x).time() for x in
                           pd.date_range(start='00:00:00', end='23:59:59', freq='H').astype(str)]
    missing_timestamps = {}
    for id_val, id_2_val in zip(df['id'], df['id_2']):
        timestamps = df[(df['id'] == id_val) & (df['id_2'] == id_2_val)]['startTime'].tolist()

        missing_timestamps[(id_val, id_2_val)] = [timestamp for timestamp in required_timestamps if
                                                  timestamp not in timestamps]
    missing_timestamps_df = pd.DataFrame.from_dict(missing_timestamps, orient='index')
    missing_timestamps_df.reset_index(inplace=True)
    missing_timestamps_df.set_index(['id', 'id_2'], inplace=True)
    print(missing_timestamps_df)
    # print(missing_timestamps_df)
    # result = missing_timestamps_df.astype(bool).any(axis=1)
    # print(result)

    return pd.Series()


if __name__ == '__main__':
    dataset_1_full_df = create_task1_df()
    dataset_2_full_df = create_task_df()
    generate_car_matrix_df = generate_car_matrix(dataset_1_full_df)
    print(generate_car_matrix_df.head(5))
    type_count_dict = get_type_count(dataset_1_full_df)
    print(type_count_dict)
    bus_indexes = get_bus_indexes(dataset_1_full_df)
    print(bus_indexes)
    routes_list = filter_routes(dataset_1_full_df)
    print(routes_list)
    matrix_df = multiply_matrix(generate_car_matrix_df)
    print(matrix_df)
    series = time_check(dataset_2_full_df)
