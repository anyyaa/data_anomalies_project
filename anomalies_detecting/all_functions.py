import pandas as pd
from typing import Dict, List, Union, Optional
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def check_duplicate_events(df: pd.DataFrame) -> Dict:
    """Проверка дубликатов событий с одинаковыми ts и event"""
    results = {'duplicates': []}
    duplicate_cols = ['ts', 'event']

    if 'counter_id' in df.columns:
        duplicate_cols.append('counter_id')

    duplicates = df[df.duplicated(subset=duplicate_cols, keep=False)]
    if not duplicates.empty:
        results['duplicates'] = []
        for _, row in duplicates.iterrows():
            results['duplicates'].append({
                'message': f"Дубликат события: ts={row['ts']}, event={row['event']}.",
                'session_id': row.get('randPAS_session_id', 'Не указано'),
                'ip': row.get('ip', 'Не указано')
            })
    else:
        results['message'] = "Дубликаты событий не найдены."
    return results


def check_session_start(df: pd.DataFrame) -> Dict:
    """Проверка, что сессии начинаются с 1"""
    results = {'start_errors': []}

    if 'randPAS_session_id' not in df.columns:
        raise KeyError("Отсутствует столбец randPAS_session_id")

    first_events = df.groupby('randPAS_session_id').first()

    start_errors = first_events[
        (first_events['page_view_order_number'] != 1) |
        (first_events['event_order_number'] != 1)
        ]

    if not start_errors.empty:
        results['start_errors'] = []
        for session_id, row in start_errors.iterrows():
            results['start_errors'].append({
                'message': f"Сессия {session_id} не начинается с 1.",
                'first_page_view': int(row['page_view_order_number']),
                'first_event': int(row['event_order_number']),
                'ip': row.get('ip', 'Не указано')
            })
    else:
        results['message'] = "Все сессии начинаются с 1."

    return results


def check_order_relation(df: pd.DataFrame) -> Dict:
    """Проверка соотношения page_view и event order numbers"""
    results = {'relation_errors': []}

    relation_errors = df[df['event_order_number'] < df['page_view_order_number']]

    if not relation_errors.empty:
        results['relation_errors'] = []
        for session_id, group in relation_errors.groupby('randPAS_session_id'):
            results['relation_errors'].append({
                'message': f"Несоответствие между номерами событий и просмотров страниц для сессии {session_id}.",
                'count': len(group),
                'ip': group['ip'].iloc[0] if 'ip' in group.columns else 'Не указано',
                'examples': group[['ts', 'page_view_order_number', 'event_order_number']].head(3).to_dict('records')
            })
    else:
        results['message'] = "Все номера событий соответствуют номерам просмотров страниц."

    return results


def check_numbering_sequence(df: pd.DataFrame) -> Dict:
    """Проверка пропусков или неправильной нумерации событий для каждого пользователя"""
    results = {'missing_numbers': []}

    grouped = df.groupby('randPAS_session_id')

    for session_id, group in grouped:
        group = group.sort_values('ts').reset_index(drop=True)

        for i in range(1, len(group)):
            current_event = group.iloc[i]
            previous_event = group.iloc[i - 1]

            if current_event['event_order_number'] != previous_event['event_order_number'] + 1:
                results['missing_numbers'].append({
                    'session_id': session_id,
                    'message': f"Номер события {current_event['event_order_number']} не соответствует {previous_event['event_order_number'] + 1} в сессии {session_id}.",
                    'ip': current_event['ip'],
                    'previous_event_ts': previous_event['ts'],
                    'current_event_ts': current_event['ts'],
                    'previous_event': previous_event['event'],
                    'current_event': current_event['event']
                })

    if not results['missing_numbers']:
        results['message'] = "Пропусков или ошибок в нумерации событий не найдено."

    return results


def detect_location_changes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Анализирует смену местоположения (geo_city_id и ip) для каждого пользователя.
    Возвращает DataFrame с информацией о сменах местоположения и временных интервалах.
    """
    df_sorted = df.sort_values(['randPAS_user_passport_id', 'ts'])
    grouped = df_sorted.groupby('randPAS_user_passport_id')

    results = []

    for user_id, group in grouped:
        user_data = group[['ts', 'ip', 'geo_city_id']].drop_duplicates()

        changes = user_data[
            (user_data['geo_city_id'].shift() != user_data['geo_city_id']) |
            (user_data['ip'].shift() != user_data['ip'])
            ].copy()
        if len(changes) > 1:
            changes['time_diff'] = changes['ts'].diff().dt.total_seconds()
            city_changes = list(zip(
                changes['geo_city_id'].astype(str),
                changes['time_diff'].astype(str)
            ))

            results.append({
                'user_id': user_id,
                'city_changes': " → ".join([f"{city} ({time}s)" for city, time in city_changes]),
                'change_count': len(changes) - 1,
                'first_change': changes['ts'].iloc[1],
                'last_change': changes['ts'].iloc[-1],
                'unique_cities': changes['geo_city_id'].nunique(),
                'unique_ips': changes['ip'].nunique()
            })

    return pd.DataFrame(results)


def analyze_city_activity(df, min_events=10, z_threshold=3, rolling_window='1H'):
    """
    Анализирует активность по городам и выявляет аномальные всплески

    Параметры:
        df: DataFrame с данными
        min_events: минимальное количество событий для анализа города
        z_threshold: порог для детектирования аномалий (в стандартных отклонениях)
        rolling_window: размер окна для скользящей статистики

    Возвращает:
        DataFrame с результатами анализа
    """

    if not pd.api.types.is_datetime64_any_dtype(df['ts']):
        df['ts'] = pd.to_datetime(df['ts'])
    city_activity = df.groupby(['geo_city_id', pd.Grouper(key='ts', freq=rolling_window)]) \
        .size() \
        .reset_index(name='event_count')
    # print(city_activity)
    city_stats = city_activity.groupby('geo_city_id')['event_count'].agg(['count', 'mean', 'std'])
    valid_cities = city_stats[city_stats['count'] > 5].index
    city_activity = city_activity[city_activity['geo_city_id'].isin(valid_cities)]

    city_activity['z_score'] = city_activity.groupby('geo_city_id')['event_count'] \
        .transform(lambda x: (x - x.mean()) / x.std())

    city_activity['is_anomaly'] = city_activity['z_score'] > z_threshold

    total_activity = df.groupby(pd.Grouper(key='ts', freq=rolling_window)) \
        .size() \
        .reset_index(name='total_events')

    city_activity = city_activity.merge(total_activity, on='ts')
    city_activity['activity_ratio'] = city_activity['event_count'] / city_activity['total_events']

    city_activity['prev_ratio'] = city_activity.groupby('geo_city_id')['activity_ratio'].shift(1)
    city_activity['ratio_change'] = (city_activity['activity_ratio'] - city_activity['prev_ratio']) / city_activity[
        'prev_ratio']

    anomalies = city_activity[city_activity['is_anomaly']].sort_values('z_score', ascending=False)

    return anomalies, city_activity


def detect_suspicious_ips(df: pd.DataFrame,
                          max_users_per_ip: int = 10) -> pd.DataFrame:
    """
    Обнаруживает IP-адреса с аномально большим количеством пользователей

    Параметры:
        df: DataFrame с данными
        max_users_per_ip: максимальное допустимое количество пользователей с одного IP

    Возвращает:
        DataFrame с подозрительными IP и статистикой
    """
    ip_stats = (
        df.groupby('ip')
        .agg(
            unique_users=('randPAS_user_passport_id', 'nunique'),
            total_actions=('randPAS_user_passport_id', 'count'),
            first_seen=('ts', 'min'),
            last_seen=('ts', 'max')
        )
        .reset_index()
    )

    suspicious_ips = ip_stats[ip_stats['unique_users'] > max_users_per_ip]
    suspicious_ips['activity_period'] = suspicious_ips['last_seen'] - suspicious_ips['first_seen']

    return suspicious_ips.sort_values('unique_users', ascending=False)


def detect_user_activity_spikes(df: pd.DataFrame,
                                time_window_sec: int = 60,
                                max_actions: int = 30) -> pd.DataFrame:
    """
    Обнаруживает пользователей с аномально высокой активностью

    Параметры:
        df: DataFrame с данными
        time_window_sec: временное окно в секундах для анализа
        max_actions: максимальное допустимое количество действий за окно

    Возвращает:
        DataFrame с подозрительными пользователями и статистикой
    """

    df_sorted = df.sort_values(['randPAS_user_passport_id', 'ts'])
    df_sorted['time_diff'] = (
        df_sorted.groupby('randPAS_user_passport_id')['ts']
        .diff()
        .dt.total_seconds()
    )
    rapid_actions = df_sorted[df_sorted['time_diff'] < time_window_sec] \
        .groupby('randPAS_user_passport_id') \
        .agg(
        rapid_actions_count=('time_diff', 'count'),
        min_time_diff=('time_diff', 'min'),
        avg_time_diff=('time_diff', 'mean'),
        ip_list=('ip', lambda x: x.unique().tolist())
    ) \
        .reset_index()
    suspicious_users = rapid_actions[rapid_actions['rapid_actions_count'] > max_actions]
    suspicious_users['ip_count'] = suspicious_users['ip_list'].apply(len)

    return suspicious_users.sort_values('rapid_actions_count', ascending=False)


def detect_anomalous_time_windows(df, threshold=1.5, window_size='1H'):
    """
        Обнаруживает аномальные временные окна, где активность резко меняется.

        Параметры:
            df (pd.DataFrame): DataFrame с данными о времени активности.
            threshold (float): Порог для изменения пропорции активности по временным окнам.
            window_size (str): Размер временного окна для анализа.

        Возвращает:
            pd.DataFrame: DataFrame с аномальными временными окнами и их статистикой.
        """
    df = df.copy()
    df['ts'] = pd.to_datetime(df['ts'])
    df['time_window'] = df['ts'].dt.floor(window_size)

    df = df.sort_values(by=['randPAS_user_passport_id', 'ts'])
    df['time_diff'] = df.groupby('randPAS_user_passport_id')['ts'].diff().dt.total_seconds()

    df['time_category'] = pd.cut(df['time_diff'],
                                 bins=[0, 30, 300, 1800, float('inf')],
                                 labels=['short', 'medium', 'long', 'very_long'])

    time_window_stats = df.groupby('time_window')['time_category'].value_counts(normalize=True).unstack().fillna(0)

    time_window_stats['short_ratio_change'] = time_window_stats['short'].pct_change().abs().fillna(0)

    anomalous_windows = time_window_stats[time_window_stats['short_ratio_change'] > threshold]

    return anomalous_windows[['short', 'medium', 'long']]


def detect_anomalous_device_shares(df, threshold=1.5, window_size='30T'):
    """
       Обнаруживает аномальные изменения в доле устройств по временным окнам.

       Параметры:
           df (pd.DataFrame): DataFrame с данными об устройствах.
           threshold (float): Порог для обнаружения аномальных изменений в доле устройств.
           window_size (str): Размер временного окна для анализа.

       Возвращает:
           pd.DataFrame: DataFrame с аномальными изменениями в доле устройств.
       """
    df = df.copy()
    df['ts'] = pd.to_datetime(df['ts'])
    df['time_window'] = df['ts'].dt.floor(window_size)

    device_shares = df.groupby(['time_window', 'ua_device_family']).size().unstack().fillna(0)
    device_shares = device_shares.div(device_shares.sum(axis=1), axis=0)
    device_shares_change = device_shares.pct_change().abs().fillna(0)

    anomalous_windows = device_shares_change[device_shares_change.max(axis=1) > threshold]
    return anomalous_windows


def detect_anomalous_page_views(df, threshold=3, window_size='30T'):
    """
    Обнаруживает аномальные изменения в количестве просмотров страниц по временным окнам.

    Параметры:
        df (pd.DataFrame): DataFrame с данными о просмотрах страниц.
        threshold (int): Порог для обнаружения аномальных изменений в количестве просмотров страниц.
        window_size (str): Размер временного окна для анализа.

    Возвращает:
        pd.DataFrame: DataFrame с аномальными изменениями в просмотрах страниц.
    """
    df = df.copy()
    df['ts'] = pd.to_datetime(df['ts'])
    df['time_window'] = df['ts'].dt.floor(window_size)
    page_views = df.groupby(['time_window', 'url'])['randPAS_user_passport_id'].nunique().unstack().fillna(0)
    page_views_change = page_views.pct_change().abs()
    page_views_change[page_views.shift(1) == 0] = np.nan
    anomalies = page_views_change[page_views_change > threshold].stack().reset_index()
    anomalies.columns = ['time_window', 'url', 'growth']

    return anomalies.dropna()


def detect_anomalous_users(df):
    """
    Обнаруживает аномальных пользователей по времени, проведенному на страницах.

    Параметры:
        df (pd.DataFrame): DataFrame с данными о пользователях и времени их активности.

    Возвращает:
        pd.DataFrame: DataFrame с аномальными пользователями.
    """
    user_page_times = df.groupby(['randPAS_user_passport_id', 'url'])['ts'].apply(lambda x: x.max() - x.min()).reset_index()


    user_avg_times = user_page_times.groupby('randPAS_user_passport_id')['ts'].mean().reset_index()
    user_avg_times.columns = ['randPAS_user_passport_id', 'avg_time_spent']
    model = IsolationForest(contamination=0.05, random_state=42)  # contamination - доля аномальных
    user_avg_times['is_anomalous'] = model.fit_predict(user_avg_times[['avg_time_spent']])
    anomalous_users = user_avg_times[user_avg_times['is_anomalous'] == -1]

    return anomalous_users


def zscore_detector(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Обнаружение аномалий с помощью Z-Score.

    Параметры:
        data: одномерный массив данных
        threshold: пороговое значение (в стандартных отклонениях)

    Возвращает:
        Бинарный массив (1 - аномалия, 0 - норма)
    """
    z_scores = np.abs(stats.zscore(data))
    return (z_scores > threshold).astype(int)


def iqr_detector(data: np.ndarray, k: float = 1.5) -> np.ndarray:
    """
    Обнаружение аномалий с помощью межквартильного размаха (IQR).

    Параметры:
        data: одномерный массив данных
        k: множитель IQR (обычно 1.5)

    Возвращает:
        Бинарный массив (1 - аномалия, 0 - норма)
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    return ((data < lower_bound) | (data > upper_bound)).astype(int)


def modified_zscore_detector(data: np.ndarray, threshold: float = 3.5) -> np.ndarray:
    """
    Обнаружение аномалий с помощью модифицированного Z-Score (более устойчив к выбросам).

    Параметры:
        data: одномерный массив данных
        threshold: пороговое значение

    Возвращает:
        Бинарный массив (1 - аномалия, 0 - норма)
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    modified_z = 0.6745 * (data - median) / mad
    return (np.abs(modified_z) > threshold).astype(int)


def isolation_forest_detector(data: np.ndarray,
                              contamination: float = 0.05) -> np.ndarray:
    """
    Обнаружение аномалий с помощью Isolation Forest.

    Параметры:
        data: одномерный массив данных
        contamination: предполагаемая доля аномалий

    Возвращает:
        Бинарный массив (1 - аномалия, 0 - норма)
    """
    clf = IsolationForest(contamination=contamination, random_state=42)
    data_reshaped = data.reshape(-1, 1)
    preds = clf.fit_predict(data_reshaped)
    return (preds == -1).astype(int)


def lof_detector(data: np.ndarray,
                 n_neighbors: int = 20,
                 contamination: float = 0.05) -> np.ndarray:
    """
    Обнаружение аномалий с помощью Local Outlier Factor (LOF).

    Параметры:
        data: одномерный массив данных
        n_neighbors: количество соседей для анализа
        contamination: предполагаемая доля аномалий

    Возвращает:
        Бинарный массив (1 - аномалия, 0 - норма)
    """
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    data_reshaped = data.reshape(-1, 1)
    preds = lof.fit_predict(data_reshaped)
    return (preds == -1).astype(int)


def percentile_detector(data: np.ndarray,
                        lower_percentile: float = 1,
                        upper_percentile: float = 99) -> np.ndarray:
    """
    Обнаружение аномалий по перцентилям.

    Параметры:
        data: одномерный массив данных
        lower_percentile: нижний перцентиль
        upper_percentile: верхний перцентиль

    Возвращает:
        Бинарный массив (1 - аномалия, 0 - норма)
    """
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    return ((data < lower_bound) | (data > upper_bound)).astype(int)


def majority_anomaly_vote(*anomaly_arrays: np.ndarray) -> np.ndarray:
    """
    Объединяет результаты нескольких методов обнаружения аномалий.
    Аномалия отмечается, если более половины методов считают точку аномальной.

    Параметры:
        *anomaly_arrays: любое количество бинарных массивов (0 - норма, 1 - аномалия)

    Возвращает:
        Бинарный массив (1 - аномалия, 0 - норма)
    """
    if len(anomaly_arrays) == 0:
        raise ValueError("Не передано ни одного массива аномалий.")
    anomaly_matrix = np.vstack(anomaly_arrays)
    anomaly_counts = np.sum(anomaly_matrix, axis=0)
    threshold = len(anomaly_arrays) / 2
    final_anomalies = (anomaly_counts > threshold).astype(int)

    return final_anomalies


def plot_anomalies_comparison(data, **anomaly_dict):
    """
    Функция для построения графика сравнения методов обнаружения аномалий.

    Параметры:
        data: массив исходных данных
        anomaly_dict: именованные массивы аномалий (где 1 - аномалия, 0 - нет)
    """
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(data)), data, c='blue', label='Данные', alpha=0.5)
    for method_name, anomalies in anomaly_dict.items():
        anomaly_indices = np.where(anomalies == 1)[0]
        plt.scatter(anomaly_indices, data[anomaly_indices], label=method_name, alpha=0.6)

    plt.legend()
    plt.title("Сравнение методов обнаружения аномалий")
    plt.show()
