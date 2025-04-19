import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


def plot_location_change_distribution(location_changes_df, threshold=45):
    """
    Строит график распределения числа изменений местоположения с логарифмической шкалой по оси Y.

    :param location_changes_df: DataFrame, содержащий столбец 'change_count' — количество изменений местоположения.
    :param threshold: int, значения больше или равные которому будут объединены в одну категорию "threshold+".
    """
    change_count_dist = location_changes_df['change_count'].value_counts().sort_index()
    above_threshold = change_count_dist[change_count_dist.index >= threshold].sum()
    change_count_dist = change_count_dist[change_count_dist.index < threshold]

    change_count_dist[threshold] = above_threshold

    plt.figure(figsize=(10, 6))
    plt.bar(change_count_dist.index, change_count_dist.values, color='skyblue')
    plt.yscale('log')  
    xticks = [i for i in range(1, threshold, 5)] + [threshold]
    plt.xticks(xticks, [str(i) if i < threshold else f'{threshold}+' for i in xticks])

    plt.xlabel('Число изменений местоположения (город/IP)')
    plt.ylabel('Количество пользователей (логарифмическая шкала)')
    plt.title('Распределение числа изменений местоположения среди пользователей')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.show()



def plot_top_anomalous_cities(city_activity_df, top_n=5):
    """
    Визуализирует временные ряды активности для топ-N городов с наибольшим числом аномалий.

    :param city_activity_df: DataFrame, полученный из analyze_city_activity
    :param top_n: количество городов для отображения
    """
    sns.set(style="whitegrid")  
    top_cities = (
        city_activity_df[city_activity_df['is_anomaly']]
        .groupby('geo_city_id')
        .size()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    num_plots = len(top_cities)
    fig, axes = plt.subplots(num_plots, 1, figsize=(14, num_plots * 4), sharex=True)

    if num_plots == 1:
        axes = [axes]

    for ax, city_id in zip(axes, top_cities):
        city_data = city_activity_df[city_activity_df['geo_city_id'] == city_id]
        anomalies = city_data[city_data['is_anomaly']]

        ax.plot(
            city_data['ts'], city_data['event_count'],
            color='steelblue', linewidth=2, label='Число событий'
        )

        ax.scatter(
            anomalies['ts'], anomalies['event_count'],
            color='crimson', s=60, edgecolors='white', linewidth=1.5,
            label='Аномалия', zorder=5
        )

        ax.set_title(f'Город ID {city_id} — активность и аномалии', fontsize=14, fontweight='bold')
        ax.set_ylabel('События', fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.xlabel('Время', fontsize=12)
    plt.tight_layout()
    plt.show()




def plot_top_suspicious_ips(suspicious_ips_df, top_n=10):
    """
    Визуализирует топ-N подозрительных IP-адресов по числу уникальных пользователей.

    :param suspicious_ips_df: DataFrame, полученный из detect_suspicious_ips
    :param top_n: Сколько IP отобразить
    """
    sns.set(style="whitegrid")
    top_ips = suspicious_ips_df.head(top_n).copy()
    top_ips = top_ips.sort_values("unique_users", ascending=True)

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=top_ips,
        x='ip',
        y='unique_users',
        color='skyblue' 
    )

    plt.xlabel('Уникальные пользователи', fontsize=12)
    plt.ylabel('IP-адрес', fontsize=12)
    plt.title(f'Топ-{top_n} подозрительных IP по числу пользователей', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()



def plot_ip_bubble_chart(suspicious_ips_df):
    """
    Строит bubble chart: пользователи vs действия, размер = длительность активности.

    :param suspicious_ips_df: DataFrame, полученный из detect_suspicious_ips
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    sizes = suspicious_ips_df['activity_period'].dt.total_seconds() / 60  # в минутах

    plt.scatter(
        suspicious_ips_df['unique_users'],
        suspicious_ips_df['total_actions'],
        s=sizes,
        alpha=0.6,
        color='tomato',
        edgecolors='black'
    )

    for _, row in suspicious_ips_df.iterrows():
        plt.text(row['unique_users'], row['total_actions'], row['ip'], fontsize=8, alpha=0.7)

    plt.xlabel('Уникальные пользователи')
    plt.ylabel('Общее количество действий')
    plt.title('Анализ подозрительных IP: активность и масштаб')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()



def plot_anomalous_time_windows(anomalous_df, full_stats_df):
    """
    Визуализирует динамику категорий сессий по времени и выделяет аномальные окна.

    :param anomalous_df: DataFrame с аномальными временными окнами (из detect_anomalous_time_windows)
    :param full_stats_df: Полный time_window_stats до фильтрации, чтобы видеть всю картину
    """
    plt.figure(figsize=(14, 6))

    plt.stackplot(full_stats_df.index,
                  full_stats_df['short'],
                  full_stats_df['medium'],
                  full_stats_df['long'],
                  labels=['short', 'medium', 'long'],
                  colors=['#7bc8f6', '#aad8b0', '#f7b89c'],
                  alpha=0.8)

    for t in anomalous_df.index:
        plt.axvline(x=t, color='red', linestyle='--', alpha=0.6)

    plt.legend(loc='upper left')
    plt.title('Распределение типов сессий по времени с выделением аномалий', fontsize=14, fontweight='bold')
    plt.xlabel('Временное окно')
    plt.ylabel('Доля сессий')
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()


def plot_anomalous_page_views(anomalies_df):
    """
    Визуализирует аномалии по title (результат detect_anomalous_page_views).

    Параметры:
        anomalies_df (pd.DataFrame): DataFrame с колонками ['time_window', 'url', 'title', 'growth'].
    """
    if anomalies_df.empty:
        print("Нет аномалий для отображения.")
        return

    plt.figure(figsize=(14, 6))
    sns.scatterplot(
        data=anomalies_df,
        x='time_window',
        y='title',
        size='growth',
        hue='growth',
        palette='Reds',
        sizes=(50, 300),
        legend='brief'
    )

    plt.title('Обнаруженные аномалии по названиям страниц (title)')
    plt.xlabel('Временное окно')
    plt.ylabel('Заголовок страницы (title)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend(title='Рост')
    plt.show()



def plot_anomalous_users(user_df):
    """
    Визуализирует среднее время, проведенное пользователями, с выделением аномалий.

    Параметры:
        user_df (pd.DataFrame): DataFrame с колонками ['randPAS_user_passport_id', 'avg_time_spent', 'is_anomalous'].
    """
    if user_df.empty:
        print("Нет данных для визуализации.")
        return

    user_df = user_df.copy()
    user_df['user_id_str'] = user_df['randPAS_user_passport_id'].astype(str)

    plt.figure(figsize=(14, 6))
    sns.scatterplot(
        data=user_df,
        x='user_id_str',
        y='avg_time_spent',
        hue='is_anomalous',
        palette={1: 'blue', -1: 'red'},
        s=100
    )

    plt.title('Анализ пользователей по среднему времени на странице')
    plt.xlabel('Пользователь')
    plt.ylabel('Среднее время на странице')
    plt.xticks([], [])  
    plt.legend(title='Аномалия', labels=['Нормальный', 'Аномалия'])
    plt.tight_layout()
    plt.show()



def plot_device_share_changes(device_shares, anomalous_windows=None):
    """
    Визуализирует изменение долей типов устройств во времени с подсветкой аномалий.

    Параметры:
        device_shares (pd.DataFrame): Доля устройств по временным окнам.
        anomalous_windows (pd.DataFrame или None): Окна, в которых были аномалии.
    """
    plt.figure(figsize=(14, 6))
    device_shares.plot(ax=plt.gca(), marker='o')

    if anomalous_windows is not None and not anomalous_windows.empty:
        for ts in anomalous_windows.index:
            plt.axvline(ts, color='red', linestyle='--', alpha=0.5)

    plt.title('Доля типов устройств во времени')
    plt.xlabel('Временное окно')
    plt.ylabel('Доля')
    plt.legend(title='Тип устройства')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

