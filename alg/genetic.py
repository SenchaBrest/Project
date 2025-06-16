import base64
import random
from io import BytesIO

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deap import base, creator, tools
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import euclidean

# matplotlib.use('Agg')  # Неинтерактивный бэкенд для серверного рендеринга


def process_csv(file_path, days):
    """
    Обрабатывает CSV-файл с данными об интенсивности движения и агрегирует их по заданным дням недели.

    Функция выполняет следующие шаги:
    1. Преобразует таблицу в формат, где строки соответствуют комбинации (дата, направление движения), 
       а столбцы представляют временные метки начала накопления интенсивности.
    2. Округляет временные метки вниз до ближайшего интервала накопления.
    3. Фильтрует данные, оставляя только записи за указанные дни недели.
    4. Интерполирует пропущенные значения линейным методом.

    Параметры:
    - file_path (str): Путь к CSV-файлу с данными.
    - days (list[int]): Список номеров дней недели, для которых необходимо оставить данные 
      (0 - понедельник, 6 - воскресенье).

    Возвращает:
    dict: Словарь с обработанными данными:
        - "id" (int): Идентификатор `dkNum`.
        - "dates" (numpy.ndarray): Уникальные даты после фильтрации.
        - "timestamps" (list): Округленные временные метки.
        - "directions" (numpy.ndarray): Уникальные направления движения.
        - "intensities" (numpy.ndarray): Трехмерный массив интенсивностей движения, 
          где размеры соответствуют (количество дней, количество направлений, количество временных меток).
    """

    df = pd.read_csv(file_path)

    # id
    id = int(df['dkNum'].iloc[0])

    df['date'] = pd.to_datetime(df['date'], format="%d-%m-%y")
    accumulationInterval = df['accumulationInterval'].iloc[0]
    df = df.pivot(index=['date', 'directionNum'],
                  columns='accumulationStartTime', values='intensity')

    def floor_to(time_str):
        dt = pd.to_datetime(time_str, format='%H:%M:%S')
        minute = dt.minute - (dt.minute % accumulationInterval)
        floored = dt.replace(minute=minute, second=0)
        return floored.strftime('%H:%M:%S')

    # timestamps
    timestamps = [col for col in df.columns if col not in [
        'accumulationStartTime', 'date', 'directionNum']]

    rename_mapping = {col: floor_to(col) for col in timestamps}

    time_df = df[timestamps].rename(columns=rename_mapping)

    aggregated_time = pd.DataFrame(index=time_df.index)
    for new_col in time_df.columns.unique():
        subset = time_df.loc[:, time_df.columns == new_col]
        aggregated_time[new_col] = subset.sum(axis=1, min_count=1)
    df = aggregated_time
    timestamps = [col for col in df.columns if col not in [
        'accumulationStartTime', 'date', 'directionNum']]

    weekdays_df = df[df.index.get_level_values('date').dayofweek.isin(days)]
    weekdays_df = weekdays_df.interpolate(
        method='linear', axis=1, limit_direction='both')

    indices = list(weekdays_df.index)
    dates, directions = zip(*indices)
    # dates
    dates = np.unique(np.array([d.strftime('%Y-%m-%d') for d in dates]))
    # directions
    directions = np.unique(np.array(directions))
    n_directions = directions.shape[0]

    weekdays_df = weekdays_df.to_numpy()
    n_rows, n_cols = weekdays_df.shape
    # intensities
    intensities = weekdays_df.reshape(
        n_rows // n_directions, n_directions, n_cols)

    return {
        "id": id,
        "dates": dates,
        "timestamps": timestamps,
        "directions": directions,
        "intensities": intensities
    }


def partition_array_ga(array, num_parts, min_width, pop_size=300, n_gen=100,
                       cxpb=0.7, mutpb=0.2, diff_weight=1.0,
                       size_penalty_weight=0.3, range_uniformity_weight=2.0,
                       early_stopping_patience=15):
    """
    Разделяет массив на несколько частей с использованием генетического алгоритма, оптимизируя:
    1. Максимальное различие между средними значениями частей
    2. Равномерность размеров частей
    3. Схожесть диапазонов значений в частях

    Данный алгоритм не предусмотрен для разбиения на 1 или 2 части.
    Также рекомендуется выбирать количество частей не больше 18, поскольку алгоритм
    не сможет находить оптимальные границы, 
    поэтому алгоритм будет выдавать ошибку на этапе создания осособей.

    Параметры:
    - array (numpy.ndarray): MxN массив для разбиения
    - num_parts (int): Количество частей
    - min_width (int): Минимальная ширина части
    - pop_size (int): Размер популяции в генетическом алгоритме
    - n_gen (int): Максимальное число поколений
    - cxpb (float): Вероятность скрещивания
    - mutpb (float): Вероятность мутации
    - diff_weight (float): Вес фактора различий между частями
    - size_penalty_weight (float): Вес штрафа за неравномерный размер
    - range_uniformity_weight (float): Вес штрафа за несоответствие диапазонов
    - early_stopping_patience (int): Количество поколений без улучшений до остановки

    Возвращает:
    - best_solution (list[int]): Границы оптимального разбиения
    - best_fitness (float): Значение фитнес-функции лучшего решения
    - logbook (deap.tools.Logbook): Лог эволюционного процесса
    """

    M, N = array.shape

    # Проверка входных параметров
    if N < num_parts * min_width:
        raise ValueError(
            f"Невозможно создать {num_parts} частей с минимальной шириной {min_width} для массива с {N} столбцами")
    # Создание классов пригодности и особей
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Создание особи
    def create_individual():
        # Распределить дополнительные столбцы (сверх минимально необходимого) случайным образом
        extra = N - num_parts * min_width
        dividers = sorted(random.sample(range(1, extra + 1), num_parts - 1))

        # Преобразовать в фактические граничные положения
        boundaries = []
        cum_sum = 0
        for i in range(num_parts - 1):
            if i == 0:
                cum_sum += min_width + dividers[0]
            else:
                cum_sum += min_width + (dividers[i] - dividers[i-1])
            boundaries.append(cum_sum)

        return boundaries

    # Регистрация генетических операторов
    toolbox.register("individual", tools.initIterate,
                     creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat,
                     list, toolbox.individual, n=pop_size)

    # Вспомогательная функция для получения индексов частей
    def get_partitions(individual, N):
        boundaries = [0] + sorted(individual) + [N]
        return [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]

    # Функция оценки пригодности
    def evaluate(individual):
        partitions = get_partitions(individual, N)

        # Проверка того, не слишком ли мала часть
        if any(end - start < min_width for start, end in partitions):
            return (-1000,)

        mean_vectors = []
        range_vectors = []
        partition_sizes = []

        # Вычисление векторов и диапазонов частей
        for start, end in partitions:
            block = array[:, start:end]
            mean_vector = np.mean(block, axis=1)
            range_vector = np.max(block, axis=1) - np.min(block, axis=1)

            mean_vectors.append(mean_vector)
            range_vectors.append(range_vector)
            partition_sizes.append(end - start)

        # Вычисление разницы между векторами частей
        total_diff = 0
        for i in range(len(mean_vectors)):
            for j in range(i+1, len(mean_vectors)):
                total_diff += np.linalg.norm(mean_vectors[i] - mean_vectors[j])

        # Штраф за разные размеры
        ideal_size = N / num_parts
        size_std = np.std(partition_sizes)
        size_ratio = size_std / ideal_size
        size_penalty = size_ratio * size_penalty_weight

        # Штраф за большие диапазоны значений
        mean_ranges = [np.mean(range_vec) for range_vec in range_vectors]
        range_std = np.std(mean_ranges)
        range_mean = np.mean(mean_ranges)

        # Коэффициент вариации как мера однородности
        range_cv = range_std / (range_mean + 1e-10)
        range_uniformity_penalty = range_cv * range_uniformity_weight

        # Нормализация
        normalized_diff = total_diff / (num_parts * (num_parts - 1) / 2)

        # Максимизация различий, минимизация штрафов
        fitness = diff_weight * normalized_diff - \
            size_penalty - range_uniformity_penalty

        return (fitness,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxOnePoint)

    # Оператор мутации
    def mutate_boundaries(individual, indpb=0.3):
        for i in range(len(individual)):
            if random.random() < indpb:
                # Вычисление допустимого диапазона для границы
                if i == 0:
                    min_val = min_width
                else:
                    min_val = individual[i-1] + min_width

                if i == len(individual) - 1:
                    max_val = N - min_width
                else:
                    max_val = individual[i+1] - min_width

                if min_val < max_val:
                    individual[i] = random.randint(min_val, max_val)

        return (individual,)

    toolbox.register("mutate", mutate_boundaries, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Инициализация отслеживания статистики
    pop = toolbox.population()
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + stats.fields

    # Оценивание начальной популяции
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    hof.update(pop)
    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(pop), **record)
    print(logbook.stream)

    # Учитываем ранний останов
    best_fitness = hof[0].fitness.values[0]
    no_improvement_counter = 0

    for gen in range(1, n_gen + 1):
        # Селекция
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Кроссинговер
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Мутация
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Оценка особей с неверной пригодностью
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Заменяем популяцию
        pop[:] = offspring
        hof.update(pop)

        # Записываем статистику
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        print(logbook.stream)

        # проверка на ранний останов
        current_best_fitness = hof[0].fitness.values[0]
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1

        if no_improvement_counter >= early_stopping_patience:
            print(
                f"Early stopping: No improvement for {early_stopping_patience} generations (at gen {gen})")
            break

    best_solution = hof[0]
    best_fitness = best_solution.fitness.values[0]

    return best_solution, best_fitness, logbook

def partition_array_agglomerative(array, min_width=4, max_clusters=10, **kwargs):
    """
    Разделяет массив на части с помощью иерархической агломеративной кластеризации.
    Итеративно увеличивает число кластеров для обработки длинных сегментов.
    
    Параметры:
    - array: numpy.ndarray (MxN)
    - min_width: минимальная ширина сегмента
    - max_clusters: максимальное количество кластеров для итерации
    
    Возвращает:
    - best_solution: список границ сегментов (индексы)
    - best_fitness: всегда 0 (для совместимости)
    - logbook: None (для совместимости)
    """
    
    def merge_small_segments(time_series, segment_boundaries, min_segment_length):
        """
        Объединяет сегменты меньше min_segment_length с ближайшими соседями
        """
        if not segment_boundaries:
            return []
            
        # Преобразуем в список списков для удобства изменения
        boundaries = [[start, end] for start, end in segment_boundaries]
        
        while True:
            small_segments = []
            for i, (start, end) in enumerate(boundaries):
                if end - start < min_segment_length:
                    small_segments.append(i)
                    
            if not small_segments:
                break
                
            idx = small_segments[0]
            start, end = boundaries[idx]
            small_segment_mean = np.mean(time_series[start:end], axis=0)
            
            closest_idx = None
            min_distance = float('inf')
            
            # Проверяем соседа слева
            if idx > 0:
                left_start, left_end = boundaries[idx-1]
                left_mean = np.mean(time_series[left_start:left_end], axis=0)
                distance = euclidean(small_segment_mean, left_mean)
                if distance < min_distance:
                    min_distance = distance
                    closest_idx = idx - 1
            
            # Проверяем соседа справа
            if idx < len(boundaries) - 1:
                right_start, right_end = boundaries[idx+1]
                right_mean = np.mean(time_series[right_start:right_end], axis=0)
                distance = euclidean(small_segment_mean, right_mean)
                if distance < min_distance:
                    min_distance = distance
                    closest_idx = idx + 1
            
            if closest_idx is None:
                # Если нет соседей, просто удаляем этот индекс из списка маленьких сегментов
                break
                
            # Объединяем сегменты
            if closest_idx < idx:
                boundaries[closest_idx][1] = end
            else:
                boundaries[closest_idx][0] = start
                
            # Удаляем объединенный сегмент
            boundaries.pop(idx)
            
        return boundaries

    def segment_time_series_iterative(time_series, min_segment_length, n_clusters=2):
        """
        Итеративно делит временной ряд на сегменты
        """
        segments_to_process = [[0, len(time_series)]]
        final_segments = []
        
        while segments_to_process:
            start, end = segments_to_process.pop(0)
            segment = time_series[start:end]
            
            # Если сегмент меньше 2*порог, добавляем его в окончательный список и не обрабатываем дальше
            if end - start < 2 * min_segment_length:
                final_segments.append([start, end])
                continue
                
            # Применяем кластеризацию для разделения на кластеры
            try:
                clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
                labels = clustering.fit_predict(segment)
                
                # Находим точки изменения
                change_points = np.where(labels[1:] != labels[:-1])[0] + 1
                
                # Создаем подсегменты на основе точек изменения
                if len(change_points) == 0:
                    # Если не удалось разделить, добавляем весь сегмент
                    final_segments.append([start, end])
                    continue
                    
                sub_segments = []
                sub_start = 0
                
                for change_point in change_points:
                    sub_segments.append([sub_start, change_point])
                    sub_start = change_point
                    
                sub_segments.append([sub_start, len(segment)])
                
                # Объединяем маленькие сегменты
                merged_sub_segments = merge_small_segments(segment, sub_segments, min_segment_length)
                
                # Если после объединения остался только один сегмент, добавляем весь исходный сегмент
                if len(merged_sub_segments) == 1:
                    final_segments.append([start, end])
                    continue
                
                # Добавляем все подсегменты в очередь для дальнейшей обработки
                for sub_start, sub_end in merged_sub_segments:
                    segments_to_process.append([start + sub_start, start + sub_end])
                    
            except Exception as e:
                # В случае ошибки кластеризации или других проблем добавляем весь сегмент
                final_segments.append([start, end])
        
        # Сортируем сегменты по начальному индексу
        final_segments.sort(key=lambda x: x[0])
        return final_segments

    def process_large_segments_iteratively(time_series, segments, min_width, max_clusters):
        """
        Итеративно обрабатывает большие сегменты, увеличивая количество кластеров
        """
        current_segments = segments.copy()
        
        # Начинаем с 3 кластеров (2 уже были использованы в первичной сегментации)
        n_clusters = 3
        
        while n_clusters <= max_clusters:
            # Проверяем наличие слишком длинных сегментов
            long_segments_exist = False
            for start, end in current_segments:
                if end - start > 2 * min_width:
                    long_segments_exist = True
                    break
            
            # Если длинных сегментов нет, завершаем процесс
            if not long_segments_exist:
                break
            
            # Обрабатываем длинные сегменты
            new_segments = []
            
            for start, end in current_segments:
                segment_width = end - start
                
                # Если сегмент больше чем 2*min_width, применяем кластеризацию
                if segment_width > 2 * min_width:
                    # Извлекаем данные текущего сегмента
                    segment_data = time_series[start:end]
                    
                    # Применяем segmentation с увеличенным количеством кластеров
                    # Только если длина сегмента позволяет разделить на такое количество кластеров
                    if segment_width >= n_clusters * min_width:
                        sub_segments = segment_time_series_iterative(segment_data, min_width, n_clusters=n_clusters)
                        
                        # Преобразуем индексы подсегментов с учетом смещения
                        for sub_start, sub_end in sub_segments:
                            new_segments.append([start + sub_start, start + sub_end])
                    else:
                        # Если сегмент слишком мал для данного количества кластеров,
                        # просто добавляем его без изменений
                        new_segments.append([start, end])
                else:
                    # Для маленьких сегментов просто добавляем их в список
                    new_segments.append([start, end])
            
            # Обновляем текущие сегменты и увеличиваем количество кластеров для следующей итерации
            current_segments = sorted(new_segments, key=lambda x: x[0])
            n_clusters += 1
        
        return current_segments

    if array.shape[0] < array.shape[1]:
        time_series = array.T
    else:
        time_series = array
        
    min_segment_length = min_width
    
    # Первичная сегментация с использованием 2 кластеров
    initial_segments = segment_time_series_iterative(time_series, min_segment_length)
    
    # Итеративно обрабатываем длинные сегменты
    final_segments = process_large_segments_iteratively(
        time_series, 
        initial_segments, 
        min_segment_length, 
        max_clusters
    )
    
    # Формируем список границ сегментов
    boundaries = [seg[0] for seg in final_segments] + [final_segments[-1][1]]
    
    return boundaries, 0, None


def get_min_max_values4parts(array, partition_boundaries):
    """
    Вычисляет минимальные и максимальные значения для каждой части разбиения 2D-массива.

    Параметры:
    - array: Двумерный NumPy массив (MxN), который необходимо разделить.
    - partition_boundaries: Границы разбиения (список индексов, определяющих начало и конец частей).

    Возвращает:
    - Список NumPy массивов, где каждый элемент списка представляет собой массив формы (2, M), содержащий:
        - В первой строке: минимальные значения каждой строки в соответствующей части.
        - Во второй строке: максимальные значения каждой строки в соответствующей части.

    Пример:
    >>> array = np.array([[1, 2, 3, 10], [4, 5, 6, 20]])
    >>> partition_boundaries = [0, 2, 4]
    >>> get_min_max_values4parts(array, partition_boundaries)
    [array([[1, 4],   # минимальные значения для первой части
            [2, 5]]),  # максимальные значения для первой части
     array([[ 3,  6],  # минимальные значения для второй части
            [10, 20]])] # максимальные значения для первой части
    """

    partition_boundaries = np.array(partition_boundaries)
    parts = np.column_stack(
        (partition_boundaries[:-1], partition_boundaries[1:]))
    min_max_vectors = []
    for start, end in parts:
        block = array[:, start:end]
        min_max_vectors.append(
            np.array([np.min(block, axis=1), np.max(block, axis=1)]))

    return min_max_vectors


def compute_difference_matrix(part_vectors):
    """
    Вычисляет матрицу попарных разностей между векторами частей.

    Эта функция рассчитывает евклидово расстояние между всеми парами векторов 
    из входного списка, создавая симметричную матрицу, где каждый элемент 
    представляет расстояние между двумя векторами.

    Параметры:
    - part_vectors: Список или массив из N NumPy-массивов, где каждый массив 
      представляет собой вектор в некотором пространстве признаков.

    Возвращает:
    - diff_matrix: NumPy-массив формы (N, N), где элемент (i, j) содержит 
      евклидово расстояние между part_vectors[i] и part_vectors[j].
    """

    num_parts = len(part_vectors)
    diff_matrix = np.zeros((num_parts, num_parts))
    for i in range(num_parts):
        for j in range(num_parts):
            diff_matrix[i, j] = np.linalg.norm(
                part_vectors[i] - part_vectors[j])
    return diff_matrix


def cluster_parts_by_threshold(diff_matrix, threshold_percentile=25):
    """
    Объединяет части на основе порогового значения, вычисленного по процентилю матрицы различий.

    Параметры:
    - diff_matrix: NumPy-массив (NxN), содержащий попарные различия между векторами частей.
    - threshold_percentile: Процентиль (от 0 до 100), используемый для определения порога слияния.

    Возвращает:
    - cluster_labels: NumPy-массив длины N, содержащий метки кластеров для каждой части.
    - threshold: Числовое значение порога, использованное для слияния.
    """

    # Вычисление порога на основе процентиля из верхнего треугольника матрицы (без диагонали)
    upper_triangle = diff_matrix[np.triu_indices_from(diff_matrix, k=1)]
    threshold = np.percentile(upper_triangle, threshold_percentile)

    # Инициализация: каждая часть изначально в своем кластере
    num_parts = diff_matrix.shape[0]
    cluster_labels = np.arange(num_parts)

    # Итеративное объединение кластеров
    while True:
        merged = False
        min_diff = float('inf')
        merge_i, merge_j = -1, -1

        # Поиск пар кластеров с минимальной разницей, которую можно объединить
        for i in range(num_parts):
            for j in range(i + 1, num_parts):
                if cluster_labels[i] != cluster_labels[j] and diff_matrix[i, j] < threshold:
                    if diff_matrix[i, j] < min_diff:
                        min_diff = diff_matrix[i, j]
                        merge_i, merge_j = i, j

        # Если нашли кластеры для объединения
        if min_diff < threshold:
            old_label = cluster_labels[merge_j]
            new_label = cluster_labels[merge_i]
            cluster_labels[cluster_labels == old_label] = new_label
            merged = True

        # Если больше объединять нечего, выходим
        if not merged:
            break

    # Нормализация меток кластеров (чтобы они шли подряд, начиная с 0)
    unique_labels = np.unique(cluster_labels)
    normalized_labels = np.zeros_like(cluster_labels)
    for i, label in enumerate(unique_labels):
        normalized_labels[cluster_labels == label] = i

    return normalized_labels, threshold


def analyze_clusters(array, partition_boundaries, cluster_labels):
    """
    Анализирует кластеры, вычисляя статистические характеристики для визуализации.

    Параметры:
    - array: NumPy-массив размером (M, N), представляющий исходные данные.
    - partition_boundaries: Список границ разбиения, содержащий (num_parts + 1) элементов.
    - cluster_labels: NumPy-массив меток кластеров длины num_parts.

    Возвращает:
    - cluster_stats: Словарь, содержащий статистику и границы регионов для каждого кластера.

    Формат `cluster_stats`:
    {
        cluster_id: {
            'mean_values': NumPy-массив (M,) — средние значения по строкам,
            'range_values': NumPy-массив (M,) — диапазон значений по строкам (макс - мин),
            'region_bounds': Список кортежей (start, end, center) — границы объединённых регионов.
        },
        ...
    }
    """

    M, N = array.shape
    num_parts = len(partition_boundaries) - 1
    num_clusters = len(np.unique(cluster_labels))

    # Инициализация структуры для хранения статистики
    cluster_stats = {cluster_id: {'parts': [], 'mean_values': np.zeros(M), 'range_values': np.zeros(M)}
                     for cluster_id in range(num_clusters)}

    # Заполнение статистики для каждого кластера
    for i in range(num_parts):
        start, end = partition_boundaries[i], partition_boundaries[i+1]
        cluster_id = cluster_labels[i]

        # Добавление интервала разбиения в список кластера
        cluster_stats[cluster_id]['parts'].append((start, end))

        # Обновление средних значений и диапазона по строкам
        for row in range(M):
            data = array[row, start:end]
            current_mean = cluster_stats[cluster_id]['mean_values'][row]
            current_range = cluster_stats[cluster_id]['range_values'][row]
            current_count = len(cluster_stats[cluster_id]['parts']) - 1

            if current_count == 0:  # Первая часть в кластере
                cluster_stats[cluster_id]['mean_values'][row] = np.mean(data)
                cluster_stats[cluster_id]['range_values'][row] = np.max(
                    data) - np.min(data)
            else:
                # Инкрементальное обновление среднего и диапазона значений
                new_count = current_count + 1
                new_mean = (current_mean * current_count +
                            np.mean(data)) / new_count
                new_range = max(current_range, np.max(data) - np.min(data))

                cluster_stats[cluster_id]['mean_values'][row] = new_mean
                cluster_stats[cluster_id]['range_values'][row] = new_range

    # Определение объединённых регионов для визуализации
    for cluster_id in range(num_clusters):
        parts = sorted(cluster_stats[cluster_id]['parts'])
        contiguous_regions = []
        current_region = [parts[0]]

        for i in range(1, len(parts)):
            if parts[i][0] == current_region[-1][1]:
                current_region.append(parts[i])
            else:
                contiguous_regions.append(current_region)
                current_region = [parts[i]]

        contiguous_regions.append(current_region)

        # Формирование границ регионов
        region_bounds = [(region[0][0], region[-1][1], (region[0][0] + region[-1][1]) / 2)
                         for region in contiguous_regions]

        cluster_stats[cluster_id]['region_bounds'] = region_bounds
        del cluster_stats[cluster_id]['parts']  # Удаляем, так как уже не нужно

    return cluster_stats


def visualize_clusters(array, partition_boundaries, cluster_labels, cluster_stats, times, directions, title=""):
    """
    Визуализирует массив данных с разбиением на кластеры.

    Отображает временные ряды, выделяя участки разбиения разными цветами в 
    соответствии с принадлежностью кластерам. Также отображает средние значения 
    и диапазоны изменений для кластеров с адаптивным расположением подписей.

    Параметры:
    - array: NumPy-массив размером (M, N), представляющий исходные данные.
    - partition_boundaries: Список границ разбиения (num_parts + 1 элементов).
    - cluster_labels: NumPy-массив меток кластеров длины num_parts.
    - cluster_stats: Словарь со статистикой кластеров, полученный из analyze_clusters.
    - times: Список временных меток (размер N).
    - directions: Список направлений (размер M).
    - title: Заголовок графика (по умолчанию пустой).

    Возвращает:
    - image_base64: График в формате base64 (PNG).
    """

    M, N = array.shape
    num_parts = len(partition_boundaries) - 1
    num_clusters = len(np.unique(cluster_labels))

    assert len(
        cluster_labels) == num_parts, f"Ожидалось {num_parts} меток кластеров, получено {len(cluster_labels)}"

    fig, axes = plt.subplots(M, 1, figsize=(15, 3*M), sharex=True)
    if M == 1:
        axes = [axes]

    # Генерируем цвета для кластеров
    cluster_colors = plt.cm.tab20(np.linspace(0, 1, num_clusters))

    # Визуализация данных
    for row in range(M):
        ax = axes[row]
        ax.plot(range(N), array[row, :], 'k-', linewidth=1)
        ax.set_ylabel(f"Направление {directions[row]}")

        # Определяем границы по оси Y
        y_min, y_max = np.min(array[row, :]), np.max(array[row, :])
        y_margin = 0.05 * (y_max - y_min)

        # Закрашиваем области кластеров
        for i in range(num_parts):
            start, end = partition_boundaries[i], partition_boundaries[i+1]
            cluster_id = cluster_labels[i]

            ax.axvspan(start-0.5, end-0.5, alpha=0.3,
                       color=cluster_colors[cluster_id])

            # Отображаем границы разбиений
            if i > 0:
                linestyle = ':' if cluster_labels[i] == cluster_labels[i-1] else '-'
                color = 'gray' if linestyle == ':' else 'red'
                ax.axvline(start-0.5, color=color,
                           linestyle=linestyle, alpha=0.7)

        # Размещаем подписи с параметрами кластеров
        used_positions = []
        min_horizontal_gap = N / 15

        for cluster_id in range(num_clusters):
            mean_val = cluster_stats[cluster_id]['mean_values'][row]
            range_val = cluster_stats[cluster_id]['range_values'][row]

            for start, end, center in cluster_stats[cluster_id]['region_bounds']:
                region_data = array[row, start:end]
                region_y_min = np.min(region_data)
                region_y_max = np.max(region_data)
                region_y_avg = np.mean(region_data)

                space_below = region_y_min - y_min
                space_above = y_max - region_y_max

                label_text = f"К{cluster_id}\nμ={mean_val:.2f}\nΔ={range_val:.2f}"

                horizontal_overlap = any(
                    abs(pos_x - center) < min_horizontal_gap for pos_x in used_positions)

                if horizontal_overlap:
                    y_text = region_y_max + y_margin if space_above > space_below else region_y_min - y_margin
                    va = 'bottom' if space_above > space_below else 'top'
                else:
                    y_text = region_y_max + 0.5 * \
                        y_margin if space_above > space_below else region_y_min - 0.5 * y_margin
                    va = 'bottom' if space_above > space_below else 'top'

                used_positions.append(center)

                ax.annotate(
                    label_text,
                    xy=(center, region_y_avg),
                    xytext=(center, y_text),
                    ha='center', va=va,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8,
                              edgecolor=cluster_colors[cluster_id], linewidth=2)
                )

    # Легенда для кластеров
    handles = [plt.Rectangle(
        (0, 0), 1, 1, color=cluster_colors[i], alpha=0.3) for i in range(num_clusters)]
    labels = [f"Кластер {i}" for i in range(num_clusters)]
    axes[0].legend(handles, labels, loc='upper right')

    axes[-1].set_xlabel("Время (часы)")
    ax.set_xticks([i * 4 for i in range(0, len(times) // 4)])
    ax.set_xticklabels([str(i) for i in range(0, len(times) // 4)])

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

    # Конвертируем изображение в base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return image_base64


def merge_parts_by_cluster(partition_boundaries, labels):
    """
    Объединяет соседние интервалы с одинаковой меткой в один общий интервал.

    Параметры:
    - partition_boundaries: 1D numpy массив с границами интервалов.
    - labels: Список или numpy массив с метками для каждого интервала.

    Возвращает:
    - merged_intervals: 2D numpy массив (K, 2) с объединенными интервалами [начало, конец].
    - merged_labels: Список меток для каждого объединенного интервала.
    """

    parts = np.column_stack(
        (partition_boundaries[:-1], partition_boundaries[1:]))
    labels = np.array(labels)

    merged_intervals = []
    merged_labels = []

    start, end = parts[0]
    current_label = labels[0]

    for i in range(len(parts)):
        if labels[i] == current_label:
            end = parts[i][1]
        else:
            merged_intervals.append((start, end))
            merged_labels.append(current_label)
            start, end = parts[i]
            current_label = labels[i]

    merged_intervals.append((start, end))
    merged_labels.append(current_label)

    return np.unique(merged_intervals), merged_labels


def get_days_names(day_numbers):
    """
    Преобразует номера дней недели в их обозначения.

    Параметры:
    - day_numbers: Список индексов дней (0 = Пн, ..., 6 = Вс).

    Возвращает:
    - Список строк с названиями дней.
    """
    days = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
    return [days[i] for i in day_numbers if 0 <= i <= 6]


def dict_to_html_table(data, directions):
    """
    Формирует HTML-таблицу с данными кластеров.

    Параметры:
    - data: Словарь с данными кластеров.
    - directions: Список направлений (заголовки столбцов).

    Возвращает:
    - Строку с HTML-таблицей.
    """

    M = len(directions)
    html = "<table border='1' cellspacing='0' cellpadding='5'>\n"

    # Первая строка
    html += "  <tr>\n"
    html += "    <th rowspan='2'>Кластер</th>\n"
    html += f"    <th colspan='{M}'>Среднее значение</th>\n"
    html += f"    <th colspan='{M}'>Диапазон значений</th>\n"
    html += "    <th rowspan='2'>Временные границы кластера</th>\n"
    html += "  </tr>\n"

    # Вторая строка
    html += "  <tr>\n"
    for d in directions:
        html += f"    <th>{d}</th>\n"
    for d in directions:
        html += f"    <th>{d}</th>\n"
    html += "  </tr>\n"

    # Заполнение строк таблицы
    for key, value in data.items():
        mean_vals = value.get('mean_values', [])
        range_vals = value.get('range_values', [])
        region_bounds = ", ".join(" - ".join(bound)
                                  for bound in value.get('region_bounds', []))

        html += "  <tr>\n"
        html += f"    <td>{key}</td>\n"

        for i in range(M):
            html += f"    <td>{round(mean_vals[i])}</td>\n"

        for i in range(M):
            html += f"    <td>{round(range_vals[i])}</td>\n"

        html += f"    <td>{region_bounds}</td>\n"
        html += "  </tr>\n"

    html += "</table>"
    return html


def analyze(file_path, days, num_parts=13, threshold=10, min_width=4):
    """
    Выполняет анализ данных, включая разбиение, кластеризацию и визуализацию.

    Параметры:
    - file_path: Путь к CSV-файлу.
    - days: Дни недели, для которых производится анализ.
    - num_parts: Количество частей при разбиении.
    - threshold: Пороговое значение для кластеризации.
    - min_width: Минимальная ширина сегмента.

    Возвращает:
    - HTML-таблицу с характеристиками кластеров.
    - Изображение графика в base64.
    """
    data = process_csv(file_path, days)
    id, date, timestamps, directions, intensities = data['id'], data[
        'dates'], data['timestamps'], data['directions'], data['intensities']

    X = np.mean(intensities, axis=0)
    X_normalized = (X - X.mean(axis=1, keepdims=True)) / \
        (X.std(axis=1, keepdims=True) + 1e-8)
    M, N = X.shape

    best_solution, best_fitness, logbook = partition_array_ga(
        X_normalized, num_parts, min_width,
        pop_size=1000, # большая популяция для большей стабильности
        n_gen=50,
        diff_weight=1.0,
        size_penalty_weight=0.2,
        range_uniformity_weight=2.5
    )
    partition_boundaries = np.array([0] + sorted(best_solution) + [N])

    diff_matrix = compute_difference_matrix(
        get_min_max_values4parts(X_normalized, partition_boundaries))
    final_cluster_labels, threshold = cluster_parts_by_threshold(
        diff_matrix, threshold_percentile=threshold)
    partition_boundaries, final_cluster_labels = merge_parts_by_cluster(
        partition_boundaries, final_cluster_labels)

    cluster_stats = analyze_clusters(
        X, partition_boundaries, final_cluster_labels)

    image_base64 = visualize_clusters(
        X, partition_boundaries, final_cluster_labels, cluster_stats, timestamps, directions,
        title=f"{id}: {get_days_names(np.unique(days))}"
    )

    # Заменяем границы в виде индексов на временные метки
    for value in cluster_stats.values():
        value['region_bounds'] = [(timestamps[max(
            int(i) - 1, 0)], timestamps[int(j) - 1]) for i, j, _ in value['region_bounds']]

    cluster_stats_html_table = dict_to_html_table(cluster_stats, directions)

    return cluster_stats_html_table, image_base64


