"""
Environmental audio analysis
COMP.SGN.120-2020-2021-1 Introduction to Audio Processing
Timo Hartikainen
Tampere University

Main file for audio analysis and plotting.
"""

from __future__ import annotations
from typing import Dict, List, Set, Generator, Callable
from datetime import datetime
import os
import csv
import pickle
import librosa
import numpy as np
import matplotlib.pyplot as plt

# files and directories
ANNOTATION_FILES_DIRECTORY = "csv_files"
AUDIO_FILES_DIRECTORY = "audio_files"
FIGURES_DIRECTORY = "figures"
MAX_AUDIO_FILES = 150
EXCLUDE_CLASSES = ["dog_barking"]
ALL_FILES_DUMP_FILE = "all_files_dump"

# pyplot colors
COLOR_MAP = "hot"
COLOR_RED = "firebrick"
COLOR_GREEN = "limegreen"


class SortableDictionary(dict):
    def sort_by_value(self, reverse=False):
        return dict(sorted(self.items(), key=lambda item: item[1], reverse=reverse))


class AudioFile:
    def __init__(self, directory: str, filename: str, classes: Set[str]):
        self.directory = directory
        self.filename = filename

        scene, city, _, _, _ = filename.split("-")
        self.scene = self._format_string(scene)
        self.city = self._format_string(city)
        self.place = "{}, {}".format(self.scene, self.city)

        self.classes = set([self._format_string(class_) for class_ in classes])

        self.mfcc = self._calculate_mfcc()
        self.mfcc_coefficients = self._calculate_mfcc_coefficients()

        self.cosine_similarities = SortableDictionary()
        self.dtw_costs = SortableDictionary()

    @staticmethod
    def _format_string(string: str):
        if not string:
            return "Other"

        return string.replace("_", " ").capitalize()

    def _calculate_mfcc(self):
        filepath = os.path.join(self.directory, self.filename)
        y, sr = librosa.load(filepath, sr=None)
        return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=14, n_fft=int(sr * 0.128), hop_length=int(sr * 0.032),
                                    window="hann", fmin=20, fmax=20000, n_mels=40, htk=True)

    def _calculate_mfcc_coefficients(self):
        mean = np.mean(self.mfcc, 1)
        std = np.std(self.mfcc, 1)
        return [*mean, *std]

    def calculate_cosine_similarity(self, other: AudioFile):
        if other == self:
            return 1

        if other in self.cosine_similarities.keys():
            return self.cosine_similarities[other]

        cosine_similarity = np.dot(self.mfcc_coefficients, other.mfcc_coefficients) / \
                            (np.linalg.norm(self.mfcc_coefficients) * np.linalg.norm(other.mfcc_coefficients))

        self.cosine_similarities[other] = cosine_similarity
        other.cosine_similarities[self] = cosine_similarity

        return cosine_similarity

    def calculate_dtw_cost(self, other: AudioFile):
        if other == self:
            return 0

        if other in self.dtw_costs.keys():
            return self.dtw_costs[other]

        D = librosa.sequence.dtw(self.mfcc, other.mfcc, metric="cosine", backtrack=False)
        dtw_cost = D[-1, -1]

        self.dtw_costs[other] = dtw_cost
        other.dtw_costs[self] = dtw_cost

        return dtw_cost

    def get_cosine_similarity_best_matches(self, n) -> Generator[AudioFile]:
        if len(self.cosine_similarities) < 1:
            raise RuntimeError("Cosine similarity matrix not calculated")

        sorted_files = self.cosine_similarities.sort_by_value(True)
        for (index, file) in enumerate(sorted_files):
            if index < n:
                yield file
            else:
                break

    def get_dtw_cost_best_matches(self, n) -> Generator[AudioFile]:
        if len(self.dtw_costs) < 1:
            raise RuntimeError("DTW distance matrix not calculated")

        sorted_files = self.dtw_costs.sort_by_value()
        for (index, file) in enumerate(sorted_files):
            if index < n:
                yield file
            else:
                break


class AudioFileCollection(Dict[str, List[AudioFile]]):
    ALL_FILES_CATEGORY = "All"

    def __init__(self, files: List[AudioFile], key: Callable[[AudioFile], str] = None):
        super().__init__()

        self._files = files

        if key is None:
            self._group_files_by_class()
        else:
            self._group_files_by_key(key)

    def _group_files_by_class(self):
        for file in self._files:
            for file_class in file.classes:
                self._add(file_class, file)

    def _group_files_by_key(self, key: Callable[[AudioFile], str]):
        for file in self._files:
            self._add(key(file), file)

    def _add(self, category: str, file: AudioFile):
        if category not in self.keys():
            self[category] = [file]
        else:
            self[category].append(file)

    def get_files_count(self):
        return len(self._files)

    def add_all_files_category(self):
        self[self.ALL_FILES_CATEGORY] = self._files

    def remove_all_files_category(self):
        self.pop(self.ALL_FILES_CATEGORY, None)


def log(message: str):
    print(datetime.now(), message)


def save_files_to_dump_file(files: List[AudioFile], dump_file):
    log("Saving files to dump: {}...".format(dump_file))

    with open(dump_file, "wb") as file:
        pickle.dump(files, file)


def load_files_from_dump_file(dump_file) -> List[AudioFile]:
    log("Loading files from dump: {}...".format(dump_file))

    with open(dump_file, "rb") as file:
        return pickle.load(file)


def filter_classes(classes: List[str], exclude_classes: List[str] = None) -> Set[str]:
    if exclude_classes is None:
        exclude_classes = []

    return set([class_.strip() for class_ in classes
                if class_.strip() and class_.strip() not in exclude_classes])


def read_annotation_file(filename: str, audio_files_directory: str, all_files: List[AudioFile],
                         max_audio_files: int = np.inf, exclude_classes: List[str] = None):
    """
    Reads one annotation file and reads audio files from it.

    :param filename: filename of annotation file
    :param audio_files_directory: path to directory audio files are stored in
    :param all_files: list of all audio files where new files will be added to
    :param max_audio_files: maximum number of audio files to be loaded
    :param exclude_classes: list of classes to remove from audio files
    """

    with open(filename) as annotation_file:
        reader = csv.reader(annotation_file)
        for index, row in enumerate(reader):
            # skip header row
            if index == 0:
                continue

            if len(all_files) >= max_audio_files:
                break

            log("Adding file {}...".format(len(all_files) + 1))

            filename = row[0].replace("mp3", "wav")
            classes = filter_classes(row[1].split(","), exclude_classes)

            # skip files without classes
            if len(classes) < 1:
                continue

            all_files.append(AudioFile(audio_files_directory, filename, classes))


def read_annotation_files(annotation_files_directory: str, audio_files_directory: str,
                          max_audio_files: int = np.inf, exclude_classes: List[str] = None) -> List[AudioFile]:
    """
    Reads annotation files in a directory specified.

    :param annotation_files_directory: the directory of the annotation files
    :param audio_files_directory: path to directory audio files are stored in
    :param max_audio_files: maximum number of audio files to be loaded
    :param exclude_classes: list of classes to remove from audio files
    :return: list of files
    """

    all_files: List[AudioFile] = []

    filenames = sorted(os.listdir(annotation_files_directory))
    for index, file in enumerate(filenames):
        if len(all_files) >= max_audio_files:
            break

        log("Reading annotation file {} ({})...".format(index + 1, file))

        filepath = os.path.join(annotation_files_directory, file)
        read_annotation_file(filepath, audio_files_directory, all_files, max_audio_files, exclude_classes)

    return all_files


def load_files(annotation_files_directory: str, audio_files_directory: str,
               max_audio_files: int = np.inf, exclude_classes: List[str] = None,
               dump_file: str = None) -> List[AudioFile]:
    """
    Loads files from a dump file specified if it exists.

    If the dump file does not exits, the files are loaded from annotation files in a directory specified.
    Then, similarity values are stored in them, and they are saved to the dump file.

    :param annotation_files_directory: the directory of the annotation files
    :param audio_files_directory: path to directory audio files are stored in
    :param max_audio_files: maximum number of audio files to be loaded
    :param exclude_classes: list of classes to remove from audio files
    :param dump_file: filename of data dump file
    :return: list of files
    """

    if dump_file and os.path.exists(dump_file):
        return load_files_from_dump_file(dump_file)

    all_files = read_annotation_files(annotation_files_directory, audio_files_directory,
                                      max_audio_files, exclude_classes)

    # cache matrix values in AudioFiles
    calculate_cosine_similarity_matrix(all_files)
    calculate_dtw_cost_matrix(all_files)

    if dump_file:
        save_files_to_dump_file(all_files, dump_file)

    return all_files


def get_all_classes(files: List[AudioFile]) -> List[str]:
    """
    Retrieves all classes appearing in a list of files.

    :param files: list of files
    :return: list of all classes
    """

    classes = set()

    for file in files:
        for class_ in file.classes:
            classes.add(class_)

    return sorted(classes)


def calculate_cosine_similarity_matrix(first_files: List[AudioFile], second_files: List[AudioFile] = None) -> np.array:
    """
    Calculates cosine similarity matrix for two sets of files.
    If the second set is not provided, the first set is used.

    :param first_files: first set of files
    :param second_files: second set of files
    :return: cosine similarity matrix
    """

    if second_files is None:
        second_files = first_files

    S = []

    for (index, first_file) in enumerate(first_files):
        first_file_similarities = []

        log("Calculating cosine similarity {}/{}...".format(index + 1, len(first_files)))

        for second_file in second_files:
            first_file_similarities.append(first_file.calculate_cosine_similarity(second_file))

        S.append(first_file_similarities)

    return np.array(S)


def calculate_dtw_cost_matrix(first_files: List[AudioFile], second_files: List[AudioFile] = None) -> np.array:
    """
    Calculates DTW cost matrix for two sets of files.
    If the second set is not provided, the first set is used.

    :param first_files: first set of files
    :param second_files: second set of files
    :return: DTW cost matrix
    """

    if second_files is None:
        second_files = first_files

    D = []

    for (index, first_file) in enumerate(first_files):
        first_file_similarities = []

        log("Calculating DTW distance {}/{}...".format(index + 1, len(first_files)))

        for second_file in second_files:
            first_file_similarities.append(first_file.calculate_dtw_cost(second_file))

        D.append(first_file_similarities)

    return np.array(D)


def calculate_matrix_average(M: np.array, lower_triangular_matrix=False):
    """
    Calculates the average of the matrix.

    :param M: matrix
    :param lower_triangular_matrix: calculate average below diagonal
    :return: matrix average
    """

    if lower_triangular_matrix:
        indices = np.tril_indices(len(M), -1)
        return np.mean(M[indices])
    return np.mean(M)


def get_heatmap_labels_and_ticks(keys: List[str]) -> (List[str], List[int]):
    """
    Generates heatmap labels so that only the first label in a subsequent series is displayed.

    :param keys: all keys
    :return: labels and tick indices
    """

    labels = []
    ticks = []

    previous = None
    for index, key in enumerate(keys):
        if key != previous:
            labels.append("{}↓".format(key))
            ticks.append(index)
        previous = key

    return labels, ticks


def save_figure(fig, directory: str):
    """
    Saves figure in a directory specified.
    The directory is created if did not existed.

    :param fig: figure
    :param directory: path to the directory the figure will be saved in
    :return: None
    """

    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = "{}.png".format(datetime.now()).replace(":", ".")
    filepath = os.path.join(directory, filename)

    fig.set_size_inches(21, 12)
    fig.savefig(filepath, dpi=200)


def analyze_file_collection(collection: AudioFileCollection, collection_name: str, figures_directory: str):
    collection.add_all_files_category()

    average_cosine_similarities = SortableDictionary()
    average_dtw_costs = SortableDictionary()

    for (category, files) in collection.items():
        S = calculate_cosine_similarity_matrix(files)
        s = calculate_matrix_average(S, True)
        average_cosine_similarities[category] = s

        D = calculate_dtw_cost_matrix(files)
        d = calculate_matrix_average(D, True)
        average_dtw_costs[category] = d

    average_cosine_similarities = average_cosine_similarities.sort_by_value()
    average_dtw_costs = average_dtw_costs.sort_by_value(True)

    # plot categories
    fig, (ax1, ax2) = plt.subplots(2)
    plt.subplots_adjust(hspace=.3)
    plt.suptitle("N = {}".format(collection.get_files_count()))
    ind = np.arange(len(average_cosine_similarities))

    ax1.barh(ind, average_cosine_similarities.values(), color=COLOR_RED)
    ax1.set_xlim(.990, 1)
    ax1.set_yticks(ind)
    ax1.set_yticklabels(average_cosine_similarities.keys(), rotation=45)
    ax1.set_xlabel("Average cosine similarity")
    ax1.set_ylabel(collection_name)

    # display numerical values next to bars
    for i, value in enumerate(average_cosine_similarities.values()):
        ax1.text(value + .00025, i - .2, str(round(value, 4)))

    ax2.barh(ind, average_dtw_costs.values(), color=COLOR_RED)
    ax2.set_xlim(0, 5)
    ax2.set_yticks(ind)
    ax2.set_yticklabels(average_dtw_costs.keys(), rotation=45)
    ax2.set_xlabel("Average DTW distance")
    ax2.set_ylabel(collection_name)

    # display numerical values next to bars
    for i, value in enumerate(average_dtw_costs.values()):
        ax2.text(value + .2, i - .2, str(round(value, 2)))

    save_figure(fig, figures_directory)


def compare_file_collection(collection: AudioFileCollection, collection_name: str, figures_directory: str):
    collection.add_all_files_category()

    categories = []
    average_cosine_similarities = []
    average_dtw_costs = []

    for first_category, first_category_files in sorted(collection.items()):
        categories.append(first_category)

        first_category_average_cosine_similarities = []
        first_category_average_dtw_costs = []

        for (second_category, second_category_files) in sorted(collection.items()):
            S = calculate_cosine_similarity_matrix(first_category_files, second_category_files)
            s = calculate_matrix_average(S, first_category == second_category)
            first_category_average_cosine_similarities.append(s)

            D = calculate_dtw_cost_matrix(first_category_files, second_category_files)
            d = calculate_matrix_average(D, first_category == second_category)
            first_category_average_dtw_costs.append(d)

        average_cosine_similarities.append(first_category_average_cosine_similarities)
        average_dtw_costs.append(first_category_average_dtw_costs)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.subplots_adjust(wspace=.4)
    plt.suptitle("N = {}".format(collection.get_files_count()))
    ind = np.arange(len(average_cosine_similarities))

    ax1.set_title("Average cosine similarity")
    ax1.set_xlabel(collection_name)
    ax1.set_ylabel(collection_name)
    ax1.set_xticks(ind)
    ax1.set_yticks(ind)
    ax1.set_xticklabels(categories, rotation=90)
    ax1.set_yticklabels(categories)
    im1 = ax1.imshow(average_cosine_similarities, cmap=COLOR_MAP, interpolation="nearest")
    fig.colorbar(im1, ax=ax1)

    # display numerical values in cells
    for i in range(len(categories)):
        for j in range(len(categories)):
            ax1.text(j, i, round(average_cosine_similarities[i][j], 3),
                     ha="center", va="center", color=COLOR_GREEN, fontsize=12)

    ax2.set_title("Average DTW distance")
    ax2.set_xlabel(collection_name)
    ax2.set_ylabel(collection_name)
    ax2.set_xticks(ind)
    ax2.set_yticks(ind)
    ax2.set_xticklabels(categories, rotation=90)
    ax2.set_yticklabels(categories)
    im2 = ax2.imshow(average_dtw_costs, cmap=COLOR_MAP, interpolation="nearest")
    fig.colorbar(im2, ax=ax2)

    # display numerical values in cells
    for i in range(len(categories)):
        for j in range(len(categories)):
            ax2.text(j, i, round(average_dtw_costs[i][j], 2),
                     ha="center", va="center", color=COLOR_GREEN, fontsize=12)

    save_figure(fig, figures_directory)


def analyze_file_group(files: List[AudioFile], sort_by_key: Callable[[AudioFile], str],
                       key_name: str, figures_directory: str):
    sorted_files = sorted(files, key=sort_by_key)
    keys = [sort_by_key(file) for file in sorted_files]
    labels, ticks = get_heatmap_labels_and_ticks(keys)

    S = calculate_cosine_similarity_matrix(sorted_files)
    s = calculate_matrix_average(S, True)

    D = calculate_dtw_cost_matrix(sorted_files)
    d = calculate_matrix_average(D, True)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.subplots_adjust(wspace=.4)
    plt.suptitle("N = {}, s = {}, d = {}".format(len(files), s, d))

    ax1.set_title("Cosine similarity")
    ax1.set_xlabel(key_name)
    ax1.set_ylabel(key_name)
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)
    ax1.set_xticklabels(labels, rotation=90)
    ax1.set_yticklabels(labels)
    im1 = ax1.imshow(S, cmap=COLOR_MAP, interpolation="nearest")
    fig.colorbar(im1, ax=ax1)

    ax2.set_title("DTW distance")
    ax2.set_xlabel(key_name)
    ax2.set_ylabel(key_name)
    ax2.set_xticks(ticks)
    ax2.set_yticks(ticks)
    ax2.set_xticklabels(labels, rotation=90)
    ax2.set_yticklabels(labels)
    im2 = ax2.imshow(D, cmap=COLOR_MAP, interpolation="nearest")
    fig.colorbar(im2, ax=ax2)

    save_figure(fig, figures_directory)


def compare_best_matches(files: List[AudioFile], figures_directory: str):
    cosine_similarity_same_class = 0
    cosine_similarity_same_scene = 0
    dtw_cost_same_class = 0
    dtw_cost_same_scene = 0

    for file in files:
        cosine_similarity_best_match = list(file.get_cosine_similarity_best_matches(1))[0]
        if len(file.classes.intersection(cosine_similarity_best_match.classes)) > 0:
            cosine_similarity_same_class += 1

        if file.scene == cosine_similarity_best_match.scene:
            cosine_similarity_same_scene += 1

        dtw_cost_best_match = list(file.get_dtw_cost_best_matches(1))[0]
        if len(file.classes.intersection(dtw_cost_best_match.classes)) > 0:
            dtw_cost_same_class += 1

        if file.scene == dtw_cost_best_match.scene:
            dtw_cost_same_scene += 1

    data_true = np.array([cosine_similarity_same_class, cosine_similarity_same_scene,
                          dtw_cost_same_class, dtw_cost_same_scene])
    data_false = len(files) - data_true
    labels = ["Cosine similarity, the best\nmatch has the same class",
              "Cosine similarity, the best\nmatch has the same scene",
              "DTW distance, the best\nmatch has the same class",
              "DTW distance, the best\nmatch has the same scene"]

    fig, ax = plt.subplots()
    # plt.title("N = {}". format(len(files)))
    ind = np.arange(4)
    p1 = plt.bar(ind, data_true, color=COLOR_GREEN)
    p2 = plt.bar(ind, data_false, bottom=data_true, color=COLOR_RED)

    # display numerical values above bars
    for i, value in enumerate(data_true):
        ax.text(i - .1, len(files) + 5, "{} %".format(round(value / len(files) * 100, 1)))

    plt.ylabel("N")
    plt.xticks(ind, labels)
    plt.yticks(np.arange(0, len(files) + 20, 10))
    plt.legend((p1[0], p2[0]), ("Yes", "No"))

    save_figure(fig, figures_directory)


def get_most_common_item(items: list):
    if len(items) < 1:
        return None

    most_common_item = max(set(items), key=items.count)
    most_common_item_count = items.count(most_common_item)

    _, counts = np.unique(items, return_counts=True)
    for (index, count) in enumerate(counts):
        if items[index] != most_common_item and count >= most_common_item_count:
            return None

    return most_common_item


def compare_n_best_matches(files: List[AudioFile], n: int, figures_directory: str):
    cosine_similarity_predict_class = 0
    cosine_similarity_predict_scene = 0
    dtw_cost_predict_class = 0
    dtw_cost_predict_scene = 0

    for file in files:
        cosine_similarity_classes = []
        cosine_similarity_scenes = []

        for match in file.get_cosine_similarity_best_matches(n):
            cosine_similarity_classes += match.classes
            cosine_similarity_scenes.append(match.scene)

        cosine_similarity_best_class = get_most_common_item(cosine_similarity_classes)
        cosine_similarity_best_scene = get_most_common_item(cosine_similarity_scenes)

        if cosine_similarity_best_class in file.classes:
            cosine_similarity_predict_class += 1

        if file.scene == cosine_similarity_best_scene:
            cosine_similarity_predict_scene += 1

        dtw_cost_classes = []
        dtw_cost_scenes = []

        for match in file.get_dtw_cost_best_matches(n):
            dtw_cost_classes += match.classes
            dtw_cost_scenes.append(match.scene)

        dtw_cost_best_class = get_most_common_item(dtw_cost_classes)
        dtw_cost_best_scene = get_most_common_item(dtw_cost_scenes)

        if dtw_cost_best_class in file.classes:
            dtw_cost_predict_class += 1

        if file.scene == dtw_cost_best_scene:
            dtw_cost_predict_scene += 1

    data_true = np.array([cosine_similarity_predict_class, cosine_similarity_predict_scene,
                          dtw_cost_predict_class, dtw_cost_predict_scene])
    data_false = len(files) - data_true
    labels = ["Cosine similarity,\npredict class",
              "Cosine similarity,\npredict scene",
              "DTW distance,\npredict class",
              "DTW distance,\npredict scene"]

    fig, ax = plt.subplots()
    plt.title("N = {}, n = {}". format(len(files), n))
    ind = np.arange(4)
    p1 = plt.bar(ind, data_true, color=COLOR_GREEN)
    p2 = plt.bar(ind, data_false, bottom=data_true, color=COLOR_RED)

    # display numerical values above bars
    for i, value in enumerate(data_true):
        ax.text(i - .1, len(files) + 5, "{} %".format(round(value / len(files) * 100, 1)))

    plt.ylabel("N")
    plt.xticks(ind, labels)
    plt.yticks(np.arange(0, len(files) + 20, 10))
    plt.legend((p1[0], p2[0]), ("Yes", "No"))

    save_figure(fig, figures_directory)


def calculate_file_collection_metrics(collection: AudioFileCollection, collection_name: str, figures_directory: str):
    """
    Calculates category frequencies in a collection provided.
    The frequencies are plotted in a bar diagram.
    The diagram is saved in a directory specified.

    :param collection: file collection
    :param collection_name: the name of the collection, x-axis title
    :param figures_directory: path to the directory the figure will be saved in
    :return: None
    """

    # hide "All" category
    collection.remove_all_files_category()

    keys = list(collection.keys())
    counts = [len(files) for files in collection.values()]

    sorted_labels_and_values = SortableDictionary(zip(keys, counts)).sort_by_value(True)
    labels = list(sorted_labels_and_values.keys())
    values = list(sorted_labels_and_values.values())

    # plot frequencies
    fig, ax = plt.subplots()
    # plt.subplots_adjust(bottom=.3)
    # plt.title("N = {}". format(collection.get_files_count()))
    ind = np.arange(len(labels))
    plt.bar(ind, values, color=COLOR_RED)

    # display numerical values above bars
    for i, value in enumerate(values):
        ax.text(i - .1, value + 1, value)

    plt.ylabel("N")
    plt.xlabel(collection_name)
    plt.xticks(ind, labels, rotation=30)
    plt.yticks(np.arange(0, max(values) + 15, 10))

    save_figure(fig, figures_directory)


def calculate_class_occurrences(collection: AudioFileCollection, collection_name: str,
                                classes: List[str], figures_directory: str):
    """
    Calculates class occurrences in collection categories.
    The occurrences are displayed in a heatmap.
    The diagram is saved in a directory specified.

    :param collection: file collection
    :param collection_name: the name of the collection, y-axis title
    :param classes: all classes
    :param figures_directory: path to the directory the figure will be saved in
    :return: None
    """

    # hide "All" category
    collection.remove_all_files_category()

    categories = sorted(collection.keys())
    collection_occurrences = []
    for category in categories:
        category_occurrences = []

        for class_ in classes:
            files_with_class = [file for file in collection[category] if class_ in file.classes]
            category_occurrences.append(len(files_with_class))

        collection_occurrences.append(category_occurrences)

    fig, ax = plt.subplots()
    plt.suptitle("N = {}".format(collection.get_files_count()))

    ax.set_xlabel("Class")
    ax.set_ylabel(collection_name)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(categories)))
    ax.set_xticklabels(classes, rotation=25)
    ax.set_yticklabels(categories)
    im2 = ax.imshow(collection_occurrences, cmap=COLOR_MAP, interpolation="nearest")
    fig.colorbar(im2, ax=ax)

    # display numerical values in cells
    for i in range(len(categories)):
        for j in range(len(classes)):
            ax.text(j, i, collection_occurrences[i][j],
                    ha="center", va="center", color=COLOR_GREEN, fontsize=12)

    save_figure(fig, figures_directory)


def main():
    # configure pyplot
    plt.rc("font", size=19)
    plt.rc("axes", titlesize=19)
    plt.rc("xtick", labelsize=16)
    plt.rc("ytick", labelsize=16)
    plt.rc("figure", titlesize=20)

    # load samples
    all_files = load_files(ANNOTATION_FILES_DIRECTORY, AUDIO_FILES_DIRECTORY, MAX_AUDIO_FILES, EXCLUDE_CLASSES,
                           ALL_FILES_DUMP_FILE)
    all_classes = get_all_classes(all_files)

    # classify samples
    file_collections = {}
    file_collections["Class"] = AudioFileCollection(all_files)
    file_collections["Scene"] = AudioFileCollection(all_files, lambda file: file.scene)
    file_collections["City"] = AudioFileCollection(all_files, lambda file: file.city)
    # file_collections["Place"] = AudioFileCollection(all_files, lambda file: file.place)

    for n in range(1, 5):
        compare_n_best_matches(all_files, n, FIGURES_DIRECTORY)

    compare_best_matches(all_files, FIGURES_DIRECTORY)

    analyze_file_group(all_files, lambda file: file.scene, "Scene", FIGURES_DIRECTORY)
    analyze_file_group(all_files, lambda file: file.city, "City", FIGURES_DIRECTORY)
    # analyze_file_group(all_files, lambda file: file.place, "Place", FIGURES_DIRECTORY)

    for (collection_name, collection) in file_collections.items():
        compare_file_collection(collection, collection_name, FIGURES_DIRECTORY)
        analyze_file_collection(collection, collection_name, FIGURES_DIRECTORY)
        calculate_file_collection_metrics(collection, collection_name, FIGURES_DIRECTORY)
        calculate_class_occurrences(collection, collection_name, all_classes, FIGURES_DIRECTORY)


if __name__ == "__main__":
    main()
