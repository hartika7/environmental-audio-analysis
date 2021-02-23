"""
Environmental audio analysis
COMP.SGN.120-2020-2021-1 Introduction to Audio Processing
Timo Hartikainen
Tampere University

Unit tests for the main file.
"""

import unittest
from typing import NamedTuple, List, Set
import numpy as np

import main
from main import SortableDictionary, AudioFileCollection

ANNOTATION_FILES_DIRECTORY = "test_csv_files"
AUDIO_FILES_DIRECTORY = "audio_files"
ALL_FILES_DUMP_FILE = "test_dump"


class Tests(unittest.TestCase):
    def test_read_annotation_files_count(self):
        files = main.read_annotation_files(ANNOTATION_FILES_DIRECTORY, AUDIO_FILES_DIRECTORY, 0)
        self.assertEqual(0, len(files))

        files = main.read_annotation_files(ANNOTATION_FILES_DIRECTORY, AUDIO_FILES_DIRECTORY, 7)
        self.assertEqual(7, len(files))

        files = main.read_annotation_files(ANNOTATION_FILES_DIRECTORY, AUDIO_FILES_DIRECTORY, 100)
        self.assertEqual(10, len(files))

    def test_read_annotation_files(self):
        file1, file2 = main.read_annotation_files(ANNOTATION_FILES_DIRECTORY, AUDIO_FILES_DIRECTORY, 2)

        self.assertEqual(file1.directory, AUDIO_FILES_DIRECTORY)
        self.assertEqual(file1.filename, "public_square-paris-117-3426-a.wav")
        self.assertEqual(file1.scene, "Public square")
        self.assertEqual(file1.city, "Paris")
        self.assertEqual(file1.place, "Public square, Paris")
        self.assertSetEqual(file1.classes, {"Siren", "Traffic noise", "Birds singing"})

        self.assertEqual(file2.directory, AUDIO_FILES_DIRECTORY)
        self.assertEqual(file2.filename, "park-helsinki-242-7205-a.wav")
        self.assertEqual(file2.scene, "Park")
        self.assertEqual(file2.city, "Helsinki")
        self.assertEqual(file2.place, "Park, Helsinki")
        self.assertSetEqual(file2.classes, {"Footsteps", "Adults talking"})

    def test_load_and_save_files(self):
        all_files = main.read_annotation_files(ANNOTATION_FILES_DIRECTORY, AUDIO_FILES_DIRECTORY, 10)
        main.calculate_cosine_similarity_matrix(all_files)
        main.calculate_dtw_cost_matrix(all_files)

        main.save_files_to_dump_file(all_files, ALL_FILES_DUMP_FILE)
        all_files_from_dump = main.load_files_from_dump_file(ALL_FILES_DUMP_FILE)

        for file, dump_file in zip(all_files, all_files_from_dump):
            self.assertEqual(file.directory, dump_file.directory)
            self.assertEqual(file.filename, dump_file.filename)
            self.assertEqual(file.scene, dump_file.scene)
            self.assertEqual(file.city, dump_file.city)
            self.assertSetEqual(file.classes, dump_file.classes)

            self.assertTrue(np.array_equal(file.mfcc, dump_file.mfcc))
            self.assertTrue(np.array_equal(file.mfcc_coefficients, dump_file.mfcc_coefficients))

            self.assertListEqual([f.filename for f in file.cosine_similarities.keys()],
                                 [f.filename for f in dump_file.cosine_similarities.keys()])
            self.assertListEqual(list(file.cosine_similarities.values()), list(dump_file.cosine_similarities.values()))

            self.assertListEqual([f.filename for f in file.dtw_costs.keys()],
                                 [f.filename for f in dump_file.dtw_costs.keys()])
            self.assertListEqual(list(file.dtw_costs.values()), list(dump_file.dtw_costs.values()))

    def test_audio_file_collection_all_files_category(self):
        file1, file2, file3, file4, file5 = main.read_annotation_files(ANNOTATION_FILES_DIRECTORY,
                                                                       AUDIO_FILES_DIRECTORY, 5)

        collection = AudioFileCollection([file1, file2, file3, file4, file5])
        self.assertTrue("All" not in collection)

        collection.add_all_files_category()
        self.assertListEqual([file1, file2, file3, file4, file5], collection["All"])

        collection.remove_all_files_category()
        self.assertTrue("All" not in collection)

    def test_group_files_by_class(self):
        file1, file2, file3, file4, file5 = main.read_annotation_files(ANNOTATION_FILES_DIRECTORY,
                                                                       AUDIO_FILES_DIRECTORY, 5)

        collection = AudioFileCollection([file1, file2, file3, file4, file5])
        self.assertListEqual([file1], collection["Siren"])
        self.assertListEqual([file1], collection["Traffic noise"])
        self.assertListEqual([file1, file4, file5], collection["Birds singing"])
        self.assertListEqual([file2, file4], collection["Footsteps"])
        self.assertListEqual([file2, file3, file4], collection["Adults talking"])
        self.assertListEqual([file5], collection["Children voices"])

    def test_group_files_by_scene(self):
        file1, file2, file3, file4, file5 = main.read_annotation_files(ANNOTATION_FILES_DIRECTORY,
                                                                       AUDIO_FILES_DIRECTORY, 5)

        collection = AudioFileCollection([file1, file2, file3, file4, file5], lambda file: file.scene)
        self.assertListEqual([file1], collection["Public square"])
        self.assertListEqual([file2, file4, file5], collection["Park"])
        self.assertListEqual([file3], collection["Airport"])

    def test_group_files_by_city(self):
        file1, file2, file3, file4, file5 = main.read_annotation_files(ANNOTATION_FILES_DIRECTORY,
                                                                       AUDIO_FILES_DIRECTORY, 5)

        collection = AudioFileCollection([file1, file2, file3, file4, file5], lambda file: file.city)
        self.assertListEqual([file1], collection["Paris"])
        self.assertListEqual([file2, file4], collection["Helsinki"])
        self.assertListEqual([file3], collection["Stockholm"])
        self.assertListEqual([file5], collection["London"])

    def test_get_all_classes(self):
        files = main.read_annotation_files(ANNOTATION_FILES_DIRECTORY, AUDIO_FILES_DIRECTORY, 5)
        classes = main.get_all_classes(files)

        self.assertListEqual(sorted(["Siren", "Traffic noise", "Birds singing", "Footsteps", "Adults talking",
                                     "Children voices"]), classes)

    def test_dtw_cost(self):
        file1, file2 = main.read_annotation_files(ANNOTATION_FILES_DIRECTORY, AUDIO_FILES_DIRECTORY, 2)

        dtw_cost = file1.calculate_dtw_cost(file2)
        self.assertEqual(dtw_cost, file1.dtw_costs[file2])
        self.assertEqual(dtw_cost, file2.dtw_costs[file1])

        self.assertEqual(0, file1.calculate_dtw_cost(file1))

    def test_filter_classes(self):
        class TestCase(NamedTuple):
            input: List[str]
            output: Set[str]
            exclude: any

        tests = [
            TestCase([""], set(), None),
            TestCase(["", " "], set(), None),
            TestCase(["class1", "class2 ", " "], {"class1", "class2"}, None),
            TestCase(["  class1", "class2 ", " "], {"class1", "class2"}, None),
            TestCase(["class1", "class2 ", "class3"], {"class2", "class3"}, ["class1"]),
            TestCase(["class1", "class2 ", "class3"], {"class3"}, ["class1", "class2"]),
            TestCase(["class1", "class1 ", "class1"], set(), ["class1"]),
        ]

        for test in tests:
            print(test)
            self.assertSetEqual(test.output, main.filter_classes(test.input, test.exclude))

    def test_get_most_common_item(self):
        class TestCase(NamedTuple):
            input: list
            output: any

        tests = [
            TestCase([], None),
            TestCase(["a"], "a"),
            TestCase(["a", "b", "c"], None),
            TestCase(["a", "b", "c", "c"], "c"),
            TestCase(["a", "a", "b", "c", "c"], None)
        ]

        for test in tests:
            self.assertEqual(test.output, main.get_most_common_item(test.input))

    def test_sortable_dictionary(self):
        dictionary = SortableDictionary()
        dictionary["a"] = 3
        dictionary["b"] = 1
        dictionary["c"] = 4
        dictionary["d"] = 2

        sorted_dictionary = dictionary.sort_by_value()
        self.assertListEqual(["b", "d", "a", "c"], list(sorted_dictionary.keys()))
        self.assertListEqual([1, 2, 3, 4], list(sorted_dictionary.values()))

        reverse_sorted_dictionary = dictionary.sort_by_value(True)
        self.assertListEqual(["c", "a", "d", "b"], list(reverse_sorted_dictionary.keys()))
        self.assertListEqual([4, 3, 2, 1], list(reverse_sorted_dictionary.values()))

    def test_get_heatmap_labels_and_ticks(self):
        keys = ["a", "a", "a", "b", "c", "c", "d"]
        labels, ticks = main.get_heatmap_labels_and_ticks(keys)

        self.assertListEqual(["a↓", "b↓", "c↓", "d↓"], labels)
        self.assertListEqual([0, 3, 4, 6], ticks)

    def test_get_matrix_average(self):
        M = np.array([
            [1, 1, 8, 8],
            [1, 1, 2, 2],
            [1, 0, 2, 2],
            [0, 0, 1, 2],
        ])

        self.assertEqual(2, main.calculate_matrix_average(M))
        self.assertEqual(.5, main.calculate_matrix_average(M, True))


if __name__ == '__main__':
    unittest.main()
