import os
import re
import shutil
import unittest

from Rerender_A_Video.blender.video_sequence import VideoSequence


class TestVideoSequence(unittest.TestCase):

    def setUp(self):
        # Sample call:
        # python video_blend.py videos/pexels-koolshooters-7322716 --beg 1 --end 101 --key keys
        # --output videos/pexels-koolshooters-7322716/blend.mp4 --fps 25.0 --n_proc 4  -ps

        base_dir = os.path.join(os.path.dirname(__file__), '../videos/antoni-vangogh-mock')
        beg = 1
        end = 101
        key_dir = 'keys'

        self.video_sequence = VideoSequence(base_dir=base_dir,
                                            beg_frame=beg,  # default=1
                                            end_frame=end,
                                            input_subdir='video',  # default
                                            key_subdir=key_dir,
                                            tmp_subdir='tmp',  # default
                                            input_format='%04d.png',  # default
                                            key_format='%04d.png')  # default

    def tearDown(self):
        """Remove all out_* directories created during VideoSequence init."""
        base_dir = self.video_sequence.get_base_dir()
        directories_removed = 0
        for dirname in os.listdir(base_dir):
            if dirname.startswith("out_") and os.path.isdir(os.path.join(base_dir, dirname)):
                dir_to_remove = os.path.join(base_dir, dirname)
                shutil.rmtree(dir_to_remove)
                directories_removed += 1
        print(f"Removed {directories_removed} directories.")

    def test_key_frames(self):
        frames = self.video_sequence.key_frames
        self.assertEqual(frames,
                         [1, 2, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                          31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 61, 62,
                          70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92,
                          93, 94, 95, 96, 97, 98, 99, 100, 101])

    def test_n_seq(self):
        self.assertEqual(self.video_sequence.n_seq, 76)

    def test_get_input_sequence(self):
        full_links = self.video_sequence.get_input_sequence(1, True)
        numeric_part = [int(re.search(r'(\d+)\.(png|jpg)$', path).group(1)) for path in full_links]
        self.assertEqual(numeric_part, [2, 3, 4, 5, 6, 7, 8, 9, 10])

    @unittest.skip("Same as get_input_sequence...?")
    def test_get_output_sequence(self):
        self.fail()

    def test_get_sequence_beg_id(self):
        frames = self.video_sequence.key_frames
        self.assertEqual(frames[1], 2)
        self.assertEqual(frames[2], 11)
