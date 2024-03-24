import logging
import os
import shutil
from typing import List


class VideoSequence:

    def __init__(self,
                 base_dir,
                 beg_frame,
                 end_frame,
                 input_subdir='videos',
                 key_subdir='keys0',
                 tmp_subdir='tmp',
                 input_format='frame%04d.jpg',
                 key_format='%04d.jpg',
                 out_subdir_format='out_%d',
                 blending_out_subdir='blend',
                 output_format='%04d.jpg'):

        self.__base_dir = base_dir
        self.__input_dir = os.path.join(base_dir, input_subdir)
        self.__key_dir = os.path.join(base_dir, key_subdir)
        self.__tmp_dir = os.path.join(base_dir, tmp_subdir)
        self.__input_format = input_format
        self.__blending_out_dir = os.path.join(base_dir, blending_out_subdir)
        self.__key_format = key_format
        self.__out_subdir_format = out_subdir_format
        self.__output_format = output_format
        self.__beg_frame = beg_frame  # absolute FIRST frame
        self.__end_frame = end_frame  # absolute LAST frame
        self.__key_frames = None
        self.__n_seq = None  # old: number of sequences to divide the video into
        self.__make_out_dirs()
        os.makedirs(self.__tmp_dir, exist_ok=True)

    @property
    def beg_frame(self):
        return self.__beg_frame

    @property
    def end_frame(self):
        return self.__end_frame

    @property
    def n_seq(self) -> int:
        """
        Get the number of sequences to divide the video into. Initialises the number of sequences if it has not been.

        Original implementation takes the floor division of the difference between the first and last frame and the
        interval. This has been replaced with the length of the list of sequences to support flexible intervals.
        - 1.
        :return: number of sequences (int)
        """
        self.key_frames
        return self.__n_seq

    @property
    def blending_dir(self):
        return os.path.abspath(self.__blending_out_dir)

    def remove_out_and_tmp(self):
        for i in range(self.n_seq + 1):
            out_dir = self.__get_out_subdir(i)
            shutil.rmtree(out_dir)
        shutil.rmtree(self.__tmp_dir)

    @property
    def key_frames(self):
        """
        Get all input key frames (from rerender.py) that fall within the range of the sequence (beg and end
        inclusive) and caches the number of sequences
        :return: list of key frames (int)
        """
        if self.__key_frames is None:
            self.__key_frames = self._find_key_frames()
            self.__n_seq = len(self.__key_frames) - 1
        return self.__key_frames

    def _find_key_frames(self):
        """
        Get all input key frames (from rerender.py) that fall within the range of the sequence (beg and end inclusive)
        :return: list of key frames (int)
        :raises FileNotFoundError: if no key frames are found in the range
        """
        all_keys = [int(x[:-4]) for x in sorted(os.listdir(self.__key_dir))]
        keys_in_range = [x for x in all_keys if self.__beg_frame <= x <= self.__end_frame]
        if len(keys_in_range) == 0:
            raise FileNotFoundError(f"No key frames found in range {self.__beg_frame} to {self.__end_frame}")
        return keys_in_range

    def get_input_sequence(self, i, j=1, is_forward=True):
        """
        Get the file addresses from the i-th key frame to the (i+j)-th key frame (exclusive) in the input directory.
        :param i: Index of key frame
        :param j: Number of key frames to include up to in the sequence
        :param is_forward: If True, get the sequence in forward order.
            Otherwise, get the sequence in backward order, in which case exclude the smallest frame number.
        :return: List of strings of file addresses of the sequence
        """
        beg_id = self.get_sequence_beg_id(i)
        end_id = self.get_sequence_beg_id(i + j)
        if is_forward:
            id_list = list(range(beg_id, end_id))
        else:
            id_list = list(range(end_id, beg_id, -1))
        path_dir = [
            os.path.join(self.__input_dir, self.__input_format % id) for id in id_list
        ]
        return path_dir

    def get_base_dir(self):
        """
        Get the base directory.
        """
        return self.__base_dir

    def get_output_sequence(self, i, j=1, is_forward=True):
        beg_id = self.get_sequence_beg_id(i)
        end_id = self.get_sequence_beg_id(i + j)
        if is_forward:
            id_list = list(range(beg_id, end_id))
        else:
            i += 1
            id_list = list(range(end_id, beg_id, -1))
        out_subdir = self.__get_out_subdir(i)
        path_dir = [
            os.path.join(out_subdir, self.__output_format % id)
            for id in id_list
        ]
        return path_dir

    def get_temporal_sequence(self, i, j=1, is_forward=True):
        beg_id = self.get_sequence_beg_id(i)
        end_id = self.get_sequence_beg_id(i + j)
        if is_forward:
            id_list = list(range(beg_id, end_id))
        else:
            i += 1
            id_list = list(range(end_id, beg_id, -1))
        tmp_dir = self.__get_tmp_out_subdir(i)
        path_dir = [
            os.path.join(tmp_dir, 'temporal_' + self.__output_format % id)
            for id in id_list
        ]
        return path_dir

    def get_edge_sequence(self, i, j=1, is_forward=True):
        beg_id = self.get_sequence_beg_id(i)
        end_id = self.get_sequence_beg_id(i + j)
        if is_forward:
            id_list = list(range(beg_id, end_id))
        else:
            i += 1
            id_list = list(range(end_id, beg_id, -1))
        tmp_dir = self.__get_tmp_out_subdir(i)
        path_dir = [
            os.path.join(tmp_dir, 'edge_' + self.__output_format % id)
            for id in id_list
        ]
        return path_dir

    def get_pos_sequence(self, i, j=1, is_forward=True):
        beg_id = self.get_sequence_beg_id(i)
        end_id = self.get_sequence_beg_id(i + j)
        if is_forward:
            id_list = list(range(beg_id, end_id))
        else:
            i += 1
            id_list = list(range(end_id, beg_id, -1))
        tmp_dir = self.__get_tmp_out_subdir(i)
        path_dir = [
            os.path.join(tmp_dir, 'pos_' + self.__output_format % id)
            for id in id_list
        ]
        return path_dir

    def get_flow_sequence(self, i, j=1, is_forward=True):
        beg_id = self.get_sequence_beg_id(i)
        end_id = self.get_sequence_beg_id(i + j)
        if is_forward:
            id_list = list(range(beg_id, end_id - 1))
            path_dir = [
                os.path.join(self.__tmp_dir, 'flow_f_%04d.npy' % id)
                for id in id_list
            ]
        else:
            id_list = list(range(end_id, beg_id + 1, -1))
            path_dir = [
                os.path.join(self.__tmp_dir, 'flow_b_%04d.npy' % id)
                for id in id_list
            ]

        return path_dir

    def get_input_img(self, i):
        return os.path.join(self.__input_dir, self.__input_format % i)

    def get_key_img(self, i):
        sequence_beg_id = self.get_sequence_beg_id(i)
        return os.path.join(self.__key_dir,
                            self.__key_format % sequence_beg_id)

    def get_blending_img(self, i):
        return os.path.join(self.__blending_out_dir, self.__output_format % i)

    def get_sequence_beg_id(self, i):
        """
        Get the first frame ID of the i-th sequence

        Previous implementation calculated this mathematically: i * self.__interval + self.__beg_frame
        :param i:
        :return:
        """
        return self.__key_frames[i] if i <= self.n_seq else self.__key_frames[-1]

    def __get_out_subdir(self, i):
        dir_id = self.get_sequence_beg_id(i)
        out_subdir = os.path.join(self.__base_dir,
                                  self.__out_subdir_format % dir_id)
        return out_subdir

    def __get_tmp_out_subdir(self, i):
        dir_id = self.get_sequence_beg_id(i)
        tmp_out_subdir = os.path.join(self.__tmp_dir,
                                      self.__out_subdir_format % dir_id)
        return tmp_out_subdir

    def __make_out_dirs(self):
        os.makedirs(self.__base_dir, exist_ok=True)
        os.makedirs(self.__blending_out_dir, exist_ok=True)
        for i in range(self.n_seq + 1):
            out_subdir = self.__get_out_subdir(i)
            tmp_subdir = self.__get_tmp_out_subdir(i)
            os.makedirs(out_subdir, exist_ok=True)
            os.makedirs(tmp_subdir, exist_ok=True)
