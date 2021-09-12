import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
from pytracking.utils.load_text import load_text
import os


class UAVDARKDataset(BaseDataset):
    """ UAVDark135 dataset.
    Download the dataset from https://github.com/vision4robotics/ADTrack/tree/journal
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.uavdark_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path, 
        sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext) for frame_num in range(start_frame+init_omit, end_frame+1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64, backend='numpy')

        return Sequence(sequence_info['name'], frames, 'uavdark', ground_truth_rect[init_omit:,:],
                        object_class=sequence_info['object_class'])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        datapath=self.base_path+'/Sequences/'
        seqs=os.listdir(datapath)
        sequence_info_list=[]
        for seq in seqs:
            seq_dic={}
            seq_dic["name"]=seq
            seq_dic["path"]="/Sequences/"+seq
            seq_dic["startFrame"]=1
            seq_dic["endFrame"]=len(os.listdir(self.base_path+seq_dic["path"]))
            seq_dic["nz"]=5 
            seq_dic["ext"]="jpg"
            seq_dic["anno_path"]="anno_revise/"+seq+'.txt'
            seq_dic["object_class"]="car"
            sequence_info_list.append(seq_dic)
        return sequence_info_list
