from .uavdark import UAVDARKDataset
from .darktrack import DarkTrackDataset
class DatasetFactory(object):
    @staticmethod
    def create_dataset(**kwargs):


        assert 'name' in kwargs, "should provide dataset name"
        name = kwargs['name']
        
        if 'uavdark' == name:
            dataset = UAVDARKDataset(**kwargs)
        elif 'DarkTrack' == name:
            dataset = DarkTrackDataset(**kwargs)   
        else:
            raise Exception("unknow dataset {}".format(kwargs['name']))
        return dataset

