# SCT

This is the official code for the paper "Tracker Meets Night: A Transformer Enhancer for UAV Tracking"

The spatial-channel Transformer (SCT) enhancer is a task-inspired low-light enhancer toward facilitating nighttime UAV tracking. Evaluations on the public UAVDark135 and the newly constructed DarkTrack2021 benchmarks demonstrate that the performance gains of SCT brought to nighttime UAV tracking surpass general low-light enhancers.

SCT has been submitted to RA-L.

## Environment Preparing
```
python 3.6
pytorch 1.8.1
```

### Testing

Run *lowlight_test.py*, the results will be saved in `./result/`
```
python lowlight_test.py 
```

### Training

Before training, you should prepare the training set of the [LOL](https://daooshee.github.io/BMVC2018website/) dataset.
Run *lowlight_train.py*. 
The model will be saved in `./log/SCT/models`

```
python lowlight_train.py --trainset_path /your/path/to/LOLdataset/
```

A great thanks to [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) for providing the basis for this code.