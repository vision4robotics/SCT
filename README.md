# SCT

This is the official code for the paper "Tracker Meets Night: A Transformer Enhancer for UAV Tracking"

The spatial-channel Transformer (SCT) enhancer is a task-inspired low-light enhancer toward facilitating nighttime UAV tracking. Evaluations on the public UAVDark135 and the newly constructed DarkTrack2021 benchmarks demonstrate that the performance gains of SCT brought to nighttime UAV tracking surpass general low-light enhancers.

SCT has been submitted to RA-L with ICRA option.

## Environment Preparing
```
python 3.6
pytorch 1.8.1
```

## Testing

Run *lowlight_test.py*, the results will be saved in `./result/`
```python
cd SCT
python lowlight_test.py 
```

## Training

Before training, you need to prepare the training set of the [LOL](https://daooshee.github.io/BMVC2018website/) dataset.
Run *lowlight_train.py*. 
The model will be saved in `./log/SCT/models`

```python
cd SCT
python lowlight_train.py --trainset_path /your/path/to/LOLdataset/
```

## SCT for Nighttime UAV Tracking

To evaluate the performance of SCT in facilitating trackers' nighttime tracking ability, you need to meet the enviroment requirements of base trackers and download their snapshots to corresponding folders at first. Details can be found in their repos. Currently supporting trackers including [HiFT](https://github.com/vision4robotics/HiFT), [SiamAPN++](https://github.com/vision4robotics/SiamAPN), [SiamRPN++](https://github.com/STVIR/pysot), [DiMP18, DiMP50, and PrDiMP50](https://github.com/visionml/pytracking).

For HiFT, SiamAPN++, and SiamRPN++, change directory to their corresponding root, and simply run trackers with “*--enhance*” option

```python
cd HiFT/SiamAPN++/pysot
python tools/test.py --dataset DarkTrack --enhance
```

For DiMP18, DiMP50, and PrDiMP50, customized your local paths in *pytracking/evaluation/local.py*

```python
cd pytracking 
python run_tracker.py --tracker_name dimp --tracker_param dimp18/dimp50/prdimp50 --enhance 
```

<img src="https://github.com/vision4robotics/SCT/blob/main/image/UAVDark135.png" width="180"><img src="https://github.com/vision4robotics/SCT/blob/main/image/DarkTrack2021.png" width="180">


## DarkTrack2021 Benchmark

Coming soon~

## Demo Video

[![Demo of SCT](https://res.cloudinary.com/marcomontalbano/image/upload/v1631463588/video_to_markdown/images/youtube--I1eZnJ_dbfg-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://youtu.be/I1eZnJ_dbfg "Demo of SCT")

## Contact

Junjie Ye
Email: ye.jun.jie@tongji.edu.cn

Changhong Fu
Email: changhongfu@tongji.edu.cn

## Acknowledgements

A great thanks to [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) for providing the basis for this code.																												