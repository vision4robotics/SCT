# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import cv2
import torch
import numpy as np

from pysot.models.model_builder_adapn import ModelBuilderADAPN
from pysot.tracker.adsiamapn_tracker import ADSiamRPNTracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from pysot.core.config_adapn import cfg
import sys
sys.path.append("../..")
from SCT import SCT_model
# from Enhancer_jour.Uformer_model import Unet_sp_ch
parser = argparse.ArgumentParser(description='siamapn tracking')
parser.add_argument('--dataset', default='DarkTrack',type=str,
        help='datasets')
parser.add_argument('--snapshot', default='./tools/snapshot/general++.pth', type=str,
        help='snapshot of models to eval')
parser.add_argument('--config', default='./experiments/config.yaml', type=str,
        help='config file')
parser.add_argument('--trackername', default='SiamAPN++', type=str,
        help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--vis', default='',action='store_true',
        help='whether visualzie result')
parser.add_argument('--dataset_root', default='/media/ye/Luck/dataset/') # customize your path to dataset root
parser.add_argument('--enhance', default='', action='store_true', 
        help='whether enable enhancer')#action：一旦出现这个参数，就将其值标为true
args = parser.parse_args()

save_name = 'SiamAPN++_SCT_' + str(args.enhance)
torch.set_num_threads(1)
@torch.no_grad()    
def main():
    # load config

    cfg.merge_from_file(args.config)
    model = ModelBuilderADAPN()
    model = load_pretrain(model, args.snapshot).cuda().eval()
    tracker = ADSiamRPNTracker(model)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = args.dataset_root
    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = args.snapshot.split('/')[-1].split('.')[0]+str(cfg.TRACK.w1)

    SCT_net = ''
    if args.enhance:
        SCT_net = SCT_model.SCT(img_size=128,embed_dim=32,win_size=4,token_embed='linear',token_mlp='resffn')
        SCT_net.load_state_dict(torch.load(os.path.join(cur_dir, '../../../SCT/log/SCT/models/model_latest.pth')))
        SCT_net.cuda()

    # OPE tracking
    for v_idx, video in enumerate(dataset):
        model_path = os.path.join('results', args.dataset, save_name)
        if os.path.exists(os.path.join(model_path, '{}.txt'.format(video.name))):
            print(video.name)
            continue
        if args.video != '':
            # test one special video
            if video.name != args.video:
                continue
        toc = 0
        pred_bboxes = []
        scores = []
        track_times = []
        for idx, (img, gt_bbox) in enumerate(video):
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                tracker.init(img, gt_bbox_, SCT_net)
                pred_bbox = gt_bbox_
                scores.append(None)
                if 'VOT2018-LT' == args.dataset:
                    pred_bboxes.append([1])
                else:
                    pred_bboxes.append(pred_bbox)
            else:
                outputs = tracker.track(img, SCT_net)
                pred_bbox = outputs['bbox']
                pred_bboxes.append(pred_bbox)
                scores.append(outputs['best_score'])
            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
            if idx == 0:
                cv2.destroyAllWindows()
            if args.vis and idx > 0:
                try:
                    gt_bbox = list(map(int, gt_bbox))
                except:
                    gt_bbox=[0,0,0,0]
                pred_bbox = list(map(int, pred_bbox))
                cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow(video.name, img)
                cv2.waitKey(1)
        toc /= cv2.getTickFrequency()
        # save results

        model_path = os.path.join('results', args.dataset, save_name)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        result_path = os.path.join(model_path, '{}.txt'.format(video.name))
        with open(result_path, 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x])+'\n')
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx+1, video.name, toc, idx / toc))
    

if __name__ == '__main__':
    main()
