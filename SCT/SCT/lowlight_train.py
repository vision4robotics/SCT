import torch
import torch.optim
import os
import argparse
import time
from utils import dataloader
from SCT_model import SCT
import numpy as np
import datetime
import random
import utils
from torch.optim.lr_scheduler import StepLR
from warmup_scheduler import GradualWarmupScheduler
from timm.utils import NativeScaler
from Myloss import perception_loss
os.environ['CUDA_VISIBLE_DEVICES']='0'
def train(config):
	######### Logs dir ###########
	dir_name = os.path.dirname(os.path.abspath(__file__))
	log_dir = os.path.join(dir_name,'log', config.note)
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	logname = os.path.join(log_dir, datetime.datetime.now().isoformat()+'.txt') 
	print("Now time is : ",datetime.datetime.now().isoformat())
	model_dir  = os.path.join(log_dir, 'models')
	utils.mkdir(model_dir)
	###############################################################
	# ######### Set Seeds ###########
	random.seed(1234)
	np.random.seed(1234)
	torch.manual_seed(1234)
	torch.cuda.manual_seed_all(1234)
	########### Loss #########################
	Loss_perception = perception_loss()
	########### Model ##############
	SCT_net = SCT(img_size=128,embed_dim=32,win_size=4,token_embed='linear',token_mlp='resffn')

	with open(logname,'a') as f:
		f.write(str(config)+'\n')
		f.write(str(SCT_net)+'\n')

	SCT_net.cuda()

	start_epoch = 1
	optimizer = torch.optim.AdamW(SCT_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

	######### Resume ###########
	if config.load_pretrain:
		path_chk_rest = config.pretrain_dir
		utils.load_checkpoint(SCT_net,path_chk_rest)
		start_epoch = utils.load_start_epoch(path_chk_rest) + 1
		lr = utils.load_optim(optimizer, path_chk_rest)

		for p in optimizer.param_groups: p['lr'] = lr
		new_lr = lr
		print('------------------------------------------------------------------------------')
		print("==> Resuming Training with learning rate:",new_lr)
		print('------------------------------------------------------------------------------')
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.num_epochs-start_epoch+1, eta_min=1e-6)

	######### Scheduler ###########
	if config.warmup:
		print("Using warmup and cosine strategy!")
		warmup_epochs = config.warmup_epochs
		scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.num_epochs-warmup_epochs, eta_min=1e-6)
		scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
		scheduler.step()
	else:
		step = 50
		print("Using StepLR,step={}!".format(step))
		scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
		scheduler.step()

	####################### dataloadder #########################################
	train_dataset = dataloader.pair_loader(config.trainset_path, config.train_ps)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size,
                                          shuffle=True, num_workers=config.num_workers,pin_memory=True, drop_last=False)

	len_trainset = train_dataset.__len__()

	######### train ###########
	print('===> Start Epoch {} End Epoch {}'.format(start_epoch,config.num_epochs))

	loss_scaler = NativeScaler()
	torch.cuda.empty_cache()

	for epoch in range(start_epoch, config.num_epochs+1):
		epoch_start_time = time.time()
		epoch_loss = 0

		for iteration, batch in enumerate(train_loader, 0):

			input_ = batch[0].cuda()
			target = batch[1].cuda()			
			out  = SCT_net(input_)
			out = torch.clamp(out,0,1)  
			loss= Loss_perception(out, target)

			loss_scaler(
					loss, optimizer,parameters=SCT_net.parameters())
			epoch_loss +=loss.item()

		scheduler.step()

		print("------------------------------------------------------------------")
		print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0]))
		print("------------------------------------------------------------------")
		with open(logname,'a') as f:
			f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0])+'\n')

		# torch.save({'epoch': epoch, 
		# 			'state_dict': SCT_net.state_dict(),
		# 			'optimizer' : optimizer.state_dict()
		# 			}, os.path.join(model_dir,"model_latest.pth"))   
		torch.save(SCT_net.state_dict(), os.path.join(model_dir,"model_latest.pth"))   
		if epoch%config.snapshot_iter == 0:
			torch.save({'epoch': epoch, 
						'state_dict': SCT_net.state_dict(),
						'optimizer' : optimizer.state_dict()
						}, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch))) 
	print("Now time is : ",datetime.datetime.now().isoformat())




if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--trainset_path', type=str, default='/media/ye/My_Passport/dataset/Image_enhancement/LOL/LOLdataset/our485/') 
	parser.add_argument('--lr', type=float, default=8e-4)
	parser.add_argument('--weight_decay', type=float, default=0.02)
	parser.add_argument('--num_epochs', type=int, default=100)
	parser.add_argument('--train_batch_size', type=int, default=32)
	parser.add_argument('--num_workers', type=int, default=16)
	parser.add_argument('--snapshot_iter', type=int, default=50)
	parser.add_argument('--load_pretrain', type=bool, default= False)
	parser.add_argument('--pretrain_dir', type=str, default= "log/model_latest.pth")
	parser.add_argument('--train_ps', type=int, default=256, help='patch size of training sample')
	parser.add_argument('--warmup', action='store_true', default=True, help='warmup') 
	parser.add_argument('--warmup_epochs', type=int,default=5, help='epochs for warmup') 
	parser.add_argument('--note', type=str, default= "SCT",
						help='Name of the log file')					
	config = parser.parse_args()
	train(config)
