import torch
import torchvision
import torch.optim
import os
import time
from SCT_model import SCT as model
import glob
import time
from utils.dataloader import load_img

os.environ['CUDA_VISIBLE_DEVICES']='0'
def lowlight(image_path):
	
	data_lowlight = load_img(image_path)

	data_lowlight = torch.from_numpy(data_lowlight).float()
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)

	SCT_net = model(img_size=128,embed_dim=32,win_size=4,token_embed='linear',token_mlp='resffn')

	SCT_net.load_state_dict(torch.load('log/SCT/models/model_latest.pth'))
	SCT_net.cuda()
	SCT_net.eval()
	start = time.time()
	enhanced_image = SCT_net(data_lowlight)

	end_time = (time.time() - start)
	print(end_time)
	image_path = image_path.replace('test_data','result')
	result_path = image_path
	if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
		os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))

	torchvision.utils.save_image(enhanced_image, result_path)

if __name__ == '__main__':
# test_images
	with torch.no_grad():
		filePath = 'test_data/'
	
		file_list = os.listdir(filePath)

		for file_name in file_list:
			test_list = glob.glob(filePath+file_name+"/*") 
			for image in test_list:
				print(image)
				lowlight(image)		
