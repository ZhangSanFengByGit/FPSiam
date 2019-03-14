

import sys
import cv2  # imread
import torch
import numpy as np
import os
from os.path import realpath, dirname, join
import time

from net import SiamRPN
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect, rect_2_cxy_wh
from data_provider import datas
import argparse


seqs_idx=None
parser = argparse.ArgumentParser(description='Evaluate the pretrained model')
parser.add_argument('--seqsIdx', type=int, default = None)
parser.add_argument('--fixSeq', type=str, default=None)
parser.add_argument('--model', type=str, default=None)
args = parser.parse_args()
fixSeq = args.fixSeq 
seqs_idx = args.seqsIdx
model = args.model


def encode_region(region):
    if isinstance(region, Polygon):
        return ','.join(['{},{}'.format(p.x,p.y) for p in region.points])
    elif isinstance(region, Rectangle):
        return '{},{},{},{}'.format(region.x, region.y, region.width, region.height)
    else:
        return ""


#custom params
load_path = (realpath(dirname(__file__)))+'/V3_AutoSaved_epoch_model/'
save_res_path = (realpath(dirname(__file__)))+'/'+model+'/'
if not os.path.exists(save_res_path):
	os.mkdir(save_res_path)


#set data provider
_data_provider = datas()

#OTB_seqs = _data_provider.get_seq_list()

'''
0['Basketball', 'Biker', 'Bird1', 'BlurBody', 'BlurCar2', 'BlurFace', 'BlurOwl', 'Bolt', \
1'Box', 'Car1', 'Car4', 'CarDark', 'CarScale', 'ClifBar', 'Couple', 'Crowds', 'David', \
2'Deer', 'Diving', 'DragonBaby', 'Dudek', 'Football', 'Freeman4', 'Girl', 'Human3', 'Human4', \
3'Human6', 'Human9', 'Ironman', 'Jump', 'Jumping', 'Liquor', 'Matrix', 'MotorRolling', 'Panda', \
4'RedTeam', 'Shaking', 'Singer2', 'Skating1', 'Skating2', 'Skiing', 'Soccer', 'Surfer', \
5'Sylvester', 'Tiger2', 'Trellis', 'Walking', 'Walking2', 'Woman', 'Bird2', 'BlurCar1', \
6'BlurCar3', 'BlurCar4', 'Board', 'Bolt2', 'Boy', 'Car2', 'Car24', 'Coke', 'Coupon', 'Crossing', \
7'Dancer', 'Dancer2', 'David2', 'David3', 'Dog', 'Dog1', 'Doll', 'FaceOcc1', 'FaceOcc2', \
8'Fish', 'FleetFace', 'Football1', 'Freeman1', 'Freeman3', 'Girl2', 'Gym', 'Human2', 'Human5', \
9'Human7', 'Human8', 'Jogging', 'KiteSurf', 'Lemming', 'Man', 'Mhyang', 'MountainBike', 'Rubik', \
10'Singer1', 'Skater', 'Skater2', 'Subway', 'Suv', 'Tiger1', 'Toy', 'Trans', 'Twinnings', 'Vase']
'''


if fixSeq!=None:
	OTB_seqs = [fixSeq]
else:
	if seqs_idx == 0:
		OTB_seqs = ['Basketball', 'Biker', 'Bird1', 'BlurBody', 'BlurCar2', 'BlurFace', 'BlurOwl', 'Bolt']
	elif seqs_idx ==1:
		OTB_seqs = ['Box', 'Car1', 'Car4', 'CarDark', 'CarScale', 'ClifBar', 'Couple', 'Crowds', 'David']
	elif seqs_idx ==2:
		OTB_seqs = ['Deer', 'Diving', 'DragonBaby', 'Dudek', 'Football', 'Freeman4', 'Girl', 'Human3', 'Human4']
	elif seqs_idx ==3:
		OTB_seqs = ['Human6', 'Human9', 'Ironman', 'Jump', 'Jumping', 'Liquor', 'Matrix', 'MotorRolling', 'Panda']
	elif seqs_idx ==4:
		OTB_seqs = ['RedTeam', 'Shaking', 'Singer2', 'Skating1', 'Skating2-1', 'Skating2-2', 'Skiing', 'Soccer', 'Surfer']
	elif seqs_idx ==5:
		OTB_seqs = ['Sylvester', 'Tiger2', 'Trellis', 'Walking', 'Walking2', 'Woman', 'Bird2', 'BlurCar1']
	elif seqs_idx ==6:
		OTB_seqs = ['BlurCar3', 'BlurCar4', 'Board', 'Bolt2', 'Boy', 'Car2', 'Car24', 'Coke', 'Coupon', 'Crossing']
	elif seqs_idx ==7:
		OTB_seqs = ['Dancer', 'Dancer2', 'David2', 'David3', 'Dog', 'Dog1', 'Doll', 'FaceOcc1', 'FaceOcc2']
	elif seqs_idx ==8:
		OTB_seqs = ['Fish', 'FleetFace', 'Football1', 'Freeman1', 'Freeman3', 'Girl2', 'Gym', 'Human2', 'Human5']
	elif seqs_idx ==9:
		OTB_seqs = ['Human7', 'Human8', 'Jogging-1', 'Jogging-2', 'KiteSurf', 'Lemming', 'Man', 'Mhyang', 'MountainBike', 'Rubik']
	elif seqs_idx ==10:
		OTB_seqs = ['Singer1', 'Skater', 'Skater2', 'Subway', 'Suv', 'Tiger1', 'Toy', 'Trans', 'Twinnings', 'Vase']
	else:
		raise RuntimeError('no seqs_idx found')

# load net
net_file = load_path + model#join(realpath(dirname(__file__)), 'SiamRPNOTB.model')
net = SiamRPN(pretrain = True)
pretrain_dict = torch.load(net_file)
model_dict = net.state_dict()
pretrain_dict = {k:v for k,v in pretrain_dict.items() if k in model_dict}
model_dict.update(pretrain_dict)
net.load_state_dict(model_dict)
net.cuda().eval()


# warm up
for i in range(10):
    net.temple(torch.autograd.Variable(torch.ones(1, 3, 127, 127)).cuda(), \
    	torch.autograd.Variable(torch.ones(1, 3, 271, 271)).cuda())
    net(torch.autograd.Variable(torch.ones(1, 3, 271, 271)).cuda())


for seq in OTB_seqs:
	_data_provider.pick_seq(seq)
	exemplar_path, exemplar_gt, cur_img_num = _data_provider.eval_pick_exemplar()

	exemplar = cv2.imread(exemplar_path)
	exemplar_pos, exemplar_sz = rect_2_cxy_wh(exemplar_gt)
	state = SiamRPN_init(exemplar, exemplar_pos, exemplar_sz, net)
	save_file = save_res_path + seq +'.txt'
	tracking_res = open(save_file, 'w')

	for idx in range(cur_img_num):

		instance_path = _data_provider.eval_pick_instance()
		instance = cv2.imread(instance_path)
		state = SiamRPN_track(state, instance)
		print('seq:{}:{} , score:{}'.format(seq, idx, state['score']))
		res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
		tracking_res.write('{},{},{},{}'.format(res[0], res[1], res[2], res[3]))
		tracking_res.write('\n')

	tracking_res.close()



