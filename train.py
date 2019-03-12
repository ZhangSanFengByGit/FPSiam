

import sys
import cv2  # imread
import torch
import numpy as np
import os
from os.path import realpath, dirname, join
import time
import math

from net import SiamRPN
from run_SiamRPN import SiamRPN_init_batch, SiamRPN_train_batch, SiamRPN_set_source_batch
from utils import get_axis_aligned_bbox, cxy_wh_2_rect, rect_2_cxy_wh, adjust_learning_rate
from data_provider import datas

#custom params
save_path = (realpath(dirname(__file__)))+'/V3_AutoSaved_epoch_model/'
if not os.path.exists(save_path):
	os.mkdir(save_path)
#save_print = open('process1.out','w')

# load net
net_file = join(realpath(dirname(__file__)), 'SiamRPNOTB.model')
net = SiamRPN()
pretrain_dict = torch.load(net_file)
model_dict = net.state_dict()
model_dict.update(pretrain_dict)
net.load_state_dict(model_dict)
net.cuda().train()



#set training toolkit
'''
optimizer = torch.optim.SGD([{'params': net.conv_xfit.parameters()}, \
							 {'params': net.conv_oldfit.parameters()}, \
							 {'params': net.kernel_pre.parameters(), 'lr': 0.01}, \
							 {'params': net.embed_net.parameters(), 'lr': 0.007} ], lr = 0.005, momentum = 0.9, weight_decay = 0.00005)
'''
optimizer = torch.optim.SGD([{'params': net.featureExtract[11].parameters()}, \
							 {'params': net.featureExtract[12].parameters()}, \
							 {'params': net.featureExtract[14].parameters()}, \
							 {'params': net.featureExtract[15].parameters()}, \
							 {'params': net.conv_r1.parameters()}, \
							 {'params': net.conv_r2.parameters()}, \
							 {'params': net.conv_cls1.parameters()}, \
							 {'params': net.conv_cls2.parameters()}, \
							 {'params': net.regress_adjust.parameters()}, \
							 {'params': net.conv_xfit.parameters(), 'lr': 0.007}, \
							 {'params': net.conv_oldfit.parameters(), 'lr': 0.007}, \
							 {'params': net.kernel_pre.parameters(), 'lr': 0.01}, \
							 {'params': net.embed_net.parameters(), 'lr': 0.007} ], lr = 0.001, momentum = 0.9, weight_decay = 0.00005)



'''
# warm up
for i in range(10):
    net.temple(torch.zeros(1, 3, 127, 127).cuda(), torch.zeros(1, 3, 271, 271).cuda())
    net(torch.zeros(1, 3, 271, 271).cuda(), set_source=True)
'''

#set data provider
_data_provider = datas()




#config
batch_size = 8
lamda = 1
epoch_rate = 20
warm_up_th = 15
decode_rate = 1
decay_step = 50
start_lr = 0.001



#begin training
epoch_num = int(_data_provider.seq_num* epoch_rate)
lr_decay = math.pow(1e-3, 1.0/int(epoch_num/decay_step))
print('lr_decay is : {}'.format(lr_decay))


for epoch in range(epoch_num):

	if epoch < warm_up_th:
		adjust_learning_rate(optimizer, decode_rate)
		start_lr *= decode_rate
		encode_rate = math.pow(1e-1, (warm_up_th-epoch))
		decode_rate = math.pow(10, (warm_up_th-epoch))
		adjust_learning_rate(optimizer, encode_rate)
		start_lr *= encode_rate
		print("current learning rate global : {}".format(start_lr))

	elif epoch == warm_up_th:
		adjust_learning_rate(optimizer, decode_rate)
		start_lr *= decode_rate
		print("current learning rate global : {} , back to normal----------------------------".format(start_lr))


	_data_provider.rand_pick_seq()

	if (epoch+1)%int(decay_step) == 0:
		adjust_learning_rate(optimizer, lr_decay)
		start_lr *= lr_decay
		print("current learning rate global : {}------------------------------------".format(start_lr))


	for jj in range(max(1, int(_data_provider.cur_img_num/200))):

		exemplar_list = [None for i in range(batch_size)]
		source_list = [None for i in range(batch_size)]
		instance_list = [None for i in range(batch_size)]
		exemplar_cxy_list = [[None, None] for i in range(batch_size)]
		source_cxy_list = [[None, None] for i in range(batch_size)]
		instance_cxy_list = [[None, None] for i in range(batch_size)]

		for batch in range(batch_size):
			pairs, gts = _data_provider.rand_pick_pair()

			exemplar = cv2.imread(pairs[0])
			source = cv2.imread(pairs[1])
			instance = cv2.imread(pairs[2])
			exemplar_pos, exemplar_sz = rect_2_cxy_wh(gts[0])
			source_pos, source_sz = rect_2_cxy_wh(gts[1])
			instance_pos, instance_sz = rect_2_cxy_wh(gts[2])

			exemplar_list[batch] = exemplar
			source_list[batch] = source
			instance_list[batch] = instance
			exemplar_cxy_list[batch][0], exemplar_cxy_list[batch][1] = exemplar_pos, exemplar_sz
			source_cxy_list[batch][0], source_cxy_list[batch][1] = source_pos, source_sz
			instance_cxy_list[batch][0], instance_cxy_list[batch][1] = instance_pos, instance_sz


		train_config = SiamRPN_init_batch(exemplar_list, exemplar_cxy_list, net)
		SiamRPN_set_source_batch(train_config, source_list, source_cxy_list)

		net.zero_grad()
		cls_loss, box_loss = SiamRPN_train_batch(train_config, instance_list, source_cxy_list, instance_cxy_list)

		loss = cls_loss + lamda*box_loss
		if jj%1==0:
			print('epoch:{} seq:{}	cls_loss:{} , box_loss:{}\n\n'.format(epoch, _data_provider.cur_seq, cls_loss.item(), box_loss.item()))
			#save_print.write('epoch:{} seq:{}	cls_loss:{} , box_loss:{}\n'.format(epoch, _data_provider.cur_seq, cls_loss.item(), box_loss.item()))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	if (epoch+1) % int(epoch_num/100) ==0:
		torch.save(net.state_dict(), save_path+'V3_epoch_'+str(epoch)+'.pth')

#save_print.close()







