#2018.12.15 zzc

import os
import random


def parse_region(string):
    if ',' in string:
        tokens = map(float, string.split(','))
    elif '\t' in string:
        tokens = map(float, string.split('\t'))
    else:
        tokens = map(float, string.split(' '))
    tokens = list(tokens)
    assert len(tokens) == 4, 'error'
    return [tokens[0], tokens[1], tokens[2], tokens[3]]
    #elif len(tokens) % 2 == 0 and len(tokens) > 4:
        #return ([Point(tokens[i],tokens[i+1]) for i in xrange(0,len(tokens),2)])
    #return None



class datas(object):
	def __init__(self):
		OTB_path = '/home/zhangzichun/OTB2015/'
		seq_list = open(OTB_path+'files.txt','r').readlines()
		for i in range(len(seq_list)):
			seq_list[i] = seq_list[i].replace('seq/','')
			seq_list[i] = seq_list[i].replace('.zip\n','')

		self.data_path = OTB_path
		self.seq_list = seq_list
		self.seq_num = len(seq_list)
		self.special_seq_start = {'BlurCar1' : 247, 'BlurCar3' : 3, 'BlurCar4' : 18}


	def get_seq_list(self):
		return self.seq_list


	def pick_seq(self, seq):
		self.cur_seq = seq
		self.rand_pick_seq(fixed_seq=True)


	def rand_pick_seq(self, fixed_seq=False):
		if fixed_seq==False:
			i = random.randrange(0, self.seq_num)
			self.cur_seq = self.seq_list[i]

		if self.cur_seq == 'Jogging-1':
			self.cur_seq_path = self.data_path + 'Jogging/'
			self.cur_imgs_path = self.cur_seq_path + 'img/'
			self.cur_gt = open(self.cur_seq_path+'groundtruth_rect.1.txt','r').readlines()

		elif self.cur_seq == 'Jogging-2':
			self.cur_seq_path = self.data_path + 'Jogging/'
			self.cur_imgs_path = self.cur_seq_path + 'img/'
			self.cur_gt = open(self.cur_seq_path+'groundtruth_rect.2.txt','r').readlines()

		elif self.cur_seq == 'Skating2-1':
			self.cur_seq_path = self.data_path + 'Skating2/'
			self.cur_imgs_path = self.cur_seq_path + 'img/'
			self.cur_gt = open(self.cur_seq_path+'groundtruth_rect.1.txt','r').readlines()

		elif self.cur_seq == 'Skating2-2':
			self.cur_seq_path = self.data_path + 'Skating2/'
			self.cur_imgs_path = self.cur_seq_path + 'img/'
			self.cur_gt = open(self.cur_seq_path+'groundtruth_rect.2.txt','r').readlines()

		else:
			self.cur_seq_path = self.data_path + self.cur_seq + '/'
			self.cur_imgs_path = self.cur_seq_path + 'img/'
			
			if self.cur_seq in ['Jogging', 'Skating2']:
				j = random.randrange(1,3)
				if j==1:
					self.cur_gt = open(self.cur_seq_path+'groundtruth_rect.1.txt','r').readlines()
				else:
					self.cur_gt = open(self.cur_seq_path+'groundtruth_rect.2.txt','r').readlines()

			elif self.cur_seq=='Human4':
				self.cur_gt = open(self.cur_seq_path+'groundtruth_rect.2.txt','r').readlines()

			else:
				self.cur_gt = open(self.cur_seq_path+'groundtruth_rect.txt','r').readlines()

		self.cur_img_num = len(self.cur_gt)
		for i in range(len(self.cur_gt)):
			self.cur_gt[i] = parse_region(self.cur_gt[i])


	def rand_pick_pair(self):
		start = 1
		if self.cur_seq in self.special_seq_start:
			start = self.special_seq_start[self.cur_seq]
		exemplar_idx = random.randrange(start, int(start+self.cur_img_num-50))
		instance_idx = random.randrange(exemplar_idx+50, int(start+self.cur_img_num))
		if instance_idx - exemplar_idx > 200:
			instance_idx = exemplar_idx + 200
		source_interval = random.randrange(1,5)
		source_idx = instance_idx - source_interval

		exemplar_gt = self.cur_gt[exemplar_idx-start]
		instance_gt = self.cur_gt[instance_idx-start]
		source_gt = self.cur_gt[source_idx-start]

		if self.cur_seq == 'Board':
			exemplar_idx = str(100000+exemplar_idx).replace('1','',1) + '.jpg'
			instance_idx = str(100000+instance_idx).replace('1','',1) + '.jpg'
			source_idx = str(100000+source_idx).replace('1','',1) + '.jpg'
		else:
			exemplar_idx = str(10000+exemplar_idx).replace('1','',1) + '.jpg'
			instance_idx = str(10000+instance_idx).replace('1','',1) + '.jpg'
			source_idx = str(10000+source_idx).replace('1','',1) + '.jpg'

		exemplar_path = self.cur_imgs_path + exemplar_idx
		instance_path = self.cur_imgs_path + instance_idx
		source_path = self.cur_imgs_path + source_idx

		assert os.path.exists(exemplar_path)==True, exemplar_path+' : exemplar img not exists'
		assert os.path.exists(instance_path)==True, instance_path+' : instance img not exists'
		assert os.path.exists(source_path)==True, source_path+' : source img not exists'

		return [exemplar_path, source_path, instance_path], [exemplar_gt, source_gt, instance_gt]


	def eval_pick_exemplar(self):
		start = 1
		if self.cur_seq in self.special_seq_start:
			start = self.special_seq_start[self.cur_seq]

		exemplar_idx = start
		self.eval_start = start
		self.eval_gt_idx = start
		exemplar_gt = self.cur_gt[exemplar_idx-start]

		if self.cur_seq == 'Board':
			exemplar_idx = str(100000+exemplar_idx).replace('1','',1) + '.jpg'
		else:
			exemplar_idx = str(10000+exemplar_idx).replace('1','',1) + '.jpg'

		exemplar_path = self.cur_imgs_path + exemplar_idx
		assert os.path.exists(exemplar_path)==True, exemplar_path+' : exemplar img not exists'

		return exemplar_path, exemplar_gt, self.cur_img_num


	def eval_pick_instance(self):
		instance_gt = self.cur_gt[self.eval_gt_idx - self.eval_start]
		if self.cur_seq == 'Board':
			instance_idx = str(100000+self.eval_gt_idx).replace('1','',1) + '.jpg'
		else:
			instance_idx = str(10000+self.eval_gt_idx).replace('1','',1) + '.jpg'

		instance_path = self.cur_imgs_path + instance_idx
		assert os.path.exists(instance_path)==True, instance_path+' : instance img not exists'

		self.eval_gt_idx +=1
		return  instance_path


