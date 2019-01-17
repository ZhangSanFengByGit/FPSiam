# coding: utf-8 
import matplotlib.pyplot as plt 
import matplotlib
import scipy.misc
from pylab import * 
from PIL import Image
import pdb

def get_row_col(num_pic): 
	squr = num_pic ** 0.5 
	row = round(squr) 
	col = row + 1 if squr - row > 0 else row 
	return row,col 


def visualize_feature_map(img_batch,a,b,layer): 
	feature_map = img_batch 
	print(feature_map.shape)
	feature_map_combination=[] 
	plt.figure() 
	num_pic = feature_map.shape[2] 
	row,col = get_row_col(num_pic) 
	for i in range(0,num_pic): 
		feature_map_split=feature_map[:,:,i] 
		feature_map_combination.append(feature_map_split) 
		plt.subplot(row,col,i+1) 
		plt.imshow(feature_map_split) 
		axis('off') 
	'''plt.convert('RGB')	
	plt.savefig('feature_map.jpg') 
	plt.show()'''
	
	# 各个特征图按1：1 叠加 
	feature_map_sum = sum(ele for ele in feature_map_combination)
	c=len(feature_map_combination)
	feature_map_sum=feature_map_sum/c
	feature_map_sum=feature_map_sum-mean(feature_map_sum)
	feature_map_sum=feature_map_sum.convert('RGB')
	plt.imshow(feature_map_sum) 
	#pdb.set_trace()
	'''im = Image.fromarray(feature_map_sum)
	if im.mode != 'RGB':
		im = im.convert('RGB')
		im.save("you.jpeg")'''
	#plt.savefig("feature_map_sum.png")
	pdb.set_trace()
	feature_map_sum=scipy.misc.imresize(feature_map_sum,[b,a])
	matplotlib.image.imsave('feature_map/'+layer+'.png',feature_map_sum)
	'''img=Image.open('feature_map/feature_map_sum.png')
	img1=img.convert('RGB')
	img1.save('data/conv5.jpg')'''
