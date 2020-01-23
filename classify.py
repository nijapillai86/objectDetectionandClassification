import cv2
import numpy as np
#import matplotlib.pyplot as plt

class object_classify():
	"""docstring for object_classify"""
	# def __init__(self, arg):
	# 	super(object_classify, self).__init__()
	# 	self.arg = arg
	#elemets = []
	
	def features(self,Matches, bin_img):
		pts =  np.array(Matches)[:, 0:2]
		pts =  pts.tolist()
		# e = []
		# print pts.ndim
		for match in pts:
			# print match
			# e.append(match)
			arr = np.concatenate(match, axis=None)
			print '----------------------'
			new_arr = np.split(arr, 1)
			print new_arr[0]
			# 	elemets.append(ex)
			x = new_arr[0][0]
			y = new_arr[0][1]
			w = new_arr[0][2]
			h = new_arr[0][3]
			self.cal_features(x,y,w,h,bin_img)

		# print bin_img
		# find frequency of pixels in range 0-255 
		# histr = cv2.calcHist(bin_img,[0],None,[256],[0,256]) 
		# plt.plot(histr) 
		# plt.show() 
		# points = np.array(Matches)[:, 0:2]
  #       points = points.tolist()
  #       for match in points:
  #       	print match

  	def cal_features(self,x,y,w,h,bin_img):
  		elemets = np.zeros((w*h,), dtype=int)
  		c = 0
  		# xstart = x
  		# ystart = y
  		# xend   = x + h
  		# yend   = y + w
  		# print bin_img.shape
  		#print w,' , ', h 				
  		

  		for i in range(y, y+h):
  			for j in range(x, x+w):
  				if bin_img[i][j] == 255:
					elemets[c] = bin_img[i][j]
					#print 'elemets at ',c,'=',elemets[c]
					c += 1
  		#print '--------------------',x,y,c
  		mean = round(np.mean(elemets),3)
  		#print mean
  		#To calculate the variance and standard deviation

		totdiff = np.power((elemets-mean), 2)
		totsum  = sum(totdiff)
		nele    = c - 1
		totvar  = round(totsum/nele, 3)
		#print 'variance :',totvar
		totstd  = round(np.sqrt(totvar), 3)
		#print 'Std deviation :',totstd
  		return c

  					