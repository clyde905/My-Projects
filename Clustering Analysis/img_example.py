import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

#########################################################################
# working with tif/png

num_image = 1

for k in range(0, num_image):	# read this many frames

 fname = 'test_dataset/virus_transport/tutorial'+str(k+1)+'.tif'

 img = mpimg.imread(fname)     # the jpg image reads directly gives you intensity data (unlike what was discussed before, when you read certain formats, it gives you a matrix where each element of the matrix is a RGB vector, but we only want a "regular" matrix where each element is one number).

#shows the RGB slice: now you can see the most intensity of pixels using a color map
 img = img[:,:,0]

 imgplot = plt.imshow(img)
 plt.show()

 B = np.zeros(img.shape)

 for i in range(100,img.shape[0]): # I started from 100 because the top left corner of the images has some text labels which I want to remove
  for j in range(0,img.shape[1]):
 
   if(img[i][j] > 100): 
    B[i][j] = 1
   else:
    B[i][j] = 0

 print('saving image '+str(k+1))

 save_name = 'thresholded_tutorial'+str(k+1)+'.tif'
 mpimg.imsave(save_name, B) 

########### working with jpg ########################################

for k in range(0, num_image):	# read this many frames

 fname = 'test_dataset/thermal_motion/Image'+str(k+1)+'.jpg'

 img = mpimg.imread(fname)     # the jpg image reads directly gives you intensity data (unlike what was discussed before, when you read certain formats, it gives you a matrix where each element of the matrix is a RGB vector, but we only want a "regular" matrix where each element is one number).

 imgplot = plt.imshow(img)
 plt.show()

 B = np.zeros(img.shape)

 for i in range(100,img.shape[0]): # I started from 100 because the top left corner of the images has some text labels which I want to remove
  for j in range(0,img.shape[1]):
 
   if(img[i][j] > 100): 
    B[i][j] = 1
   else:
    B[i][j] = 0

 print('saving image '+str(k+1))

 save_name = 'thresholded_Image'+str(k+1)+'.jpg'
 mpimg.imsave(save_name, B) 

 
