import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

######### functions
######### Copy uf_makeset(), uf_union(), uf_find() and hoshen_kopelman() into your own code!

def uf_makeset():
 labels[0]+=1
 ind = int(labels[0])
 labels[ind] = labels[0]
 return labels[0]

def uf_find(x):
 y=int(x)
 while labels[y] != y:
  y=int(labels[y])

 while labels[int(x)] != x:
  z = labels[int(x)]
  labels[int(x)] = y
  x = int(z)

 return y

def uf_union(x,y):
 ind = int(uf_find(x))
 labels[ind] = uf_find(y)
 return labels[uf_find(x)]

def hoshen_kopelman(M,Nr,Nc):	#input arguments: M is the matrix of 0's and 1's, and Nr, Nc are its row and column numbers.

 N_labels = int(Nr*Nc/2)

 for i in range(0,Nr):
  for j in range(0,Nc): 

   if M[i][j]: # if M[i][j] is occupied (=1)
 
    if i==0:
     up = 0
    elif i!=0: 
     up = M[i-1][j]

    if j==0:
     left = 0
    elif j!=0:
     left = M[i][j-1]

    if np.sign(up)+np.sign(left)==0:
     M[i][j] = uf_makeset()
    elif np.sign(up)+np.sign(left)==1:
     M[i][j] = max(up,left) 
    elif np.sign(up)+np.sign(left)==2:
     M[i][j] = uf_union(up,left)

 new_labels = np.zeros(N_labels)
 p=0

 for i in range(0,Nr):
  for j in range(0,Nc): 
   if(M[i][j]):
    p = uf_find(M[i][j])
    if(new_labels[p]==0):
     new_labels[0]+=1
     new_labels[p]=new_labels[0]

    M[i][j] = new_labels[p]
 
   
 total_clusters = new_labels[0]  
   
 return total_clusters

#########################################################################
# remember to change all the file directories
#%%
num_image = 120

for k in range(0, num_image):	# read this many frames
     if k < 9:
         fname = 'data4/Image#000'+str(k+1)+'.jpg'
     elif k >=10 and k < 99:
         fname = 'data4/Image#00'+str(k+1)+'.jpg'
     elif k>=100:
         fname = 'data4/Image#0'+str(k+1)+'.jpg'

     img = mpimg.imread(fname)     # the jpg image reads directly gives you intensity data (unlike what was discussed before, when you read certain formats, it gives you a matrix where each element of the matrix is a RGB vector, but we only want a "regular" matrix where each element is one number).

#shows the RGB slice: now you can see the most intensity of pixels using a color map
# imgplot = plt.imshow(img)
# plt.show()

     B = np.zeros(img.shape)
     C = np.zeros(img.shape)

     for i in range(100,img.shape[0]): # I started from 100 because the top left corner of the images has some text labels which I want to remove
                   for j in range(0,img.shape[1]):
 
                       if(img[i][j] > 70): 
                           B[i][j] = 1
                           C[i][j] = 1

     N_labels = 0
     labels = np.zeros(int(img.shape[0]*img.shape[1]/2))

# to call the HK algorithm you just need to give it the array, and the number of pixels in each direction
     num_clusters = hoshen_kopelman(C,img.shape[0],img.shape[1])
# print("total number of clusters identified is ", num_clusters)

     print('saving image '+str(k+1))

     save_name = 'thresholded_Image'+str(k+1)+'.jpg'
     mpimg.imsave(save_name, B) 

     save_name = 'labeled_Image'+str(k+1)+'.jpg'
     mpimg.imsave(save_name, C) 

     save_name = 'labeled_Image'+str(k+1)+'.dat'
     np.savetxt(save_name, C, fmt='%d')

#%%
for k in range(0,num_image):
    matrix = np.loadtxt('labeled_Image'+str(k+1)+'.dat')
    my_labels = np.unique(matrix)
    
    for i in my_labels:
      if int(i) !=0:
       rg_2 = 0
       label_locations = np.transpose(np.array(np.where(matrix==i)))
       cm_y = np.mean(0.1155*label_locations[:,0])
       cm_x = np.mean(0.1155*label_locations[:,1])
       r_cm = np.sqrt(cm_y**2 + cm_x**2)
       
       for j in range(len(label_locations)):
        x = 0.1155*label_locations[j][1]
        y = 0.1155*label_locations[j][0]
        rg_2 += (1.0/len(label_locations)) * ((x - cm_x)**2 + (y-cm_y)**2)
       print("cm for label ", i, ": x=%3.2f, y=%3.2f" %(cm_x,cm_y), "micrometers from top left origin", "rg=%3.2f" %np.sqrt(rg_2))
