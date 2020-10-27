#!/usr/bin/env python
# coding: utf-8

# In[90]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import pylab
# pylab.rcParams['figure.figsize'] = (10, 5)
import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 150


# In[2]:


get_ipython().run_line_magic('matplotlib', 'qt')

# Better to avoid inline mode, as the particles are quite small most of the times.


# #### Step 1 : Get the scale, if you don't already know it.

# In[43]:


def get_tem_scale(img_path,y1=None,y2=None,x1=None,x2=None, threshold_type = cv2.THRESH_BINARY,lower_thresh = 220):
    
    '''
    This function takes in an image and tries to find the scale by measuring the line segment usually given at the bottom left 
    if the TEM micrograph. It approximates a rectangle for this and reports the width of rectangle as the scale.
    
    Parameters:
    img_path :  Path of the TEM image.
    y1 = start index for cropping along vertical axis, must be an integer.
    y2 = end index for cropping along vertical axis, must be an integer.
    x1 = start index for cropping along horizontal axis, must be an integer.
    x2 = end index for cropping along horizontal axis, must be an integer.
    
    Prints width of the approximating rectangle. 
    
    '''
    
    img = cv2.imread(img_path)
    if np.any(img):
        pass
    else:
        print('Image could not be read, check path or check if image is corrupt.')
        return None
    
    if y1 == None:
        y1 = int(0.80*img.shape[0])
    if y2 == None:
        y2 = int(0.95*img.shape[0])
    if x1 == None:
        x1 = 0
    if x2 == None:
        x2 = int(0.5*img.shape[1])
    
    crop_img = img[y1:y2,0:x2]
    
    if np.any(crop_img):
        pass
    else:
        print('Cropped Image is empty, check crop dimensions.')
        
    cv2.imshow('cropped',crop_img)
    scale_gray = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
    
    # choose threshold type  as cv2.THRESH_BINARY_INV if scale region is black.
    if threshold_type == cv2.THRESH_BINARY_INV:
        lower_thresh = 0
    ret, thresh = cv2.threshold(scale_gray, lower_thresh, 255, threshold_type)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # filter noisy detection
    contours = [c for c in contours if cv2.contourArea(c) > 300]
    contours.sort(key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))
    cv2.rectangle(crop_img, cv2.boundingRect(contours[-1]), (0,255,0), 2)
    x,y,w,h = cv2.boundingRect(contours[-1])
    print('x : %f , y = %f , w = %f , h = %f '%(x,y,w,h))
    print('The width in pixels is : ',w)
    cv2.imshow('scale_marked',crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


# #### Step 2: Find all features, irrespective of size and generate contours by thresholding.

# In[4]:


def find_particles(img_path,scale=None,thresh = 45,save_images = False):
    
    '''
    This function finds particles by contouring, Image which is loade converted to grayscale.
    A gaussian Blur is then used to remove a bit of noise in the images, which helps with over-detection.
    Threshold style by default is cv2.THRESH_BINARY.
    
    Threshold by default is 45. Change this according to your data.
    
    Parameters:
    img_path :  Path of the TEM image.
    scale : Either already known or found from get_tem_scale()
    thresh :  lower limit of threshold.
    save_images : If true, the images are saved to same directory as the original images, False by default.
    
    '''
    # Read the image
    img = cv2.imread(img_path)
    if np.any(img):
        pass
    else:
        print('Image could not be read, check path or check if image is corrupt.')
        return None
    
    #img = cv2.resize(img,(0,0),fx=0.25,fy=0.25)
    
    # Gaussian Blur ot reduce noise
    img = cv2.GaussianBlur(img,(5,5),0)
    
    # convert to grascale
    gray= cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
    # thresholding, making a binary image.
    ret,thresh = cv2.threshold(gray,thresh,255,cv2.THRESH_BINARY)

    #img_thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,1151,1)
    cv2.namedWindow("Threshold image", cv2.WINDOW_NORMAL)
    cv2.imshow('Threshold image',thresh)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    img2 = img.copy()
    
    index = -1
    thickness = 1
    color = (255,0,255)
    
    # drawing contours.
    cv2.drawContours(img2,contours,index,color,thickness)
    cv2.namedWindow("contours", cv2.WINDOW_NORMAL)
    
    cv2.imshow('contours',img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # saving images
    if save_images==True:
        cv2.imwrite(img_path+'_thresh.tif',thresh)
        cv2.imwrite(img_path+'_contours.tif',img2)
    return contours


# #### Step 3: Plot the size distribution.

# In[93]:


def size_distribution_plot(img_path,contours=None,scale=None,r_min=None,r_max=None, bins=None, save_fig = False):
    
    '''
    This generates a size ditribtution plot, the sizes can be managed by r_min and r_max.
    
    Parameters : 
    img_path :  Path of the TEM image.
    contours: Obatained from find_particles()
    scale : Already known or found from get_tem_scale()
    r_min :  Minimum radius of particles to be marked. By default, it is equal to scale.
    r_max = Maximum radius of particles to be marked. By default, it is 100 times the scale.
    bins :  bins in the histogram which is plotted.
    
    Prints the mean particles size. (Note that the diameters are returned.)
    If you want to exclude all particles under 10 nm, give r_max as 5.
    Returns an array of particle sizes.
    
    '''
    
    # Change this according to your data.
    if r_min == None:
        r_min = 0.5*scale
    if r_max == None:
        r_max = 50*scale
    #######
    
    img = cv2.imread(img_path)
    
    if np.any(img):
        pass
    else:
        print('Image could not be read, check path or check if image is corrupt.')
        return None
    
    fig,ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].imshow(img)
    ax[1].imshow(img)
    sizes = []
   
    for i in range(len(contours)):
        cnt = contours[i]
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        center = (int(x),int(y))
        radius=int(radius)
        if radius > r_min*scale and radius < r_max*scale:
            sizes.append(radius)
            c=plt.Circle((x,y),radius,color='r' ,linewidth=0.3, fill=False)
            
            # commented out in case of a lot of particles, it is a very messy plot.
            #ax[1].annotate(str(i),(x,y))   
            
            ax[1].add_patch(c)
    sizes = (np.array(sizes)*2)/scale # Note that these are diameters 
    fig1,ax1 = plt.subplots()
    if bins == None:
        bins = len(set(sizes))
    
    ax1.hist(sizes,color='#98ff98', histtype='bar', ec='black', bins=bins)
    
    plt.xlabel('Diameter (nm)')
    plt.ylabel('Number of particles')
    print('Mean diameter : %0.2f'%(np.mean(sizes)))
    print('Total Particles : %g'%(len(sizes)))
    
    if save_fig == True:
        plt.savefig(img_path+'_analysis.png')
    
    plt.tight_layout()
    plt.show()
    return np.sort(sizes)


# In[ ]:




