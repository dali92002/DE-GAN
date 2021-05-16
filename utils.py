import numpy as np
import math




def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if (mse == 0):
        return (100)
    PIXEL_MAX = 1.0
    return (20 * math.log10(PIXEL_MAX / math.sqrt(mse)))

def split2(dataset,size,h,w):
    newdataset=[]
    nsize1=256
    nsize2=256
    for i in range (size):
        im=dataset[i]
        for ii in range(0,h,nsize1): #2048
            for iii in range(0,w,nsize2): #1536
                newdataset.append(im[ii:ii+nsize1,iii:iii+nsize2,:])
    
    return np.array(newdataset) 
def merge_image2(splitted_images, h,w):
    image=np.zeros(((h,w,1)))
    nsize1=256
    nsize2=256
    ind =0
    for ii in range(0,h,nsize1):
        for iii in range(0,w,nsize2):
            image[ii:ii+nsize1,iii:iii+nsize2,:]=splitted_images[ind]
            ind=ind+1
    return np.array(image)  



def getPatches(watermarked_image,clean_image,mystride):
    watermarked_patches=[]
    clean_patches=[]
    
    
    h =  ((watermarked_image.shape [0] // 256) +1)*256 
    w =  ((watermarked_image.shape [1] // 256 ) +1)*256
    image_padding=np.ones((h,w))
    image_padding[:watermarked_image.shape[0],:watermarked_image.shape[1]]=watermarked_image
    
    for j in range (0,h-256,mystride):  #128 not 64
        for k in range (0,w-256,mystride):
            watermarked_patches.append(image_padding[j:j+256,k:k+256])
    
    
    h =  ((clean_image.shape [0] // 256) +1)*256 
    w =  ((clean_image.shape [1] // 256 ) +1)*256
    image_padding=np.ones((h,w))*255
    image_padding[:clean_image.shape[0],:clean_image.shape[1]]=clean_image

    for j in range (0,h-256,mystride):    #128 not 64
        for k in range (0,w-256,mystride):
            clean_patches.append(image_padding[j:j+256,k:k+256]/255)  
            
    return np.array(watermarked_patches),np.array(clean_patches)