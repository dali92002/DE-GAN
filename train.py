from tensorflow import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras import metrics
import scipy.misc
import os
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import math
from PIL import Image
from tqdm import tqdm
import random
import os.path
import imageio



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



input_size = (256,256,1)


def unet(pretrained_weights = None,input_size = input_size):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))

    merge6 = concatenate ([drop4,up6])
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate ([conv3,up7])

    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate ([conv2,up8])

    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))

    merge9 = concatenate ([conv1,up9])  
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)
    
    return model

def get_optimizer():
    return Adam(lr=1e-4)

def build_discriminator(input_size = input_size):

    def d_layer(layer_input, filters, f_size=4, bn=True):

        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    img_A = Input(input_size)
    img_B = Input(input_size)

    df=64

    combined_imgs = Concatenate(axis=-1)([img_A, img_B])

    d1 = d_layer(combined_imgs, df, bn=False)
    d2 = d_layer(d1, df*2)
    d3 = d_layer(d2, df*4)
    d4 = d_layer(d3, df*4)
    
    
    
    validity = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='sigmoid')(d4)

    discriminator = Model([img_A, img_B], validity)
    discriminator.compile(loss='mse', optimizer=Adam(lr=1e-4), metrics = ['accuracy'])
    return discriminator


def train_gan(generator,discriminator, ep_start=1, epochs=1, batch_size=128):
    
    list_deg_images= os.listdir('data/A/')
    list_clean_images= os.listdir('data/A/')
    
    list_deg_images.sort()
    list_clean_images.sort()

    adam = get_optimizer()
    gan = get_gan_network(discriminator, generator, adam)
    
    for e in range(ep_start, epochs+1):
        print ('\n Epoch:' ,e)
        
        for im in tqdm(range (len(list_deg_images))):
            


            deg_image_path = ('data/A/'+list_deg_images[im])
            deg_image = Image.open(deg_image_path)# /255.0
            deg_image = deg_image.convert('L')
            deg_image.save('curr_deg_image.png')

            deg_image = plt.imread('curr_deg_image.png')

            clean_image_path = ('data/B/'+list_clean_images[im])
            clean_image = Image.open(clean_image_path)# /255.0
            clean_image = clean_image.convert('L')
            clean_image.save('curr_clean_image.png')

            clean_image = plt.imread('curr_clean_image.png')#[:,:,0]


            wat_batch, gt_batch = getPatches(deg_image,clean_image,mystride=128+64)

            batch_count = wat_batch.shape[0] // batch_size




            for b in (range(batch_count)):
                seed= range(b*batch_size, (b*batch_size) + batch_size)
                b_wat_batch = wat_batch[seed].reshape(batch_size,256,256,1)
                b_gt_batch = gt_batch[seed].reshape(batch_size,256,256,1)

                generated_images = generator.predict(b_wat_batch)


                valid = np.ones((b_gt_batch.shape[0],) + (16, 16, 1))
                fake = np.zeros((b_gt_batch.shape[0],) + (16, 16, 1))
    
                discriminator.trainable = True          
                discriminator.train_on_batch([b_gt_batch, b_wat_batch], valid)
                discriminator.train_on_batch([generated_images, b_wat_batch], fake)


                discriminator.trainable = False
                gan.train_on_batch([b_wat_batch], [valid, b_gt_batch])

        # if (e == 1 or e % 2 == 0):
        #     evaluate(generator,discriminator,e)
    # return generator,discriminator

def get_gan_network(discriminator, generator, optimizer,input_size = input_size):
    discriminator.trainable = False
    
    gan_input2 = Input(input_size)
    
    x = generator(gan_input2)
    valid = discriminator([x,gan_input2])
    gan = Model(inputs=[gan_input2], outputs=[valid,x])
    gan.compile(loss=['mse','binary_crossentropy'],loss_weights=[1, 100], optimizer=optimizer,metrics = ['accuracy'])
    return gan



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

def predic(generator, epoch):
    if not os.path.exists('Results/epoch'+str(epoch)):
        os.makedirs('Results/epoch'+str(epoch))
    for i in range(0,31):
        watermarked_image_path = ('CLEAN/VALIDATION/DATA/'+ str(i+1) + '.png')
        test_image = plt.imread(watermarked_image_path)
        
        h =  ((test_image.shape [0] // 256) +1)*256 
        w =  ((test_image.shape [1] // 256 ) +1)*256
        
        test_padding=np.zeros((h,w))+1
        test_padding[:test_image.shape[0],:test_image.shape[1]]=test_image
        
        test_image_p=split2(test_padding.reshape(1,h,w,1),1,h,w)
        predicted_list=[]
        for l in range(test_image_p.shape[0]):
            predicted_list.append(generator.predict(test_image_p[l].reshape(1,256,256,1)))
        
        predicted_image = np.array(predicted_list)#.reshape()
        predicted_image=merge_image2(predicted_image,h,w)
        
        predicted_image=predicted_image[:test_image.shape[0],:test_image.shape[1]]
        predicted_image=predicted_image.reshape(predicted_image.shape[0],predicted_image.shape[1])
        predicted_image = (predicted_image[:,:])*255
        
        predicted_image =predicted_image.astype(np.uint8)
        imageio.imwrite('Results/epoch'+str(epoch)+'/predicted'+str(i+1)+'.png', predicted_image)




### if you want to evaluate each epoch:


# def  evaluate(generator,discriminator,epoch):
#     predic(generator,epoch)
#     avg_psnr=0
#     qo=0

#     for i in range (0,31):
        
#         test_image= plt.imread('CLEAN/VALIDATION/GT/'+ str(i+1) + '.png')

#         predicted_image= plt.imread('Results/epoch'+str(epoch)+'/predicted'+ str(i+1) + '.png')
#         avg_psnr= avg_psnr + psnr(test_image,predicted_image)
#         qo=qo+1
#     avg_psnr=avg_psnr/qo
#     print('psnr= ',avg_psnr)
#     if not os.path.exists('Results/epoch'+str(epoch)+'/weights'):
#         os.makedirs('Results/epoch'+str(epoch)+'/weights')
#     discriminator.save_weights("Results/epoch"+str(epoch)+"/weights/discriminator_weights.h5")
#     generator.save_weights("Results/epoch"+str(epoch)+"/weights/generator_weights.h5")


##################################

epo = 1

generator = unet()
discriminator = build_discriminator()


### to  load pretrained models  ################"" 
# epo = 41

# generator.load_weights("Results/epoch"+str(epo-1)+"/weights/generator_weights.h5")
# discriminator.load_weights("Results/epoch"+str(epo-1)+"/weights/discriminator_weights.h5")


###############################################

train_gan(generator,discriminator, ep_start =epo, epochs=80, batch_size=4)