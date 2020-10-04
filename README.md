# DE-GAN: A Conditional Generative Adversarial Network for Document Enhancement
## Description
DE-GAN is a conditional generative adversarial network designed to enhance the document quality before the recognition process. It could be used for document cleaning, binarization, deblurring and watermark removal. 
## Requirements
will be added
## Installation

- Clone this repo:
```bash
git clone https://github.com/dali92002/DE-GAN
cd DE-GAN
```
- Then, download the trained weghts to directly use the model for document enhancement, it is important to save these weights in the subfolder named weight, in the DE-GAN folder. The link of the weightsis : https://drive.google.com/file/d/1J_t-TzR2rxp94SzfPoeuJniSFLfY3HM-/view?usp=sharing
## Using DE-GAN
### Document binarization
- To binarize an image use the followng command: 
```bash
python enhance.py binarize ./image_to_binarize ./directory_to_binarized_imge
```

image:<br /><br />
![alt text](https://github.com/dali92002/DE-GAN/blob/master/images/2.bmp?raw=true)<br /><br />
binarized image:<br /><br />
![alt text](https://github.com/dali92002/DE-GAN/blob/master/images/2cleaned.bmp?raw=true)<br /><br />
### Document deblurring
blurred image:<br /><br />
![alt text](https://github.com/dali92002/DE-GAN/blob/master/images/4014.png?raw=true)<br /><br />
enhanced image:<br /><br />
![alt text](https://github.com/dali92002/DE-GAN/blob/master/images/4014cleaned.png?raw=true)<br /><br />
### Watermark removal
watermarked image:<br /><br />
![alt text](https://github.com/dali92002/DE-GAN/blob/master/images/960.png?raw=true)<br /><br />
clean image:<br /><br />
![alt text](https://github.com/dali92002/DE-GAN/blob/master/images/960cleaned.png?raw=true)<br /><br />
### Document cleaning
degraded image:<br /><br />
![alt text](https://github.com/dali92002/DE-GAN/blob/master/images/1.png?raw=true)<br /><br />
cleaned image:<br /><br />
![alt text](https://github.com/dali92002/DE-GAN/blob/master/images/1cleaned.png?raw=true)<br /><br />
## Training with your own data
will be added
