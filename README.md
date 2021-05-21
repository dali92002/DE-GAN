# DE-GAN: A Conditional Generative Adversarial Network for Document Enhancement
## Description
This is an implementation for the paper [DE-GAN: A Conditional Generative Adversarial Network for Document Enhancement](https://ieeexplore.ieee.org/document/9187695)<br>
DE-GAN is a conditional generative adversarial network designed to enhance the document quality before the recognition process. It could be used for document cleaning, binarization, deblurring and watermark removal. The weights are available to test the enhancement. 
## License
This work is only allowed for academic research use. For commercial use, please contact the author.
## Requirements
- install the requirements.txt
## Download

- Clone this repo:
```bash
git clone https://github.com/dali92002/DE-GAN
cd DE-GAN
```
- Then, download the trained weghts to directly use the model for document enhancement, it is important to save these weights in the subfolder named weights, in the DE-GAN folder. The link to download the weights is : https://drive.google.com/file/d/1J_t-TzR2rxp94SzfPoeuJniSFLfY3HM-/view?usp=sharing
## Using DE-GAN
### Document binarization
- To binarize an image use the followng command: 
```bash
python enhance.py binarize ./image_to_binarize ./directory_to_binarized_image
```
image:<br /><br />
![alt text](https://github.com/dali92002/DE-GAN/blob/master/images/2.bmp?raw=true)<br /><br />
binarized image:<br /><br />
![alt text](https://github.com/dali92002/DE-GAN/blob/master/images/2cleaned.bmp?raw=true)<br /><br />
### Document deblurring
- To deblur an image use the followng command: 
```bash
python enhance.py deblur ./image_to_deblur ./directory_to_deblurred_image
```

blurred image:<br /><br />
![alt text](https://github.com/dali92002/DE-GAN/blob/master/images/4014.png?raw=true)<br /><br />
enhanced image:<br /><br />
![alt text](https://github.com/dali92002/DE-GAN/blob/master/images/4014cleaned.png?raw=true)<br /><br />
### Watermark removal
- To remove a watermark from  an image use the followng command: 
```bash
python enhance.py unwatermark ./image_to_unwatermark ./directory_to_unwatermarked_image
```
watermarked image:<br /><br />
![alt text](https://github.com/dali92002/DE-GAN/blob/master/images/960.png?raw=true)<br /><br />
clean image:<br /><br />
![alt text](https://github.com/dali92002/DE-GAN/blob/master/images/960cleaned.png?raw=true)<br /><br />
### Document cleaning
- Will be added: 
degraded image:<br /><br />
![alt text](https://github.com/dali92002/DE-GAN/blob/master/images/1.png?raw=true)<br /><br />
cleaned image:<br /><br />
![alt text](https://github.com/dali92002/DE-GAN/blob/master/images/1cleaned.png?raw=true)<br /><br />
## Training with your own data
- To train with your own data, place your degraded images in the folder "images/A/" and the corresponding ground-truth in the folder "images/B/". It is necessary that each degraded image and its corresponding gt are having the same name (could have different extentions), also, the number images  should be the same in both folders.
- Command to train:
```bash
python train.py 
```
- Specifying the batch size and the number of epochs could be done inside the code.
## Citation
- If this work was useful for you, please cite it as: 
```
@ARTICLE{Souibgui2020,
  author={Mohamed Ali Souibgui  and Yousri Kessentini},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={DE-GAN: A Conditional Generative Adversarial Network for Document Enhancement}, 
  year={2020},
  doi={10.1109/TPAMI.2020.3022406}}
```
