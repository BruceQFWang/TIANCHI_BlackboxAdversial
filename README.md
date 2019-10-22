# TIANCHI_BlackboxAdversial




## Environment & Run
- python3
- pytorch>=1.0
- pillow>=5.0
- dlib ver.19.17   only support python3.5    with [shape_predictor_68_face_landmarks.dat_百度云盘提取码：4qjg](https://pan.baidu.com/s/1LMhhW2tXa8a1m2dx8-mCzQ&shfl=shareset) or [shape_predictor_68_face_landmarks.dat_Google drive](https://drive.google.com/open?id=1iMXiyvu3nYcNumtUHifVauU3-P_I_ssV)
- scikit-image>=0.14
- [models_百度云盘提取码：u46u](https://pan.baidu.com/s/1USe0e12jyeVj49AELL7KLw&shfl=shareset) or [models_Google drive](https://drive.google.com/open?id=1KrBN9-vlpmcbX5N-vc0QtKVsXuxF0jXd)

Download and unzip models
```bash
$ python target_iteration.py
```
If you only add noise to the face area, you need to leverage dlib to crop the face, which will be elaborated later.

## Methods
### Ensemble models
First of all, this problem is based on black-box face attack. We integrate the common model structure, including IR50, IR101, IR152 (model depth is different), and the code called by the model is in the file model_irse.py. The specific algorithm flow chart is shown in Figure 1. Considering that the online evaluation system may determine the category of the image by similarity, we employ the target attack. Cal_likehood.py in the code file calculates the similarity between the images through multi-model integration. We select the second similar image as the target image to attack. At the same time, our loss function has three parts, the classic distance calculation loss such as L2, cos loss. TV loss is to maintain the smoothness of the image, which will be elaborated later. The resulting noise will be convolved by gauss kernel and finally superimposed on the original image. The above process is iterated until the current picture is terminated with its own matrix similarity of more than 0.25.

In addition, our model still adopts multi-process multi-graphics acceleration. We utilize two 1080Ti multi-processing calculations, and it takes less than one hour to generate 712 samples.

### TV loss
In the process of noise cancelling, the large noise on the image may have a very large impact on the result. At this time, we need to add some regularizaiton to the model of the optimization problem to restrain the image smooth. TV loss is A commonly used regularizaiton in the CV. The integration of the continuous domain becomes the summation in the discrete region of the pixel. The specific calculation process is as follows:
$$ tvloss= ∑_{i,j}((x_{i,j-1}-x_{i,j} )^2+(x_{i+1,j}-x_{i,j} )^2 )^{β/2} $$

### Gaussian filtering
Gaussian filtering combines image frequency domain processing with time domain processing under the image processing concept. As a low-pass filter, it can filter low-frequency energy (such as noise) to smooth the image.

Gaussian filtering is performed on the generated interference noise, so that the generated noise of each pixel has correlation with surrounding pixels, which reduces the difference between the interference noise generated by different models (because different models have similar classification boundaries), effectively improving fight against the success rate of sample attacks. At the same time, considering that the online test may have a defense mechanism such as Gaussian filtering, adding Gaussian filtering when the algorithm generates noise can also invalidate the defense mechanism to improve the sample attack rate. This can be done by convolution using a Gaussian kernel function. The Gaussian kernel is as follows:
$$G(x,y)=1/{(2πσ^2)} e^{{-(x^2+y^2)}/2σ^2} $$

### Noise restriction region
The existing neural network model is sensitive to important parts when training face data. In the "Face Attention Maps Visualization.ipynb" code, we try to generate an attention map on the image, and find that the face area color is more prominent.
 ![image](https://github.com/BruceQFWang/TIANCHI_BlackboxAdversial/blob/master/assets/attention%20map%20init.png)  ![image](https://github.com/BruceQFWang/TIANCHI_BlackboxAdversial/blob/master/assets/attention%20map%20final.png) 
 
 Therefore, the noise we add is only for the facial features. The specific implementation process, we use dlib to calibrate the 68 landmarks of the face, select 17 points to form a non-mask area, and finally we will save the generated image to the mask file, for a few pictures that cannot be used to calibrate the mapmark with dlib , we manually frame the face range.
 ![image](https://github.com/BruceQFWang/TIANCHI_BlackboxAdversial/blob/master/assets/dlib%2068%20face%20landmarks.png) 
 
 

### momentum trick
By integrating the momentum term into the iterative process of the attack, adding the momentum term stabilizes the update direction and leaves the poor local maximum during the iteration, resulting in more migrating adversarial samples. In order to further improve the success rate of black box attacks, we apply the momentum iteration algorithm to the integration. Experiments show that the black box attack is better after adding the momentum term. The formula for the calculation is as follows:
$$ g_{n+1}= μ*g_n+(∇_x L(X_n^{adv},y^{true};θ))/{||∇_x L(X_n^{adv},y^{true};θ)||_1 } $$

$$X_{n+1}^{adv}=Clip_X^ϵ (X_n^{adv}+α*sign(g_{n+1}) )  $$

### input diversity
When training the lfw dataset, in addition to directly cropping the face portion of 112*112, we also employ a random padding similar to data augmentation, random resizing operation, to create a more hard and diverse input mode.
The algorithm computation process is as follows:

$$X_{n+1}^{adv}=Clip_X^ϵ ( X_n^{adv}+α*sign(∇_x L(T(X_n^{adv};p),y^{true};θ)) )$$


## Reference

