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

In addition, our model still uses multi-process multi-graphics acceleration. We utilize two 1080Ti multi-processing calculations, and it takes less than one hour to generate 712 samples.


## input diversity

$$
X_{n+1}^{adv}=Clip_X^ϵ {{ X_n^{adv}+α*sign(∇_x L(T(X_n^{adv};p),y^{true};θ)) }}
$$
