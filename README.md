# ImageAI

[![Build Status](https://travis-ci.org/Borda/ImageAI.svg?branch=master)](https://travis-ci.org/Borda/ImageAI)
[![codecov](https://codecov.io/gh/Borda/ImageAI/branch/master/graph/badge.svg)](https://codecov.io/gh/Borda/ImageAI)
[![CodeFactor](https://www.codefactor.io/repository/github/borda/imageai/badge)](https://www.codefactor.io/repository/github/borda/imageai)

A python library built to empower developers to build applications and systems with self-contained Deep Learning and Computer Vision capabilities using simple
 and few lines of code.

<img src="logo1.png" style="width: 500px; height: auto; margin-left: 50px; " />

An **AI Commons** project <a href="https://aicommons.science" >https://aicommons.science </a>
Developed and Maintained by [Moses Olafenwa](https://twitter.com/OlafenwaMoses) and [John Olafenwa](https://twitter.com/johnolafenwa), brothers, creators of [TorchFusion](https://github.com/johnolafenwa/TorchFusion)
and Authors of [Introduction to Deep Computer Vision](https://john.aicommons.science/deepvision)

---

Built with simplicity in mind, **ImageAI** supports a list of state-of-the-art Machine Learning algorithms for image prediction, custom image prediction, object detection, video detection, video object tracking and image predictions trainings. **ImageAI** currently supports image prediction and training using 4 different Machine Learning algorithms trained on the ImageNet-1000 dataset. **ImageAI** also supports object detection, video detection and object tracking  using RetinaNet, YOLOv3 and TinyYOLOv3 trained on COCO dataset.
    
Eventually, **ImageAI** will provide support for a wider and more specialized aspects of Computer Vision including and not limited to image recognition in special environments and special fields.


**New Release : ImageAI 2.0.2**

What's new:

- Option to state image size during custom image prediction model trainings
- Object Detection and Video Object detection now returns bounding box coordinates **('box points')** (x1,y1,x2, y2) for each object detected in addition to object's 'name' and 'percentage probability' <br>
- Options to hide 'percentage probability' and/or object 'name' from being shown in detected image or video
- Support for video object detection on video live stream from device camera, connected camera and IP camera <br>
- Support for **YOLOv3** and **TinyYOLOv3** for all object detection and video object detection tasks.
- Video object detection for all input types (video file and camera) now allows defining custom functions to execute after each frame, each second and each minute of the video is detected and processed. Also include option to specify custom function at once video is fully detected and processed <br>
- For each custom function specified, **ImageAI** returns the **frame**/**seconds**/**minute**/**full video analysis** of the detections that include the objects' details ( **name** , **percentage** **probability**, **box_points**), number of instance of each unique object detected (counts) and overall average count of the number of instance of each unique object detected in the case of **second** / **minute** / **full video analysis**<br>
- Options to return detected frame at every frame, second or minute processed as a **Numpy array**.


### TABLE OF CONTENTS
* <a href="#dependencies" >&#9635 Dependencies</a>
* <a href="#installation" >&#9635 Installation</a>
* <a href="#prediction" >&#9635 Image Prediction</a>
* <a href="#detection" >&#9635 Object Detection</a>
* <a href="#videodetection" >&#9635 Video Object Detection, Tracking & Analysis</a>
* <a href="#customtraining" >&#9635 Custom Model Training</a>
* <a href="#customprediction" >&#9635 Custom Image Prediction</a>
* <a href="#documentation" >&#9635 Documentation</a>
* <a href="#sample" >&#9635 Projects Built on ImageAI</a>
* <a href="#recommendation" >&#9635 AI Practice Recommendations</a>
* <a href="#contact" >&#9635 Contact Developers</a>
* <a href="#contributors" >&#9635 Contributors</a>
* <a href="#ref" >&#9635 References</a>


<br><br>

## Dependencies
<div id="dependencies"></div>

To use **ImageAI** in your application developments, you must have installed the following dependencies before you install **ImageAI** : 

* Python 3.5.1 (and later versions)
* listed libraries and minimal version
    ```bash
    pip install -r requirements.txt
    ``` 
 
## Installation
<div id="installation"></div>

To install ImageAI, run the python installation instruction below in the command line: 
* compiled wheel
    ```bash
    pip3 install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.2/imageai-2.0.2-py3-none-any.whl
    ```
* using source 
    ```bash
    pip3 install git+https://github.com/OlafenwaMoses/ImageAI.git
    ```
* clone and install locally
    ```bash
    git clone https://github.com/OlafenwaMoses/ImageAI.git
    cd ImageAI
    python3 setup.py install
    
    ``` 


## Image Prediction
<div id="prediction"></div>

![Car](sample_images/1.jpg) 

```
convertible : 52.459555864334106
sports_car : 37.61284649372101
pickup : 3.1751200556755066
car_wheel : 1.817505806684494
minivan : 1.7487050965428352
```

**ImageAI** provides 4 different algorithms and model types to perform image prediction, trained on the ImageNet-1000 dataset.
The 4 algorithms provided for image prediction include **SqueezeNet**, **ResNet**, **InceptionV3** and **DenseNet**. 

Click the link below to see the full sample codes, explanations and best practices guide.

<a href="imageai/Prediction/" ><button style="font-size: 20px; color: white; background-color: steelblue; height: 50px; border-radius: 10px; " > >>> Tutorial & Guide </button></a>


## Object Detection
<div id="detection"></div>

![Input Image](sample_images/image2.jpg)
![Output Image](sample_images/image2new.jpg)

```
person : 91.946941614151
--------------------------------
person : 73.61021637916565
--------------------------------
laptop : 90.24320840835571
--------------------------------
laptop : 73.6881673336029
--------------------------------
laptop : 95.16398310661316
--------------------------------
person : 87.10319399833679
--------------------------------
```

**ImageAI** provides very convenient and powerful methods to perform object detection on images and extract each object from the image. The object detection class provides support for RetinaNet, YOLOv3 and TinyYOLOv3, with options to adjust for state of the art performance or real time processing.

Click the link below to see the full sample codes, explanations and best practices guide.


<a href="imageai/Detection/" ><button style="font-size: 20px; color: white; background-color: steelblue; height: 50px; border-radius: 10px; " > >>> Tutorial & Guide</button></a>

## Video Object Detection and Tracking
<div id="videodetection"></div>


**Video Object Detection & Analysis**
_Below is a snapshot of a video with objects detected._
![](sample_images/video1.jpg) style="width: 500px; height: auto; margin-left: 50px; " /> <br>

**Video Custom Object Detection (Object Tracking)**
_Below is a snapshot of a video with only person, bicycle and motorcyle detected._
![](sample_images/video2.jpg)

**Video Analysis Visualization**
_Below is a visualization of video analysis returned by **ImageAI** into a 'per_second' function._
![](sample_images/video_analysis_visualization.jpg)

**ImageAI** provides very convenient and powerful methods to perform object detection in videos and track specific object(s). The video object detection class provided only supports the current state-of-the-art RetinaNet, but with options to adjust for state of the art performance or real time processing.
Click the link to see the full videos, sample codes, explanations and best practices guide.


<a href="imageai/Detection/VIDEO.md" ><button style="font-size: 20px; color: white; background-color: steelblue; height: 50px; border-radius: 10px; " > >>> Tutorial & Guide </button></a>


## Custom Model Training
<div id="customtraining"></div>

_A sample from the IdenProf Dataset used to train a Model for predicting professionals._
![](sample_images/idenprof.jpg)

**ImageAI** provides classes and methods for you to train a new model that can be used to perform prediction on your own custom objects.
You can train your custom models using SqueezeNet, ResNet50, InceptionV3 and DenseNet in  ** 5 ** lines of code.
Click the link below to see the guide to preparing training images, sample training codes, explanations and best practices.

<a href="imageai/Prediction/CUSTOMTRAINING.md" ><button style="font-size: 20px; color: white; background-color: steelblue; height: 50px; border-radius: 10px; " > >>> Tutorials & Documentation </button></a>


## Custom Image Prediction
<div id="customprediction"></div>

_Prediction from a sample model trained on IdenProf, for predicting professionals_
![](sample_images/4.jpg)
```
mechanic : 76.82620286941528
chef : 10.106072574853897
waiter : 4.036874696612358
police : 2.6663416996598244
pilot : 2.239348366856575</pre>
```

**ImageAI** provides classes and methods for you to run image prediction your own custom objects using your own model trained with **ImageAI** Model Training class.
You can use your custom models trained with SqueezeNet, ResNet50, InceptionV3 and DenseNet and the JSON file containing the mapping of the custom object names.
Click the link below to see the guide to sample training codes, explanations, and best practices guide.
<br>


<a href="imageai/Prediction/CUSTOMPREDICTION.md" ><button style="font-size: 20px; color: white; background-color: steelblue; height: 50px; border-radius: 10px; " > >>> Tutorials & Documentation </button></a>


## Documentation
<div id="documentation"></div>

We have provided full documentation for all **ImageAI** classes and functions in 2 major languages. Find links below: <br>

* Documentation - **English Version**  [https://imageai.readthedocs.io](https://imageai.readthedocs.io)
* Documentation - **Chinese Version**  [https://imageai-cn.readthedocs.io](https://imageai-cn.readthedocs.io)
* Documentation - **French Version**  [https://imageai-fr.readthedocs.io](https://imageai-fr.readthedocs.io)


## Real-Time and High Performance Implementation
<div id="performance"></div>

**ImageAI** provides abstracted and convenient implementations of state-of-the-art Computer Vision technologies. All of **ImageAI** implementations and code can work on any computer system with moderate CPU capacity. However, the speed of processing for operations like image prediction, object detection and others on CPU is slow and not suitable for real-time applications. To perform real-time Computer Vision operations with high performance, you need to use GPU enabled technologies.

**ImageAI** uses the Tensorflow backbone for it's Computer Vision operations. Tensorflow supports both CPUs and GPUs ( Specifically NVIDIA GPUs.  You can get one for your PC or get a PC that has one) for machine learning and artificial intelligence algorithms' implementations. To use Tensorflow that supports the use of GPUs, follow the link below :

* [FOR WINDOWS](https://www.tensorflow.org/install/install_windows)
* [FOR macOS](https://www.tensorflow.org/install/install_mac)
* [FOR UBUNTU](https://www.tensorflow.org/install/install_linux)

## Sample Applications
<div id="sample"></div>

As a demonstration of  what you can do with ImageAI, we have 
 built a complete AI powered Photo gallery for Windows called **IntelliP** ,  using **ImageAI** and UI framework **Kivy**. Follow this [link](https://github.com/OlafenwaMoses/IntelliP) to download page of the application and its source code.

 We also welcome submissions of applications and systems built by you and powered by ImageAI for listings here. Should you want your ImageAI powered developments listed here, you can reach to us via our <a href="#contact" >Contacts</a> below.


## AI Practice Recommendations
<div id="recommendation"></div>

 For anyone interested in building AI systems and using them for business, economic,  social and research purposes, it is critical that the person knows the likely positive, negative and unprecedented impacts the use of such technologies will have. They must also be aware of approaches and practices recommended by experienced industry experts to ensure every use of AI brings overall benefit to mankind. We therefore recommend to everyone that wishes to use ImageAI and other AI tools and resources to read Microsoft's January 2018 publication on AI titled "The Future Computed : Artificial Intelligence and its role in society ".
Kindly follow the link below to download the publication.

[Futurec computed artificial intelligence role society](https://blogs.microsoft.com/blog/2018/01/17/future-computed-artificial-intelligence-role-society)

## Contact Developers
<div id="contact"></div>

- **Moses Olafenwa**
    * _Email:_ guymodscientist@gmail.com
    * _Website:_ [https://moses.aicommons.science](https://moses.aicommons.science)
    * _Twitter:_ [@OlafenwaMoses](https://twitter.com/OlafenwaMoses)
    * _Medium:_ [@guymodscientist](https://medium.com/@guymodscientist)
    * _Facebook:_ [moses.olafenwa](https://facebook.com/moses.olafenwa)
- **John Olafenwa**
    * _Email:_ johnolafenwa@gmail.com
    * _Website:_ [https://john.aicommons.science](https://john.aicommons.science)
    * _Twitter:_ [@johnolafenwa](https://twitter.com/johnolafenwa)
    * _Medium:_ [@johnolafenwa]("https://medium.com/@johnolafenwa)
    * _Facebook:_ [olafenwajohn](https://facebook.com/olafenwajohn)


### Contributors
<div id="contact"></div>

We are inviting anyone who wishes to contribute to the **ImageAI** project to reach to us. We primarily need contributions in translating the documentation of the project's code to major languages that includes but not limited to French, Spanish, Portuguese, Arabian and more. We want every developer and researcher around the world to benefit from this project irrespective of their native languages. <br>

We give special thanks to **[Kang vcar](https://github.com/kangvcar/)** for his incredible and excellent work in translating **ImageAI**'s documentation to the Chinese language. Find below the contact details of those who have contributed immensely to the **ImageAI** project.

- **Kang vcar**
    * _Email:_ kangvcar@mail.com
    * _Website:_ [http://www.kangvcar.com](http://www.kangvcar.com)
    * _Twitter:_ [@kangvcar](https://twitter.com/kangvcar)


## References
 <div id="ref"></div>

 1. Somshubra Majumdar, [DenseNet Implementation of the paper, Densely Connected Convolutional Networks in Keras](https://github.com/titu1994/DenseNet)
 2. Broad Institute of MIT and Harvard, [Keras package for deep residual networks](https://github.com/broadinstitute/keras-resnet)
 3. Fizyr, [Keras implementation of RetinaNet object detection](https://github.com/fizyr/keras-retinanet)
 4. Francois Chollet, [Keras code and weights files for popular deeplearning models](https://github.com/fchollet/deep-learning-models)
 5. Forrest N. et al, [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)
 6. Kaiming H. et al, [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
 7. Szegedy. et al, [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)
 8. Gao. et al, [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
 9. Tsung-Yi. et al, [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
 10. O Russakovsky et al, [ImageNet Large Scale Visual Recognition Challenge](https://arxiv.org/abs/1409.0575)
 11. TY Lin et al, [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312)
 12. Moses & John Olafenwa, [A collection of images of identifiable professionals.](https://github.com/OlafenwaMoses/IdenProf)
 13. Joseph Redmon and Ali Farhadi, [YOLOv3: An Incremental Improvement.](https://arxiv.org/abs/1804.02767)
