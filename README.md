# FINGERPRINT RECOGNITION USING DEEP LEARNING 

## A. PROJECT SUMMARY

**Project Title:** Fingerprint Recognition using Deep Learning

**Team Members:** 
- Ashraful Islam
- X
- Y
- Z

- [ ] **Objectives:**
- Break out the project goal into more specific objectives
- Fingerprint Algorithm
- Biometric Identification
- Tenprint Reader

##  B. ABSTRACT 
Performance of modern automated fingerprint recognition systems is heavily influenced by accuracy of their feature extraction algorithm. Nowadays, there are more approaches to fingerprint feature extraction with acceptable results. Problems start to arise in low quality conditions where majority of the traditional methods based on analyzing texture of fingerprint cannot tackle this problem so effectively as artificial neural networks. Many papers have demonstrated uses of neural networks in fingerprint recognition, but there is a little work on using them as Level-2 feature extractors.
Our goal was to contribute to this field and develop a novel algorithm employing neural networks as extractors of discriminative Level-2 features commonly used to match fingerprints. In this work, we investigated possibilities of incorporating artificial neural networks into fingerprint recognition process, implemented and documented our own software solution for fingerprint identification based on neural networks whose impact on feature extraction accuracy and overall recognition rate was evaluated. The result of this research is a fully functional software system for fingerprint recognition that consists of fingerprint sensing module using high resolution sensor, image enhancement module responsible for image quality restoration, Level-1 and Level-2 feature extraction module based on neural network, and finally fingerprint matching module using the industry standard BOZORTH3 matching algorithm.
For purposes of evaluation we used more fingerprint databases with varying image quality, and the performance of our system was evaluated using FMR/FNMR and ROC indicators. From the obtained results, we may draw conclusions about a very positive impact of neural networks on overall recognition rate, specifically in low quality.


![Coding]!![Fingerprint](https://user-images.githubusercontent.com/82527948/114950397-812d4a00-9e85-11eb-8008-161ceca99a00.jpg)
Figure 1 shows the AI output of detecting which user is using fingerprint.


## C.  DATASET

In this project, we’ll discuss our three-phase fingerprint detector, detailing how our computer vision/deep learning pipeline will be implemented.

Level-1 pattern represents overall fingerprint ridge flow. These patterns are usually divided into 5 categories (left loop, right loop, whorl, arch and tented arch) [13]. Global ridge flow is a well-defined pattern and can be retrieved easily even when the image quality is not sufficient. After successfully resolving Level-1 pattern category the whole search space in fingerprint database is narrowed down to only specific fingerprint pattern subset what drastically reduces computation time [13]. In [5], authors extensively studied and compared performance of more state-of-the-art Level-1 classification techniques. We refer to this process as coarse fingerprint classification and it is discussed in Section 3.2 in more detail.

Level-2 features or minutiae are local ridge characteristics that make every fingerprint a unique pattern. The premise of fingeprint uniqueness has been generally accepted, but still lacks proper scientific validation. Individuality of fingerprints based on Level-2 features as well as probability of correspondence of random fingerprints is discussed in [15]. These features are used by our fingerprint recognition system presented in this paper. More Level-2 feature types are distinguished from basic shapes to very complex and rare patterns with extremely high discrimination power in fingerprint matching process (see Figure 2).
Level-2 features are mainly characterized by the spatial location in th  image,their orientation and their shape type [13]. The two Level-2 features mostly used are the ridge ending, referred to as termination, and the bifurcation, which refers to a ridge splitting into two ridges. Despite such a wide range of Level-2 patterns, only ridge terminations and bifurcations are preferred in majority of commercial or civil fingerprint identification systems thanks to simplicity of their automated extraction. Fingerprints are considered identical when a matching algorithm finds sufficient number of correspondences of Level-2 features. Probably the best known and most widely used matching algorithm is called BOZORTH3 that is a part of NIST Biometric Image Software. It uses two sets of Level-2 features to establish a similarity score between two fingerprints [21].

Level-3 features are microscopic level patterns that are almost exclusively used by forensic examiners. They consist of sweat pore locations, ridge geometric details, scars and other very small characteristics. Lately, their computer automated extraction has been seriously considered as more and more biometric system vendors begin to adopt 1000 PPI (pixels per inch) sensing resolution of fingerprint images in their recognition systems [22]. Latent fingerprints often lack a large portion of fingerprint pattern. This is the case where Level-3 features step in. They make identification possible even with limited amount of information. Patterns made of Level-2 features are highly individual and differ even between identical twins [1]. Importance and complexity of Level-2 features is the main source of inspiration that resulted into our software implementation of
complete biometric system that performs fingerprint identification. 
At early stages of development, we used multiple structural approaches for feature extraction like analyzing patterns in fingerprint skeletons or time consuming computation of local ridge curvature [18]. All of them turned out to be ineffective either in terms of time needed for computation or insufficient accuracy of recognition. To overcome this issue, we implemented a new neural networkbased algorithm for extraction of Level-2 features. Firstly, instead of exploring the entire image, our algorithm detects only critical image regions with high probability of Level-2 feature occurrence using Crossing Number method [13].

![Figure2](https://user-images.githubusercontent.com/82527948/114956387-5c3ed400-9e91-11eb-83a8-6108aaddae14.png)
![All Figure](https://user-images.githubusercontent.com/82527948/114956713-01f24300-9e92-11eb-83de-3c8ad0e535da.png)


## D.   PROJECT STRUCTURE

The following directory is our structure of our project:
- $ tree --dirsfirst --filelimit 10
- .
- ├── dataset
- │   ├── 
- │   └── 
- ├── examples
- │   ├── example_01.png
- │   ├── example_02.png
- │   └── example_03.png
- ├── fingerprint_detector
- │   ├── deploy.prototxt
- │   └── res10_300x300_ssd_iter_140000.caffemodel
- ├── detect_fingerprint_image.py
- ├── detect_fingerprint_video.py
- ├── fingerprint_detector.model
- ├── plot.png
- └── train_fingerprint_detector.py

The dataset/ directory contains the data described in fingerprint detection dataset” section.

In the next two sections, we will train our fingerprint detector.


## E   TRAINING THE FINGERPRINT DETECTION

We are now ready to train our fingerprint detector using Keras, TensorFlow, and Deep Learning.

From there, open up a terminal, and execute the following command:

- $ python train_mask_detector.py --dataset dataset
- [INFO] loading images...
- [INFO] compiling model...
- [INFO] training head...
- Train for 34 steps, validate on 276 samples
- Epoch 1/20
- 34/34 [==============================] - 30s 885ms/step - loss: 0.6431 - accuracy: 0.6676 - val_loss: 0.3696 - val_accuracy: 0.8242
- Epoch 2/20
- 34/34 [==============================] - 29s 853ms/step - loss: 0.3507 - accuracy: 0.8567 - val_loss: 0.1964 - val_accuracy: 0.9375
- Epoch 3/20
- 34/34 [==============================] - 27s 800ms/step - loss: 0.2792 - accuracy: 0.8820 - val_loss: 0.1383 - val_accuracy: 0.9531
- Epoch 4/20
- 34/34 [==============================] - 28s 814ms/step - loss: 0.2196 - accuracy: 0.9148 - val_loss: 0.1306 - val_accuracy: 0.9492
- Epoch 5/20
- 34/34 [==============================] - 27s 792ms/step - loss: 0.2006 - accuracy: 0.9213 - val_loss: 0.0863 - val_accuracy: 0.9688
- ...
- Epoch 16/20
- 34/34 [==============================] - 27s 801ms/step - loss: 0.0767 - accuracy: 0.9766 - val_loss: 0.0291 - val_accuracy: 0.9922
- Epoch 17/20
- 34/34 [==============================] - 27s 795ms/step - loss: 0.1042 - accuracy: 0.9616 - val_loss: 0.0243 - val_accuracy: 1.0000
- Epoch 18/20
- 34/34 [==============================] - 27s 796ms/step - loss: 0.0804 - accuracy: 0.9672 - val_loss: 0.0244 - val_accuracy: 0.9961
- Epoch 19/20
- 34/34 [==============================] - 27s 793ms/step - loss: 0.0836 - accuracy: 0.9710 - val_loss: 0.0440 - val_accuracy: 0.9883
- Epoch 20/20
- 34/34 [==============================] - 28s 838ms/step - loss: 0.0717 - accuracy: 0.9710 - val_loss: 0.0270 - val_accuracy: 0.9922
- [INFO] evaluating network...

|      |    precision    | recall| f1-score | support |
|------|-----------------|-------|----------|---------|

|accuracy| | |0.99|276|
|macro avg|0.99|0.99|0.99|276|
|weighted avg|0.99|0.99|0.99|276|


![Figure 4](https://www.pyimagesearch.com/wp-content/uploads/2020/04/fingerprint_detector_plot.png)

Figure 4: Figure 10: Fingerprint detector training accuracy/loss curves demonstrate high accuracy and little signs of overfitting on the data

As you can see, we are obtaining ~99% accuracy on our test set.

Looking at Figure 4, we can see there are little signs of overfitting, with the validation loss lower than the training loss. 

Given these results, we are hopeful that our model will generalize well to images outside our training and testing set.


## F.  RESULT AND CONCLUSION

Detecting Fingerprint masks with OpenCV in real-time

You can then launch the fingerprint detector in real-time video streams using the following command:
- $ python detect_mask_video.py
- [INFO] loading face detector model...
- [INFO] loading face mask detector model...
- [INFO] starting video stream...

[![Figure5](https://img.youtube.com/vi/57tSzqRunFI/0.jpg)](https://www.youtube.com/watch?v=57tSzqRunFI"Figure5")

Figure 5: Fingerprint detector in real-time video streams

In Figure 5, you can see that our fingerprint detector is capable of running in real-time (and is correct in its predictions as well.


## G.   PROJECT PRESENTATION 

In this project, you learned how to create a fingerprint detector using OpenCV, Keras/TensorFlow, and Deep Learning.

We fine-tuned MobileNetV2 on our fingerprint dataset and obtained a classifier that is ~99% accurate.

We then took this fingerprint classifier and applied it to both images and real-time video streams by:

- Detecting fingerprint in images/video
- Extracting each individual fingerprint
- Applying our fingerprint classifier

Our fingerprint detector is accurate, and since we used the MobileNetV2 architecture, it’s also computationally efficient, making it easier to deploy the model to embedded systems (Raspberry Pi, Google Coral, Jetosn, Nano, etc.).

[![demo](https://img.youtube.com/vi/Zaq0QAsfXOs/0.jpg)](https://www.youtube.com/watch?v=Zaq0QAsfXOs "demo")




