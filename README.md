# Face-Recognition

Face Recognition Based on Facenet

Built using [Facenet](https://github.com/davidsandberg/facenet)'s state-of-the-art face recognition built with deep learning. The model has an accuracy of 98.6% on the [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) benchmark.

### Features

- Out of Box Working Face Recognition
- Choose Any Pre-Trained Model from Facenet
- For training just provide the proper folder structure

### Prerequisites

- You need Python(2.6 to 3.5) installed

## Let's Begin

#### For Facial Recognition we need to align images as follows:

```
import facenet_recognition
facenet_recognition.align_input('input_images','aligned_images')
```
*Above command will create our input images into aligned format and save it in given aligned images folder*

#### Train & Test Classifier on Images
After we have aligned images now we can train our classifier.
```
pre_model='./pretrained_facenet_model/20170511-185253.pb' #locaiton of pret-trained model from Facenet
my_class ='./my_classifier/my_classifier.pkl' #location where we want to save
test_classifier_type = 'svm' #type of model either svm or nn
weight= './my_classifier/model_small.yaml' #local stored weights

facenet_recognition.test_train_classifier(aligned_images,pre_model,my_class,weight,test_classifier_type,nrof_train_images_per_class=30, seed=102)
```
*Mininum Required Image per person: 30

#### Train Classifer on Images(only Training)
This API is used to Train our Classifier on Aligned Images
```
pre_model='./pretrained_facenet_model/20170511-185253.pb' #locaiton of pret-trained model from Facenet
my_class ='./my_classifier/my_classifier.pkl' #location where we want to save
test_classifier_type = 'nn' #type of model either svm or nn
weight= './my_classifier/model_small.yaml' #local stored weights

facenet_recognition.create_classifier(aligned_images,pre_model,my_class,weight,test_classifier_type)
```
*Mininum Required Image per person: 30*

#### Test Classifer on Images
This API is used to test our Trained Classifer
```
pre_model='./pretrained_facenet_model/20170511-185253.pb' #locaiton of pret-trained model from Facenet
my_class ='./my_classifier/my_classifier.pkl' #location where we want to save
test_classifier_type = 'nn' #type of model either svm or nn
weight= './my_classifier/model_small.yaml' #local stored weights

facenet_recognition.test_classifier(aligned_images,pre_model,my_class,weight,test_classifier_type)
```
*Mininum Required Image per person: 1*

## References

- FaceNet: A Unified Embedding for Face Recognition and Clustering : https://arxiv.org/abs/1503.03832.
- https://github.com/davidsandberg/facenet
