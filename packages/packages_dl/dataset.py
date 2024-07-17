
from PIL import Image
from osgeo import gdal

import random
import numpy as np

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
 
    def __call__(self, image, mask=None):
        for t in self.transforms:
            image, mask = t(image, mask)
        return {'image':image, 'mask':mask}


def flip(image,label,**params):
    label_flag=params['segmentation']
    if label_flag==0:
      print('classification')
      label_temp=label
      label=np.zeros((image.shape[0],image.shape[1],1))
    flip = np.random.choice([0,1,2,3])
    if(flip==1):
      label = np.fliplr(label)
      image = np.fliplr(image)
    elif flip==2:
      image=np.flipud(image)
      label=np.flipud(label)
    elif flip==3:
      label = np.flipud(np.fliplr(label))
      image=np.flipud(np.fliplr(image))
    if label_flag==0:
      return image,label_temp
    else:
      return image,label

def flip_test(image,label,**params):
  image=np.flipud(image)
  label=np.flipud(label)
  return image,label

def rot(image,label,**params):
    label_flag=params['segmentation']
    if label_flag==0:
      label_temp=label
      label=np.zeros((image.shape[0],image.shape[1],1))
    rot=np.random.choice([0,-1,1])
    if rot!=0:
      flag=2
    if rot==-1:
      image = np.rot90(image, -1, (0,1))
      label=np.rot90(label,-1)
    elif rot==1:
      image = np.rot90(image, 1, (0,1))
      label=np.rot90(label,1)
    if label_flag==0:
      return image,label_temp
    else:
      return image,label

def linear_stretch(image,label,**params):
    label_flag=params['segmentation']
    if label_flag==0:
      label_temp=label
      label=np.zeros((image.shape[0],image.shape[1],1))
    stretch = np.random.choice([True, False],p=[0.7,0.3])
    if(stretch):
      def truncated_linear_stretch(image, truncated_value, max_out = 1, min_out = 0):
          def gray_process(gray):
              truncated_down = np.percentile(gray, truncated_value)#5%
              truncated_up = np.percentile(gray, 100 - truncated_value)#95%
              gray = (gray - truncated_down) / (truncated_up - truncated_down) * (max_out - min_out) + min_out
              # print(gray)
              gray = np.clip(gray, min_out, max_out)
              # print(gray)
              return gray

          image_stretch = []
          for i in range(image.shape[0]):
              gray = gray_process(image[i,:,:])
              image_stretch.append(gray)
          image_stretch = np.array(image_stretch)
          return image_stretch
      image = truncated_linear_stretch(image, 0.5)
      if label_flag==0:
        return image,label_temp
      else:
        return image,label

def normalization(image,label,**params):
    label_flag=params['segmentation']
    if label_flag==0:
      label_temp=label
      label=np.zeros((image.shape[0],image.shape[1],1))
    class TorchvisionNormalize():
      def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
          self.mean = mean
          self.std = std

      def __call__(self, img):
          imgarr = np.asarray(img)
          proc_img = np.empty_like(imgarr, np.float32)
          proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
          proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
          proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]
          return proc_img
    image=TorchvisionNormalize(image)
    label=label
    if label_flag==0:
      return image,label_temp
    else:
      return image,label


def one_hot(image,label,**params):
  num_classes=params['num_classes']
  label=torch.tensor(label,dtype=torch.int64)
  return image,F.one_hot(label,num_classes).numpy()

def one_hot_segmentation(image,label,**params):
  num_classes=params['num_classes']
  label=np.squeeze(label,axis=-1)
  labels=[]
  for i in range(num_classes):
    labels.append(np.where(label==i,1,0))
  label_new=np.stack(labels,axis=-1)
  return image,label_new

def one_hot_segmentation_binary(image,label,**params):
  label=np.squeeze(label,axis=-1)
  labels_pos=np.where(label==1,1,0)
  labels_neg=np.where(label==0,1,0)
  label_new=np.stack([labels_neg,labels_pos],axis=-1)
  return image,label_new
   
