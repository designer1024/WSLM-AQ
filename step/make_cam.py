#Environmental setting
#pytorch
import torch
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.utils.model_zoo
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
from collections import OrderedDict
from sklearn.model_selection import train_test_split

#Image processing
import cv2
from osgeo import gdal

#Others
import numpy as np
import random
import math
from glob import glob
import pandas as pd
import yaml
from tqdm import tqdm
import time
import os
import sys
import numpy as np
import random
import math
import pandas as pd
import argparse
from collections import OrderedDict
import time
import datetime

#Set random seed
# torch.manual_seed(1337)
# torch.cuda.manual_seed(1337)
# np.random.seed(1337)
# random.seed(1337)

if __name__=='__main__':

    '''set parameters'''
    def str2bool(v):
        if v.lower() in ['true', 1]:
            return True
        elif v.lower() in ['false', 0]:
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser = argparse.ArgumentParser()
    #basic
    parser.add_argument('--device',help='Use GPU or CPU for training',choices=['cpu','cuda']
        ,default='cuda')
    parser.add_argument('--num_workers',help='',type=int
        ,default=0)
    #Path
    parser.add_argument('--package_dirs',help='Directory for packages'
        ,default="[]")
    parser.add_argument('--pretrain_dir',help='Folder for pre-trained model'
        ,default='')
    parser.add_argument('--model_name',help='Pre-trained model path'
        ,default='model_loss.pth')
    parser.add_argument('--outputdir_name',help='Folder for output results'
        ,default='cam')
    #Dataset
    parser.add_argument('--img_cam_dir',help='Folder of input imagery for generating pseudo-labels'
        ,default='')
    parser.add_argument('--img_ext',help='image file extension'
        ,default='.tif')
    parser.add_argument('--output_type',help='Format of output results',choices=['.npy','.tif']
        ,default='.tif')
    parser.add_argument('--classes_list',help='list of classes to generate CAMs'
        ,default="[1]")
    parser.add_argument('--classes_pos',help='list of positive classes'
        ,default="[1]")
    parser.add_argument('--water_channels',help='index of CF_NDWI channels',type=int
        ,default=-2)
    parser.add_argument('--label_channels',help='index of label channels',type=int
        ,default=-1)
    parser.add_argument('--scales',help='scale sizes for multi-scale CAM generation'
        ,default='(1.0, 0.5, 0.75, 1.25, 1.5, 1.75, 2.0)')
    #CAM
    parser.add_argument('--cam',help='method to generate CAM'
        ,default='models.segmentation.gradcam')
    parser.add_argument('--target_layer',help='list of model layers to generate CAM'
        ,default="['resnet.layer4.2.bn3']")
    parser.add_argument('--target_layer_weight',help='list of weights for layers to generate CAM; the sum of weights must equal to 1'
        ,default="[1,]")#"[0.5,0.5]"
    parser.add_argument('--adv_iter',help='number of iteration',type=int
        ,default=1)
    parser.add_argument('--score_th',help='threshold that distinguishes between discriminative and non-discriminative regions (referred to as τCAM in the paper',type=float
        ,default=0.5)
    parser.add_argument('--AD_coeff',help='hyperparameters of the regularization term for discriminative water regions (referred to as λ1 in the paper)',type=int
        ,default=7)
    parser.add_argument('--AD_stepsize',type=float
        ,default=0.08)
    parser.add_argument('--flip_augmentation',help='whether to use multi-scale CAM generation method'
        ,default='True')
    parser.add_argument('--use_water',help='whether to use CF_NDWI to constrain pseudo-label generation'
        ,default='True')
    parser.add_argument('--water_threshold',help='threshold that distinguish water from non-water (referred to as τCF in the paper)',type=float
        ,default=0.3)
    parser.add_argument('--suppressing_classes',help='whether to use non-aquaculture water class suppression'
        ,default='True')
    parser.add_argument('--add_discriminative',help='whether to use discriminative suppression'
        ,default='True')
    parser.add_argument('--water_for_discriminative',help='whether to use CF_NDWI for discriminative suppression, i.e. discriminative water region suppression'
        ,default='True')
    parser.add_argument('--non_water_for_discriminative',help='whether to use the regularization term for discriminative non-water regions'
        ,default='True')
    parser.add_argument('--AD_coeff2',help='hyperparameters of the regularization term for discriminative non-water regions (referred to as λ2 in the paper)',type=int
        ,default=7)    
    config = parser.parse_args()
    config=vars(config)

    config['device'] = config['device'] if torch.cuda.is_available() else "cpu"
    config['package_dirs']=eval(config['package_dirs'])
    config['target_layer']=eval(config['target_layer'])
    config['classes_list']=eval(config['classes_list'])
    config['classes_pos']=eval(config['classes_pos'])
    config['scales']=eval(config['scales'])
    config['target_layer_weight']=eval(config['target_layer_weight'])
    config['flip_augmentation']=eval(config['flip_augmentation'])
    config['use_water']=eval(config['use_water'])
    config['suppressing_classes']=eval(config['suppressing_classes'])
    config['add_discriminative']=eval(config['add_discriminative'])
    config['water_for_discriminative']=eval(config['water_for_discriminative'])
    config['non_water_for_discriminative']=eval(config['non_water_for_discriminative'])

    config['outputdir']=os.path.join(config['pretrain_dir'],config['outputdir_name'])
    os.makedirs(config['outputdir'],exist_ok=True)
    config['npy_dir']=os.path.join(config['outputdir'],'npy')
    config['log_path']=os.path.join(config['outputdir'],'log.csv')
    config['fig_path']=os.path.join(config['outputdir'],'fig.jpg')
    config['config_path']=os.path.join(config['outputdir'],'config_make_cam.yml')

    #save config
    os.makedirs(os.path.join(config['pretrain_dir'],config['outputdir_name']), exist_ok=True)
    with open(config['config_path'], 'w') as f:
        yaml.dump(config, f)
        
    #load config from pre-trained model
    # import yaml
    with open(os.path.join(config['pretrain_dir'],'config.yml'), 'r') as f:
        config_pretrain = yaml.load(f, Loader=yaml.FullLoader)
    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)
    print('+'*20)
    for key in config_pretrain.keys():
        print('%s: %s' % (key, str(config_pretrain[key])))
    print('+'*20)
    '''load modules from packages'''
    for item in config['package_dirs']:
        sys.path.append(item)
    import importlib
    from dataset import *
    from losses import *
    from utils import *
    from metrics import *
    import utils

    '''load dataset'''
    class ClassificationDataset(Dataset):
        def __init__(self, img_paths, label_ids,img_ext, num_classes, input_channel_list,transform=None,transform_methods=None):
            self.img_paths = img_paths
            self.label_ids = label_ids
            self.img_ext = img_ext
            self.num_classes = num_classes
            self.transform = transform
            self.transform_methods=transform_methods
            self.input_channel_list=input_channel_list

        def __len__(self):
            return len(self.img_paths)

        def __getitem__(self, idx):
            '''load image as (H,W,C)'''
            img_path = self.img_paths[idx]
            label_id=self.label_ids[idx]
            if (self.img_ext=='.tif'):
                img=gdal.Open(img_path)
                img=img.ReadAsArray()
                if(len(img.shape)==2):
                    img=img[None,...]
                img=img.transpose(1,2,0)
            else:
                img = cv2.imread(img_path)

            channels=[np.take(img,i,axis=2) for i in self.input_channel_list]
            img=np.stack(channels,axis=2)
            
            '''Data agumentation'''
            label=label_id
            if self.transform:
                for key in self.transform_methods:
                    method=getattr(importlib.import_module('dataset'),key)
                    img,label=method(img,label,**self.transform_methods[key])

            '''(H,W,C) to (C,H,W)'''
            img = img.transpose(2, 0, 1)
            img = img.astype('float32')
            img=np.nan_to_num(img)
            result={'label':label, 'img_path': img_path,'img':img}

            return result

    class ClassificationDatasetMSF(ClassificationDataset):
        '''convert single-scale input to multi-scale input'''
        def __init__(self, img_paths, label_ids,img_ext, num_classes, input_channel_list:list,water_channels,label_channels=-1,transform=None,transform_methods=None,
                    scales=(1.0,)):
            super().__init__(img_paths, label_ids,img_ext, num_classes, input_channel_list,transform=None,transform_methods=None)
            self.scales = scales
            self.water_channels=water_channels
            self.label_channels=label_channels
        def __getitem__(self, idx):
            img_path = self.img_paths[idx]
            out=dict()
            if (self.img_ext=='.tif'):
                ds=gdal.Open(img_path)
                out['GeoTransform']=ds.GetGeoTransform()
                out['Projection']=ds.GetProjection()
                img=ds.ReadAsArray()
                if(len(img.shape)==2):
                    img=img[None,...]
                img=img.transpose(1,2,0)
            else:
                img = cv2.imread(img_path)

            channels=[np.take(img,i,axis=2) for i in self.input_channel_list+[self.water_channels,self.label_channels]]
            img=np.stack(channels,axis=2)
            # print(channels)
            # print(img.shape)

            #transform
            def one_hot(image,label,others=None):
                num_classes=others
                return image,F.one_hot(torch.tensor(label),num_classes)
            img,label=one_hot(img,self.label_ids[idx],others=2)
            ms_img_list = []
            for s in self.scales:
                if s == 1:
                    s_img = img
                else:
                    def pil_resize(img, size, order):
                        if size[0] == img.shape[0] and size[1] == img.shape[1]:
                            return img
                        if order == 3:
                            resample = cv2.INTER_CUBIC
                        elif order == 0:
                            resample = cv2.INTER_NEAREST
                        return cv2.resize(img,(size[0],size[1]),interpolation=resample)
                    def pil_rescale(img, scale, order):
                        height, width = img.shape[:2]
                        target_size = (int(np.round(height*scale)), int(np.round(width*scale)))
                        return pil_resize(img, target_size, order)
                    s_img = pil_rescale(img, s, order=3)
                s_img = np.transpose(s_img, (2, 0, 1))
                ms_img_list.append(np.stack([s_img, np.flip(s_img, -1)], axis=0))
            img_water=[np.expand_dims(ms_img[:,-2,:,:],axis=1) for ms_img in ms_img_list]
            label_seg=ms_img_list[0][0,-1,:,:][None,...]
            out["img_path"]=img_path
            out['img_water']=img_water
            out['size']=(img.shape[0], img.shape[1])
            out['label']=label
            out['label_seg']=label_seg
            out['scales']=self.scales

            ms_img_list_new=[]
            for item in ms_img_list:
                ms_img_list_new.append(item[:,0:-2,:,:].astype(np.float32))
            out['img']=ms_img_list_new
            return out

    from glob import glob
    def get_imglabel_by_classlist(classes_list,img_train_dir,img_ext):
        '''
        get the paths and classes of the input images
        '''
        img_train_paths=[]
        label_train_ids=[]
        for i in classes_list:
            t=glob(os.path.join(img_train_dir, str(i),'*' + img_ext))
            print(os.path.join(img_train_dir, str(i),'*' + img_ext))
            img_train_paths=img_train_paths+t
            label_train_ids=label_train_ids+[i for j in range(len(t))]
        return img_train_paths,label_train_ids

    img_test_pos_paths,label_test_pos_ids=get_imglabel_by_classlist(config['classes_list'],config['img_cam_dir'],config['img_ext'])

    val_dataset_pos = ClassificationDatasetMSF(
        img_paths=img_test_pos_paths,
        label_ids=label_test_pos_ids,
        img_ext=config_pretrain['img_ext'],
        num_classes=config_pretrain['num_classes'],
        input_channel_list=config_pretrain['input_channel_list'],
        water_channels=config['water_channels'],
        label_channels=config['label_channels'],
        transform=config_pretrain['transform'],
        transform_methods=config_pretrain['val_transform_methods'],
        scales=config['scales']
    )

    n_gpus = torch.cuda.device_count()
    val_dataloader_pos = DataLoader(val_dataset_pos, shuffle=False, num_workers=config['num_workers'] // n_gpus, pin_memory=True)

    '''model building and training'''
    # create model
    print("=> creating model %s" % config_pretrain['arch'])
    model=getattr(importlib.import_module(config_pretrain['arch']), 'CAM')(**config_pretrain['arch_parameters'])
    try:
        model.to(config['device'])
    except:
        model=model.model
        model.to(config['device'])
    model_data=torch.load(os.path.join(config['pretrain_dir'],config['model_name']), map_location = 'cpu')
    checkpoint  = model_data['model']
    state_dict={}
    for key,v in checkpoint.items():
        new_key=key
        if key.startswith('module.'):
            new_key=key[7:]
        state_dict[new_key]=v
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    '''make CAM'''
    def get_cam_and_output(config,gcam,step,target_layer,img_single,class_index,it):
        def gap2d(x, keepdims=False):
            out = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
            if keepdims:
                out = out.view(out.size(0), out.size(1), 1, 1)
            return out

        img_single.requires_grad = True
        outputs = gcam.forward(img_single.to(config['device'], non_blocking =True), step=step)
        gcam.backward(ids=class_index)
        cam_img = gcam.generate(target_layer=target_layer)
        if config['flip_augmentation']:
            cam_img = cam_img[0] + cam_img[1].flip(-1)
        else:
            cam_img=cam_img[0]
        return cam_img,outputs

    def get_loss(outputs,class_index,cam_img,init_cam,cam_mask,cam_mask2):
        def gap2d(x, keepdims=False):
            out = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
            if keepdims:
                out = out.view(out.size(0), out.size(1), 1, 1)
            return out
        logit = F.relu(outputs)
        logit = gap2d(logit, keepdims=True)[:, :, 0, 0]
        if config['suppressing_classes']:
            logit_loss = logit[:, class_index].sum() - (torch.sum(logit)- (logit[:, class_index]).sum() )
        else:
            logit_loss= logit[:, class_index].sum()
        L_AD = torch.sum((torch.abs(cam_img - init_cam))*cam_mask.to(config['device']))
        loss = logit_loss - L_AD * config['AD_coeff']
        if config['non_water_for_discriminative']:
            L_AD2=torch.sum((torch.abs(cam_img))*cam_mask2.to(config['device']))
            loss=loss-L_AD2*config['AD_coeff2']
        return loss

    def get_cam_adv_iter(config,gcam,pack,target_layer,step,size_idx,img_single,class_index,valid_class_index,total_adv_iter,mul_for_scale):
        for it in range(total_adv_iter):
            cam_img,outputs=get_cam_and_output(config,gcam,step,target_layer,img_single,class_index,it)
            if it==0:
                cam_this_class=torch.zeros([valid_class_index.shape[0], outputs.shape[2], outputs.shape[3]])
            cam_this_class += cam_img[0].data.cpu()*mul_for_scale
            def add_discriminative(regions, score_th,size_idx):
                '''
                discriminative regions suppression
                '''
                expanded_mask2=torch.zeros(regions.shape)

                region_ = regions / regions.max()
                expanded_mask=torch.where(region_>score_th,torch.tensor(1),torch.tensor(0))

                if config['water_for_discriminative'] or config['non_water_for_discriminative']:
                    '''discriminative water regions suppression'''
                    water=pack['img_water'][size_idx][0][0].cpu().numpy()
                    initial_size=water.shape[1]
                    new_size=expanded_mask.shape[1]
                    water=water.reshape(1,new_size,int(initial_size/new_size),new_size,int(initial_size/new_size))
                    water=water.mean(axis=(2,4))
                    water=np.where(water>=config['water_threshold'],10,0)                   
                    mask=expanded_mask+torch.tensor(water).to(config['device'])
                    if config['water_for_discriminative']:
                        expanded_mask=torch.where(mask==11,torch.tensor(1),torch.tensor(0))
                    if config['non_water_for_discriminative']:
                        expanded_mask2=torch.where(mask==1,torch.tensor(1),torch.tensor(0))
                return expanded_mask,expanded_mask2
            if config['add_discriminative']:
                cam_mask,cam_mask2=add_discriminative(cam_img,config['score_th'],size_idx)
            else:
                cam_mask=torch.ones(cam_img.shape)
                cam_mask2=torch.ones(cam_img.shape)

            if it == 0:
                init_cam = cam_img.detach()
            loss=get_loss(outputs,class_index,cam_img,init_cam,cam_mask,cam_mask2)
            model.zero_grad()
            img_single.grad.zero_()
            loss.backward()
            def anti_adv(image, epsilon, data_grad):
                '''
                Iterative anti-adversarial attacks
                '''
                sign_data_grad = data_grad / (torch.max(torch.abs(data_grad))+1e-12)
                perturbed_image = image + epsilon*sign_data_grad
                perturbed_image = torch.clamp(perturbed_image, image.min().data.cpu().float(), image.max().data.cpu().float()) 
                return perturbed_image
            perturbed_data=anti_adv(img_single, config['AD_stepsize'], img_single.grad.data)
            img_single = perturbed_data.detach()
        return cam_this_class

    def get_cam_class(config,gcam,pack,target_layer,step,size_idx,img_org,total_adv_iter,mul_for_scale):
        valid_class_index = torch.nonzero(pack['label'][0])[:, 0]
        cam_all_classes = []
        for i, class_index in enumerate(list(valid_class_index)):
            pack['img'][size_idx] = img_org
            img_single = pack['img'][size_idx].detach()[0]
            cam_this_class=get_cam_adv_iter(config,gcam,pack,target_layer,step,size_idx,img_single,class_index,valid_class_index,total_adv_iter,mul_for_scale)
            cam_all_classes.append(cam_this_class)
        return torch.stack(cam_all_classes,dim=0)[0]

    def get_cam_size(config,gcam,pack,target_layer,step):
        outputs_cam = []
        size_idx_list=list(np.array(config['scales']).argsort())

        for i,size_idx in enumerate(size_idx_list):
            if pack['scales'][i]==1:
                if config['adv_iter'] > 10:
                    total_adv_iter = config['adv_iter'] // 2
                    mul_for_scale = 2
                elif config['adv_iter'] < 6:
                    total_adv_iter = config['adv_iter']
                    mul_for_scale = 1
                else:
                    total_adv_iter = 5
                    mul_for_scale = float(total_adv_iter) / 5
            else:
                total_adv_iter = config['adv_iter']
            img_org=pack['img'][size_idx].clone()
            cam_all_classes=get_cam_class(config,gcam,pack,target_layer,step,size_idx,img_org,total_adv_iter,mul_for_scale)
            outputs_cam.append(cam_all_classes)
        cam = [F.interpolate(torch.unsqueeze(item, 1), pack['size'],mode='bilinear', align_corners=False) for item in outputs_cam]
        cam = torch.sum(torch.stack(cam, 0), 0)[:, 0, :pack['size'][0], :pack['size'][1]]
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5
        return cam

    def get_cam_step(config,gcam,pack):
        for i,layer in enumerate(config['target_layer']):
            cam=get_cam_size(config,gcam,pack,target_layer=layer,step=i+1)
            if i==0:
                cam_weight=cam*config['target_layer_weight'][i]
            else:
                cam_weight=cam_weight+cam*config['target_layer_weight'][i]
        return cam_weight.cpu().numpy()

    def get_path(output_dir,img_path):
        dir1=os.path.basename(os.path.dirname(img_path))#'0/1
        dir2=os.path.basename(os.path.dirname(os.path.dirname(img_path)))#train
        img_name=os.path.splitext(os.path.basename(img_path))[0]#'aqua_0999.tif)
        newdir=os.path.join(output_dir,dir2,dir1)
        os.makedirs(newdir,exist_ok=True)
        return os.path.join(newdir, img_name + config['output_type'])

    def get_pseudo_label(cam,score_th,pack):
        '''pseudo-label generation'''
        if int(os.path.basename(os.path.dirname(pack['img_path'][0]))) in config['classes_pos']:
            cam_mask = np.where(cam > score_th,1,0).astype(np.uint8)
        else:
            cam_mask = np.where(cam < score_th,1,0).astype(np.uint8)
        if config['use_water']:
            water=pack['img_water'][0][0][0].cpu().numpy()
            water=np.where(water>=config['water_threshold'],1,0)
            cam_mask=cam_mask*water
        return cam_mask

    def get_cam_dataloader(config,gcam,dataloader,output_dir):
        log = OrderedDict([
            ('path', []),
            ('class', []),
            ('iou',[])
        ])
        npy_list=[]
        for _iter, pack in enumerate(dataloader):
            cam= get_cam_step(config,gcam,pack)

            data={
                "label":torch.nonzero(pack['label'][0])[:, 0]
                ,'label_seg':pack['label_seg'][0].cpu().numpy()
                ,"cam":cam
                ,"pseudo_label":get_pseudo_label(cam,config['score_th'],pack)
                ,'rgb':pack['img'][0][0][0][0:3]
            }
            path=get_path(output_dir,pack['img_path'][0])
            iou=iou_score(data['pseudo_label'],data['label_seg'])
            print('iou:',iou)
            if config['output_type']=='.npy':
                npy_data=data
                npy_data['iou']=iou
                np.save(path,npy_data)
                npy_data=np.load(path,allow_pickle=True).item()
            elif config['output_type']=='.tif':
                im_data=pack['img'][0][0][0].cpu().numpy()
                water=pack['img_water'][0][0][0].cpu().numpy()
                label=data["pseudo_label"]
                im_data = np.concatenate([im_data,water,label], axis=0)
                if 'int8' in im_data.dtype.name:
                    datatype = gdal.GDT_Byte
                elif 'int16' in im_data.dtype.name:
                    datatype = gdal.GDT_UInt16
                else:
                    datatype = gdal.GDT_Float32
                im_bands, im_height, im_width = im_data.shape
                driver = gdal.GetDriverByName("GTiff")
                dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
                if(dataset!= None):
                    dataset.SetGeoTransform(pack['GeoTransform']) 
                    dataset.SetProjection(pack['Projection'][0])
                for i in range(im_bands):
                    dataset.GetRasterBand(i+1).WriteArray(im_data[i])
                dataset=None    
            npy_list.append(path)
            
            log['path'].append(path)
            log['class'].append(data['label'])
            log['iou'].append(iou)
            pd.DataFrame(log).to_csv(config['log_path'], index=False)

            if config['device']=='cuda':
                torch.cuda.empty_cache()
        return npy_list

    gcam = getattr(importlib.import_module(config['cam']), 'CAM')(model=model, candidate_layers=config['target_layer'])
    start_time=time.time()
    npy_list=get_cam_dataloader(config,gcam,dataloader=val_dataloader_pos,output_dir=config['npy_dir'])
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
