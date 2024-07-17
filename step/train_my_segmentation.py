#Environmental setting
#pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
import torch.utils.model_zoo
from sklearn.model_selection import train_test_split

#Image processing
import cv2
from osgeo import gdal

#Others
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
    parser.add_argument('--name',help='Name of folder to storage trained model and other files'
        ,default='20240215_unet')
    parser.add_argument('--device',help='Use GPU or CPU for training',choices=['cpu','cuda']
        ,default='cuda')
    parser.add_argument('--num_workers',help='',type=int
        ,default=0)
    #Directory
    parser.add_argument('--dataset',help='Folder for the input dataset'
        ,default='dataset_seg')
    parser.add_argument('--work_dir',help='Wording directory'
        ,default='')
    parser.add_argument('--package_dirs',help='Directory for packages'
        ,default="[]")
    parser.add_argument('--img_train_dir',help='Directory for training dataset'
        ,default='')
    parser.add_argument('--img_val_dir',help='Directory for validation dataset'
        ,default='')
    parser.add_argument('--img_test_dir',help='Directory for test dataset'
        ,default='')
    parser.add_argument('--output_path',help='Directory for output results'
        ,default='')
    parser.add_argument('--config_path',help='Path for model parameters'
        ,default='')
    parser.add_argument('--log_path',help='Path for traning log'
        ,default='')
    parser.add_argument('--model_iou_path',help='Path for well-traning model (best iou)'
        ,default='')
    parser.add_argument('--model_loss_path',help='Path for well-traning model (best loss)'
        ,default='')
    parser.add_argument('--fig_path',help='Path for result figure'
        ,default='')
    #Dataset
    parser.add_argument('--img_ext',help='image file extension'
        ,default='.tif')
    parser.add_argument('--input_channel_list',help='The channels of the imagery you plan to input'
        ,default="[0,1,2,3]")
    parser.add_argument('--label_channel',help='index of label channels',type=int
        ,default=-1)
    parser.add_argument('--pos_classes',help='list of positive classes'
        ,default="[1]")
    parser.add_argument('--neg_classes',help='list of negative classes'
        ,default="[0]")
    parser.add_argument('--num_classes',help='number of classes',type=int
        ,default=2)
    parser.add_argument('--transform',help='Whether to perform data augment'
        ,default='False')
    parser.add_argument('--train_transform_methods',help='Augmentation method for training set'
        ,default="{'one_hot_segmentation_binary':{'test':None}}")
    parser.add_argument('--val_transform_methods',help='Augmentation method for validation and test sets'
        ,default="{'one_hot_segmentation_binary':{'test':None}}")
    parser.add_argument('--batch_size_train','-b',metavar='N',help='batch size for training set',type=int
        ,default=10)
    parser.add_argument('--batch_size_val',metavar='N',help='batch size for validation and test sets',type=int
        ,default=1)
    #Model
    parser.add_argument('--arch', '-a', metavar='ARCH',help='Net architecture'
        ,default='models.segmentation.unet_vgg')
    parser.add_argument('--arch_parameters',help='Parameters for the net'
        ,default="{'input_size':(4,256,256),'num_classes':2,'model_name':'vgg19'}")
    parser.add_argument('--pretrain_backbone',help='Whether to use pre-trained model'
        ,default='False')
    parser.add_argument('--pretrain_path',help='Path for pre-trained model'
        ,default='')
    #Training
    parser.add_argument('--train_mode',help='True as training,False as validation'
        ,default='True')
    parser.add_argument('--epochs',metavar='N',help='epoch',type=int
        ,default=40)
    parser.add_argument('--optimizer',help='Optimize method for loss function'
        ,default='Adam')
    parser.add_argument('--optimizer_params',help="Parameters for optimizer"
        ,default="{'lr':0.001}")
    parser.add_argument('--loss',help='loss function'
        ,default='BinaryCrossEntropyWithLogits')
    parser.add_argument('--early_stopping',metavar='N', help='early stopping (default: -1)',type=int
        ,default=10)
    parser.add_argument('--neg_for_train',help='whether use negative samples for model training'
        ,default='True')
    parser.add_argument('--update',help='whether update labels while training'
        ,default='True')
    parser.add_argument('--update_epoch',help='which epoch to update',type=int
        ,default=20)
    parser.add_argument('--alpha',help=' adjust the loss function for original labels during training with new labels, within the range of 0 to 1',type=float
        ,default=0)
    parser.add_argument('--best_metric',help='Use best_loss or best_iou for early stopping',choices=['loss','iou']
        ,default='loss')
    config = parser.parse_args()
    print(config)
    config=vars(config)
    config['device'] = config['device'] if torch.cuda.is_available() else "cpu"
    config['img_train_dir'] =os.path.join(config['work_dir'],'inputs',config['dataset'],'train')
    config['img_val_dir'] =os.path.join(config['work_dir'],'inputs',config['dataset'],'val')
    config['img_test_dir'] =os.path.join(config['work_dir'],'inputs',config['dataset'],'test')
    config['output_path']=os.path.join(config['work_dir'],'models',config['name'])
    config['config_path']=os.path.join(config['output_path'],'config.yml')
    config['model_loss_path']=os.path.join(config['output_path'],'model_loss.pth')
    config['model_iou_path']=os.path.join(config['output_path'],'model_iou.pth')
    config['fig_path']=os.path.join(config['output_path'],'result.jpg')
    config['output_dir']=os.path.join(config['output_path'],'output')

    config['input_channel_list']=eval(config['input_channel_list'])
    config['transform']=eval(config['transform'])
    config['pos_classes']=eval(config['pos_classes'])
    config['neg_classes']=eval(config['neg_classes'])
    config['num_classes']=len(config['pos_classes'])+len(config['neg_classes'])
    config['pretrain_backbone']=eval(config['pretrain_backbone'])
    config['package_dirs']=eval(config['package_dirs'])
    config['train_transform_methods']=eval(config['train_transform_methods'])
    config['val_transform_methods']=eval(config['val_transform_methods'])
    config['train_mode']=eval(config['train_mode'])
    config['arch_parameters']=eval(config['arch_parameters'])
    config['optimizer_params']=eval(config['optimizer_params'])
    config['neg_for_train']=eval(config['neg_for_train'])
    config['update']=eval(config['update'])
    if config['train_mode']:
        config['log_path']=os.path.join(config['output_path'],'log_train.csv')
    else:
        config['log_path']=os.path.join(config['output_path'],'log_validation.csv')
    #save config
    import yaml
    os.makedirs(config['output_path'], exist_ok=True)
    with open(config['config_path'], 'w') as f:
        yaml.dump(config, f)
    print(config)

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
    class SegmentationDataset(Dataset):
        def __init__(self, img_paths, img_ext, num_classes, input_channel_list,label_channel,transform=None,transform_methods=None):
            self.img_paths = img_paths
            self.img_ext = img_ext
            self.num_classes = num_classes
            self.transform = transform
            self.transform_methods=transform_methods
            self.input_channel_list=input_channel_list
            self.label_channel=label_channel

        def __len__(self):
            return len(self.img_paths)

        def __getitem__(self, idx):
            '''load image as (H,W,C)'''
            img_path = self.img_paths[idx]
            out=dict()
            if (self.img_ext=='.tif'):
                img=gdal.Open(img_path)
                if config['train_mode']==False:
                    out['GeoTransform']=img.GetGeoTransform()
                    out['Projection']=img.GetProjection()
                img=img.ReadAsArray()
                if(len(img.shape)==2):
                    img=img[None,...]
                img=img.transpose(1,2,0)
            else:
                img = cv2.imread(img_path)

            mask=img[...,self.label_channel][...,None]
            channels=[np.take(img,i,axis=2) for i in self.input_channel_list]
            img=np.stack(channels,axis=2)
            
            '''Data agumentation'''
            if self.transform:
                for key in self.transform_methods:
                    method=getattr(importlib.import_module('dataset'),key)
                    img,mask=method(img,mask,**self.transform_methods[key])
                    
            '''(H,W,C) to (C,H,W)'''
            img = img.transpose(2, 0, 1)
            img = img.astype('float32')
            img=np.nan_to_num(img)
            mask = mask.transpose(2, 0, 1)
            mask = mask.astype('float32')
            mask=np.nan_to_num(mask)
            out['label']=mask
            out['img_path']=img_path
            out['img']=img
            out['label_update']=mask           
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
            # print(len(t))
            img_train_paths=img_train_paths+t
            label_train_ids=label_train_ids+[i for j in range(len(t))]
        return img_train_paths,label_train_ids

    img_train_pos_paths,label_train_pos_ids=get_imglabel_by_classlist(config['pos_classes'],config['img_train_dir'],config['img_ext'])
    img_train_neg_paths,label_train_neg_ids=get_imglabel_by_classlist(config['neg_classes'],config['img_train_dir'],config['img_ext'])
    img_val_pos_paths,label_val_pos_ids=get_imglabel_by_classlist(config['pos_classes'],config['img_val_dir'],config['img_ext'])
    img_val_neg_paths,label_val_neg_ids=get_imglabel_by_classlist(config['neg_classes'],config['img_val_dir'],config['img_ext'])
    img_test_pos_paths,label_test_pos_ids=get_imglabel_by_classlist(config['pos_classes'],config['img_test_dir'],config['img_ext'])
    img_test_neg_paths,label_test_neg_ids=get_imglabel_by_classlist(config['neg_classes'],config['img_test_dir'],config['img_ext'])

    train_dataset = SegmentationDataset(
        img_paths=img_train_pos_paths+img_train_neg_paths,
        img_ext=config['img_ext'],
        num_classes=config['num_classes'],
        input_channel_list=config['input_channel_list'],
        label_channel=config['label_channel'],
        transform=config['transform'],
        transform_methods=config['train_transform_methods']
    )
    train_dataset_pos = SegmentationDataset(
        img_paths=img_train_pos_paths,
        img_ext=config['img_ext'],
        num_classes=config['num_classes'],
        input_channel_list=config['input_channel_list'],
        label_channel=config['label_channel'],
        transform=config['transform'],
        transform_methods=config['train_transform_methods']
    )
    train_dataset_neg = SegmentationDataset(
        img_paths=img_train_neg_paths,
        img_ext=config['img_ext'],
        num_classes=config['num_classes'],
        input_channel_list=config['input_channel_list'],
        label_channel=config['label_channel'],
        transform=config['transform'],
        transform_methods=config['train_transform_methods']
    )

    val_dataset = SegmentationDataset(
        img_paths=img_val_pos_paths+img_val_neg_paths,
        img_ext=config['img_ext'],
        num_classes=config['num_classes'],
        input_channel_list=config['input_channel_list'],
        label_channel=config['label_channel'],
        transform=config['transform'],
        transform_methods=config['val_transform_methods']
        )
    val_dataset_pos = SegmentationDataset(
        img_paths=img_val_pos_paths,
        img_ext=config['img_ext'],
        num_classes=config['num_classes'],
        input_channel_list=config['input_channel_list'],
        label_channel=config['label_channel'],
        transform=config['transform'],
        transform_methods=config['val_transform_methods']
        )
    val_dataset_neg = SegmentationDataset(
        img_paths=img_val_neg_paths,
        img_ext=config['img_ext'],
        num_classes=config['num_classes'],
        input_channel_list=config['input_channel_list'],
        label_channel=config['label_channel'],
        transform=config['transform'],
        transform_methods=config['val_transform_methods']
        )

    test_dataset = SegmentationDataset(
        img_paths=img_test_pos_paths+img_test_neg_paths,
        img_ext=config['img_ext'],
        num_classes=config['num_classes'],
        input_channel_list=config['input_channel_list'],
        label_channel=config['label_channel'],
        transform=config['transform'],
        transform_methods=config['val_transform_methods']
        )
    test_dataset_pos = SegmentationDataset(
        img_paths=img_test_pos_paths,
        img_ext=config['img_ext'],
        num_classes=config['num_classes'],
        input_channel_list=config['input_channel_list'],
        label_channel=config['label_channel'],
        transform=config['transform'],
        transform_methods=config['val_transform_methods']
        )
    test_dataset_neg = SegmentationDataset(
        img_paths=img_test_neg_paths,
        img_ext=config['img_ext'],
        num_classes=config['num_classes'],
        input_channel_list=config['input_channel_list'],
        label_channel=config['label_channel'],
        transform=config['transform'],
        transform_methods=config['val_transform_methods']
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size_train'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False
    )
    train_loader_pos = torch.utils.data.DataLoader(
        train_dataset_pos,
        batch_size=config['batch_size_train'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False
    )
    train_loader_neg = torch.utils.data.DataLoader(
        train_dataset_neg,
        batch_size=config['batch_size_train'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size_val'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)
    val_loader_pos = torch.utils.data.DataLoader(
        val_dataset_pos,
        batch_size=config['batch_size_val'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)
    val_loader_neg = torch.utils.data.DataLoader(
        val_dataset_neg,
        batch_size=config['batch_size_val'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size_val'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)
    test_loader_pos = torch.utils.data.DataLoader(
        test_dataset_pos,
        batch_size=config['batch_size_val'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)
    test_loader_neg = torch.utils.data.DataLoader(
        test_dataset_neg,
        batch_size=config['batch_size_val'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)  

    '''Model building and training'''
    # create model
    print("=> creating model %s" % config['arch'])
    model =  getattr(importlib.import_module(config['arch']), 'Net')(**config['arch_parameters'])
    try:
        model.to(config['device'])
    except:
        model=model.model
        model.to(config['device'])

    if config['pretrain_backbone']:
        checkpoint  = torch.load(config['pretrain_path'], map_location = 'cpu')['model']
        state_dict={}
        for key,v in checkpoint.items():
            new_key=key
            if key.startswith('module.'):
                new_key=key[7:]
            state_dict[new_key]=v
        model.load_state_dict(state_dict, strict=False)

    #optimizer
    param_groups=model.trainable_parameters()
    if config['optimizer']=='customized':
        config['len(train_dataset)']=len(train_dataset_pos)+len(train_dataset_neg)
        optimizer=model.optimizer(config,**config['optimizer_params'])
    else:
        optimizer=getattr(importlib.import_module('torch.optim'), config['optimizer'])(model.trainable_parameters(),**config['optimizer_params'])
    
    #scheduler
    def lr_lambda(epoch):
        if epoch < config['update_epoch']:
            return 1.0
        else:
            return 0.1 ** ((epoch - (config['update_epoch']-1)) / (config['epochs'] - config['update_epoch']))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    #loss function
    if config['loss'] == 'customized':
        criterion =None
    else:
        try:#导入的是类
            criterion = getattr(importlib.import_module('losses'), config['loss'])()
        except:#导入的是函数
            criterion = getattr(importlib.import_module('losses'), config['loss'])
        try:
            criterion=criterion.to(config['device'])
        except:
            pass

    start_epoch = 0
    from tqdm import tqdm
    def iou_score(pred, target, nclass=2):
        if torch.is_tensor(pred):
            pred = pred.data.cpu().numpy()
        if torch.is_tensor(target):
            target = target.data.cpu().numpy()
        ious = []
        for i in range(nclass):
            pred_ins = pred == i
            target_ins = target == i
            inser = pred_ins[target_ins].sum()
            union = pred_ins.sum() + target_ins.sum() - inser
            iou = inser / union
            ious.append(iou)
        return ious

    def get_mean(lst):
        return sum(lst) / len(lst)

    def get_mean_list(nested_list):
        sum_a = sum_b = 0
        for sublist in nested_list:
            sum_a += sublist[0]
            sum_b += sublist[1]
        mean_a = sum_a / len(nested_list)
        mean_b = sum_b / len(nested_list)
        return [mean_a, mean_b]    
        
    def get_log():
        log = OrderedDict([
            ('path',[]),
            ('iou',[]),
            ('oa',[]),
            ('precision',[]),
            ('recall', []),
            ('f1', [])
        ])
        return log

    def get_data_by_class(data_lst,index_lst):
        data_lst_new=[]
        for i in index_lst:
            data_lst_new.append(data_lst[i])
        return sum(data_lst_new)/len(data_lst_new)

    def log_record(log,item,iou,oa,precision,recall,f1,flag):
        if torch.is_tensor(iou):
            iou = iou.data.cpu().numpy()
        if torch.is_tensor(oa):
            oa = oa.data.cpu().numpy()
        if torch.is_tensor(precision):
            precision = precision.data.cpu().numpy()
        if torch.is_tensor(recall):   
            recall = recall.data.cpu().numpy()
        if torch.is_tensor(f1):
            f1 = f1.data.cpu().numpy()
        if flag=='pos':
            log['path'].append(item['img_path'][0])
            log['iou'].append(get_data_by_class(iou,config['pos_classes']))
            log['oa'].append(oa)
            log['precision'].append(get_data_by_class(precision,config['pos_classes']))
            log['recall'].append(get_data_by_class(recall,config['pos_classes']))
            log['f1'].append(get_data_by_class(f1,config['pos_classes']))
        else:
            log['path'].append(item['img_path'][0])
            log['iou'].append(get_data_by_class(iou,config['neg_classes']))
            log['oa'].append(oa)
            log['precision'].append(get_data_by_class(precision,config['neg_classes']))
            log['recall'].append(get_data_by_class(recall,config['neg_classes']))
            log['f1'].append(get_data_by_class(f1,config['neg_classes']))
        return log

    def get_path(output_dir,img_path):
        dir1=os.path.basename(os.path.dirname(img_path))#'0/1
        dir2=os.path.basename(os.path.dirname(os.path.dirname(img_path)))#train
        img_name=os.path.splitext(os.path.basename(img_path))[0]#'aqua_0999.tif)
        newdir=os.path.join(output_dir,dir2,dir1)
        os.makedirs(newdir,exist_ok=True)
        return os.path.join(newdir, img_name + '.tif')

    def save_tif(im_data,geotransform,projection,path):
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
            dataset.SetGeoTransform(geotransform) 
            dataset.SetProjection(projection)
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])
        dataset=None
        
        
    def train(config, train_loader_list, model, criterion, optimizer):
        avg_meters = {'loss': AverageMeter()}
        acc_total = ClassificationMetric(numClass=config['num_classes'], device=config['device'])
        model.train()
        pbar = tqdm(total=len(train_loader_list[0]))
        if config['neg_for_train']:
            train_loader2 = train_loader_list[1]
            train_loader2=iter(train_loader2)
        for step,item in enumerate(train_loader_list[0]):
            if config['neg_for_train']:
                inputs1=item['img']
                item2=next(train_loader2)
                inputs2=item2['img']
                inputs=torch.cat((inputs1,inputs2),dim=0).to(config['device'], non_blocking =True)
                target1=item['label']
                target2=item2['label']
                target=torch.cat((target1,target2),dim=0).to(config['device'], non_blocking =True)
            else:
                inputs=item['img'].to(config['device'], non_blocking =True)
                target=item['label'].to(config['device'], non_blocking =True)
            optimizer.zero_grad()
            if criterion==None:
                output,loss=model.loss(inputs,target,**config)
            else:
                output = model(inputs)
                loss = criterion(output['out'],target)
            loss.backward()
            optimizer.step()
            avg_meters['loss'].update(loss.item(), inputs.size(0))
            temp_output=output['out'].detach()
            temp_target=target.detach()
            temp_output=torch.sigmoid(temp_output).argmax(axis=1)
            if temp_output.shape != target.shape:
                temp_target=temp_target.argmax(axis=1)
            iou = iou_score(temp_output.cpu().numpy(), temp_target.cpu().numpy(),nclass=config['num_classes'])
            postfix = OrderedDict([('loss', avg_meters['loss'].avg),('iou',iou)])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()
        return postfix

    def update_label(model, data_loader,config):
        model.eval()
        data=[]
        with torch.no_grad():
            for step,item in enumerate(data_loader):
                inputs=item['img'].to(config['device'], non_blocking =True)
                target=item['label'].to(config['device'], non_blocking =True)
                output = model(inputs)
                output_pred = (torch.sigmoid(output['out'])>0.5).long()
                for i in range(config['batch_size_train']):
                    data_temp=dict()
                    data_temp['label_update']=output_pred[i].cpu().numpy()
                    data_temp['img_path']=item['img_path'][i]
                    data.append(data_temp)
        data_loader_new=torch.utils.data.DataLoader(
            data,
            batch_size=config['batch_size_train'],
            shuffle=False,
            num_workers=config['num_workers'],
            drop_last=True
        )
        return data_loader_new

    def save_data_loader(dataloader,dir):
        import shutil
        from PIL import Image
        import numpy as np
        def create_dir(folder_path):
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
            os.makedirs(folder_path)
        create_dir(dir)
        for step,item in enumerate(dataloader):
            path_list=item['img_path']
            label_update_list=item['label_update']
            for i,path in enumerate(path_list):
                name=os.path.splitext(os.path.basename(path))[0]
                new_path=os.path.join(dir,name+'.png')
                label_update=label_update_list[i].cpu().numpy()
                label_update=np.transpose(label_update,(1,2,0))
                image = Image.fromarray((label_update).astype('uint8'))
                image.save(new_path)

    def train_and_update(config, flag_data_loader,flag_update2,train_loader_list,train_loader_pos_new,model, criterion, optimizer):
        if flag_update2==1:
            train_loader_pos_new=update_label(model, train_loader_list[0], config)
            flag_data_loader=flag_data_loader+1
            dir=os.path.join(config['output_path'],'update_label',str(flag_data_loader))
            save_data_loader(train_loader_pos_new,dir)
            flag_update2=0

        avg_meters = {'loss': AverageMeter()}
        acc_total = ClassificationMetric(numClass=config['num_classes'], device=config['device'])

        train_loader_pos_new_iter=iter(train_loader_pos_new)
        if config['neg_for_train']:
            train_loader2 = train_loader_list[1]
            train_loader2=iter(train_loader2)


        pbar = tqdm(total=len(train_loader_list[0]))
        for step,item in enumerate(train_loader_list[0]):
            if config['neg_for_train']:
                inputs1=item['img']
                item2=next(train_loader2)
                inputs2=item2['img']
                inputs=torch.cat((inputs1,inputs2),dim=0).to(config['device'], non_blocking =True)
                target1=item['label']
                target2=item2['label']
                target=torch.cat((target1,target2),dim=0).to(config['device'], non_blocking =True)
                item3=next(train_loader_pos_new_iter)
                target_new=torch.cat((item3['label_update'],target2)).to(config['device'], non_blocking =True)
            else:
                inputs=item['img'].to(config['device'], non_blocking =True)
                target=item['label'].to(config['device'], non_blocking =True)
                target_new=item['label_update'].to(config['device'], non_blocking =True)
            model.train()
            optimizer.zero_grad()

            output = model(inputs)
            if config['alpha']==0:
                loss=criterion(output['out'], target_new)
            else:
                loss = config['alpha']*criterion(output['out'], target) + criterion(output['out'], target_new)#.float()
            loss.backward()
            optimizer.step()
            temp_output=output['out'].detach()
            temp_target=target.detach()
            temp_output=torch.sigmoid(temp_output).argmax(axis=1)
            if temp_output.shape != target.shape:
                temp_target=temp_target.argmax(axis=1)
            iou = iou_score(temp_output.cpu().numpy(), temp_target.cpu().numpy())
            acc_total.addBatch(temp_output,temp_target)
            avg_meters['loss'].update(loss.item(), inputs.size(0))

            oa = acc_total.OverallAccuracy()
            precision=acc_total.Precision()
            recall=acc_total.Recall()
            f1 = acc_total.F1score()
            postfix = OrderedDict([('loss', avg_meters['loss'].avg)
                                    ,('iou',iou)
                                    ,('oa',oa.cpu().numpy())
                                    ,('precision',precision.cpu().numpy())
                                    ,('recall',recall.cpu().numpy())
                                    ,('f1',f1.cpu().numpy())])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()
        return postfix,train_loader_pos_new,flag_data_loader,flag_update2

    def validate(config, val_loader, model, criterion,flag='pos'):
        avg_meters = {'loss': AverageMeter()}
        acc_total = ClassificationMetric(numClass=config['num_classes'], device=config['device'])
        log=get_log()
        model.eval()
        with torch.no_grad():
            pbar = tqdm(total=len(val_loader))
            for item in val_loader:
                inputs=item['img'].to(config['device'], non_blocking =True)
                target=item['label'].to(config['device'], non_blocking =True)
                if criterion==None:
                    output,loss=model.loss(inputs,target,**config)
                else:
                    output = model(inputs)
                    loss = criterion(output['out'],target)
                temp_output=output['out'].detach()
                temp_target=target.detach()
                temp_output=torch.sigmoid(temp_output).argmax(axis=1)
                if temp_output.shape != target.shape:
                    temp_target=temp_target.argmax(axis=1)
                acc_total.addBatch(temp_output,temp_target)
                avg_meters['loss'].update(loss.item(), inputs.size(0))
                oa=acc_total.OverallAccuracy()
                precision=acc_total.Precision()
                recall=acc_total.Recall()
                f1=acc_total.F1score()
                iou=iou_score(temp_output.cpu().numpy(), temp_target.cpu().numpy())
                log=log_record(log,item,iou,oa,precision,recall,f1,flag)
                postfix = OrderedDict([('loss', avg_meters['loss'].avg)
                                        ,('iou',get_mean(log['iou']))
                                        ,('oa',get_mean(log['oa']))
                                        ,('precision',get_mean(log['precision']))
                                        ,('recall',get_mean(log['recall']))
                                        ,('f1',get_mean(log['f1']))])
                pbar.set_postfix(postfix)
                pbar.update(1)
            pbar.close()
        return postfix

    def validate2(config, val_loader, model, criterion,flag='pos'):
        avg_meters = {'loss': AverageMeter()}
        acc_total = ClassificationMetric(numClass=config['num_classes'], device=config['device'])
        log=get_log()
        model.eval()
        with torch.no_grad():
            pbar = tqdm(total=len(val_loader))
            for item in val_loader:
                inputs=item['img'].to(config['device'], non_blocking =True)
                target=item['label'].to(config['device'], non_blocking =True)
                if criterion==None:
                    output,loss=model.loss(inputs,target,**config)
                else:
                    output = model(inputs)
                    loss = criterion(output['out'],target)
                temp_output=output['out'].detach()
                temp_target=target.detach()
                temp_output=torch.sigmoid(temp_output).argmax(axis=1)
                if temp_output.shape != target.shape:
                    temp_target=temp_target.argmax(axis=1)
                acc_total.addBatch(temp_output,temp_target)
                avg_meters['loss'].update(loss.item(), inputs.size(0))
                oa = acc_total.OverallAccuracy()
                precision=acc_total.Precision()
                recall=acc_total.Recall()
                f1 = acc_total.F1score()
                iou = iou_score(temp_output.cpu().numpy(), temp_target.cpu().numpy(),nclass=config['num_classes'])
                for i,path in enumerate(item['img_path']):
                    label=torch.argmax(item['label'][i],dim=0)[None,...].cpu().numpy()
                    label2=torch.argmax(output['out'][i],dim=0)[None,...].cpu().numpy()
                    im_data = np.concatenate([label,label2], axis=0)
                    geotransform=[t[i] for t in item['GeoTransform']]
                    projection=item['Projection'][i]
                    new_path=get_path(config['output_dir'],path)
                    save_tif(im_data,geotransform,projection,new_path)
                log=log_record(log,item,iou,oa,precision,recall,f1,flag)
                postfix = OrderedDict([('loss', avg_meters['loss'].avg)
                                        ,('iou',get_mean(log['iou']))
                                        ,('oa',get_mean(log['oa']))
                                        ,('precision',get_mean(log['precision']))
                                        ,('recall',get_mean(log['recall']))
                                        ,('f1',get_mean(log['f1']))])
                pbar.set_postfix(postfix)
                pbar.update(1)
            pbar.close()
        return log
        
    if config['train_mode']==False:
        log_val_pos = validate2(config, val_loader_pos, model, criterion,flag='pos') 
        pd.DataFrame(log_val_pos).to_csv(os.path.join(config['output_path'],'log_val_pos.csv'), index=False)
        print('iou %.9f - oa %.9f - precision %.9f - recall %.9f - f1 %.9f'
        % (get_mean(log_val_pos['iou']),get_mean(log_val_pos['oa']),get_mean(log_val_pos['precision']),get_mean(log_val_pos['recall']),get_mean(log_val_pos['f1'])))        
        
        log_val_neg = validate2(config, val_loader_neg, model, criterion,flag='neg')
        pd.DataFrame(log_val_neg).to_csv(os.path.join(config['output_path'],'log_val_neg.csv'), index=False)
        print('iou %.9f - oa %.9f - precision %.9f - recall %.9f - f1 %.9f'
        % (get_mean(log_val_neg['iou']),get_mean(log_val_neg['oa']),get_mean(log_val_neg['precision']),get_mean(log_val_neg['recall']), get_mean(log_val_neg['f1'])))  
        
        log_test_pos = validate2(config, test_loader_pos, model, criterion,flag='pos')
        pd.DataFrame(log_test_pos).to_csv(os.path.join(config['output_path'],'log_test_pos.csv'), index=False)
        print('iou %.9f - oa %.9f - precision %.9f - recall %.9f - f1 %.9f'
        % (get_mean(log_test_pos['iou']),get_mean(log_test_pos['oa']),get_mean(log_test_pos['precision']),get_mean(log_test_pos['recall']), get_mean(log_test_pos['f1'])))  

        log_test_neg = validate2(config, test_loader_neg, model, criterion,flag='neg')
        pd.DataFrame(log_test_neg).to_csv(os.path.join(config['output_path'],'log_test_neg.csv'), index=False)
        print('iou %.9f - oa %.9f - precision %.9f - recall %.9f - f1 %.9f'
        % (get_mean(log_test_neg['iou']),get_mean(log_test_neg['oa']),get_mean(log_test_neg['precision']),get_mean(log_test_neg['recall']), get_mean(log_test_neg['f1'])))  

    else:
        log = OrderedDict([
            ('epoch', []),
            ('lr', []),
            ('iou',[]),
            ('val_iou_pos',[]),
            ('val_iou_neg',[]),
            ('loss', []),
            ('val_loss_pos', []),
            ('val_loss_neg', []),
            ('val_oa_pos',[]),
            ('val_oa_neg',[]),
            ('val_precision_pos',[]),
            ('val_precision_neg',[]),
            ('val_recall_pos',[]),
            ('val_recall_neg',[]),
            ('val_f1_pos',[]),
            ('val_f1_neg',[]),
            ('flag',[])
        ])
        print('start training %s' % config['name'])
        best_loss=100
        best_iou=0
        trigger = 0
        start_time = time.time()
        train_loader_pos_new=None
        flag_data_loader=0
        flag_update=0
        flag_update2=1
        dir=os.path.join(config['output_path'],'update_label',str(flag_data_loader))
        save_data_loader(train_loader_pos,dir)
  
        for epoch in range(config['epochs']):
            log['epoch'].append(epoch)
            log['lr'].append(optimizer.param_groups[0]['lr'])
            print('Epoch [%d/%d], lr: %.6f' % (epoch, config['epochs'],optimizer.param_groups[0]['lr']))
            if config['neg_for_train']:
                train_loader_list=[train_loader_pos,train_loader_neg]
            else:
                train_loader_list=[train_loader_pos]
            if epoch < config['update_epoch']:
                train_log = train(config,train_loader_list, model, criterion, optimizer)
                scheduler.step()
            else:
                train_log,train_loader_pos_new,flag_data_loader,flag_update2 = train_and_update(config,flag_data_loader,flag_update2,train_loader_list,train_loader_pos_new, model, criterion, optimizer)
                scheduler.step()
            val_log_pos = validate(config, val_loader_pos, model, criterion,flag='pos')
            val_log_neg = validate(config, val_loader_neg, model, criterion,flag='neg')
            print('iou %.4f - val_iou_pos %.4f - val_iou_neg %.4f| loss %.4f - val_loss_pos %.4f - val_loss_neg %.4f'% 
            (train_log['iou'][config['pos_classes'][0]],val_log_pos['iou'],val_log_neg['iou'],train_log['loss'], val_log_pos['loss'], val_log_neg['loss']))
            log['iou'].append(train_log['iou'])
            log['val_iou_pos'].append(val_log_pos['iou'])
            log['val_iou_neg'].append(val_log_neg['iou'])
            log['loss'].append(train_log['loss'])
            log['val_loss_pos'].append(val_log_pos['loss'])
            log['val_loss_neg'].append(val_log_neg['loss'])
            log['val_oa_pos'].append(val_log_pos['oa'])
            log['val_oa_neg'].append(val_log_neg['oa'])
            log['val_precision_pos'].append(val_log_pos['precision'])
            log['val_precision_neg'].append(val_log_neg['precision'])
            log['val_recall_pos'].append(val_log_pos['recall'])
            log['val_recall_neg'].append(val_log_neg['recall'])
            log['val_f1_pos'].append(val_log_pos['f1'])
            log['val_f1_neg'].append(val_log_neg['f1'])
            
            trigger += 1
            if config['best_metric']=='loss':
                current_loss=(val_log_pos['loss']+val_log_neg['loss'])/2
                if  current_loss< best_loss:
                    save_file = {"model": model.state_dict()
                            ,"state_dict":model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                            ,"optimizer": optimizer.state_dict()
                            ,"epoch": epoch
                            ,"args": config
                            ,'best_loss':best_loss
                            }
                    torch.save(save_file, config['model_loss_path'])
                    best_loss = current_loss
                    print("=> saved best loss model")
                    trigger = 0
                    log['flag'].append(1)
                else:
                    log['flag'].append(0)
            elif config['best_metric']=='iou':
                current_iou=(val_log_pos['iou']+val_log_neg['iou'])/2
                if  current_iou> best_iou:
                    save_file = {"model": model.state_dict()
                            ,"state_dict":model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                            ,"optimizer": optimizer.state_dict()
                            ,"epoch": epoch
                            ,"args": config
                            ,'best_iou':best_iou
                            }
                    torch.save(save_file, config['model_iou_path'])
                    best_iou = current_iou
                    print("=> saved best iou model")
                    trigger = 0
                    log['flag'].append(1)
                else:
                    log['flag'].append(0)

            # early stopping
            if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
                    print("=> early stopping")
                    break
            pd.DataFrame(log).to_csv(config['log_path'], index=False)
            if config['device']=='cuda':
                torch.cuda.empty_cache()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("training time {}".format(total_time_str))

        path=config['log_path']
        data=pd.read_csv(path,header=0)
        data
        import matplotlib.pyplot as plt
        fig=plt.figure(dpi=300)
        plt.plot(data['epoch'],data['iou'],label='training_iou')
        plt.plot(data['epoch'],data['val_iou_pos'],label='testing_iou_pos')
        plt.plot(data['epoch'],data['val_iou_neg'],label='testing_iou_neg')
        plt.plot(data['epoch'],data['loss'],label='training_loss')
        plt.plot(data['epoch'],data['val_loss_pos'],label='testing_loss_pos')
        plt.plot(data['epoch'],data['val_loss_neg'],label='testing_loss_neg')
        plt.legend()
        fig.savefig(config['fig_path'])
        fig.show()
