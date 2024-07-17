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
        ,default='model_cls')
    parser.add_argument('--device',help='Use GPU or CPU for training',choices=['cpu','cuda']
        ,default='cuda')
    parser.add_argument('--num_workers',help='',type=int
        ,default=0)
    #Directory
    parser.add_argument('--dataset',help='Folder for the input dataset'
        ,default='dataset_cls')
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
    parser.add_argument('--model_oa_path',help='Path for well-traning model (best oa)'
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
    parser.add_argument('--num_classes',help='Number of categories',type=int
        ,default=2)
    parser.add_argument('--transform',help='Whether to perform data augment'
        ,default='False')
    parser.add_argument('--train_transform_methods',help='Augmentation method for training set'
        ,default="{'1':a}")
    parser.add_argument('--val_transform_methods',help='Augmentation method for test set'
        ,default="{'1':a}")
    parser.add_argument('--batch_size','-b',metavar='N',help='batch size',type=int
        ,default=12)
    #Model
    parser.add_argument('--arch', '-a', metavar='ARCH',help='Net architecture'
        ,default='models.classification.vgg')
    parser.add_argument('--arch_parameters',help='Parameters for the net'
        ,default="{'test':None}")
    parser.add_argument('--pretrain_backbone',help='Whether to use pre-trained model'
        ,default='False')
    parser.add_argument('--pretrain_path',help='Path for pre-trained model'
        ,default='')
    #Training
    parser.add_argument('--train_mode',help='True as training,False as validation'
        ,default='True')
    parser.add_argument('--epochs',metavar='N',help='epoch',type=int
        ,default=100)
    parser.add_argument('--optimizer',help='Optimize method for loss function'
        ,default='customized')
    parser.add_argument('--optimizer_params',help="Parameters for optimizer"
        ,default="{'test':None}")
    parser.add_argument('--scheduler'
        ,default='CosineAnnealingLR')
    parser.add_argument('--scheduler_parameters'
        ,default="{'test':None}")
    parser.add_argument('--loss',help='loss function'
        ,default='customized')
    parser.add_argument('--early_stopping',metavar='N', help='early stopping (default: -1)',type=int
        ,default=-1)
    parser.add_argument('--best_metric',help='Use best_loss or best_oa for early stopping',choices=['loss','oa']
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
    config['log_path']=os.path.join(config['output_path'],'log.csv')
    config['model_loss_path']=os.path.join(config['output_path'],'model_loss.pth')
    config['model_oa_path']=os.path.join(config['output_path'],'model_oa.pth')
    config['fig_path']=os.path.join(config['output_path'],'result.jpg')
    
    config['input_channel_list']=eval(config['input_channel_list'])
    print(config['input_channel_list'])
    config['transform']=eval(config['transform'])
    config['pretrain_backbone']=eval(config['pretrain_backbone'])
    config['package_dirs']=eval(config['package_dirs'])
    config['train_transform_methods']=eval(config['train_transform_methods'])
    config['val_transform_methods']=eval(config['val_transform_methods'])
    config['train_mode']=eval(config['train_mode'])
    config['arch_parameters']=eval(config['arch_parameters'])
    config['optimizer_params']=eval(config['optimizer_params'])
    config['scheduler_parameters']=eval(config['scheduler_parameters'])
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

    from glob import glob
    def get_img_label(num_classes,img_train_dir,img_ext):
        '''
        get the paths and classes of the input images
        '''
        img_train_paths=[]
        label_train_ids=[]
        for i in range(num_classes):
            t=glob(os.path.join(img_train_dir, str(i),'*' + img_ext))
            print(os.path.join(img_train_dir, str(i),'*' + img_ext))
            img_train_paths=img_train_paths+t
            label_train_ids=label_train_ids+[i for j in range(len(t))]
        return img_train_paths,label_train_ids
    img_train_paths,label_train_ids=get_img_label(config['num_classes'],config['img_train_dir'],config['img_ext'])
    img_test_paths,label_test_ids=get_img_label(config['num_classes'],config['img_test_dir'],config['img_ext'])
    img_val_paths,label_val_ids=get_img_label(config['num_classes'],config['img_val_dir'],config['img_ext'])

    train_dataset = ClassificationDataset(
        img_paths=img_train_paths,
        label_ids=label_train_ids,
        img_ext=config['img_ext'],
        num_classes=config['num_classes'],
        input_channel_list=config['input_channel_list'],
        transform=config['transform'],
        transform_methods=config['train_transform_methods']
    )

    val_dataset = ClassificationDataset(
        img_paths=img_val_paths,
        label_ids=label_val_ids,
        img_ext=config['img_ext'],
        num_classes=config['num_classes'],
        input_channel_list=config['input_channel_list'],
        transform=config['transform'],
        transform_methods=config['val_transform_methods']
        )
    
    test_dataset = ClassificationDataset(
        img_paths=img_test_paths,
        label_ids=label_test_ids,
        img_ext=config['img_ext'],
        num_classes=config['num_classes'],
        input_channel_list=config['input_channel_list'],
        transform=config['transform'],
        transform_methods=config['val_transform_methods']
        )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
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
        print('use customized optimizer')
        config['len(train_dataset)']=len(train_dataset)
        optimizer=model.optimizer(config,**config['optimizer_params'])
    else:
        optimizer=getattr(importlib.import_module('torch.optim'), config['optimizer'])(model.trainable_parameters(),**config['optimizer_params'])
        
    # scheduler
    def get_scheduler(config):
        def create_lr_scheduler(optimizer,
                num_step: int,
                epochs: int,
                warmup=True,
                warmup_epochs=1,
                warmup_factor=1e-3):
            '''warmup'''
            assert num_step > 0 and epochs > 0
            if warmup is False:
                warmup_epochs = 0
            def f(x):
                if warmup is True and x <= (warmup_epochs * num_step):
                    alpha = float(x) / (warmup_epochs * num_step)
                    return warmup_factor * (1 - alpha) + alpha
                else:
                    return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
        if config['scheduler']=='warmup':
            scheduler=create_lr_scheduler(optimizer, len(train_loader), config['epochs'])
        elif config['scheduler']=='None':
            class IdentityLR(torch.optim.lr_scheduler._LRScheduler):
                def __init__(self, optimizer, last_epoch=-1, verbose=False):
                    super().__init__(optimizer, last_epoch, verbose)
                def get_lr(self):
                    return [base_lr for base_lr in self.base_lrs]
            scheduler = IdentityLR(optimizer)
        else:
            scheduler=getattr(importlib.import_module('torch.optim.lr_scheduler'), config['scheduler'])(optimizer,**config['scheduler_parameters'])
        return scheduler

    #loss function
    if config['loss'] == 'customized':
        print('use customized loss function')
        criterion =None
    else:
        try:
            criterion = getattr(importlib.import_module('losses'), config['loss'])()
        except:
            criterion = getattr(importlib.import_module('losses'), config['loss'])
        try:
            criterion=criterion.to(config['device'])
        except:
            pass
            

    from tqdm import tqdm
    def train(config, train_loader, model, criterion, optimizer):
        avg_meters = {'loss': AverageMeter()}
        acc_total = ClassificationMetric(numClass=config['num_classes'], device=config['device'])
        model.train()
        pbar = tqdm(total=len(train_loader))
        for step,item in enumerate(train_loader):
            inputs=item['img'].to(config['device'], non_blocking =True)
            target=item['label'].to(config['device'], non_blocking =True)
            optimizer.zero_grad()
            if criterion==None:#use customized loss function
                output,loss=model.loss(inputs,target,**config)
            else:
                output = model(inputs)
                loss = criterion(output['out'],target)
            loss.backward()
            optimizer.step()
            temp_output=output['out'].argmax(axis=1)
            temp_target=target.detach()
            if temp_output.shape != target.shape:
                temp_target=temp_target.argmax(axis=1)
            acc_total.addBatch(temp_output,temp_target)
            avg_meters['loss'].update(loss.item(), inputs.size(0))

            oa = acc_total.OverallAccuracy()
            f1 = acc_total.F1score()
            postfix = OrderedDict([('loss', avg_meters['loss'].avg),('oa',oa.cpu().numpy()),('F1',f1.cpu())])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()
        return postfix

    def validate(config, val_loader, model, criterion):
        avg_meters = {'loss': AverageMeter()}
        acc_total = ClassificationMetric(numClass=config['num_classes'], device=config['device'])
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
                temp_output=output['out'].argmax(axis=1)
                temp_target=target.detach()
                if temp_output.shape != target.shape:
                    temp_target=temp_target.argmax(axis=1)
                acc_total.addBatch(temp_output,temp_target)
                avg_meters['loss'].update(loss.item(), inputs.size(0))
                oa = acc_total.OverallAccuracy()
                f1 = acc_total.F1score()
                postfix = OrderedDict([('loss', avg_meters['loss'].avg),('oa',oa.cpu().numpy()),('F1',f1.cpu())])
                pbar.set_postfix(postfix)
                pbar.update(1)
            pbar.close()
        return postfix

    if config['train_mode']==False:
        val_log = validate(config, test_loader, model, criterion)
        print('val_oa %.4f | val_loss %.4f'% (val_log['oa'], val_log['loss']))
    else:
        log = OrderedDict([
            ('epoch', []),
            ('lr', []),
            ('oa',[]),
            ('val_oa',[]),
            ('loss', []),
            ('val_loss', [])
        ])
        print('start training %s' % config['name'])
        best_loss=100
        best_oa=0
        trigger = 0#for early stopping
        start_time = time.time()
        model.train()
        for epoch in range(config['epochs']):
            print('Epoch [%d/%d], lr: %.6f' % (epoch, config['epochs'],optimizer.param_groups[0]['lr']))
            train_log = train(config,train_loader, model, criterion, optimizer)
            val_log = validate(config, val_loader, model, criterion)
            print('oa %.4f - val_oa %.4f | loss %.4f - val_loss %.4f'% (train_log['oa'],val_log['oa'],train_log['loss'], val_log['loss']))
            log['epoch'].append(epoch)
            log['lr'].append(optimizer.param_groups[0]['lr'])
            log['oa'].append(train_log['oa'])
            log['val_oa'].append(val_log['oa'])
            log['loss'].append(train_log['loss'])
            log['val_loss'].append(val_log['loss'])
            pd.DataFrame(log).to_csv(config['log_path'], index=False)
    
            trigger += 1
            
            if val_log['loss'] < best_loss:
                save_file = {"model": model.state_dict()
                        ,"state_dict":model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                        ,"optimizer": optimizer.state_dict()
                        ,"epoch": epoch
                        ,"args": config
                        ,'best_loss':best_loss
                            }
                torch.save(save_file, config['model_loss_path'])
                best_loss = val_log['loss']
                print("=> saved best loss model")
                if config['best_metric']=='loss':
                    trigger = 0                
            if val_log['oa'] > best_oa:
                save_file = {"model": model.state_dict()
                        ,"state_dict":model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                        ,"optimizer": optimizer.state_dict()
                        ,"epoch": epoch
                        ,"args": config
                        ,'best_oa':best_oa
                            }
                torch.save(save_file, config['model_oa_path'])
                best_oa = val_log['oa']
                print("=> saved best oa model")
                if config['best_metric']=='oa':
                    trigger = 0    
    
            # early stopping
            if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
                print("=> early stopping")
                break
    
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
        plt.plot(data['epoch'],data['oa'],label='training_oa')
        plt.plot(data['epoch'],data['val_oa'],label='testing_oa')
        plt.plot(data['epoch'],data['loss'],label='training_loss')
        plt.plot(data['epoch'],data['val_loss'],label='testing_loss')
        plt.legend()
        fig.savefig(config['fig_path'])
        fig.show()

