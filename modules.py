import torch
from torch import nn,optim
import torch.nn.functional as F
import torchvis
from matplotlib import cm
import mne
import numpy as np
from envelope import envelop
from torch.nn import init
import os
from visdom import Visdom
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score
import scipy.signal as Signal
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
from statsmodels.tsa.ar_model import AR
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import cv2
# import warnings
# warnings.filterwarnings('ignore', 'statsmodels.tsa.ar_model.AR', FutureWarning)


#have to be deleted after
def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data,0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0,0.01)
        m.bias.data.zero_()

def compute_RF(kernels,strides):
    """

    :param kernels:list, deeper layers place first
    :param strides: list, deeper layers place first
    :return: last rf
    """
    RF=1
    RFs=[]
    for i,k in enumerate(kernels):
        RF = (RF-1)*strides[i]+k
        RFs.append(RF)
    return RF

def band_pass_EEGdata(eegdata,lf,hf,fs):
    """

    :param eegdata:should be np.ndarray  shape:(chans,time_steps)
    :param lf: low band pass frequency
    :param hf: hign band pass frequency
    :param fs: sample frequency
    :return: band pass filtered eegdata
    """
    sos = Signal.cheby2(12, 20, [lf, hf], 'bp', fs=fs, output='sos')
    filtered = Signal.sosfilt(sos, eegdata)
    return filtered

class FBCSPNet(nn.Module):

    def __init__(self,in_chans,time_steps,classes,env=None,linear_init_std=0.1,eps=1e-3,pace_1_ratio = 0.05,fs=250):
        #hyperparameters
        self.eps = eps
        self.time_steps = time_steps
        # hyperparameters
        self.pace_1_ratio = pace_1_ratio
        self.pace_1 = round(self.time_steps * self.pace_1_ratio)

        self.env = env
        self.s1 = 1
        self.s2 = 1
        # hyperparameters
        self.linear_init_std = linear_init_std
        self.in_chans = in_chans

        self.classes=classes
        super(FBCSPNet, self).__init__()

        self.n_filters_time = 25
        self.n_filters_spat = 25
        self.pool_kernel_ratio = 0.25
        self.pool_kernel = round(self.time_steps*self.pool_kernel_ratio)
        self.pool_stride_ratio = 0.05
        self.pool_stride = round(self.time_steps*self.pool_stride_ratio)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=self.n_filters_time,kernel_size=(1,self.pace_1),stride=self.s1,padding=0))
        self.sconv = nn.Sequential(nn.Conv2d(in_channels=self.n_filters_time,out_channels=self.n_filters_spat,kernel_size=(self.in_chans,1),stride=self.s2)
                                   ,nn.BatchNorm2d(num_features=self.n_filters_spat))

        self.maxpool = nn.MaxPool2d(kernel_size=(1,self.pool_kernel),stride=(1,self.pool_stride))

        dummy = torch.Tensor(1,1,self.in_chans,self.time_steps)
        dummy = self.maxpool(self.sconv(self.conv1(dummy)))
        # input_dim =round (self.n_filters_spat * (self.time_steps-(self.pace_1-1)-(self.pool_kernel-1))/self.pool_stride )
        input_dim = dummy.shape[1]*dummy.shape[3]

        self.linear = nn.Sequential(nn.Linear(input_dim,round(input_dim*0.4)),nn.Linear(round(input_dim*0.4),self.classes))

        self.kernels = [1,self.pace_1]
        self.strides = [self.s2,self.s1]
        self.fs = fs

    def weigth_init(self,m):
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            init.constant_(m.bias.data, 0.1)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, self.linear_init_std)
            m.bias.data.zero_()

    def get_sconv_kernel_input_feature(self,x,channel,fl,fh):

        if x.dim() != 4:
            x = torch.unsqueeze(x,dim=1)
        if x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)
        x = self.conv1(x)
        features = self.sconv(x)

        kernels = self.kernels[:]
        strides = self.strides[:]
        RF = compute_RF(kernels=kernels,strides=strides)
        x_ch = x[:][0][0][channel]
        x_ch = band_pass_EEGdata(x_ch.detach().numpy(),fl,fh,self.fs)
        x_ens=[]
        r = 0
        for i in range(features.shape[-1]):
            x_en = x_ch[r:r+RF]
            en = envelop(x_en).mean()
            x_ens.append(en)
            r += self.s1
        x_ens = np.hstack([x for x in x_ens])
        x_ens = np.expand_dims(x_ens,axis=0)
        return x_ens

    def plot_sconv_kernel_input_features(self,datasets,channel,fq=[(7,13),(13,31),(31,100)],plot=True,save_name=None):
        dl = DataLoader(datasets,1)
        (x, y) = next(iter(dl))
        inputs= []
        for i,(fl,fh) in enumerate(fq):
            input = self.get_sconv_kernel_input_feature(x=x,channel=channel,fl=fl,fh=fh)
            inputs.append(input)
        inputs = np.vstack([x for x in inputs])
        if plot==True:
            f, ax1 = plt.subplots(figsize=(6, 6), nrows=1)
            sns.heatmap(inputs, annot=True, ax=ax1,cmap=cm.coolwarm)
            ax1.set_yticklabels([str(l)+'-'+str(h)+'Hz' for (l,h) in fq])
        if save_name!=None:
            plt.savefig(save_name)
        return inputs

    def get_sconv_kernel_unit_output(self,x,filter_idx):
        if x.dim() != 4:
            x = torch.unsqueeze(x,dim=1)
        if x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)
        x = self.conv1(x)
        features = self.sconv(x)
        return features[:][0][filter_idx].detach().numpy()

    def plot_sconv_kernel_unit_output(self,datasets,filter_idxs=[0,1,2],plot=True,save_name=None):
        dl = DataLoader(datasets,1)
        (x, y) = next(iter(dl))
        outputs= []
        for i,filter_idx in enumerate(filter_idxs):
            output = self.get_sconv_kernel_unit_output(x,filter_idx)
            outputs.append(output)
        outputs = np.vstack([x for x in outputs])
        if plot==True:
            f, ax1 = plt.subplots(figsize=(6, 6), nrows=1)
            sns.heatmap(outputs, annot=True, ax=ax1,cmap=cm.coolwarm)
            ax1.set_yticklabels(['filter'+str(i) for i in filter_idxs])
        if save_name!=None:
            plt.savefig(save_name)
        return outputs

    def plot_cor_map(self,datasets,chan,fq=[(1,4),(4,8),(8,20)],filter_idxs=[0,1,2,3],save_name=None,plot=True):
        plot = plot
        inputs = self.plot_sconv_kernel_input_features(datasets=datasets,fq=fq,channel=chan,plot=False)
        outputs = self.plot_sconv_kernel_unit_output(datasets=datasets,filter_idxs=filter_idxs,plot=False)
        inputs = (inputs-inputs.mean())/inputs.std()
        outputs = (outputs-outputs.mean())/outputs.std()
        cor = np.cov(inputs,outputs,rowvar=True)[0:len(inputs),len(outputs):]
        if plot == True:
            f, ax1 = plt.subplots(figsize=(6, 6), nrows=1)
            sns.heatmap(cor, annot=True, ax=ax1,cmap=cm.coolwarm)
            ax1.set_yticklabels([str(l)+'-'+str(h)+'Hz' for (l,h) in fq])
            ax1.set_xticklabels(['filter' + str(i) for i in filter_idxs])
        if save_name!=None:
            plt.savefig(save_name)
        return cor

    def plot_cor_map_mean_filter(self,datasets,chan,fq=[(1,4),(4,8),(8,20)],filter_idxs=[0,1,2,3],save_name=None,plot=True):
        inputs = self.plot_sconv_kernel_input_features(datasets=datasets,fq=fq,channel=chan,plot=False)
        outputs = self.plot_sconv_kernel_unit_output(datasets=datasets,filter_idxs=filter_idxs,plot=False)
        inputs = (inputs-inputs.mean())/inputs.std()
        outputs = (outputs-outputs.mean())/outputs.std()
        cor = np.cov(inputs,outputs,rowvar=True)[0:len(inputs),len(outputs):]
        cor_mean = cor.mean(axis=0)
        cor_mean = np.expand_dims(cor_mean,axis=1)
        if plot==True:
            f, ax1 = plt.subplots(figsize=(6, 6), nrows=1)
            sns.heatmap(cor_mean, annot=True, ax=ax1,cmap=cm.coolwarm)
            ax1.set_yticklabels([str(l)+'-'+str(h)+'Hz' for (l,h) in fq])
        if save_name!=None:
            plt.savefig(save_name)
        return cor_mean


    def preprocess(self,x):
        x = (x - x.mean())/x.std()
        return x

    def forward(self, x):
        """
            x:input:[batch_size,1,in_chans,time_steps]
            conv1_kernel:(1,pace_1)
            conv1_output:[batch_size,n_filters_time,in_chans,time_steps-(pace_1-1)]
            sconv_kernel:(in_chans,1)
            sconv_output:[batch_size,n_filters_spat,1,time_steps-(pace_1-1)]
            maxpool_kernel:(1,pool_kernel)
            maxpool_stride:(1,pool_stride)
            maxpool_output:[batch_size,n_filters_spat,1,(time_steps-(pace_1-1)-(pool_kernel-1))/pool_stride)]
            linear_input:n_filters_spat*(time_steps-(pace_1-1)-(pool_kernel-1))/pool_stride)
            linear_output:classes
        """
        if x.dim() != 4:
            x = torch.unsqueeze(x,dim=1)
        if x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)
        batch_size = x.shape[0]
        x = self.preprocess(x)
        x = self.conv1(x)
        x = self.sconv(x)
        x = torch.mul(x,x)
        x = self.maxpool(x)
        x = torch.log(x)
        x = x.view(batch_size,-1)
        x = self.linear(x)
        x = F.softmax(x,dim=1)
        return x

    def predict(self, dataset):
        dl = DataLoader(dataset, len(dataset))
        for idx, (x, y) in enumerate(dl):
            if type(x) == type(np.array((1))):
                x = torch.Tensor(x)
            logits = self.forward(x)
            if idx == 0:
                pred = logits.argmax(dim=1)
            else:
                pred = torch.cat((pred, logits.argmax(dim=1)), 0)
        return pred

    def transform(self, dataset):
        dl = DataLoader(dataset, len(dataset))
        for idx, (x, y) in enumerate(dl):
            if type(x) == type(np.array((1))):
                x = torch.Tensor(x)
            logits = self.forward(x)
            if idx == 0:
                out = logits
            else:
                out = torch.cat((out, logits), 0)
        return out

    def graph(self):
        dummy = torch.Tensor(2,1,25,176)
        model = FBCSPNet(in_chans=25,time_steps=176,classes=4)
        ans = model(dummy)

        vis_graph = torchvis.make_dot(model(dummy), params=dict(model.named_parameters()))
        vis_graph.view()

    def evaluate_train(self, test_dataset, batch_size):
        self.eval()
        with torch.no_grad():
            #     training accuracy
            total_num = 0
            total_correct = 0
            total_loop = 0
            total_kappa = 0
            data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            for (x, y) in data_loader:
                logits = self.forward(x)
                pred = logits.argmax(dim=1)

                y = (y - 1).to(dtype=torch.long)
                kappa = cohen_kappa_score(y, pred)

                total_correct += torch.eq(pred, y).float().sum()
                total_num += int(x.shape[0])
                total_loop += 1
                total_kappa += kappa

            accuracy = total_correct / total_num
            kappa = total_kappa / total_loop
            return accuracy, kappa

    def fit(self,train_dataset, batch_size, max_epoch, valid_dataset=None, test_dataset=None, log_path=None ,reg =None,save_name='fbcspnet.pth'):
        """
        fit the model to the dataset
        :param dataset:torch.utils.data.Dateset
        :param model: torch.nn.module
        :return: model
        """
        if log_path == None:
            log_path = os.getcwd()
        if self.env==None:
            viz = Visdom()
        else:
            viz = Visdom(env=self.env)
        viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
        viz.line([0.], [0.], win='train_acc', opts=dict(title='train accuracy'))
        viz.line([0.], [0.], win='test_acc', opts=dict(title='test accuracy'))
        viz.line([0.], [0.], win='train_kappa', opts=dict(title='train kappa value'))
        max_epoch = max_epoch
        batch_size = batch_size
        optimizer = optim.Adam(self.parameters(),weight_decay=reg)
        self.apply(self.weigth_init)
        # if reg == None:
        criteria = nn.NLLLoss()
        best_acc = -1
        # 定义学习率策略
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100,
                                                               verbose=True,
                                                               threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                               min_lr=0,
                                                               eps=self.eps)
        dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        global_step = 0
        for epoch in range(max_epoch):

            # train a epoch
            for idx_batch, (x, y) in enumerate(dl):
                logits = self.forward(x)
                y = (y - 1).to(dtype=torch.long)
                loss = criteria(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step(loss)

                global_step += 1

                viz.line([loss.item()], [global_step], win='train_loss', update='append')

            # test trainning acuracy
            train_acc, kappa = self.evaluate_train(test_dataset=train_dataset, batch_size=batch_size)
            test_acc, _ = self.evaluate_train(test_dataset=test_dataset, batch_size=batch_size)
            viz.line([train_acc.item()], [global_step], win='train_acc', update='append')
            viz.line([test_acc.item()], [global_step], win='test_acc', update='append')
            viz.line([kappa.item()], [global_step], win='train_kappa', update='append')

            # save model
            if test_acc.item() > best_acc:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, 'ckpt_fbcspnet.mdl')
                best_acc = test_acc
                torch.save(self,save_name)
            self.train()

class FBCSPNet_for_show(nn.Module):

    def __init__(self,in_chans,time_steps,classes,env=None,linear_init_std=0.1,eps=1e-3,pace_1_ratio = 0.05,fs=250):
        #hyperparameters
        self.eps = eps
        self.time_steps = time_steps
        # hyperparameters
        self.pace_1_ratio = pace_1_ratio
        self.pace_1 = round(self.time_steps * self.pace_1_ratio)

        self.env = env
        self.s1 = 1
        self.s2 = 1
        # hyperparameters
        self.linear_init_std = linear_init_std
        self.in_chans = in_chans

        self.classes=classes
        super(FBCSPNet_for_show, self).__init__()

        self.n_filters_time = 25
        self.n_filters_spat = 25
        self.pool_kernel_ratio = 0.25
        self.pool_kernel = round(self.time_steps*self.pool_kernel_ratio)
        self.pool_stride_ratio = 0.05
        self.pool_stride = round(self.time_steps*self.pool_stride_ratio)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=self.n_filters_time,kernel_size=(1,self.pace_1),stride=self.s1,padding=0))
        self.sconv = nn.Sequential(nn.Conv2d(in_channels=self.n_filters_time,out_channels=self.n_filters_spat,kernel_size=(self.in_chans,1),stride=self.s2)
                                   ,nn.BatchNorm2d(num_features=self.n_filters_spat))

        self.maxpool = nn.MaxPool2d(kernel_size=(1,self.pool_kernel),stride=(1,self.pool_stride))

        dummy = torch.Tensor(1,1,self.in_chans,self.time_steps)
        dummy = self.maxpool(self.sconv(self.conv1(dummy)))
        # input_dim =round (self.n_filters_spat * (self.time_steps-(self.pace_1-1)-(self.pool_kernel-1))/self.pool_stride )
        input_dim = dummy.shape[1]*dummy.shape[3]

        self.linear = nn.Sequential(nn.Linear(input_dim,round(input_dim*0.4)),nn.Linear(round(input_dim*0.4),self.classes))

        self.kernels = [1,self.pace_1]
        self.strides = [self.s2,self.s1]
        self.fs = fs

    def weigth_init(self,m):
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            init.constant_(m.bias.data, 0.1)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, self.linear_init_std)
            m.bias.data.zero_()

    def get_sconv_kernel_input_feature(self,x,channel,fl,fh):

        if x.dim() != 4:
            x = torch.unsqueeze(x,dim=1)
        if x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)
        x = self.conv1(x)
        features = self.sconv(x)

        kernels = self.kernels[:]
        strides = self.strides[:]
        RF = compute_RF(kernels=kernels,strides=strides)
        x_ch = x[:][0][0][channel]
        x_ch = band_pass_EEGdata(x_ch.detach().numpy(),fl,fh,self.fs)
        x_ens=[]
        r = 0
        for i in range(features.shape[-1]):
            x_en = x_ch[r:r+RF]
            en = envelop(x_en).mean()
            x_ens.append(en)
            r += self.s1
        x_ens = np.hstack([x for x in x_ens])
        x_ens = np.expand_dims(x_ens,axis=0)
        return x_ens

    def plot_sconv_kernel_input_features(self,datasets,channel,fq=[(7,13),(13,31),(31,100)],plot=True,save_name=None):
        dl = DataLoader(datasets,1)
        (x, y) = next(iter(dl))
        inputs= []
        for i,(fl,fh) in enumerate(fq):
            input = self.get_sconv_kernel_input_feature(x=x,channel=channel,fl=fl,fh=fh)
            inputs.append(input)
        inputs = np.vstack([x for x in inputs])
        if plot==True:
            f, ax1 = plt.subplots(figsize=(6, 6), nrows=1)
            sns.heatmap(inputs, annot=True, ax=ax1,cmap=cm.coolwarm)
            ax1.set_yticklabels([str(l)+'-'+str(h)+'Hz' for (l,h) in fq])
        if save_name!=None:
            plt.savefig(save_name)
        return inputs

    def get_sconv_kernel_unit_output(self,x,filter_idx):
        if x.dim() != 4:
            x = torch.unsqueeze(x,dim=1)
        if x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)
        x = self.conv1(x)
        features = self.sconv(x)
        return features[:][0][filter_idx].detach().numpy()

    def plot_sconv_kernel_unit_output(self,datasets,filter_idxs=[0,1,2],plot=True,save_name=None):
        dl = DataLoader(datasets,1)
        (x, y) = next(iter(dl))
        outputs= []
        for i,filter_idx in enumerate(filter_idxs):
            output = self.get_sconv_kernel_unit_output(x,filter_idx)
            outputs.append(output)
        outputs = np.vstack([x for x in outputs])
        if plot==True:
            f, ax1 = plt.subplots(figsize=(6, 6), nrows=1)
            sns.heatmap(outputs, annot=True, ax=ax1,cmap=cm.coolwarm)
            ax1.set_yticklabels(['filter'+str(i) for i in filter_idxs])
        if save_name!=None:
            plt.savefig(save_name)
        return outputs

    def plot_cor_map(self,datasets,chan,fq=[(1,4),(4,8),(8,20)],filter_idxs=[0,1,2,3],save_name=None,plot=True):
        plot = plot
        inputs = self.plot_sconv_kernel_input_features(datasets=datasets,fq=fq,channel=chan,plot=False)
        outputs = self.plot_sconv_kernel_unit_output(datasets=datasets,filter_idxs=filter_idxs,plot=False)
        inputs = (inputs-inputs.mean())/inputs.std()
        outputs = (outputs-outputs.mean())/outputs.std()
        cor = np.cov(inputs,outputs,rowvar=True)[0:len(inputs),len(outputs):]
        if plot == True:
            f, ax1 = plt.subplots(figsize=(6, 6), nrows=1)
            sns.heatmap(cor, annot=True, ax=ax1,cmap=cm.coolwarm)
            ax1.set_yticklabels([str(l)+'-'+str(h)+'Hz' for (l,h) in fq])
            ax1.set_xticklabels(['filter' + str(i) for i in filter_idxs])
        if save_name!=None:
            plt.savefig(save_name)
        return cor

    def plot_cor_map_mean_filter(self,datasets,chan,fq=[(1,4),(4,8),(8,20)],filter_idxs=[0,1,2,3],save_name=None,plot=True):
        inputs = self.plot_sconv_kernel_input_features(datasets=datasets,fq=fq,channel=chan,plot=False)
        outputs = self.plot_sconv_kernel_unit_output(datasets=datasets,filter_idxs=filter_idxs,plot=False)
        inputs = (inputs-inputs.mean())/inputs.std()
        outputs = (outputs-outputs.mean())/outputs.std()
        cor = np.cov(inputs,outputs,rowvar=True)[0:len(inputs),len(outputs):]
        cor_mean = cor.mean(axis=0)
        cor_mean = np.expand_dims(cor_mean,axis=1)
        if plot==True:
            f, ax1 = plt.subplots(figsize=(6, 6), nrows=1)
            sns.heatmap(cor_mean, annot=True, ax=ax1,cmap=cm.coolwarm)
            ax1.set_yticklabels([str(l)+'-'+str(h)+'Hz' for (l,h) in fq])
        if save_name!=None:
            plt.savefig(save_name)
        return cor_mean


    def preprocess(self,x):
        x = (x - x.mean())/x.std()
        return x

    def forward(self, x):
        """
            x:input:[batch_size,1,in_chans,time_steps]
            conv1_kernel:(1,pace_1)
            conv1_output:[batch_size,n_filters_time,in_chans,time_steps-(pace_1-1)]
            sconv_kernel:(in_chans,1)
            sconv_output:[batch_size,n_filters_spat,1,time_steps-(pace_1-1)]
            maxpool_kernel:(1,pool_kernel)
            maxpool_stride:(1,pool_stride)
            maxpool_output:[batch_size,n_filters_spat,1,(time_steps-(pace_1-1)-(pool_kernel-1))/pool_stride)]
            linear_input:n_filters_spat*(time_steps-(pace_1-1)-(pool_kernel-1))/pool_stride)
            linear_output:classes
        """
        if x.dim() != 4:
            x = torch.unsqueeze(x,dim=1)
        if x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)
        batch_size = x.shape[0]
        x = self.preprocess(x)
        x = self.conv1(x)
        x = self.sconv(x)
        x = torch.mul(x,x)
        x = self.maxpool(x)
        x = torch.log(x)
        x = x.view(batch_size,-1)
        x = self.linear(x)
        x = F.softmax(x,dim=1)
        return x

    def predict(self, dataset):
        dl = DataLoader(dataset, len(dataset))
        for idx, (x, y) in enumerate(dl):
            if type(x) == type(np.array((1))):
                x = torch.Tensor(x)
            logits = self.forward(x)
            if idx == 0:
                pred = logits.argmax(dim=1)
            else:
                pred = torch.cat((pred, logits.argmax(dim=1)), 0)
        return pred

    def transform(self, dataset):
        dl = DataLoader(dataset, len(dataset))
        for idx, (x, y) in enumerate(dl):
            if type(x) == type(np.array((1))):
                x = torch.Tensor(x)
            logits = self.forward(x)
            if idx == 0:
                out = logits
            else:
                out = torch.cat((out, logits), 0)
        return out

    def graph(self):
        dummy = torch.Tensor(2,1,25,176)
        model = FBCSPNet(in_chans=25,time_steps=176,classes=4)
        ans = model(dummy)

        vis_graph = torchvis.make_dot(model(dummy), params=dict(model.named_parameters()))
        vis_graph.view()

    def evaluate_train(self, test_dataset, batch_size):
        self.eval()
        with torch.no_grad():
            #     training accuracy
            total_num = 0
            total_correct = 0
            total_loop = 0
            total_kappa = 0
            data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            for (x, y) in data_loader:
                logits = self.forward(x)
                pred = logits.argmax(dim=1)

                y = (y - 1).to(dtype=torch.long)
                kappa = cohen_kappa_score(y, pred)

                total_correct += torch.eq(pred, y).float().sum()
                total_num += int(x.shape[0])
                total_loop += 1
                total_kappa += kappa

            accuracy = total_correct / total_num
            kappa = total_kappa / total_loop
            return accuracy, kappa

    def get_change_image(self,datasets,pos):
        in_chan = self.in_chans
        time_steps = self.time_steps
        cor_v_scalp = []
        for ch in range(in_chan):
            value = self.plot_cor_map_mean_filter(datasets=datasets, chan=ch, plot=False)
            cor_v_scalp.append(value)
        cors_bands_scalp_0 = np.hstack([x for x in cor_v_scalp])
        cor_v_scalp = []
        for ch in range(in_chan):
            value = self.plot_cor_map_mean_filter(datasets=datasets, chan=ch, plot=False)
            cor_v_scalp.append(value)
        cors_bands_scalp = np.hstack([x for x in cor_v_scalp])
        vs = cors_bands_scalp - cors_bands_scalp_0
        fig, ax = plt.subplots(3, 1)
        fqs = ['7-13Hz', '13-30Hz', '30-100Hz']
        for i, v in enumerate(vs):
            max_v = np.max(v)
            mne.viz.plot_topomap(v, pos,
                                 vmin=-max_v, vmax=max_v, contours=0,
                                 cmap=cm.Reds, axes=ax[i], show=True)
            ax[i].set_title(fqs[i])
        save_name = 'secret.jpg'
        fig.savefig(save_name)
        img = cv2.imread(save_name)
        return img.transpose(2, 0, 1)[::-1,...]



    def fit(self,train_dataset, batch_size, max_epoch, pos,valid_dataset=None, test_dataset=None, log_path=None ,reg =None):
        """
        fit the model to the dataset
        :param dataset:torch.utils.data.Dateset
        :param model: torch.nn.module
        :return: model
        """
        if log_path == None:
            log_path = os.getcwd()
        if self.env==None:
            viz = Visdom()
        else:
            viz = Visdom(env=self.env)
        viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
        viz.line([0.], [0.], win='train_acc', opts=dict(title='train accuracy'))
        viz.line([0.], [0.], win='test_acc', opts=dict(title='test accuracy'))
        viz.line([0.], [0.], win='train_kappa', opts=dict(title='train kappa value'))
        max_epoch = max_epoch
        batch_size = batch_size
        optimizer = optim.Adam(self.parameters(),weight_decay=reg)
        self.apply(self.weigth_init)
        # if reg == None:
        criteria = nn.NLLLoss()
        best_acc = -1
        # 定义学习率策略
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100,
                                                               verbose=True,
                                                               threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                               min_lr=0,
                                                               eps=self.eps)
        dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        global_step = 0
        for epoch in range(max_epoch):

            # train a epoch
            for idx_batch, (x, y) in enumerate(dl):
                logits = self.forward(x)
                y = (y - 1).to(dtype=torch.long)
                loss = criteria(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step(loss)

                global_step += 1

                viz.line([loss.item()], [global_step], win='train_loss', update='append')

            # test trainning acuracy
            train_acc, kappa = self.evaluate_train(test_dataset=train_dataset, batch_size=batch_size)
            test_acc, _ = self.evaluate_train(test_dataset=test_dataset, batch_size=batch_size)
            viz.line([train_acc.item()], [global_step], win='train_acc', update='append')
            viz.line([test_acc.item()], [global_step], win='test_acc', update='append')
            viz.line([kappa.item()], [global_step], win='train_kappa', update='append')

            #visualize topo-scalp
            img = self.get_change_image(datasets=train_dataset,pos=pos)
            viz.image(img=img,win='visualization')


            self.train()

def plot_definetion():
    from BCICdatasets import BCIC4_2a
    from matplotlib import cm
    datasets = BCIC4_2a(uv=True)
    datasets.load_data()
    in_chan = datasets.signal_shape()[0]
    cor_v_scalp = []
    savelog = 'vis_demo'
    for ch in range(in_chan):
        in_chan = datasets.signal_shape()[0]
        time_steps = datasets.signal_shape()[1]
        model = FBCSPNet(in_chans=in_chan,time_steps=time_steps,classes=datasets.n_classes)
        # model.plot_sconv_kernel_unit_output(datasets=datasets,save_name=os.path.join(savelog,str(ch)+'_unit_output.jpg'))
        # model.plot_sconv_kernel_input_features(datasets=datasets,channel=ch,save_name=os.path.join(savelog,str(ch)+'_input_feature.jpg'))
        # model.plot_cor_map(datasets=datasets,chan=ch,save_name=os.path.join(savelog,str(ch)+'_cor_map.jpg'))
        value = model.plot_cor_map_mean_filter(datasets=datasets,chan=ch)
        cor_v_scalp.append(value)
    cors_bands_scalp = np.hstack([x for x in cor_v_scalp])
    from scalp import ax_scalp
    v = cors_bands_scalp[0]
    # ax_scalp(v,datasets.names,dataset='BCIC42a',save_name=os.path.join(savelog,'scalp'),colormap=cm.coolwarm)
    import mne
    max_v = np.max(v)
    import scalp
    positions = [x for x in scalp.BCIC4_2a_scalp_position.values()]
    positions = np.array(positions)*0.10
    fig, ax = plt.subplots(1, 1)
    mne.viz.plot_topomap(v, positions,
                         vmin=-max_v, vmax=max_v, contours=0,
                        cmap=cm.coolwarm, axes=ax, show=False);



def plot_33a_change():
    from BCICdatasets import BCIC3_3a
    from matplotlib import cm
    import mne
    import scalp
    datasets = BCIC3_3a(uv=True)
    datasets.load_data()
    from BCI_database import Subject_dataset
    sub_data = Subject_dataset(datasets,0)
    in_chan = datasets.signal_shape()[0]
    time_steps = datasets.signal_shape()[1]
    cor_v_scalp = []
    savelog = 'vis_demo'
    model = FBCSPNet(in_chans=in_chan, time_steps=time_steps, classes=datasets.n_classes)
    for ch in range(in_chan):
        # model.plot_cor_map(datasets=datasets,chan=ch,save_name=os.path.join(savelog,str(ch)+'_cor_map.jpg'),plot=False)
        # value = model.plot_cor_map_mean_filter(datasets=datasets,chan=ch,plot=False)
        # model_e = torch.load(r'G:\undergraduate\MIdecode\multiclass_classification\BCIC3_3a\subject0_fbcspnet.pth')
        # model_e.plot_cor_map(datasets=datasets, chan=ch, save_name=os.path.join(savelog, str(ch) + '_cor_map.jpg'),
        #                    plot=False)
        value = model.plot_cor_map_mean_filter(datasets=datasets, chan=ch, plot=False)
        cor_v_scalp.append(value)
    cors_bands_scalp_0 = np.hstack([x for x in cor_v_scalp])

    n_iter = 10
    for j in range(n_iter):
        cor_v_scalp = []
        for ch in range(in_chan):
            # model.plot_cor_map(datasets=datasets,chan=ch,save_name=os.path.join(savelog,str(ch)+'_cor_map.jpg'),plot=False)
            # value = model.plot_cor_map_mean_filter(datasets=datasets,chan=ch,plot=False)
            # model_e = torch.load(r'G:\undergraduate\MIdecode\multiclass_classification\BCIC3_3a\subject0_fbcspnet.pth')
            # model_e.plot_cor_map(datasets=datasets, chan=ch, save_name=os.path.join(savelog, str(ch) + '_cor_map.jpg'),
            #                    plot=False)
            value = model.plot_cor_map_mean_filter(datasets=datasets, chan=ch, plot=False)
            cor_v_scalp.append(value)
        cors_bands_scalp = np.hstack([x for x in cor_v_scalp])
        vs = cors_bands_scalp - cors_bands_scalp_0
        fig, ax = plt.subplots(3, 1)
        fqs = ['7-13Hz', '13-30Hz', '30-100Hz']
        for i, v in enumerate(vs):
            max_v = np.max(v)
            mne.viz.plot_topomap(v, datasets.positions,
                                 vmin=-max_v, vmax=max_v, contours=0,
                                 cmap=cm.Reds, axes=ax[i], show=True)
            ax[i].set_title(fqs[i])

        model.fit(train_dataset=sub_data,test_dataset=sub_data,batch_size=16,max_epoch=1,reg=1)
def plot_42a_change():
    from BCICdatasets import BCIC4_2a
    from matplotlib import cm
    import mne
    import scalp
    datasets = BCIC4_2a(uv=True)
    datasets.load_data()
    from BCI_database import Subject_dataset
    sub_data = Subject_dataset(datasets,0)
    in_chan = datasets.signal_shape()[0]
    time_steps = datasets.signal_shape()[1]
    cor_v_scalp = []
    savelog = 'vis_demo'
    model = FBCSPNet(in_chans=in_chan, time_steps=time_steps, classes=datasets.n_classes)
    for ch in range(in_chan):
        # model.plot_cor_map(datasets=datasets,chan=ch,save_name=os.path.join(savelog,str(ch)+'_cor_map.jpg'),plot=False)
        # value = model.plot_cor_map_mean_filter(datasets=datasets,chan=ch,plot=False)
        # model_e = torch.load(r'G:\undergraduate\MIdecode\multiclass_classification\BCIC3_3a\subject0_fbcspnet.pth')
        # model_e.plot_cor_map(datasets=datasets, chan=ch, save_name=os.path.join(savelog, str(ch) + '_cor_map.jpg'),
        #                    plot=False)
        value = model.plot_cor_map_mean_filter(datasets=datasets, chan=ch, plot=False)
        cor_v_scalp.append(value)
    cors_bands_scalp_0 = np.hstack([x for x in cor_v_scalp])

    n_iter = 10
    for j in range(n_iter):
        cor_v_scalp = []
        for ch in range(in_chan):
            # model.plot_cor_map(datasets=datasets,chan=ch,save_name=os.path.join(savelog,str(ch)+'_cor_map.jpg'),plot=False)
            # value = model.plot_cor_map_mean_filter(datasets=datasets,chan=ch,plot=False)
            # model_e = torch.load(r'G:\undergraduate\MIdecode\multiclass_classification\BCIC3_3a\subject0_fbcspnet.pth')
            # model_e.plot_cor_map(datasets=datasets, chan=ch, save_name=os.path.join(savelog, str(ch) + '_cor_map.jpg'),
            #                    plot=False)
            value = model.plot_cor_map_mean_filter(datasets=datasets, chan=ch, plot=False)
            cor_v_scalp.append(value)
        cors_bands_scalp = np.hstack([x for x in cor_v_scalp])
        vs = cors_bands_scalp - cors_bands_scalp_0
        fig, ax = plt.subplots(3, 1)
        fqs = ['7-13Hz', '13-30Hz', '30-100Hz']
        for i, v in enumerate(vs):
            max_v = np.max(v)
            mne.viz.plot_topomap(v, datasets.positions,
                                 vmin=-max_v, vmax=max_v, contours=0,
                                 cmap=cm.Reds, axes=ax[i], show=True)
            ax[i].set_title(fqs[i])

        model.fit(train_dataset=sub_data,test_dataset=sub_data,batch_size=16,max_epoch=1,reg=1)



class deep_FBCSPNet(nn.Module):

    def __init__(self,in_chans,time_steps,classes,fs = 250,linear_init_std=0.1,eps=1e-3,env=None):
        self.fs = fs
        self.eps = eps
        self.linear_init_std = linear_init_std
        self.env = env
        self.in_chans = in_chans
        self.time_steps = time_steps
        self.classes=classes
        super(deep_FBCSPNet, self).__init__()
        self.pace_1_ratio = 0.05
        self.pace_1 = round(self.time_steps*self.pace_1_ratio)
        self.n_filters_time = 25
        self.n_filters_spat = 25
        self.pool_kernel_ratio = 0.02
        self.pool_kernel = round(self.time_steps*self.pool_kernel_ratio) + 1
        self.pool_stride_ratio = 0.01
        self.pool_stride = round(self.time_steps*self.pool_stride_ratio) + 1
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=self.n_filters_time,kernel_size=(1,self.pace_1),stride=1,padding=0))
        self.sconv = nn.Sequential(nn.Conv2d(in_channels=self.n_filters_time,out_channels=self.n_filters_spat,kernel_size=(in_chans,1),stride=1),
                                   nn.BatchNorm2d(num_features=self.n_filters_spat))
        self.maxpool = nn.MaxPool2d(kernel_size=(1,self.pool_kernel),stride=(1,self.pool_stride))

        dummy = torch.Tensor(1,1,self.in_chans,self.time_steps)
        dummy = self.maxpool(self.sconv(self.conv1(dummy)))

        self.input_dim = dummy.shape
        #[batch_size,n_filters_spat,1,time_wise]
        self.expand_ratio = 2
        self.conv_kernel_ratio = 0.02
        kernel_length = round(self.input_dim[3]*self.conv_kernel_ratio) + 1
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=self.input_dim[1],out_channels=self.input_dim[1]*self.expand_ratio,
                                             kernel_size=(1,kernel_length),stride=1),
                                   nn.MaxPool2d(kernel_size=(1,3),stride=(1,3)))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim[1] * pow(self.expand_ratio,1), out_channels=self.input_dim[1] * pow(self.expand_ratio,2),
                      kernel_size=(1, kernel_length), stride=1),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim[1] * pow(self.expand_ratio,2), out_channels=self.input_dim[1] * pow(self.expand_ratio,3),
                      kernel_size=(1, kernel_length), stride=1),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)))
        dummy = self.conv4(self.conv3(self.conv2(dummy)))
        self.output_dim = dummy.shape

        self.linear = nn.Linear(in_features=self.output_dim[1]*self.output_dim[3],out_features=self.classes)

    def forward(self, x):
        """
            x:input:[batch_size,1,in_chans,time_steps]
            conv1_kernel:(1,pace_1)
            conv1_output:[batch_size,n_filters_time,in_chans,time_steps-(pace_1-1)]
            sconv_kernel:(in_chans,1)
            sconv_output:[batch_size,n_filters_spat,1,time_steps-(pace_1-1)]
            maxpool_kernel:(1,pool_kernel)
            maxpool_stride:(1,pool_stride)
            maxpool_output:[batch_size,n_filters_spat,1,(time_steps-(pace_1-1)-(pool_kernel-1))/pool_stride)]

            待更
        """
        if x.dim() != 4:
            x = torch.unsqueeze(x,dim=1)
        if x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.sconv(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = F.elu(x)
        x = self.conv3(x)
        x = F.elu(x)
        x = self.conv4(x)
        x = F.elu(x)

        x = x.view(batch_size,-1)
        x = self.linear(x)
        x = F.softmax(x,dim=1)
        return x

    def weigth_init(self,m):
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            init.constant_(m.bias.data, 0.1)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, self.linear_init_std)
            m.bias.data.zero_()

    def predict(self,dataset):
        dl = DataLoader(dataset,self.batch_size)
        for idx,(x,y) in enumerate(dl):
            if x.shape[0] != self.batch_size:
                continue
            if type(x) == type(np.array((1))):
                x = torch.Tensor(x)
            logits = self.forward(x)
            if idx == 0:
                pred = logits.argmax(dim=1)
            else:
                pred = torch.cat((pred,logits.argmax(dim=1)),0)
        return pred

    def transform(self, dataset):
        dl = DataLoader(dataset, self.batch_size)
        for idx, (x, y) in enumerate(dl):
            if x.shape[0] != self.batch_size:
                continue
            if type(x) == type(np.array((1))):
                x = torch.Tensor(x)
            logits = self.forward(x)
            if idx == 0:
                out = logits
            else:
                out = torch.cat((out, logits), 0)
        return out

    def graph(self):
        dummy = torch.Tensor(2,1,25,176)
        model = deep_FBCSPNet(in_chans=25,time_steps=176,classes=4)
        ans = model(dummy)

        vis_graph = torchvis.make_dot(model(dummy), params=dict(model.named_parameters()))
        vis_graph.view()


    def evaluate_train(self, test_dataset, batch_size):
        self.eval()
        with torch.no_grad():
            #     training accuracy
            total_num = 0
            total_correct = 0
            total_loop = 0
            total_kappa = 0
            data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            for (x, y) in data_loader:
                logits = self.forward(x)
                pred = logits.argmax(dim=1)

                y = (y - 1).to(dtype=torch.long)
                kappa = cohen_kappa_score(y, pred)

                total_correct += torch.eq(pred, y).float().sum()
                total_num += x.shape[0]
                total_loop += 1
                total_kappa += kappa

            accuracy = total_correct / total_num
            kappa = total_kappa / total_loop
            return accuracy, kappa

    def fit(self, train_dataset, batch_size, max_epoch, valid_dataset=None, test_dataset=None, log_path=None, reg=None,save_name='deep_fbcspnet.pth'):
        """
        fit the model to the dataset
        :param dataset:torch.utils.data.Dateset
        :param model: torch.nn.module
        :return: model
        """
        if log_path == None:
            log_path = os.getcwd()
        if self.env == None:
            viz = Visdom()
        else:
            viz = Visdom(env=self.env)
        viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
        viz.line([0.], [0.], win='train_acc', opts=dict(title='train accuracy'))
        viz.line([0.], [0.], win='test_acc', opts=dict(title='test accuracy'))
        viz.line([0.], [0.], win='train_kappa', opts=dict(title='train kappa value'))
        max_epoch = max_epoch
        batch_size = batch_size
        optimizer = optim.Adam(self.parameters())
        self.apply(self.weigth_init)
        if reg == None:
            criteria = nn.NLLLoss()
        best_acc = -1
        # 定义学习率策略
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100,
                                                               verbose=True,
                                                               threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                               min_lr=0,
                                                               eps=self.eps)
        dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        global_step = 0
        for epoch in range(max_epoch):

            # train a epoch
            for idx_batch, (x, y) in enumerate(dl):
                # x = x * 1e6
                logits = self.forward(x)
                y = (y - 1).to(dtype=torch.long)
                loss = criteria(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step(loss)

                global_step += 1

                viz.line([loss.item()], [global_step], win='train_loss', update='append')

            # test trainning acuracy
            train_acc, kappa = self.evaluate_train(test_dataset=train_dataset, batch_size=batch_size)
            test_acc, _ = self.evaluate_train(test_dataset=test_dataset, batch_size=batch_size)
            viz.line([train_acc.item()], [global_step], win='train_acc', update='append')
            viz.line([test_acc.item()], [global_step], win='test_acc', update='append')
            viz.line([kappa.item()], [global_step], win='train_kappa', update='append')

            # save model
            if test_acc.item() > best_acc:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, 'ckpt_deepfbcsp.mdl')
                best_acc = test_acc
                torch.save(self,save_name)
            self.train()


class EEGNet(nn.Module):
    def __init__(self,in_chans,time_steps,fs,n_classes,F1=4,D=2,F2=6,linear_init_std=0.1,eps=1e-3,env=None):
        self.env = env
        self.eps = eps
        self.linear_init_std = linear_init_std
        super(EEGNet, self).__init__()
        self.in_chans = in_chans
        self.time_steps = time_steps
        self.F1 = F1
        self.D = D
        self.fs = fs
        self.F2 = F2
        self.sptr_filter_length = int(0.5*self.fs)
        self.avgp2d0_length = 4
        self.temp_filter_length = int(0.25*self.fs)
        self.n_classes = n_classes
        self.dummy = torch.zeros((1,1,in_chans,time_steps))
        # spectral conv
        self.conv_sptr = nn.Conv2d(in_channels=1, out_channels=self.F1, kernel_size=(1,self.sptr_filter_length), padding=(0,self.sptr_filter_length//2))

        # spatial conv
        self.conv_spat = nn.Conv2d(in_channels=self.F1, out_channels=self.D*F1, kernel_size=(self.in_chans,1),groups=self.F1,padding_mode='same')
        self.batchnorm0 = nn.BatchNorm2d(F1*D)
        self.elu0 = nn.ELU()
        self.averagepool2d0 = nn.AvgPool2d(kernel_size=(1,self.avgp2d0_length))

        self.dropout0 = nn.Dropout(0.5)

        # temperal conv
        self.conv_temp0 = nn.Conv2d(in_channels=self.D*F1,out_channels=self.D*F1,kernel_size=(1,self.temp_filter_length),padding=(0,self.temp_filter_length//2),groups=D*F1)
        self.conv_temp1 = nn.Conv2d(in_channels=self.D*F1,out_channels=self.F2,kernel_size=(1,1))
        self.batchnorm1 = nn.BatchNorm2d(self.F2)
        self.elu1 = nn.ELU()
        self.averagepool2d1 = nn.AvgPool2d(kernel_size=(1,8))
        self.dropout1 = nn.Dropout(0.25)

        self.ts = self.dropout1(self.averagepool2d1(self.elu1(self.batchnorm1(self.conv_temp1(self.conv_temp0(self.dropout0(self.averagepool2d0(self.elu0(self.batchnorm0(self.conv_spat(self.conv_sptr(self.dummy))))))))))))
        self.n_feature = self.ts.shape[1]*self.ts.shape[3]
        self.linear = nn.Linear(self.n_feature,self.n_classes)

    def forward(self, x):
        if x.dim() != 4:
            x = torch.unsqueeze(x,dim=1)
        if x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)
        batch_size = x.shape[0]
        x = self.conv_sptr(x)
        x = self.conv_spat(x)
        x = self.batchnorm0(x)
        x = self.elu0(x)
        x = self.averagepool2d0(x)
        x = self.dropout0(x)

        x = self.conv_temp0(x)
        x = self.conv_temp1(x)
        x = self.batchnorm1(x)
        x = self.elu1(x)
        x = self.averagepool2d1(x)
        x = self.dropout1(x)

        x = x.view(batch_size,-1)
        x = self.linear(x)
        return x

    def weigth_init(self,m):
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            init.constant_(m.bias.data, 0.1)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, self.linear_init_std)
            m.bias.data.zero_()

    def graph(self):
        dummy = torch.Tensor(2,1,25,176)
        model = EEGNet(in_chans=25,time_steps=176,fs=250,n_classes=4)
        ans = model(dummy)

        vis_graph = torchvis.make_dot(model(dummy), params=dict(model.named_parameters()))
        vis_graph.view()

    def fit(self,train_dataset, batch_size, max_epoch, valid_dataset=None, test_dataset=None, log_path=None,save_name='eegnet.pth'):
        """
        fit the model to the dataset
        :param dataset:torch.utils.data.Dateset
        :param model: torch.nn.module
        :return: model
        """
        if log_path == None:
            log_path = os.getcwd()
        if self.env == None:
            viz = Visdom()
        else:
            viz = Visdom(env=self.env)
        viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
        viz.line([0.], [0.], win='train_acc', opts=dict(title='train accuracy'))
        viz.line([0.], [0.], win='test_acc', opts=dict(title='test accuracy'))
        viz.line([0.], [0.], win='train_kappa', opts=dict(title='train kappa value'))
        max_epoch = max_epoch
        batch_size = batch_size
        optimizer = optim.Adam(self.parameters())
        self.apply(self.weigth_init)
        criteria = nn.NLLLoss()
        best_acc = -1
        # 定义学习率策略
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100,
                                                               verbose=True,
                                                               threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                               min_lr=0,
                                                               eps=self.eps)
        dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        global_step = 0
        for epoch in range(max_epoch):

            # train a epoch
            for idx_batch, (x, y) in enumerate(dl):

                logits = self.forward(x)
                y = (y - 1).to(dtype=torch.long)
                loss = criteria(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step(loss)

                global_step += 1

                viz.line([loss.item()], [global_step], win='train_loss', update='append')

            # test trainning acuracy
            train_acc, kappa = self.evaluate_train(test_dataset=train_dataset, batch_size=batch_size)
            test_acc, _ = self.evaluate_train(test_dataset=test_dataset, batch_size=batch_size)
            viz.line([train_acc.item()], [global_step], win='train_acc', update='append')
            viz.line([test_acc.item()], [global_step], win='test_acc', update='append')
            viz.line([kappa.item()], [global_step], win='train_kappa', update='append')

            # save model
            if test_acc.item() > best_acc:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, 'ckpt_eegnet.mdl')
                best_acc = test_acc
                torch.save(self, save_name)
            self.train()

    def evaluate_train(self, test_dataset, batch_size):
        self.eval()
        with torch.no_grad():
            #     training accuracy
            total_num = 0
            total_correct = 0
            total_loop = 0
            total_kappa = 0
            data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            for (x, y) in data_loader:
                logits = self.forward(x)
                pred = logits.argmax(dim=1)

                y = (y - 1).to(dtype=torch.long)
                kappa = cohen_kappa_score(y, pred)

                total_correct += torch.eq(pred, y).float().sum()
                total_num += x.shape[0]
                total_loop += 1
                total_kappa += kappa

            accuracy = total_correct / total_num
            kappa = total_kappa / total_loop
            return accuracy, kappa


class multi_deep_FBCSPNet(nn.Module):

    def __init__(self,batch_size,in_chans,time_steps,classes):
        self.batch_size = batch_size
        self.in_chans = in_chans
        self.time_steps = time_steps
        self.classes=classes
        super(multi_deep_FBCSPNet, self).__init__()
        self.pace_1_ratio = 0.05
        self.pace_1 = round(self.time_steps*self.pace_1_ratio)
        self.n_filters_time = 25
        self.n_filters_spat = 25
        self.pool_kernel_ratio = 0.02
        self.pool_kernel = round(self.time_steps*self.pool_kernel_ratio) + 1
        self.pool_stride_ratio = 0.01
        self.pool_stride = round(self.time_steps*self.pool_stride_ratio) + 1
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=self.n_filters_time,kernel_size=(1,self.pace_1),stride=1,padding=0))
        self.sconv = nn.Sequential(nn.Conv2d(in_channels=self.n_filters_time,out_channels=self.n_filters_spat,kernel_size=(in_chans,1),stride=1),
                                   nn.BatchNorm2d(num_features=self.n_filters_spat))
        self.maxpool = nn.MaxPool2d(kernel_size=(1,self.pool_kernel),stride=(1,self.pool_stride))

        dummy = torch.Tensor(self.batch_size,1,self.in_chans,self.time_steps)
        dummy = self.maxpool(self.sconv(self.conv1(dummy)))

        self.input_dim = dummy.shape
        #[batch_size,n_filters_spat,1,time_wise]
        self.expand_ratio = 2
        self.conv_kernel_ratio = 0.02
        kernel_length = round(self.input_dim[3]*self.conv_kernel_ratio) + 1

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=self.input_dim[1],out_channels=self.input_dim[1]*self.expand_ratio,
                                             kernel_size=(1,kernel_length),stride=1),
                                   nn.MaxPool2d(kernel_size=(1,3),stride=(1,3)))
        self.conv2_target = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim[1], out_channels=self.input_dim[1] * self.expand_ratio,
                      kernel_size=(1, kernel_length), stride=1),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim[1] * pow(self.expand_ratio,1), out_channels=self.input_dim[1] * pow(self.expand_ratio,2),
                      kernel_size=(1, kernel_length), stride=1),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)))
        self.conv3_target = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim[1] * pow(self.expand_ratio, 1),
                      out_channels=self.input_dim[1] * pow(self.expand_ratio, 2),
                      kernel_size=(1, kernel_length), stride=1),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim[1] * pow(self.expand_ratio,2), out_channels=self.input_dim[1] * pow(self.expand_ratio,3),
                      kernel_size=(1, kernel_length), stride=1),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)))
        self.conv4_target = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim[1] * pow(self.expand_ratio,2), out_channels=self.input_dim[1] * pow(self.expand_ratio,3),
                      kernel_size=(1, kernel_length), stride=1),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)))

        dummy = self.conv4(self.conv3(self.conv2(dummy)))
        self.output_dim = dummy.shape

        self.linear = nn.Linear(in_features=self.output_dim[1]*self.output_dim[3],out_features=self.classes)
        self.linear_target = nn.Linear(in_features=self.output_dim[1]*self.output_dim[3],out_features=1)

    def forward(self, x):
        """
            x:input:[batch_size,1,in_chans,time_steps]
            conv1_kernel:(1,pace_1)
            conv1_output:[batch_size,n_filters_time,in_chans,time_steps-(pace_1-1)]
            sconv_kernel:(in_chans,1)
            sconv_output:[batch_size,n_filters_spat,1,time_steps-(pace_1-1)]
            maxpool_kernel:(1,pool_kernel)
            maxpool_stride:(1,pool_stride)
            maxpool_output:[batch_size,n_filters_spat,1,(time_steps-(pace_1-1)-(pool_kernel-1))/pool_stride)]

            待更
        """
        if x.dim() != 4:
            x = torch.unsqueeze(x,dim=1)
        if x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)

        x = self.conv1(x)
        x = self.sconv(x)
        feature = self.maxpool(x)

        feature1 = self.conv2(feature)
        feature1 = F.elu(feature1)
        feature1 = self.conv3(feature1)
        feature1 = F.elu(feature1)
        feature1 = self.conv4(feature1)
        feature1 = F.elu(feature1)
        feature1 = feature1.view(self.batch_size,-1)
        feature1 = self.linear(feature1)
        pred_class = F.softmax(feature1,dim=1)

        feature2 = self.conv2_target(feature)
        feature2 = F.elu(feature2)
        feature2 = self.conv3_target(feature2)
        feature2  = F.elu(feature2)
        feature2 = self.conv4_target(feature2)
        feature2 = F.elu(feature2)
        feature2 = feature2.view(self.batch_size,-1)
        pred_target_interval = self.linear_target(feature2)
        return (pred_class,pred_target_interval)

    def graph(self):
        dummy = torch.Tensor(16,1,22,1050)
        model = multi_deep_FBCSPNet(batch_size=16,in_chans=22,time_steps=1050,classes=4)
        ans = model(dummy)

        vis_graph = torchvis.make_dot(model(dummy), params=dict(model.named_parameters()))
        vis_graph.view()

    def evaluate_train(self, test_dataset, batch_size):
        self.eval()
        with torch.no_grad():
            #     training accuracy
            total_num = 0
            total_correct = 0
            total_loop = 0
            total_kappa = 0
            data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            for (x, y) in data_loader:
                if x.shape[0] != batch_size:
                    continue
                y_class = y[0]
                y_interval = y[1].to(torch.float32)

                x = x * 1e6
                (logits, interval) = self.forward(x)
                pred = logits.argmax(dim=1)

                y_class = (y_class - 7).to(dtype=torch.long)

                kappa = cohen_kappa_score(y_class, pred)

                total_correct += torch.eq(pred, y_class).float().sum()
                total_num += batch_size
                total_loop += 1
                total_kappa += kappa

            accuracy = total_correct / total_num
            kappa = total_kappa / total_loop
            return accuracy, kappa

    def fit(self,train_dataset, batch_size, max_epoch, lambda_cls=0.9999999,valid_dataset=None, test_dataset=None, log_path=None ,reg =None,):
        """
        fit the model to the dataset
        :param dataset:torch.utils.data.Dateset
        :param model: torch.nn.module
        :return: model
        """
        if log_path == None:
            log_path = os.getcwd()
        viz = Visdom()
        viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
        viz.line([0.], [0.], win='train_acc', opts=dict(title='train accuracy'))
        viz.line([0.], [0.], win='test_acc', opts=dict(title='test accuracy'))
        viz.line([0.], [0.], win='train_kappa', opts=dict(title='train kappa value'))
        max_epoch = max_epoch
        batch_size = batch_size
        optimizer = optim.Adam(self.parameters())
        self.apply(weigth_init)
        if reg == None:
            criteria_class = nn.NLLLoss()
            criteria_target_interval = nn.MSELoss()
            weight_class = lambda_cls
            weight_interval = 1 - weight_class
        best_acc = -1
        # 定义学习率策略
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100,
                                                               verbose=True,
                                                               threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                               min_lr=0,
                                                               eps=1e-10)
        dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        global_step = 0
        for epoch in range(max_epoch):

            # train a epoch
            for idx_batch, (x, y) in enumerate(dl):
                # drop the last batch of a epoch if not %batch_size != 0
                if x.shape[0] != batch_size:
                    continue

                y_class = y[0]
                y_interval = y[1].to(torch.float32)

                x = x * 1e6
                (logits, interval) = self.forward(x)
                y_class = (y_class - 7).to(dtype=torch.long)
                loss_class = criteria_class(logits, y_class)
                loss_interval = criteria_target_interval(interval, y_interval)
                loss = weight_class * loss_class + weight_interval * loss_interval

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step(loss)

                global_step += 1

                viz.line([loss.item()], [global_step], win='train_loss', update='append')

            # test trainning acuracy
            train_acc, kappa = self.evaluate_train(test_dataset=train_dataset, batch_size=batch_size)
            test_acc, _ = self.evaluate_train(test_dataset=test_dataset, batch_size=batch_size)
            viz.line([train_acc.item()], [global_step], win='train_acc', update='append')
            viz.line([test_acc.item()], [global_step], win='test_acc', update='append')
            viz.line([kappa.item()], [global_step], win='train_kappa', update='append')

            # save model
            if test_acc.item() > best_acc:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, 'ckpt_multideepfbcspnet.mdl')
                best_acc = test_acc

            self.train()

    def record(self,train_dataset,batch_size):
        with open('results_0.0001_100.txt', 'w') as f:
            for x, y in DataLoader(train_dataset, 16):
                if x.shape[0] != batch_size:
                    continue
                x = x * 1e6
                f.writelines('pred interval' + str(self.forward(x)[1]) + '\n')
                f.writelines('true interval' + str(y[1]) + '\n')
                f.writelines('pred class' + str(torch.argmax(self.forward(x)[0], dim=1)) + '\n')
                f.writelines('true class' + str(y[0] - 7) + '\n')
                f.writelines('——————————————\n')



class Point_wiseLSTM(nn.Module):
    def __init__(self,in_chans,emd_dim,classes,eps=1e-3,linear_init_std=0.001):
        super(Point_wiseLSTM, self).__init__()
        self.eps = eps
        self.linear_init_std = linear_init_std
        self.in_chans = in_chans
        self.classes = classes
        self.h_dim = classes
        self.emd_dim = emd_dim
        self.linear_1 = nn.Sequential(nn.Linear(self.in_chans,emd_dim),
                                      nn.ReLU())
        self.lstm = nn.LSTM(self.emd_dim,self.h_dim,4)

    def preprocess(self,x):
        x = (x - x.mean())/x.std()
        return x

    def weigth_init(self,m):
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            init.constant_(m.bias.data, 0.1)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, self.linear_init_std)
            m.bias.data.zero_()

    def forward(self, x):
        #x[batch_size,in_chans,time_steps]
        x = x.to(dtype=torch.float32)
        x = self.preprocess(x)
        x = x.permute(2,0,1)
        # print(x.shape)
        x = self.linear_1(x)
        # print(x.shape)
        ht = torch.ones(4,x.shape[1],self.h_dim)
        ct = ht.clone()
        out,(ht,ct) = self.lstm(x, [ht, ct])
        out = F.softmax(out,dim=2)
        out = out.permute(1,2,0)
        return out

    def graph(self):
        dummy = torch.Tensor(16,22,1)
        model = Point_wiseLSTM(in_chans=22,emd_dim=30,classes=4)
        ans = model(dummy)

        vis_graph = torchvis.make_dot(model(dummy), params=dict(model.named_parameters()))
        vis_graph.view()

    def evaluate_train(self, test_dataset, batch_size):
        with torch.no_grad():
            #     training accuracy
            total_num = 0
            total_correct = 0
            total_loop = 0
            total_kappa = 0
            data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            for (x_trial, y_trial) in data_loader:
                if x_trial.shape[0] != batch_size:
                    continue
                x_trial = x_trial * 1e6
                logits = self.forward(x_trial)
                pred = logits.argmax(dim=1)
                y_trial = y_trial.squeeze(dim=1)
                for (p, y) in zip(pred.t(), y_trial.t()):
                    y = y.to(dtype=torch.long)
                    kappa = cohen_kappa_score(y, p)

                    total_correct += torch.eq(p, y).float().sum()
                    total_num += batch_size
                    total_loop += 1
                    total_kappa += kappa

            accuracy = total_correct / total_num
            kappa = total_kappa / total_loop
            return accuracy, kappa

    def fit(self,train_dataset, batch_size, max_epoch, valid_dataset=None, test_dataset=None, log_path=None ,reg =None,env = None ):
        """
        fit the model to the dataset
        :param dataset:torch.utils.data.Dateset
        :param model: torch.nn.module
        :return: model
        """
        if log_path == None:
            log_path = os.getcwd()
        if env==None:
            viz = Visdom()
        else:
            viz = Visdom(env)
        viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
        viz.line([0.], [0.], win='train_acc', opts=dict(title='train accuracy'))
        viz.line([0.], [0.], win='test_acc', opts=dict(title='test accuracy'))
        viz.line([0.], [0.], win='train_kappa', opts=dict(title='train kappa value'))
        max_epoch = max_epoch
        batch_size = batch_size
        optimizer = optim.Adam(self.parameters())
        self.apply(self.weigth_init)
        if reg == None:
            criteria = nn.NLLLoss2d()
        best_acc = -1
        # 定义学习率策略
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100,
                                                               verbose=True,
                                                               threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                               min_lr=0,
                                                               eps=self.eps)
        dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        global_step = 0
        for epoch in range(max_epoch):

            # train a epoch
            for idx_batch, (x, y) in enumerate(dl):
                # drop the last batch of a epoch if not %batch_size != 0
                if x.shape[0] != batch_size:
                    continue

                x = x * 1e6
                logits = self.forward(x)

                pred = logits.unsqueeze(dim=2).to(dtype=torch.float32)
                y = y.to(dtype=torch.int64)
                loss = criteria(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step(loss)

                global_step += 1

                viz.line([loss.item()], [global_step], win='train_loss', update='append')

            # test trainning acuracy
            train_acc, kappa = self.evaluate_train(test_dataset=train_dataset, batch_size=batch_size)
            test_acc, _ = self.evaluate_train(test_dataset=test_dataset, batch_size=batch_size)
            viz.line([train_acc.item()], [global_step], win='train_acc', update='append')
            viz.line([test_acc.item()], [global_step], win='test_acc', update='append')
            viz.line([kappa.item()], [global_step], win='train_kappa', update='append')

            # save model
            if test_acc.item() > best_acc:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, 'ckpt_pointwiselstm.mdl')
                best_acc = test_acc

            self.train()



class CSP_estimator(BaseEstimator, ClassifierMixin):
    def __init__(self,fs=250,feature_idx='all'):
        self.fs = fs
        self.csps = []
        self.n_fs = 4
        self.fq_interval = 4
        for i in range(self.n_fs):
            csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
            self.csps.append(csp)
        # self.clf = LinearDiscriminantAnalysis()
        self.clf = SVC(kernel='rbf', probability=True)
        self.feature_idx = feature_idx


    def fit(self,x,y):
        features = []
        for f in range(self.n_fs):
            fq = (f+1)*self.fq_interval
            x_f = band_pass_EEGdata(x,fq,fq+self.fq_interval,self.fs)
            feature = self.csps[f].fit_transform(x_f,y)
            features.append(feature)
        self.feature_all_f = np.hstack([x for x in features])
        if self.feature_idx != 'all':
            feature_selected = self.feature_all_f[:,self.feature_idx]
        else:
            feature_selected = self.feature_all_f
        self.clf.fit(feature_selected,y)

    def predict(self,x):
        features = []
        for f in range(self.n_fs):
            fq = (f+1)*self.fq_interval
            x_f = band_pass_EEGdata(x, fq, fq + self.fq_interval, self.fs)
            feature = self.csps[f].transform(x_f)
            features.append(feature)
        feature_all_f = np.hstack([x for x in features])
        if self.feature_idx != 'all':
            feature_selected = feature_all_f[:,self.feature_idx]
        else:
            feature_selected = feature_all_f
        pred = self.clf.predict(feature_selected)
        return pred

    def transform(self,x):
        print('use transform')
        features = []
        for f in range(self.n_fs):
            fq = (f+1)*self.fq_interval
            x_f = band_pass_EEGdata(x, fq , fq + self.fq_interval,self.fs)
            feature = self.csps[f].transform(x_f)
            features.append(feature)
        feature_all_f = np.hstack([x for x in features])
        if self.feature_idx != 'all':
            feature_selected = feature_all_f[:,self.feature_idx]
        else:
            feature_selected = feature_all_f
        logits = self.clf.transform(feature_selected)
        return logits

    def score(self, X, y, sample_weight=None):
        pred = self.predict(X)
        return accuracy_score(y,pred)

class CSP_estimator_SVC(BaseEstimator, ClassifierMixin):
    def __init__(self,fs=250,f_l=4,f_h=40,feature_idx='all'):
        self.fs = fs
        self.csps = []
        self.f_l = f_l
        self.f_h = f_h
        self.csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        self.clf = SVC(kernel='rbf', probability=True)
        self.feature_idx = feature_idx


    def fit(self,x,y):
        x_f = band_pass_EEGdata(x,self.f_l,self.f_h,self.fs)
        feature = self.csp.fit_transform(x_f,y)
        self.feature_all_f = feature
        if self.feature_idx != 'all':
            feature_selected = self.feature_all_f[:,self.feature_idx]
        else:
            feature_selected = self.feature_all_f
        self.clf.fit(feature_selected,y)

    def predict(self,x):
        x_f = band_pass_EEGdata(x, self.f_l, self.f_h, self.fs)
        feature = self.csp.transform(x_f)
        self.feature_all_f = feature
        if self.feature_idx != 'all':
            feature_selected = self.feature_all_f[:, self.feature_idx]
        else:
            feature_selected = self.feature_all_f
        pred = self.clf.predict(feature_selected)
        return pred

    def transform(self,x):
        features = []
        x_f = band_pass_EEGdata(x, self.f_l, self.f_h, self.fs)
        feature = self.csp.transform(x_f)
        self.feature_all_f = feature
        if self.feature_idx != 'all':
            feature_selected = self.feature_all_f[:, self.feature_idx]
        else:
            feature_selected = self.feature_all_f
        logits = self.clf.transform(feature_selected)
        return logits

    def score(self, X, y, sample_weight=None):
        pred = self.predict(X)
        return accuracy_score(y,pred)

class CSSP_estimator_LDA(BaseEstimator, ClassifierMixin):
    def __init__(self,tao=5,fs=250,f_l=4,f_h=40,feature_idx='all'):
        self.fs = fs
        self.tao = tao
        self.csps = []
        self.f_l = f_l
        self.f_h = f_h
        self.csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        self.clf = LinearDiscriminantAnalysis()
        self.feature_idx = feature_idx


    def fit(self,x,y):
        """

        :param x: have to be [trails,in_chan,time_step]
        :param y:
        :return:
        """
        x_f = band_pass_EEGdata(x,self.f_l,self.f_h,self.fs)
        x_f = np.hstack([x_f[...,self.tao:],x_f[...,:-self.tao]])
        feature = self.csp.fit_transform(x_f,y)
        self.feature_all_f = feature
        if self.feature_idx != 'all':
            feature_selected = self.feature_all_f[:,self.feature_idx]
        else:
            feature_selected = self.feature_all_f
        self.clf.fit(feature_selected,y)

    def predict(self,x):
        x_f = band_pass_EEGdata(x, self.f_l, self.f_h, self.fs)
        x_f = np.hstack([x_f[...,self.tao:],x_f[...,:-self.tao]])
        feature = self.csp.transform(x_f)
        self.feature_all_f = feature
        if self.feature_idx != 'all':
            feature_selected = self.feature_all_f[:, self.feature_idx]
        else:
            feature_selected = self.feature_all_f
        pred = self.clf.predict(feature_selected)
        return pred

    def transform(self,x):
        features = []
        x_f = band_pass_EEGdata(x, self.f_l, self.f_h, self.fs)
        x_f = np.hstack([x_f[...,self.tao:],x_f[...,:-self.tao]])
        feature = self.csp.transform(x_f)
        self.feature_all_f = feature
        if self.feature_idx != 'all':
            feature_selected = self.feature_all_f[:, self.feature_idx]
        else:
            feature_selected = self.feature_all_f
        logits = self.clf.transform(feature_selected)
        return logits

    def score(self, X, y, sample_weight=None):
        pred = self.predict(X)
        return accuracy_score(y,pred)

class CSSSP_estimator_LDA(BaseEstimator, ClassifierMixin):
    def __init__(self, T=5, fs=250, f_l=4, f_h=40, feature_idx='all'):
        self.fs = fs
        self.T = T
        self.csps = []
        self.f_l = f_l
        self.f_h = f_h
        self.csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        self.clf = LinearDiscriminantAnalysis()
        self.feature_idx = feature_idx

    def get_x_f(self,x):
        for i in range(self.T):
            if i==0:
                x_f = x[..., self.T:]
            else:
                x_f = np.hstack([x_f, x[..., self.T-i:-i]])
        return x_f

    def fit(self, x, y):
        """

        :param x: have to be [trails,in_chan,time_step]
        :param y:
        :return:
        """
        x_f = band_pass_EEGdata(x, self.f_l, self.f_h, self.fs)
        x_f = self.get_x_f(x_f)
        feature = self.csp.fit_transform(x_f, y)
        self.feature_all_f = feature
        if self.feature_idx != 'all':
            feature_selected = self.feature_all_f[:, self.feature_idx]
        else:
            feature_selected = self.feature_all_f
        self.clf.fit(feature_selected, y)

    def predict(self, x):
        x_f = band_pass_EEGdata(x, self.f_l, self.f_h, self.fs)
        x_f = self.get_x_f(x_f)
        feature = self.csp.transform(x_f)
        self.feature_all_f = feature
        if self.feature_idx != 'all':
            feature_selected = self.feature_all_f[:, self.feature_idx]
        else:
            feature_selected = self.feature_all_f
        pred = self.clf.predict(feature_selected)
        return pred

    def transform(self, x):
        features = []
        x_f = band_pass_EEGdata(x, self.f_l, self.f_h, self.fs)
        x_f = self.get_x_f(x_f)
        feature = self.csp.transform(x_f)
        self.feature_all_f = feature
        if self.feature_idx != 'all':
            feature_selected = self.feature_all_f[:, self.feature_idx]
        else:
            feature_selected = self.feature_all_f
        logits = self.clf.transform(feature_selected)
        return logits

    def score(self, X, y, sample_weight=None):
        pred = self.predict(X)
        return accuracy_score(y, pred)

class SBSSP_estimator_LDA(BaseEstimator, ClassifierMixin):
    def __init__(self,f_l=8,f_h=40, T=5, fs=250, feature_idx='all'):
        self.fs = fs
        self.T = T
        self.csps = []
        self.clfs = []
        self.f_l = f_l
        self.f_h = f_h
        self.subbands=[(8,12),(12,16),(16,20),(20,24),(24,28),(28,32)]
        for i in range(len(self.subbands)):
            csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
            self.csps.append(csp)
            clf = LinearDiscriminantAnalysis()
            self.clfs.append(clf)
        self.feature_idx = feature_idx

    def fit(self, x, y):
        """

        :param x: have to be [trails,in_chan,time_step]
        :param y:
        :return:
        """
        for i,(fls,flh) in enumerate(self.subbands):
            x_f = band_pass_EEGdata(x, fls, flh, self.fs)
            feature = self.csps[i].fit_transform(x_f, y)
            self.feature_all_f = feature
            if self.feature_idx != 'all':
                feature_selected = self.feature_all_f[:, self.feature_idx]
            else:
                feature_selected = self.feature_all_f
            self.clfs[i].fit(feature_selected, y)
        self.ys = np.unique(y)

    def predict(self, x):
        trans = self.transform(x)
        trans = trans.sum(axis=0)/len(trans)
        #should be [n_trial,1]
        pred = np.zeros_like(trans)
        for i,trial in enumerate(trans):
            if trial>0:
                pred[i] = self.ys[0]
            else:
                pred[i] = self.ys[1]
        return pred


    def transform(self, x):
        trans = np.zeros((len(self.subbands),x.shape[0],1))
        for i,(fls,flh) in enumerate(self.subbands):
            x_f = band_pass_EEGdata(x, fls, flh, self.fs)
            feature = self.csps[i].transform(x_f)
            self.feature_all_f = feature
            if self.feature_idx != 'all':
                feature_selected = self.feature_all_f[:, self.feature_idx]
            else:
                feature_selected = self.feature_all_f
            trans_sb = self.clfs[i].transform(feature_selected)
            trans[i] = trans_sb
        return trans


    def score(self, X, y, sample_weight=None):
        pred = self.predict(X)
        return accuracy_score(y, pred)

class SBSSP_estimator_LDA_m(BaseEstimator, ClassifierMixin):
    def __init__(self,f_l=8,f_h=40, T=5, fs=250, feature_idx='all'):
        self.fs = fs
        self.T = T
        self.csps = []
        self.clfs = []
        self.f_l = f_l
        self.f_h = f_h
        self.subbands=[(8,12),(12,16),(16,20),(20,24),(24,28),(28,32)]
        for i in range(len(self.subbands)):
            csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
            self.csps.append(csp)
            clf = LinearDiscriminantAnalysis()
            self.clfs.append(clf)
        self.feature_idx = feature_idx

    def fit(self, x, y):
        """

        :param x: have to be [trails,in_chan,time_step]
        :param y:
        :return:
        """
        for i,(fls,flh) in enumerate(self.subbands):
            x_f = band_pass_EEGdata(x, fls, flh, self.fs)
            feature = self.csps[i].fit_transform(x_f, y)
            self.feature_all_f = feature
            if self.feature_idx != 'all':
                feature_selected = self.feature_all_f[:, self.feature_idx]
            else:
                feature_selected = self.feature_all_f
            self.clfs[i].fit(feature_selected, y)
        self.ys = np.unique(y)

    def predict(self, x):
        trans = self.transform(x)
        trans = trans.sum(axis=0)/len(trans)
        #should be [n_trial,1]
        pred = np.zeros_like(trans)
        for i,trial in enumerate(trans):
            if trial>0:
                pred[i] = self.ys[0]
            else:
                pred[i] = self.ys[1]
        return pred


    def transform(self, x):
        trans = np.zeros((len(self.subbands),x.shape[0],1))
        for i,(fls,flh) in enumerate(self.subbands):
            x_f = band_pass_EEGdata(x, fls, flh, self.fs)
            feature = self.csps[i].transform(x_f)
            self.feature_all_f = feature
            if self.feature_idx != 'all':
                feature_selected = self.feature_all_f[:, self.feature_idx]
            else:
                feature_selected = self.feature_all_f
            trans_sb = self.clfs[i].transform(feature_selected)
            trans[i] = trans_sb
        return trans


    def score(self, X, y, sample_weight=None):
        pred = self.predict(X)
        return accuracy_score(y, pred)

class FBCSP_estimator_LDA(BaseEstimator, ClassifierMixin):
    def __init__(self, T=5, fs=250,csp_cpn = 6,select_ratio=0.1, feature_idx='all'):
        self.fs = fs
        self.T = T
        self.csps = []
        self.f_l = 8
        self.f_h = 40
        self.select_ratio = select_ratio
        self.subbands = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)]
        self.csp_cpn = csp_cpn
        for i in range(len(self.subbands)):
            csp = CSP(n_components=self.csp_cpn, reg=None, log=True, norm_trace=False)
            self.csps.append(csp)
        self.clf = LinearDiscriminantAnalysis()
        self.feature_idx = feature_idx

    def select_feature(self,X,Y,class1,class2,num='default'):
            #select best featuers from 2-class feature vector
            #Filter bank ... 2.3.1 Mutual information ...
            n_feature = X.shape[1]
            if num == 'default':
                num = int(n_feature*self.select_ratio)+1
            if num>n_feature:
                print('num can not be larger than'+str(n_feature))
                return
            try:
                sigma = np.std(Y)
            except:
                print(Y)
            h = pow(4/3/len(Y),0.2)*sigma
            phi = lambda y,h:np.exp(-y**2/2/h**2)/pow(2*np.pi,0.5)
            p1 = len([y for y in Y if int(y) == int(class1)])/len(Y)
            p2 = 1-p1
            h_omega = -(p1*np.log2(p1))-(p2*np.log2(p2))
            x_1 = [x for idx,x in enumerate(X) if int(Y[idx]) ==int(class1)]
            x_1 = np.vstack(x_1).transpose()
            x_2 = [x for idx,x in enumerate(X) if int(Y[idx]) ==int(class2)]
            x_2 = np.vstack(x_2).transpose()
            V = X.transpose()
            feature_MI_values = np.zeros([V.shape[0],1])
            for idx,f_j in enumerate(V):
                h_omega_f_j = 0
                for f_j_i in f_j:
                    p_f_j_i_1 =sum([phi(f_j_i-f_j_k,h) for f_j_ in x_1 for f_j_k in f_j_ ])/len(x_1)
                    p_f_j_i_2 =sum([phi(f_j_i-f_j_k,h) for f_j_ in x_2 for f_j_k in f_j_ ])/len(x_2)
                    p_1_f_j_i = p_f_j_i_1 * p1 / (p_f_j_i_1 * p1 + p_f_j_i_2 * p2)
                    p_2_f_j_i = p_f_j_i_1 * p2 / (p_f_j_i_1 * p1 + p_f_j_i_2 * p2)
                    h_omega_f_j += p_1_f_j_i*np.log2(p_1_f_j_i)
                    h_omega_f_j += p_2_f_j_i*np.log2(p_2_f_j_i)
                I_f_j_omega = h_omega - h_omega_f_j
                feature_MI_values[idx] = I_f_j_omega
            sort = np.argsort(feature_MI_values, axis=0)
            X_selected = [x for idx,x in enumerate(V) if int(sort[idx]) >= len(sort)-num]
            index = sort[0:num]
            X_selected = np.vstack(X_selected).transpose()
            return X_selected,index

    def fit(self, x, y):
        """

        :param x: have to be [trails,in_chan,time_step]
        :param y:
        :return:
        """
        self.ys = np.unique(y)
        self.y_0 = self.ys[0]
        self.y_1 = self.ys[1]
        self.feature_all_f = np.zeros((len(x),self.csp_cpn*len(self.subbands)))
        for i, (fls, flh) in enumerate(self.subbands):
            x_f = band_pass_EEGdata(x, fls, flh, self.fs)
            feature = self.csps[i].fit_transform(x_f, y)
            self.feature_all_f[:,i*(self.csp_cpn):(i+1)*(self.csp_cpn)] = feature
        self.feature_selected,self.indexs = self.select_feature(self.feature_all_f,y,self.y_0,self.y_1)
        self.clf.fit(self.feature_selected, y)

    def predict(self, x):
        feature_all_f = np.zeros((len(x), self.csp_cpn * len(self.subbands)))
        for i, (fls, flh) in enumerate(self.subbands):
            x_f = band_pass_EEGdata(x, fls, flh, self.fs)
            feature = self.csps[i].transform(x_f)
            feature_all_f[:, i * (self.csp_cpn):(i + 1) * (self.csp_cpn)] = feature
        feature_selected = feature_all_f[:, self.indexs].squeeze(axis=2)
        return self.clf.predict(feature_selected)

    def transform(self, x):
        feature_all_f = np.zeros((len(x),self.csp_cpn*len(self.subbands)))
        for i, (fls, flh) in enumerate(self.subbands):
            x_f = band_pass_EEGdata(x, fls, flh, self.fs)
            feature = self.csps[i].transform(x_f)
            feature_all_f[:,i*(self.csp_cpn):(i+1)*(self.csp_cpn)] = feature
        feature_selected = feature_all_f[:,self.indexs].squeeze(axis=2)
        return self.clf.transform(feature_selected)


    def score(self, X, y, sample_weight=None):
        pred = self.predict(X)
        return accuracy_score(y, pred)

class DFBCSP_FR_estimator_LDA(BaseEstimator, ClassifierMixin):
    def __init__(self,C3_idx,C4_idx, T=5, fs=250,csp_cpn = 4,select_ratio=0.2, feature_idx='all'):
        self.C3_idx = C3_idx
        self.C4_idx = C4_idx
        self.fs = fs
        self.T = T
        self.csps = []
        self.f_l = 8
        self.f_h = 40
        self.select_ratio = select_ratio
        self.subbands = [(8, 11), (11, 14), (14, 17), (17, 20), (20, 23), (23, 26),(26, 29),(29,32),(32,35),(35,38),(38,41)]
        self.n_select_bands = int(len(self.subbands)*self.select_ratio)+1
        self.csp_cpn = csp_cpn
        for i in range(len(self.subbands)):
            csp = CSP(n_components=self.csp_cpn, reg=None, log=True, norm_trace=False)
            self.csps.append(csp)
        self.clf = LinearDiscriminantAnalysis()
        self.feature_idx = feature_idx

    def select_bands(self,X,Y,class1,class2):
            #select best bands based on Fisher ratio of frequency estimates of C3 and C4
            x_c3 = X[:,self.C3_idx,:]
            x_c4 = X[:, self.C4_idx, :]
            self.Frs= []
            for i, (fls, flh) in enumerate(self.subbands):
                ps = np.zeros((len(X), 2))
                for j,trial in enumerate(x_c3):
                    x_c3_f = band_pass_EEGdata(trial,fls,flh,self.fs)
                    ps[j,0] = np.power(x_c3_f,2).mean()
                    x_c4_f = band_pass_EEGdata(x_c4[j], fls, flh, self.fs)
                    ps[j, 1] = np.power(x_c4_f, 2).mean()
                ps_0 = [x for idx, x in enumerate(ps) if int(Y[idx]) == int(self.y_0)]
                ps_0 = np.vstack(ps_0)
                ps_1 = [x for idx, x in enumerate(ps) if int(Y[idx]) == int(self.y_1)]
                ps_1 = np.vstack(ps_1)
                self.Sb = np.power(ps_0.mean(axis =0)-ps_1.mean(axis=0),2).sum()
                self.Sw = np.power(ps_0 - ps_0.mean(axis=0),2).sum()/len(ps_0)/2 + np.power(ps_1 - ps_1.mean(axis=0),2).sum()/len(ps_1)/2
                Fr = self.Sb/(self.Sw)
                self.Frs.append(Fr)
            sort = np.argsort(self.Frs, axis=0)
            index = sort[0:self.n_select_bands]
            self.indexs = index



    def fit(self, x, y):
        self.ys = np.unique(y)
        self.y_0 = self.ys[0]
        self.y_1 = self.ys[1]
        self.select_bands(x,y,self.y_0,self.y_1)
        self.feature_all_f = np.zeros((len(x), self.csp_cpn * len(self.indexs)))
        for i,band_i in enumerate(self.indexs):
            (fl,fh) = self.subbands[band_i]
            x_f = band_pass_EEGdata(x, fl, fh, self.fs)
            feature = self.csps[band_i].fit_transform(x_f, y)
            self.feature_all_f[:, i * (self.csp_cpn):(i + 1) * (self.csp_cpn)] = feature
        self.clf.fit(self.feature_all_f, y)

    def transform(self, x):
        feature_all_f = np.zeros((len(x), self.csp_cpn * len(self.indexs)))
        for i, band_i in enumerate(self.indexs):
            (fl, fh) = self.subbands[band_i]
            x_f = band_pass_EEGdata(x, fl, fh, self.fs)
            feature = self.csps[band_i].transform(x_f)
            feature_all_f[:, i * (self.csp_cpn):(i + 1) * (self.csp_cpn)] = feature
        trans = self.clf.transform(feature_all_f)
        return trans

    def predict(self,x):
        feature_all_f = np.zeros((len(x), self.csp_cpn * len(self.indexs)))
        for i, band_i in enumerate(self.indexs):
            (fl, fh) = self.subbands[band_i]
            x_f = band_pass_EEGdata(x, fl, fh, self.fs)
            feature = self.csps[band_i].transform(x_f)
            feature_all_f[:, i * (self.csp_cpn):(i + 1) * (self.csp_cpn)] = feature
        pred = self.clf.predict(feature_all_f)
        return pred

    def score(self, X, y, sample_weight=None):
        pred = self.predict(X)
        return accuracy_score(y, pred)

# model = DFBCSP_FR_estimator_LDA(C3_idx = 7,C4_idx = 11)
# model.fit(x1.numpy(),y1.numpy())
# model.score(x1.numpy(),y1.numpy())

class AAR_estimator_LDA(BaseEstimator, ClassifierMixin):
    """
    feature:AAR
    clf:LDA
    """
    def __init__(self,in_chan,classes,maxlag=3,fs=250,feature_idx='all'):
        self.in_chan = in_chan
        self.classes = classes
        self.fs = fs
        self.maxlag = maxlag
        self.clf = LinearDiscriminantAnalysis()

    def find_aar_para(self,x):
        features_x = np.zeros((len(x), self.in_chan * (self.maxlag + 1)))
        for j, each_trial in enumerate(x):
            features_trial = np.zeros((self.in_chan, self.maxlag + 1))
            for i in range(self.in_chan):
                x_chan = each_trial[i]
                para_estimator = AR(x_chan)
                para_estimator = para_estimator.fit(maxlag=self.maxlag)
                features_trial[i] = para_estimator.params
            features_trial = features_trial.flatten()
            features_x[j] = features_trial
        return features_x

    def fit(self,x,y):
        features_x = self.find_aar_para(x)
        self.clf.fit(features_x,y)

    def predict(self,x):
        features_x = self.find_aar_para(x)
        pred = self.clf.predict(features_x)
        return pred


    def transform(self,x):
        features_x = self.find_aar_para(x)
        tran = self.clf.transform(features_x)
        return tran


    def score(self, X, y, sample_weight=None):
        pred = self.predict(X)
        return accuracy_score(y,pred)

class AAR_estimator_SVC(BaseEstimator, ClassifierMixin):
    """
    feature:AAR
    clf:SVM
    """
    def __init__(self,in_chan,classes,maxlag=3,fs=250):
        self.in_chan = in_chan
        self.classes = classes
        self.fs = fs
        self.maxlag = maxlag
        self.clf = SVC(kernel='rbf', probability=True)

    def find_aar_para(self,x):
        features_x = np.zeros((len(x), self.in_chan * (self.maxlag + 1)))
        for j, each_trial in enumerate(x):
            features_trial = np.zeros((self.in_chan, self.maxlag + 1))
            for i in range(self.in_chan):
                x_chan = each_trial[i]
                para_estimator = AR(x_chan)
                para_estimator = para_estimator.fit(maxlag=self.maxlag)
                features_trial[i] = para_estimator.params
            features_trial = features_trial.flatten()
            features_x[j] = features_trial
        return features_x

    def fit(self,x,y):
        features_x = self.find_aar_para(x)
        self.clf.fit(features_x,y)

    def predict(self,x):
        features_x = self.find_aar_para(x)
        pred = self.clf.predict(features_x)
        return pred


    # def transform(self,x):
    #     features_x = self.find_aar_para(x)
    #     tran = self.clf.transform(features_x)
    #     return tran


    def score(self, X, y, sample_weight=None):
        pred = self.predict(X)
        return accuracy_score(y,pred)

class AAR_estimator_KNN(BaseEstimator, ClassifierMixin):
    """
    feature:AAR
    clf:KNN
    """
    def __init__(self,in_chan,classes,k=6,maxlag=3,fs=250,feature_idx='all'):
        self.in_chan = in_chan
        self.classes = classes
        self.fs = fs
        self.maxlag = maxlag
        self.k = k
        self.clf = KNeighborsClassifier(self.k)

    def find_aar_para(self,x):
        features_x = np.zeros((len(x), self.in_chan * (self.maxlag + 1)))
        for j, each_trial in enumerate(x):
            features_trial = np.zeros((self.in_chan, self.maxlag + 1))
            for i in range(self.in_chan):
                x_chan = each_trial[i]
                para_estimator = AR(x_chan)
                para_estimator = para_estimator.fit(maxlag=self.maxlag)
                features_trial[i] = para_estimator.params
            features_trial = features_trial.flatten()
            features_x[j] = features_trial
        return features_x

    def fit(self,x,y):
        features_x = self.find_aar_para(x)
        self.clf.fit(features_x,y)

    def predict(self,x):
        features_x = self.find_aar_para(x)
        pred = self.clf.predict(features_x)
        return pred


    # def transform(self,x):
    #     features_x = self.find_aar_para(x)
    #     tran = self.clf.transform(features_x)
    #     return tran


    def score(self, X, y, sample_weight=None):
        pred = self.predict(X)
        return accuracy_score(y,pred)

# import time
# t = time.time()
# c = CSP_estimator([1,4,5,10])
# c.fit(train_dataset,250)
# pred= c.predict(tp_x,250)
# confusion_matrix(tp_y,pred)
# inte = time.time()-t





"""

from BCICdatasets import BCIC4_2a_lrh,Subject_dataset
train_dataset = BCIC4_2a_lrh()
train_dataset.load_data()
train_dataset = Subject_dataset(train_dataset,0)
from experiment_utils import split_dataset

tp_x,tp_y = next(iter(DataLoader(train_dataset,len(train_dataset))))
tp_x = tp_x*1e6

L = list(range(8))

def all_subsets_without_none(L):
    List = [[]]
    for i in range(len(L)):  # 定长
        for j in range(len(List)):  # 变长
            sub_List = List[j] + [L[i]]
            if sub_List not in L:
                if sub_List!=[]:
                    List.append(sub_List)
    List.pop(0)
    return List

List = all_subsets_without_none(L)

params = {
    'feature_idx': List
}
gs = GridSearchCV(CSP_estimator(), param_grid=params, cv=4)
gs.fit(tp_x.numpy(),tp_y.numpy())


"""


#gs.best_estimator_

"""
names = ['EEG-Fz',
 'EEG-0',
 'EEG-1',
 'EEG-2',
 'EEG-3',
 'EEG-4',
 'EEG-5',
 'EEG-C3',
 'EEG-6',
 'EEG-Cz',
 'EEG-7',
 'EEG-C4',
 'EEG-8',
 'EEG-9',
 'EEG-10',
 'EEG-11',
 'EEG-12',
 'EEG-13',
 'EEG-14',
 'EEG-Pz',
 'EEG-15',
 'EEG-16']
scalp.ax_scalp(c.csps[0].patterns_[0].data,names,'BCIC4_2a')
"""



