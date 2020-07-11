from torch.utils.data import DataLoader
import torch
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(42)

def get_one_para(paras):
    """
    from paras return list of para
    :param paras:dict like {para1:[value1,value2],para2:[value1]}
    :return: list of combination of paras
    """
    dim = len(paras)
    paras_ret = []
    if dim == 2:
        key0 = list(paras.keys())[0]
        key1 = list(paras.keys())[1]
        for i in range(len(paras[key0])):
            for j in range(len(paras[key1])):
                paras_ret.append(({key0:paras[key0][i]},{key1:paras[key1][j]}))
    elif dim == 3:
        key0 = list(paras.keys())[0]
        key1 = list(paras.keys())[1]
        key2 = list(paras.keys())[2]
        for i in range(len(paras[key0])):
            for j in range(len(paras[key1])):
                for k in range(len(paras[key2])):
                    paras_ret.append(({key0:paras[key0][i]},{key1:paras[key1][j]},{key2:paras[key2][k]}))
    elif dim == 4:
        key0 = list(paras.keys())[0]
        key1 = list(paras.keys())[1]
        key2 = list(paras.keys())[2]
        key3 = list(paras.keys())[3]
        for i in range(len(paras[key0])):
            for j in range(len(paras[key1])):
                for k in range(len(paras[key2])):
                    for l in range(len(paras[key3])):
                        paras_ret.append(({key0:paras[key0][i]},{key1:paras[key1][j]},{key2:paras[key2][k]},{key3:paras[key3][l]}))
    return paras_ret

def plot_module_selection(title,hyperparas,means,stds,width=0.25,save=False):
    """

    :param title:
    :param hyperparas: names of [h0,h1,h2,h3]
    :param means:[[h0 s0,h0 s1],[h1 s0,h1 s1],[h2 s0,h2 s1],[h3 s0,h3 s1]]
    :param stds: same as means
    :return: a plot
    """
    n_subjects = len(means[0])
    n_hp = len(means)
    labels = []
    width = width
    for i in range(n_subjects):
        labels.append('S'+str(i+1))
    x = np.arange(len(labels)) # the label locations

    means = list(np.array(means))

    fig, ax = plt.subplots()
    err_attr = {"elinewidth": 2, "ecolor": "black", "capsize": n_hp}
    rects = []
    for i in range(n_hp):
        rect = ax.bar(x + ((i+1)-n_hp/2)*(width)/n_hp, means[i],yerr=stds[i],error_kw=err_attr,width=width/n_hp,label=str(hyperparas[i]))
        for rec in rect:
            height = rec.get_height()
            ax.annotate('{:.2f}'.format(height),
                        xy=(rec.get_x() + rec.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        rects.append(rect)

    ax.set_ylabel('Accuracy')
    ax.set_title(str(title))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc=4)
    fig.tight_layout()
    plt.savefig(title+'.jpg')
    plt.show()

def logits_to_pred(model,x):
    if type(x) != type(torch.ones((1,1))):
        x = torch.Tensor(x)
    logits = model(x)
    pred = torch.argmax(logits,dim=1)
    return pred


def plot_acc_bars(title,model_names,means,width=0.25,save=False):
    """

    :param title:
    :param hyperparas: names of [h0,h1,h2,h3]
    :param means:[[h0 s0,h0 s1],[h1 s0,h1 s1],[h2 s0,h2 s1],[h3 s0,h3 s1]]
    :param stds: same as means
    :return: a plot
    """
    n_subjects = len(means[0])
    n_hp = len(means)
    labels = []
    width = width
    for i in range(n_subjects):
        labels.append('S'+str(i+1))
    x = np.arange(len(labels)) # the label locations

    means = list(np.array(means))

    fig, ax = plt.subplots()
    rects = []
    for i in range(n_hp):
        rect = ax.bar(x + ((i+1)-n_hp/2)*(width)/n_hp, means[i],width=width/n_hp,label=str(model_names[i]))
        for rec in rect:
            height = rec.get_height()
            ax.annotate('{:.2f}'.format(height),
                        xy=(rec.get_x() + rec.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        rects.append(rect)

    ax.set_ylabel('Accuracy')
    ax.set_title(str(title))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc=4)
    fig.tight_layout()
    plt.savefig(title+'.jpg')
    plt.show()


def plot_acc_bars_no_anno(title,model_names,means,width=0.25,save=False):
    """

    :param title:
    :param hyperparas: names of [h0,h1,h2,h3]
    :param means:[[h0 s0,h0 s1],[h1 s0,h1 s1],[h2 s0,h2 s1],[h3 s0,h3 s1]]
    :param stds: same as means
    :return: a plot
    """
    n_subjects = len(means[0])
    n_hp = len(means)
    labels = []
    width = width
    for i in range(n_subjects):
        labels.append('S'+str(i+1))
    x = np.arange(len(labels)) # the label locations

    means = list(np.array(means))

    fig, ax = plt.subplots()
    rects = []
    for i in range(n_hp):
        rect = ax.bar(x + ((i+1)-n_hp/2)*(width)/n_hp, means[i],width=width/n_hp,label=str(model_names[i]))
        rects.append(rect)

    ax.set_ylabel('Accuracy')
    ax.set_title(str(title))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc=4)
    fig.tight_layout()
    plt.savefig(title+'.jpg')
    plt.show()
# means = [[100,68,79],[120,70,75],[44, 55,66],[31,42,63],[100,68,79],[120,70,75],[44, 55,66],[31,42,63]]
# plot_acc_bars('test',['h0','h1','h2','h3','h0','h1','h2','h3'],means,width=0.94)

# # e.g.
# means = [[100,68,79],[120,70,75],[44, 55,66],[31,42,63],[100,68,79],[120,70,75],[44, 55,66],[31,42,63]]
# stds = [[7,2,6],[5,1,6],[3,4,5],[1,3,4],[7,2,6],[5,1,6],[3,4,5],[1,3,4]]
#
# plot_module_selection('test',['h0','h1','h2','h3','h0','h1','h2','h3'],means,stds,width=0.94)

def split_dataset(dataset,train_rate,valid_rate):
    """

    :param dataset: have to be subclass of torch.utils.data.Dataset
    :param train_rate:
    :param valid_rate:
    :return: train dataset,test dataset if valid_rate=0
             train dataset,valid dataset,test dataset if valid_rate != 0
    """
    total_len = len(dataset)
    train_len = round(total_len * train_rate)
    valid_len = round(total_len * valid_rate)
    test_len = total_len - train_len - valid_len
    try:
        if valid_rate==0:
            train_dataset, test_dataset = torch.utils.data.random_split(dataset,[train_len, test_len])
            return train_dataset,test_dataset
        else:
            train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset,[train_len, valid_len, test_len])
            return train_dataset,valid_dataset,test_dataset
    except:
        print('have to call EEGdataset.load_data first')

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

def reload(model,file_name,optimizer=None):
    checkpoint = torch.load(file_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer!=None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch,loss

def plot_ROC(y_true,pred):
    """

    :param y_true: label
    :param pred: transform score of x
    :return:
    """
    fpr, tpr, thresholds = roc_curve(y_true, pred, pos_label=2)
    plt.plot(fpr,tpr,label = 'deepfbcspnet_ROC',linestyle=':', linewidth=4)

# tran_deepfbcspnet = model_deep_fbcspnet.transform(test_data).detach().numpy()[:,0]
# plot_ROC(y_test_c,tran_deepfbcspnet)
"""
#examples
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold

from BCICdatasets import BCIC4_2a,Subject_dataset,BCIC4_2a_plus,BCIC3_5
from modules import FBCSPNet,deep_FBCSPNet,Point_wiseLSTM,multi_deep_FBCSPNet

data = BCIC4_2a()
data.load_data()
data = Subject_dataset(database=data,subject_id=0)

train_dataset,valid_dataset,test_dataset = split_dataset(data,0.8,0.1)
batch_size = 16
assert batch_size<len(test_dataset)

model = FBCSPNet(batch_size=batch_size, in_chans=22, time_steps=176, classes=4)
model.fit(train_dataset=train_dataset,max_epoch=5,batch_size=batch_size,test_dataset= test_dataset)

model_deep = deep_FBCSPNet(batch_size=batch_size, in_chans=22, time_steps=176, classes=4)
model_deep.fit(train_dataset=train_dataset,max_epoch=5,batch_size=batch_size,test_dataset= test_dataset)

data_multi = BCIC4_2a_plus()
data_multi.load_data()
data_multi = Subject_dataset(database=data_multi,subject_id=0)
train_dataset,valid_dataset,test_dataset = split_dataset(data_multi,0.8,0.1)
batch_size = 16
assert batch_size<len(test_dataset)

model_multi = multi_deep_FBCSPNet(batch_size=16,in_chans=22,time_steps=1050,classes=4)
model_multi.fit(train_dataset=train_dataset,max_epoch=5,batch_size=batch_size,test_dataset= test_dataset,lambda_cls=0.0001)

data_point = BCIC3_5()
data_point.load_data()
data_point = Subject_dataset(database=data_point,subject_id=0)
train_dataset,valid_dataset,test_dataset = split_dataset(data_point,0.8,0.1)
batch_size = 16
assert batch_size<len(test_dataset)

model_point = Point_wiseLSTM(in_chans=32,emd_dim=32,classes=3)
model_point.fit(train_dataset=train_dataset,max_epoch=5,batch_size=batch_size,test_dataset= test_dataset)

"""




