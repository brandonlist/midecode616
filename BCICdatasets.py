from scipy.io import loadmat
from BCI_database import EEG_database
import numpy as np
import numpy.ma as npm
import os
import mne
import scalp

class BCIC2_3(EEG_database):
    """
    datasets:BCI Competition II 3
    see   for more information
    """
    def __init__(self,path=r'G:\undergraduate\MIdatabase\BCIC2\3\dataset_BCIcomp1.mat'):
        super(BCIC2_3, self).__init__(name='BCIC23',n_subject=1)
        self.path = path
        self.fs = 128
        self.ys = (1,2)
        self.n_classes = 2
        self.chi_names = ['左手','右手']

    def load_data(self,*args):
        data_all = loadmat(self.path)
        x_data = data_all['x_train']
        y_data = data_all['y_train']
        x_data = np.swapaxes(x_data, 0, 2)
        # x_data:[140(trials),3(channels),1152(time_steps)]
        super(BCIC2_3, self).load_data(*args)
        for i in range(self.n_subject):
            "operate on self.subjects_data[i] "
            self.subjects_data[i].load_data(n_trial=140)
            for idx,trial in enumerate(x_data):
                self.subjects_data[i].subject_trials[idx].signal = trial
                self.subjects_data[i].subject_trials[idx].target = y_data[idx][0]
#(3,1152)
#class 2

class BCIC3_3a(EEG_database):
    """
    datasets:BCI Competition III 3a
    see   for more information
    """
    def __init__(self,path=r'G:\undergraduate\MIdatabase\BCIC3\3\3a',uv=False):
        super(BCIC3_3a, self).__init__(name='BCIC33a',n_subject=3)
        self.path = path
        self.eventDescription = {'276': 'eyesOpen', '277': 'eyesClosed', '768': 'startTrail', '769': 'cueOnsetLeft',
                                 '770': 'cueOnsetRight'
            , '771': 'cueOnsetFoot', '772': 'cueOnsetTongue', '783': 'cueUnknown', '1023': 'rejectedTrial',
                                 '1072': 'eyeMovements'
            , '32766': 'startOfNewRun','785':'really unknown1','786':'really unknown2'}
        self.classes = ['cueOnsetLeft','cueOnsetRight','cueOnsetTongue','cueOnsetFoot']
        self.class_dict ={'cueOnsetLeft':1,'cueOnsetRight':2,'cueOnsetTongue':4,'cueOnsetFoot':3}
        self.uv = uv
        self.fs = 250
        self.names = ['#  1', '#  2', '#  3', '#  4', '#  5', '#  6', '#  7', '#  8', '#  9', '# 10', '# 11', '# 12', '# 13', '# 14',
         '# 15', '# 16', '# 17', '# 18', '# 19', '# 20', '# 21', '# 22', '# 23', '# 24', '# 25', '# 26', '# 27', '# 28',
         '# 29', '# 30', '# 31', '# 32', '# 33', '# 34', '# 35', '# 36', '# 37', '# 38', '# 39', '# 40', '# 41', '# 42',
         '# 43', '# 44', '# 45', '# 46', '# 47', '# 48', '# 49', '# 50', '# 51', '# 52', '# 53', '# 54', '# 55', '# 56',
         '# 57', '# 58', '# 59', '# 60']
        self.C3 = 28
        self.C4 = 34
        self.n_classes = 4
        self.chi_names = ['左手', '右手','脚','舌头']
        positions = [x for x in scalp.BCIC3_3a_scalp_position.values()]
        positions = np.array(positions) * 0.10
        self.positions = positions

    def load_data(self,*args):
        super(BCIC3_3a, self).load_data(*args)
        for i in range(self.n_subject):
            "operate on self.subjects_data[i] "

            path_s = os.path.join(self.path,'s'+str(i+1))
            for file_path in os.listdir(path_s):
                if file_path.split('.')[-1] == 'gdf':
                    path_s = os.path.join(path_s,file_path)

            raw = mne.io.read_raw_gdf(path_s)
            event, _ = mne.events_from_annotations(raw)
            event_id = {}
            for code in _:
                event_id[self.eventDescription[code]] = _[code]
            # e.g.:{'cueOnsetLeft':7,'cueOnsetRight':8}
            epochs = mne.Epochs(raw, event, event_id, event_repeated='merge',tmin=-0.1, tmax=4)
            #compute n_trail
            count = 0
            for class_i in self.classes:
                for trial in epochs[class_i]:
                    count = count+1
            self.subjects_data[i].load_data(n_trial=count)

            count = 0
            for class_i in self.classes:
                for trial in epochs[class_i]:
                    if self.uv:
                        self.subjects_data[i].subject_trials[count].signal = trial*1e6
                    else:
                        self.subjects_data[i].subject_trials[count].signal = trial
                    self.subjects_data[i].subject_trials[count].target = self.class_dict[class_i]
                    count = count+1
#(60,1026)
#class 4


class BCIC3_3b(EEG_database):
    """
    datasets:BCI Competition III 3b
    see   for more information
    """
    def __init__(self,path=r'G:\undergraduate\MIdatabase\BCIC3\3\3b',uv=False):
        super(BCIC3_3b, self).__init__(name='BCIC33b',n_subject=3)
        self.path = path
        self.eventDescription = {'276': 'eyesOpen', '277': 'eyesClosed', '768': 'startTrail', '769': 'cueOnsetLeft',
                                 '770': 'cueOnsetRight'
            , '771': 'cueOnsetFoot', '772': 'cueOnsetTongue', '783': 'cueUnknown', '1023': 'rejectedTrial',
                                 '1072': 'eyeMovements'
            , '32766': 'startOfNewRun','785':'really unknown1','781':'really unknown2'}
        self.classes = ['cueOnsetLeft','cueOnsetRight']
        self.classcode = {'cueOnsetLeft':1,'cueOnsetRight':2}
        self.uv = uv
        self.fs = 125
        self.n_classes = 2
        self.chi_names = ['左手', '右手']

    def load_data(self,*args):
        super(BCIC3_3b, self).load_data(*args)
        for i in range(self.n_subject):
            "operate on self.subjects_data[i] "

            path_s = os.path.join(self.path,'s'+str(i+1))
            for file_path in os.listdir(path_s):
                if file_path.split('.')[-1] == 'gdf':
                    path_s = os.path.join(path_s,file_path)

            raw = mne.io.read_raw_gdf(path_s)
            event, _ = mne.events_from_annotations(raw)
            event_id = {}
            for code in _:
                event_id[self.eventDescription[code]] = _[code]
            # e.g.:{'cueOnsetLeft':7,'cueOnsetRight':8}
            epochs = mne.Epochs(raw, event, event_id, event_repeated='merge',tmin=-0.1, tmax=4)
            #compute n_trail
            count = 0
            for class_i in self.classes:
                for trial in epochs[class_i]:
                    count = count+1
            self.subjects_data[i].load_data(n_trial=count)

            count = 0
            for class_i in self.classes:
                for trial in epochs[class_i]:
                    if self.uv:
                        self.subjects_data[i].subject_trials[count].signal = trial*1e6
                    else:
                        self.subjects_data[i].subject_trials[count].signal = trial
                    self.subjects_data[i].subject_trials[count].target = self.classcode[class_i]
                    count = count+1
#(2,1026)
#class 2

class BCIC3_4a(EEG_database):
    """
    datasets:BCI Competition III 4a
    see   for more information
    bug0: in load_data, the time length of a trial (length) can not be too big due to menmery failure
        in fact it can only be small numbers like 1 now (wkl) future adaption of preload-false
        method is severely needed
    """
    def __init__(self,path=r'G:\undergraduate\MIdatabase\BCIC3\4\4a\1000Hz'):
        super(BCIC3_4a, self).__init__(name='BCIC34a',n_subject=5)
        self.path = path
        self.classes=[1,2]
        self.fs = 1000
        self.names = ['Fp1', 'AFp1', 'Fpz', 'AFp2', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'FAF5', 'FAF1', 'FAF2', 'FAF6',
                      'F7',
                      'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FFC7', 'FFC5', 'FFC3', 'FFC1', 'FFC2', 'FFC4',
                      'FFC6',
                      'FFC8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'CFC7',
                      'CFC5',
                      'CFC3', 'CFC1', 'CFC2', 'CFC4', 'CFC6', 'CFC8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                      'T8',
                      'CCP7', 'CCP5', 'CCP3', 'CCP1', 'CCP2', 'CCP4', 'CCP6', 'CCP8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
                      'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'PCP7', 'PCP5', 'PCP3', 'PCP1', 'PCP2', 'PCP4', 'PCP6',
                      'PCP8', 'P9', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10', 'PPO7', 'PPO5', 'PPO1',
                      'PPO2', 'PPO6', 'PPO8', 'PO7', 'PO3', 'PO1', 'POz', 'PO2', 'PO4', 'PO8', 'OPO1', 'OPO2', 'O1',
                      'Oz',
                      'O2', 'OI1', 'OI2', 'I1', 'I2']
        self.C3 = 51
        self.C4 = 55
        self.n_classes = 2
        self.chi_names = ['左手', '右手']
        positions = [x for x in scalp.BCIC3_4a_scalp_position.values()]
        positions = np.array(positions) * 0.10
        self.positions = positions

    def load_data(self,*args,length = 3200):
        super(BCIC3_4a, self).load_data(*args)
        for i in range(self.n_subject):
            "operate on self.subjects_data[i] "
            file_mat_path = [p for p in os.listdir(os.path.join(self.path,'s'+str(i+1))) if p.split('.')[-1] =='mat'][0]
            file_mat_path = os.path.join(self.path,'s'+str(i+1),file_mat_path)
            data_subject_i = loadmat(file_mat_path)
            mask_all = data_subject_i['mrk']['y'].item() < 5
            n_trial = mask_all.sum()
            count = 0
            self.subjects_data[i].load_data(n_trial=n_trial)
            for class_i in self.classes:
                count_i =0
                mask_i = data_subject_i['mrk']['y'].item() == int(class_i)
                n_trial_i = mask_i.sum()
                index_list = npm.array(data_subject_i['mrk']['pos'].item(), mask=mask_i)
                index_list = [ind for ind in index_list[0] if ind != False]
                for trial_i in range(n_trial_i):
                    signal = data_subject_i['cnt'][index_list[count_i]:index_list[count_i]+length,:]
                    signal = np.swapaxes(signal,0,1)
                    self.subjects_data[i].subject_trials[count].signal = signal
                    self.subjects_data[i].subject_trials[count].target = class_i
                    count = count+1
                    count_i = count_i+1
#(118,length) length=3200
#class 2


class BCIC3_4b(EEG_database):
    """
    datasets:BCI Competition III 4b
    see   for more information
    bug0: in load_data, the time length of a trial (length) can not be too big due to menmery failure
        in fact it can only be small numbers like 1 now (wkl) future adaption of preload-false
        method is severely needed
    """
    def __init__(self,path=r'G:\undergraduate\MIdatabase\BCIC3\4\4b\1000Hz'):
        super(BCIC3_4b, self).__init__(name='BCIC34b',n_subject=1)
        self.path = path
        self.classes=['1','-1']
        self.class_dict={'1':1,'-1':2}
        self.fs = 1000
        self.names = ['Fp1','AFp1','Fpz', 'AFp2', 'Fp2', 'AF7', 'AF3', 'AF4','AF8', 'FAF5', 'FAF1', 'FAF2', 'FAF6','F7',
                      'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FFC7', 'FFC5', 'FFC3', 'FFC1', 'FFC2', 'FFC4', 'FFC6',
                      'FFC8', 'FT9', 'FT7','FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'CFC7', 'CFC5',
                      'CFC3', 'CFC1', 'CFC2', 'CFC4', 'CFC6', 'CFC8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
                      'CCP7', 'CCP5', 'CCP3', 'CCP1', 'CCP2', 'CCP4', 'CCP6', 'CCP8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
                      'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'PCP7', 'PCP5', 'PCP3', 'PCP1', 'PCP2', 'PCP4', 'PCP6',
                      'PCP8', 'P9', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10', 'PPO7', 'PPO5', 'PPO1',
                      'PPO2', 'PPO6', 'PPO8', 'PO7', 'PO3','PO1', 'POz', 'PO2', 'PO4', 'PO8', 'OPO1', 'OPO2', 'O1', 'Oz',
                      'O2', 'OI1', 'OI2', 'I1', 'I2']
        self.C3 = 51
        self.C4 = 55
        self.n_classes = 2
        self.chi_names = ['左手', '右手']
        positions = [x for x in scalp.BCIC3_4b_scalp_position.values()]
        positions = np.array(positions) * 0.10
        self.positions = positions

    def load_data(self,*args,length = 3200):
        super(BCIC3_4b, self).load_data(*args)
        for i in range(self.n_subject):
            "operate on self.subjects_data[i] "
            file_mat_path_train = os.path.join(self.path,'data_set_IVb_al_train.mat')
            data_subject_i = loadmat(file_mat_path_train)
            n_trial = abs(data_subject_i['mrk']['y'].item()).sum()
            count = 0
            self.subjects_data[i].load_data(n_trial=n_trial)
            for class_i in self.classes:
                count_i =0
                mask_i = data_subject_i['mrk']['y'].item() == int(class_i)
                n_trial_i = mask_i.sum()
                index_list = npm.array(data_subject_i['mrk']['pos'].item(), mask=mask_i)
                index_list = [ind for ind in index_list[0] if ind != False]
                for trial_i in range(n_trial_i):
                    signal = data_subject_i['cnt'][index_list[count_i]:index_list[count_i]+length,:]
                    signal = np.swapaxes(signal,0,1)
                    self.subjects_data[i].subject_trials[count].signal = signal
                    self.subjects_data[i].subject_trials[count].target = self.class_dict[class_i]
                    count = count+1
                    count_i = count_i+1
#(118,length) length=3200
#class 2


class BCIC3_5(EEG_database):
    """
    datasets:BCI Competition III 5
    see   for more information
    Be aware: This datasets provide label for every sample point,

    """
    def __init__(self,path=r'G:\undergraduate\MIdatabase\BCIC3\5'):
        super(BCIC3_5, self).__init__(name='BCIC35',n_subject=3)
        self.path = path
        self.classes=[2,3,7]
        self.classcode={2:0,3:1,7:2}
        self.fs = 512
        self.n_classes = 3
        self.chi_names = ['左手', '右手','想象单词']

    def load_data(self,*args,length = 8000):
        super(BCIC3_5, self).load_data(*args)
        print('Loading data in continues manner, every trial contains different labels')
        for i in range(self.n_subject):
            "operate on self.subjects_data[i] "
            file_mat_path_train = os.path.join(self.path,'s'+str(i+1))
            file_names = [p for p in os.listdir(file_mat_path_train) if p.split('.')[-1] == 'mat' and p.split('_')[0] == 'train']
            # self.subjects_data[i].load_data(n_trial=3)
            subject_data_x =[]
            subject_data_y =[]
            for j,file_name in enumerate(file_names):
                file_path = os.path.join(file_mat_path_train,file_name)
                data_subject_i = loadmat(file_path)
                #operate data in each file
                subject_data_x.append(data_subject_i['X'].transpose())
                for key in self.classcode:
                    value = self.classcode[key]
                    data_subject_i['Y'][data_subject_i['Y']==key]=value
                subject_data_y.append(data_subject_i['Y'].transpose())
            self.subject_data_x = np.hstack([x for x in subject_data_x])
            self.subject_data_y = np.hstack([y for y in subject_data_y])
            n_trial = int(np.floor((self.subject_data_y.shape[1])/length))
            self.subjects_data[i].load_data(n_trial=n_trial)
            count = 0
            for k in range(n_trial):
                self.subjects_data[i].subject_trials[k].signal=self.subject_data_x[:,count:(count+length)]
                self.subjects_data[i].subject_trials[k].target=self.subject_data_y[:,count:(count+length)]
                count += length
#(32,length) length=8000
#class 3


class BCIC4_1(EEG_database):
    """
    datasets:BCI Competition IV 1
    see   for more information
    """
    def __init__(self,path=r'G:\undergraduate\MIdatabase\BCIC4\1\1_1000Hz'):
        super(BCIC4_1, self).__init__(name='BCIC41',n_subject=7)
        self.path = path
        self.classes=['1','-1']
        self.fs = 1000
        self.class_dict = {'1':1,'-1':2}
        self.names = ['AF3','AF4','F5','F3','F1','Fz','F2','F4','F6','FC5','FC3','FC1','FCz','FC2','FC4','FC6','CFC7',
                    'CFC5','CFC3','CFC1','CFC2','CFC4','CFC6','CFC8','T7','C5','C3','C1','Cz','C2','C4','C6','T8','CCP7',
                      'CCP5','CCP3','CCP1','CCP2','CCP4','CCP6','CCP8','CP5','CP3','CP1','CPz','CP2','CP4','CP6','P5',
                      'P3','P1','Pz','P2','P4','P6','PO1','PO2','O1', 'O2']
        self.C3 = 26
        self.C4 = 30
        self.n_classes = 2
        self.chi_names = ['左手', '右手']
        positions = [x for x in scalp.BCIC4_1_scalp_position.values()]
        positions = np.array(positions) * 0.10
        self.positions = positions

    def load_data(self,*args,length = 3200):
        super(BCIC4_1, self).load_data(*args)
        file_names = [p for p in os.listdir(self.path) if p.split('.')[-1] =='mat']
        for i in range(self.n_subject):
            "operate on self.subjects_data[i] "
            file_mat_path_train = os.path.join(self.path,file_names[i])
            data_subject_i = loadmat(file_mat_path_train)
            n_trial = abs(data_subject_i['mrk']['y'].item()).sum()
            count = 0
            self.subjects_data[i].load_data(n_trial=n_trial)
            for class_i in self.classes:
                count_i =0
                mask_i = data_subject_i['mrk']['y'].item() == int(class_i)
                n_trial_i = mask_i.sum()
                index_list = npm.array(data_subject_i['mrk']['pos'].item(), mask=mask_i)
                index_list = [ind for ind in index_list[0] if ind != False]
                for trial_i in range(n_trial_i):
                    signal = data_subject_i['cnt'][index_list[count_i]:index_list[count_i]+length,:]
                    signal = np.swapaxes(signal,0,1)
                    self.subjects_data[i].subject_trials[count].signal = signal
                    self.subjects_data[i].subject_trials[count].target = self.class_dict[class_i]
                    count = count+1
                    count_i = count_i+1
#(59,length) length=3200
#2 class

class BCIC4_2a(EEG_database):
    """
    datasets:BCI Competition IV 2a
    see   for more information
    """
    def __init__(self,path=r'G:\undergraduate\MIdatabase\BCIC4\2\a',uv=False):
        super(BCIC4_2a, self).__init__(name='BCIC42a',n_subject=9)
        self.uv = uv
        self.path = path
        self.eventDescription = {'276': 'eyesOpen', '277': 'eyesClosed', '768': 'startTrail', '769': 'cueOnsetLeft',
                                 '770': 'cueOnsetRight'
            , '771': 'cueOnsetFoot', '772': 'cueOnsetTongue', '783': 'cueUnknown', '1023': 'rejectedTrial',
                                 '1072': 'eyeMovements'
            , '32766': 'startOfNewRun'}
        self.eventDescription_e = {'1023': 'rejectedTrial','1072': 'eyeMovements','276': 'eyesOpen', '277': 'eyesClosed',
                                   '32766': 'startOfNewRun','768': 'startTrail','783': 'cueUnknown'}
        self.classes = ['cueOnsetLeft','cueOnsetRight','cueOnsetTongue','cueOnsetFoot']
        self.class_dict={1:'cueOnsetLeft',2:'cueOnsetRight',4:'cueOnsetTongue',3:'cueOnsetFoot'}
        self.class_dict_t= {'cueOnsetLeft':1,'cueOnsetRight':2, 'cueOnsetTongue':4, 'cueOnsetFoot':3}
        self.fs = 250
        self.names = ['EEG-Fz','EEG-0','EEG-1','EEG-2','EEG-3','EEG-4','EEG-5','EEG-C3', 'EEG-6', 'EEG-Cz','EEG-7', 'EEG-C4','EEG-8',
                      'EEG-9','EEG-10','EEG-11','EEG-12','EEG-13', 'EEG-14','EEG-Pz', 'EEG-15','EEG-16']
        self.C3 = 7
        self.C4 = 11
        self.n_classes = 4
        self.chi_names = ['左手', '右手','脚','舌头']
        positions = [x for x in scalp.BCIC4_2a_scalp_position.values()]
        positions = np.array(positions) * 0.17
        self.positions = positions


    def load_data(self,*args):
        super(BCIC4_2a, self).load_data(*args)
        for i in range(self.n_subject):
            "operate on self.subjects_data[i] "
            path_s = os.path.join(self.path,'s'+str(i+1))
            for file_path in os.listdir(path_s):
                if file_path.split('.')[-1] == 'gdf' and file_path.find('T') !=-1:
                    path_s_t = os.path.join(path_s,file_path)
                if file_path.split('.')[-1] == 'gdf' and file_path.find('E') !=-1:
                    path_s_e = os.path.join(path_s,file_path)
            raw_t = mne.io.read_raw_gdf(path_s_t)
            event_t, _t = mne.events_from_annotations(raw_t)
            event_id = {}
            for code in _t:
                event_id[self.eventDescription[code]] = _t[code]
            # e.g.:{'cueOnsetLeft':7,'cueOnsetRight':8}
            raw_t.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']
            picks = mne.pick_types(raw_t.info, eeg=True, exclude='bads')
            epochs_t = mne.Epochs(raw_t, event_t, event_id, event_repeated='merge', picks=picks,tmin=-0.1, tmax=4)

            #compute n_trai of A0XT.gdf
            count = 0
            for class_i in self.classes:
                for trial in epochs_t[class_i]:
                    count = count+1

          #A0XE
            # raw_e = mne.io.read_raw_gdf(path_s_e)
            # event_e, _e = mne.events_from_annotations(raw_e)
            # event_id_e = {}
            # for code in _e:
            #     event_id_e[self.eventDescription_e[code]] = _e[code]
            # # e.g.:{'cueOnsetLeft':7,'cueOnsetRight':8}
            # label_mat = loadmat(path_s_e.replace('gdf','mat'))
            # labels = list(label_mat['classlabel'])
            # # compute n_trai of A0XE.gdf
            # count += len(labels)

            # epochs_e = mne.Epochs(raw_e, event_e, event_id_e, event_repeated='merge')

            self.subjects_data[i].load_data(n_trial=count)

            count = 0
            for class_i in self.classes:
                for trial in epochs_t[class_i]:
                    if self.uv:
                        self.subjects_data[i].subject_trials[count].signal = trial*1e6
                    else:
                        self.subjects_data[i].subject_trials[count].signal = trial
                    self.subjects_data[i].subject_trials[count].target = np.array(self.class_dict_t[class_i])
                    count = count+1
         #
         #    for idx,trial in enumerate(epochs_e['cueUnknown']):
         #        self.subjects_data[i].subject_trials[count].signal = trial
         # #       self.subjects_data[i].subject_trials[count].target = self.class_dict[int(labels[idx])]
         #        self.subjects_data[i].subject_trials[count].target = np.array(labels[idx])
         #        count = count + 1
    def get_data(self):
        data_signals = []
        data_targets = []
        for data_subject in self.subjects_data:
            for data_trial in data_subject.subject_trials:
                data_signals.append(data_trial.signal)
                data_targets.append(data_trial.target)
        return data_signals,data_targets
#(22,1026)
#4 class


class BCIC4_2a_lrh(EEG_database):
    """
    datasets:BCI Competition IV 2a
    see   for more information
    extract only 2 classes
    """
    def __init__(self,path=r'G:\undergraduate\MIdatabase\BCIC4\2\a',fs=250,uv=False):
        super(BCIC4_2a_lrh, self).__init__(name='BCIC42alrh',n_subject=9)
        self.uv = uv
        self.fs = fs
        self.path = path
        self.eventDescription = {'276': 'eyesOpen', '277': 'eyesClosed', '768': 'startTrail', '769': 'cueOnsetLeft',
                                 '770': 'cueOnsetRight'
            , '771': 'cueOnsetFoot', '772': 'cueOnsetTongue', '783': 'cueUnknown', '1023': 'rejectedTrial',
                                 '1072': 'eyeMovements'
            , '32766': 'startOfNewRun'}
        self.eventDescription_e = {'1023': 'rejectedTrial','1072': 'eyeMovements','276': 'eyesOpen', '277': 'eyesClosed',
                                   '32766': 'startOfNewRun','768': 'startTrail','783': 'cueUnknown'}
        self.classes = ['cueOnsetLeft','cueOnsetRight']
        self.class_dict={1:'cueOnsetLeft',2:'cueOnsetRight'}
        self.class_dict_t= {'cueOnsetLeft':1,'cueOnsetRight':2}
        self.fs = 250
        self.names = ['EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5', 'EEG-C3', 'EEG-6', 'EEG-Cz',
                      'EEG-7', 'EEG-C4', 'EEG-8',
                      'EEG-9', 'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-Pz', 'EEG-15', 'EEG-16']
        self.n_classes = 4


    def load_data(self,*args):
        super(BCIC4_2a_lrh, self).load_data(*args)
        for i in range(self.n_subject):
            "operate on self.subjects_data[i] "
            path_s = os.path.join(self.path,'s'+str(i+1))
            for file_path in os.listdir(path_s):
                if file_path.split('.')[-1] == 'gdf' and file_path.find('T') !=-1:
                    path_s_t = os.path.join(path_s,file_path)
                if file_path.split('.')[-1] == 'gdf' and file_path.find('E') !=-1:
                    path_s_e = os.path.join(path_s,file_path)
            raw_t = mne.io.read_raw_gdf(path_s_t)
            event_t, _t = mne.events_from_annotations(raw_t)
            event_id = {}
            for code in _t:
                event_id[self.eventDescription[code]] = _t[code]
            # e.g.:{'cueOnsetLeft':7,'cueOnsetRight':8}
            raw_t.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']
            picks = mne.pick_types(raw_t.info, eeg=True, exclude='bads')
            epochs_t = mne.Epochs(raw_t, event_t, event_id, event_repeated='merge', picks=picks,tmin=-0.1, tmax=4)

            #compute n_trai of A0XT.gdf
            count = 0
            for class_i in self.classes:
                for trial in epochs_t[class_i]:
                    count = count+1

          #A0XE
            # raw_e = mne.io.read_raw_gdf(path_s_e)
            # event_e, _e = mne.events_from_annotations(raw_e)
            # event_id_e = {}
            # for code in _e:
            #     event_id_e[self.eventDescription_e[code]] = _e[code]
            # # e.g.:{'cueOnsetLeft':7,'cueOnsetRight':8}
            # label_mat = loadmat(path_s_e.replace('gdf','mat'))
            # labels = list(label_mat['classlabel'])
            # # compute n_trai of A0XE.gdf
            # count += len(labels)

            # epochs_e = mne.Epochs(raw_e, event_e, event_id_e, event_repeated='merge')

            self.subjects_data[i].load_data(n_trial=count)

            count = 0
            for class_i in self.classes:
                for trial in epochs_t[class_i]:
                    if self.uv:
                        self.subjects_data[i].subject_trials[count].signal = trial*1e6
                    else:
                        self.subjects_data[i].subject_trials[count].signal = trial
                    self.subjects_data[i].subject_trials[count].target = np.array(self.class_dict_t[class_i])
                    count = count+1
         #
         #    for idx,trial in enumerate(epochs_e['cueUnknown']):
         #        self.subjects_data[i].subject_trials[count].signal = trial
         # #       self.subjects_data[i].subject_trials[count].target = self.class_dict[int(labels[idx])]
         #        self.subjects_data[i].subject_trials[count].target = np.array(labels[idx])
         #        count = count + 1
    def get_data(self):
        data_signals = []
        data_targets = []
        for data_subject in self.subjects_data:
            for data_trial in data_subject.subject_trials:
                data_signals.append(data_trial.signal)
                data_targets.append(data_trial.target)
        return data_signals,data_targets
#(22,176)
#2 class


class BCIC4_2b(EEG_database):
    """
    datasets:BCI Competition IV 2b
    see   for more information
    6 channels , 176 time_steps
    """
    def __init__(self,path=r'G:\undergraduate\MIdatabase\BCIC4\2\b'):
        super(BCIC4_2b, self).__init__(name='BCIC42b',n_subject=9)
        self.path = path
        self.eventDescription = {'276': 'eyesOpen', '277': 'eyesClosed', '768': 'startTrail', '769': 'cueOnsetLeft',
                                 '770': 'cueOnsetRight', '1023': 'rejectedTrial',
                                 '32766': 'startOfNewRun','1077':'unKnown_1','1078':'unKnown_2','1079':'unKnown_3','1081':'unKnown_4'}
        self.eventDescription_e = {'1023': 'rejectedTrial','1072': 'eyeMovements','276': 'eyesOpen', '277': 'eyesClosed',
                                   '32766': 'startOfNewRun','768': 'startTrail','783': 'cueUnknown'}
        self.classes = ['cueOnsetLeft','cueOnsetRight']
        self.class_dict={1:'cueOnsetLeft',3:'cueOnsetRight'}
        self.class_dict_t= {'cueOnsetLeft':1,'cueOnsetRight':2}
        self.fs = 250
        self.n_classes = 2
        self.chi_names = ['左手', '右手']

    def load_data(self,*args):
        super(BCIC4_2b, self).load_data(*args)
        for i in range(self.n_subject):
            "operate on self.subjects_data[i] "
            path_s = os.path.join(self.path,'s'+str(i+1))
            self.path_s_ts = []
            self.raws = []
            for file_path in os.listdir(path_s):
                if file_path.split('.')[-1] == 'gdf' and file_path.find('T') !=-1:
                    path_s_t = os.path.join(path_s,file_path)
                    self.path_s_ts.append(path_s_t)
                if file_path.split('.')[-1] == 'gdf' and file_path.find('E') !=-1:
                    path_s_e = os.path.join(path_s,file_path)
            for file_path in self.path_s_ts:
                raw = mne.io.read_raw_gdf(file_path)
                self.raws.append(raw)

            raw_t = mne.concatenate_raws([r for r in self.raws])
            event_t, _t = mne.events_from_annotations(raw_t)
            event_id = {}
            for code in _t:
                event_id[self.eventDescription[code]] = _t[code]
            # e.g.:{'cueOnsetLeft':7,'cueOnsetRight':8}
            epochs_t = mne.Epochs(raw_t, event_t, event_id, event_repeated='merge',tmin=-0.1, tmax=4)

            #compute n_trai of A0XT.gdf
            count = 0
            for class_i in self.classes:
                for trial in epochs_t[class_i]:
                    count = count+1

          #A0XE
            # raw_e = mne.io.read_raw_gdf(path_s_e)
            # event_e, _e = mne.events_from_annotations(raw_e)
            # event_id_e = {}
            # for code in _e:
            #     event_id_e[self.eventDescription_e[code]] = _e[code]
            # # e.g.:{'cueOnsetLeft':7,'cueOnsetRight':8}
            # label_mat = loadmat(path_s_e.replace('gdf','mat'))
            # labels = list(label_mat['classlabel'])
            # # compute n_trai of A0XE.gdf
            # count += len(labels)

            # epochs_e = mne.Epochs(raw_e, event_e, event_id_e, event_repeated='merge')

            self.subjects_data[i].load_data(n_trial=count)

            count = 0
            for class_i in self.classes:
                for trial in epochs_t[class_i]:
                    self.subjects_data[i].subject_trials[count].signal = trial
                    self.subjects_data[i].subject_trials[count].target = np.array(self.class_dict_t[class_i])
                    count = count+1
         #
         #    for idx,trial in enumerate(epochs_e['cueUnknown']):
         #        self.subjects_data[i].subject_trials[count].signal = trial
         # #       self.subjects_data[i].subject_trials[count].target = self.class_dict[int(labels[idx])]
         #        self.subjects_data[i].subject_trials[count].target = np.array(labels[idx])
         #        count = count + 1
    def get_data(self):
        data_signals = []
        data_targets = []
        for data_subject in self.subjects_data:
            for data_trial in data_subject.subject_trials:
                data_signals.append(data_trial.signal)
                data_targets.append(data_trial.target)
        return data_signals,data_targets
#(6,1026)
#2 class


class BCIC4_2a_plus(EEG_database):
    """
        datasets:BCI Competition IV 2a
        see   for more information
        provide class label and start position of each trial
        """

    def __init__(self, path=r'G:\undergraduate\MIdatabase\BCIC4\2\a',uv=False):
        super(BCIC4_2a_plus, self).__init__(name='BCIC42aplus', n_subject=9)
        self.uv =uv
        self.path = path
        self.eventDescription = {'276': 'eyesOpen', '277': 'eyesClosed', '768': 'startTrail', '769': 'cueOnsetLeft',
                                 '770': 'cueOnsetRight'
            , '771': 'cueOnsetFoot', '772': 'cueOnsetTongue', '783': 'cueUnknown', '1023': 'rejectedTrial',
                                 '1072': 'eyeMovements'
            , '32766': 'startOfNewRun'}
        self.eventDescription_e = {'1023': 'rejectedTrial', '1072': 'eyeMovements', '276': 'eyesOpen',
                                   '277': 'eyesClosed',
                                   '32766': 'startOfNewRun', '768': 'startTrail', '783': 'cueUnknown'}
        self.classes = ['cueOnsetLeft', 'cueOnsetRight', 'cueOnsetTongue', 'cueOnsetFoot']
        self.class_dict = {1: 'cueOnsetLeft', 2: 'cueOnsetRight', 4: 'cueOnsetTongue', 3: 'cueOnsetFoot'}
        self.class_dict_t = {'cueOnsetLeft': 1, 'cueOnsetRight': 2, 'cueOnsetTongue': 4, 'cueOnsetFoot': 3}
        self.fs = 250
        self.names = ['EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5', 'EEG-C3', 'EEG-6', 'EEG-Cz',
                      'EEG-7', 'EEG-C4', 'EEG-8',
                      'EEG-9', 'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-Pz', 'EEG-15', 'EEG-16']
        self.n_classes = 4

    def load_data(self, *args):
        super(BCIC4_2a_plus, self).load_data(*args)
        for i in range(self.n_subject):
            "operate on self.subjects_data[i] "
            path_s = os.path.join(self.path, 's' + str(i + 1))
            for file_path in os.listdir(path_s):
                if file_path.split('.')[-1] == 'gdf' and file_path.find('T') != -1:
                    path_s_t = os.path.join(path_s, file_path)
                if file_path.split('.')[-1] == 'gdf' and file_path.find('E') != -1:
                    path_s_e = os.path.join(path_s, file_path)
            raw_t = mne.io.read_raw_gdf(path_s_t)
            event_t, _t = mne.events_from_annotations(raw_t)
            #compute n_trials
            count = 0
            for idx, (time, _, label) in enumerate(event_t):
                if label == 6:
                    if event_t[idx + 1][2] == 7 or event_t[idx + 1][2] == 8 or event_t[idx + 1][2] == 9 or event_t[idx + 1][2] == 10:
                        try:
                            start_interval = event_t[idx + 2][0] - time
                            start_interval = start_interval * np.random.random()
                            start_time = int(time + start_interval)
                            try:
                                _tp = raw_t[0:22, start_time:start_time + 176][0]*1e6
                                count+=1
                            except:
                                continue
                        except:
                            continue
            self.subjects_data[i].load_data(n_trial=count)

            count = 0
            for idx, (time, _, label) in enumerate(event_t):
                if label == 6:
                    if event_t[idx + 1][2] == 7 or event_t[idx + 1][2] == 8 or event_t[idx + 1][2] == 9 or event_t[idx + 1][2] == 10:
                        try:
                            start_interval = event_t[idx + 2][0] - time
                            start_interval = start_interval * np.random.random()
                            start_time = int(time + start_interval)
                            if start_time > event_t[idx+1][0]:
                                target_interval = 0
                            else:
                                target_interval = 1
                            try:
                                if self.uv:
                                    self.subjects_data[i].subject_trials[count].signal = raw_t[0:22, start_time:start_time + 176][0]*1e6
                                else:
                                    self.subjects_data[i].subject_trials[count].signal = raw_t[0:22, start_time:start_time + 176][0]
                                self.subjects_data[i].subject_trials[count].target = (np.array(event_t[idx+1][2]),np.array(target_interval))
                                count+=1
                            except:
                                continue
                        except:
                            continue


class BCIC4_2a_3_2(EEG_database):
    """
        datasets:BCI Competition IV 2a and BCI Competition III 2
        see   for more information
        provide class label and start position of each trial
        """

    def __init__(self, path=r'G:\undergraduate\MIdatabase\BCIC4\2\a',uv=False):
        super(BCIC4_2a_plus, self).__init__(name='BCIC42aplus', n_subject=9)
        self.uv =uv
        self.path = path
        self.eventDescription = {'276': 'eyesOpen', '277': 'eyesClosed', '768': 'startTrail', '769': 'cueOnsetLeft',
                                 '770': 'cueOnsetRight'
            , '771': 'cueOnsetFoot', '772': 'cueOnsetTongue', '783': 'cueUnknown', '1023': 'rejectedTrial',
                                 '1072': 'eyeMovements'
            , '32766': 'startOfNewRun'}
        self.eventDescription_e = {'1023': 'rejectedTrial', '1072': 'eyeMovements', '276': 'eyesOpen',
                                   '277': 'eyesClosed',
                                   '32766': 'startOfNewRun', '768': 'startTrail', '783': 'cueUnknown'}
        self.classes = ['cueOnsetLeft', 'cueOnsetRight', 'cueOnsetTongue', 'cueOnsetFoot']
        self.class_dict = {1: 'cueOnsetLeft', 2: 'cueOnsetRight', 4: 'cueOnsetTongue', 3: 'cueOnsetFoot'}
        self.class_dict_t = {'cueOnsetLeft': 1, 'cueOnsetRight': 2, 'cueOnsetTongue': 4, 'cueOnsetFoot': 3}
        self.fs = 250
        self.names = ['EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5', 'EEG-C3', 'EEG-6', 'EEG-Cz',
                      'EEG-7', 'EEG-C4', 'EEG-8',
                      'EEG-9', 'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-Pz', 'EEG-15', 'EEG-16']
        self.n_classes = 4

    def load_data(self, *args):
        super(BCIC4_2a_plus, self).load_data(*args)
        for i in range(self.n_subject):
            "operate on self.subjects_data[i] "
            path_s = os.path.join(self.path, 's' + str(i + 1))
            for file_path in os.listdir(path_s):
                if file_path.split('.')[-1] == 'gdf' and file_path.find('T') != -1:
                    path_s_t = os.path.join(path_s, file_path)
                if file_path.split('.')[-1] == 'gdf' and file_path.find('E') != -1:
                    path_s_e = os.path.join(path_s, file_path)
            raw_t = mne.io.read_raw_gdf(path_s_t)
            event_t, _t = mne.events_from_annotations(raw_t)
            #compute n_trials
            count = 0
            for idx, (time, _, label) in enumerate(event_t):
                if label == 6:
                    if event_t[idx + 1][2] == 7 or event_t[idx + 1][2] == 8 or event_t[idx + 1][2] == 9 or event_t[idx + 1][2] == 10:
                        try:
                            start_interval = event_t[idx + 2][0] - time
                            start_interval = start_interval * np.random.random()
                            start_time = int(time + start_interval)
                            try:
                                _tp = raw_t[0:22, start_time:start_time + 176][0]*1e6
                                count+=1
                            except:
                                continue
                        except:
                            continue
            self.subjects_data[i].load_data(n_trial=count)

            count = 0
            for idx, (time, _, label) in enumerate(event_t):
                if label == 6:
                    if event_t[idx + 1][2] == 7 or event_t[idx + 1][2] == 8 or event_t[idx + 1][2] == 9 or event_t[idx + 1][2] == 10:
                        try:
                            start_interval = event_t[idx + 2][0] - time
                            start_interval = start_interval * np.random.random()
                            start_time = int(time + start_interval)
                            if start_time > event_t[idx+1][0]:
                                target_interval = 0
                            else:
                                target_interval = 1
                            try:
                                if self.uv:
                                    self.subjects_data[i].subject_trials[count].signal = raw_t[0:22, start_time:start_time + 176][0]*1e6
                                else:
                                    self.subjects_data[i].subject_trials[count].signal = raw_t[0:22, start_time:start_time + 176][0]
                                self.subjects_data[i].subject_trials[count].target = (np.array(event_t[idx+1][2]),np.array(target_interval))
                                count+=1
                            except:
                                continue
                        except:
                            continue



# db = BCIC4_2a_plus()
# db.load_data()
# db.signal_shape()

# def test_BCIC_dataset(name):
#     t = name()
#     t.load_data()
#     t.subjects_data[0].subject_trials[0].signal
#     type(t)  # BCIC2_3
#     type(t.subjects_data)  # list
#     type(t.subjects_data[0])  # BCI_database.EEG_data_subject
#     type(t.subjects_data[0].subject_trials)  # list
#     type(t.subjects_data[0].subject_trials[0])  # BCI_database.Trial
#     type(t.subjects_data[0].subject_trials[0].signal)  # numpy.ndarray
#     type(t.subjects_data[0].subject_trials[0].target)  # numpy.ndarray
#     # iteration
#     for s in t.subjects_data:
#         for trial in s.subject_trials:
#             print(trial.target)
#     return

#test whether all data loaded without problem
# for i,s in enumerate(db.subjects_data):
#         for j,trial in enumerate(s.subject_trials):
#             try:
#                 _ = trial.signal
#                 _ = trial.target
#             except:
#                 print(i,j)
#                 continue

# db_train = BCIC4_2a()
# db_train.load_data()
# db_train = DataLoader(db_train,batch_size=10,shuffle=True)
#
# for (x, y) in db_train:
#     print(x.shape)
#     print(y.shape)
# #

# db = BCIC4_2a_lrh()
# db.load_data()
# db_s1 = Subject_dataset(database=db,subject_id=0)

"""
gigad
from scipy.io import loadmat
ans = loadmat(r'G:\download\gigadb\s1.mat')
ans['eeg'][0][0][7].shape
#Out[64]: (68, 358400)
ans['eeg'][0][0][9]
#Out[85]: array([[100]], dtype=uint8)
ans['eeg'][0][0][11][0].sum()
#Out[95]: 100
[i for i,k in enumerate(ans['eeg'][0][0][11][0]) if k!=0]
# [1023,
#  4607,
#  8191,
#  11775,
#  15359,
#  18943,...
[i for i,j in enumerate(idx) if idx[i]-idx[i-1] != 3584]
#Out[103]: [0]
ans['eeg'].dtype.names
#从0开始排

"""



