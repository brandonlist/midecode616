import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np




class Trial(object):
    """
    Totally abstract object
    attributes can be added as needed
    We recommend the following attribute names to be identical when reading different datasets:
        signal
        target
        sample_rate...
    for continues data, might contain multiple labels in one trial
    """
    pass


class EEG_data_subject(object):
    """
    EEG data for one subject
    expected updating to succeed self-define general form of data
    """
    def __init__(self,preload,id):
        self.preload = preload
        self.id = id

    def load_data(self,*args,**kwargs):
        "parasing different form of data"
        self.n_trial = kwargs['n_trial']
        self.subject_trials = []
        for i in range(self.n_trial):
            trial = Trial()
            self.subject_trials.append(trial)

#must load data first!

class EEG_database(Dataset):
    """
    General EEG data baseform, designed for multiple general-purpose use of the data, including:
    1.creat machine learning database
    2.analysis of the data including visualization
    3....
    Things like sample frequency are not included in the baseform.
    They can be added to the baseform or the Trial object as needed.
    """
    def __init__(self,name,n_subject,preload=True):
        super(EEG_database, self).__init__()
        self.preload = preload
        self.n_subject = n_subject
        self.name = name
        self.data_loaded = False

    def brief(self):
        if self.data_loaded != True:
            print('data not loaded')
            return
        else:
            for idx,each_s in enumerate(self.subjects_data):
                print('loaded {0} trails for subject {1}'.format(each_s.n_trial,idx))

    def load_data(self,*args):
        print("loading datasets: {0} ".format(self.name))
        for par in args:
            if par=='element_wise':
                self.load_mode = par
                print('load in element_wise_mode')
        self.subjects_data = []
        for i in range(self.n_subject):
            subject = EEG_data_subject(preload=True,id=i)
            self.subjects_data.append(subject)
        self.data_loaded = True

    def signal_shape(self):
        if self.data_loaded:
            return self.subjects_data[0].subject_trials[0].signal.shape
        else:
            print('data not loaded')
            return


    def __len__(self):
        count = 0
        for subject in self.subjects_data:
            for trial in subject.subject_trials:
                count = count+1
        return count

    def __getitem__(self, idx):
        count = 0
        for subject_n,subject_data in enumerate(self.subjects_data):
            for trial_n,trial_data in enumerate(subject_data.subject_trials):
                if count == idx:
                    return trial_data.signal,trial_data.target
                count = count+1


    # def create_ml_datasets(self,base_type = None):
    #     self.ml_dataset_base_type = base_type
    #     if self.ml_dataset_base_type == 'torch.utils.data.datasets':
    #         pass


class Subject_dataset(EEG_database):
    def __init__(self,database,subject_id):
        """
        dataset for one subject
        :param database: must be subclass of EEG_database
        :param subject_id: first to be 0
        """
        super(Subject_dataset, self).__init__(name=database.name,n_subject=1)
        self.fs = database.fs
        try:
            database.data_loaded ==True
        except:
            print('database not loaded, sorry man')
            return
        self.load_data()
        self.subjects_data[0] = database.subjects_data[subject_id]