from BCI_database import Subject_dataset
from experiment_utils import split_dataset,plot_module_selection,plot_acc_bars,get_one_para,logits_to_pred,plot_acc_bars_no_anno
from torch.utils.data import DataLoader
from sklearn.grid_search import GridSearchCV
from modules import AAR_estimator_KNN,AAR_estimator_LDA,AAR_estimator_SVC,CSSSP_estimator_LDA,CSSP_estimator_LDA,\
    CSP_estimator_SVC,FBCSP_estimator_LDA,DFBCSP_FR_estimator_LDA,SBSSP_estimator_LDA,FBCSPNet,deep_FBCSPNet,\
    multi_deep_FBCSPNet,EEGNet,Point_wiseLSTM
from sklearn.externals import joblib
import numpy as np
import torch
batch_size = 16
from result2 import cm_plot,all_cm_plot,get_cm

def fit_AAR_module(log_name,datasets,model,out_file,**kwargs):
    ch_dict = {'AAR_LDA':'使用线性判别分析的自回归模型','AAR_SVC':'使用支持向量机的自回归模型','AAR_KNN':'使用k最近邻分类器的自回归模型'}

    subject_datasets = []
    in_chan = datasets.signal_shape()[0]
    time_steps = datasets.signal_shape()[1]
    classes = datasets.n_classes
    fs = datasets.fs

    #default values
    maxlag_lda = [6]
    maxlag_svc = [7]
    maxlag_knn = [7]
    k = [7]
    for key in kwargs:
        if key == 'maxlag_lda':
            maxlag_lda = kwargs[key]
        if key == 'maxlag_svc':
            maxlag_svc = kwargs[key]
        if key == 'maxlag_knn':
            maxlag_knn = kwargs[key]
        if key == 'k':
            k = kwargs[key]

    with open(log_name, 'w') as f:
        for i in range(datasets.n_subject):
            datasets_i = Subject_dataset(database=datasets, subject_id=i)
            subject_datasets.append(datasets_i)

        means_AAR = []
        stds_AAR = []
        paras_AAR = []

        for j, exp_dataset in enumerate(subject_datasets):
            train_data, test_data = split_dataset(exp_dataset, 0.8, 0)
            x_train, y_train = next(iter(DataLoader(train_data, len(train_data))))
            batch_size = 16
            if model == 'AAR_LDA':
                "1.AARestimatorlda"
                f.write('estimate AAR_lda for subject' + str(j) + '\n')
                # train and module selection

                params = {
                    'maxlag': maxlag_lda
                }
                f.write('maxlag' + str(maxlag_lda) + '\n')
                f.write('4-fold cv to select model' + '\n')
                gs = GridSearchCV(AAR_estimator_LDA(in_chan=in_chan, classes=classes), param_grid=params, cv=4)
                gs.fit(x_train.numpy(), y_train.numpy())
                f.write(str(gs.grid_scores_) + '\n')
                clf_0 = gs.best_estimator_
                joblib.dump(clf_0, 'subject'+str(j)+'_'+'best_AAR_estimator_LDA.m')
                # test
                x_test, y_test = next(iter(DataLoader(test_data, len(test_data))))
                score = clf_0.score(x_test.numpy(), y_test.numpy())
                # 0.68
                f.write('AAR_estimator_LDA: test score for subject' + str(i) + ':' + str(score) + '\n')
                f.write('\n')

            if model == 'AAR_SVC':
                "2.AARestimatorSVC"
                f.write('estimate AAR_SVC for subject' + str(j) + '\n')
                # train and module selection

                params = {
                    'maxlag': maxlag_svc
                }
                f.write('maxlag' + str(maxlag_svc) + '\n')
                f.write('4-fold cv to select model' + '\n')
                gs = GridSearchCV(AAR_estimator_SVC(in_chan=in_chan, classes=classes), param_grid=params, cv=4)
                gs.fit(x_train.numpy(), y_train.numpy())
                f.write(str(gs.grid_scores_) + '\n')
                clf_1 = gs.best_estimator_
                joblib.dump(clf_1, 'subject'+str(j)+'_'+'best_AAR_estimator_SVC.m')

                # test
                x_test, y_test = next(iter(DataLoader(test_data, len(test_data))))
                score = clf_1.score(x_test.numpy(), y_test.numpy())
                # 0.68
                f.write('AAR_estimator_SVC: test score for subject' + str(i) + ':' + str(score) + '\n')
                f.write('\n')

            if model == 'AAR_KNN':
                "3.AARestimatorKNN"
                f.write('estimate AAR_KNN for subject' + str(j) + '\n')
                # train and module selection

                params = {
                    'maxlag': maxlag_knn,
                    'k': k
                }
                f.write('maxlag' + str(maxlag_knn) + '\n')
                f.write('k' + str(k) + '\n')
                f.write('4-fold cv to select model' + '\n')
                gs = GridSearchCV(AAR_estimator_KNN(in_chan=in_chan, classes=classes), param_grid=params, cv=4)
                gs.fit(x_train.numpy(), y_train.numpy())
                f.write(str(gs.grid_scores_) + '\n')
                clf_2 = gs.best_estimator_
                joblib.dump(clf_2, 'subject'+str(j)+'_'+'best_AAR_estimator_KNN.m')

                # test
                x_test, y_test = next(iter(DataLoader(test_data, len(test_data))))
                score = clf_2.score(x_test.numpy(), y_test.numpy())
                # 0.68
                f.write('AAR_estimator_LDA: test score for subject' + str(i) + ':' + str(score) + '\n')
                f.write('\n')


            means_AAR_sub = []
            stds_AAR_sub = []
            paras_AAR_sub = []
            for (para, mean, values) in gs.grid_scores_:
                means_AAR_sub.append(mean)
                stds_AAR_sub.append(values.std())
                paras_AAR_sub.append(para)

            means_AAR.append(means_AAR_sub)
            stds_AAR.append(stds_AAR_sub)

        means_AAR = list(np.array(means_AAR).transpose())
        max_mean_para = np.array(means_AAR).mean(axis=1).max()
        max_arg_para = np.array(means_AAR).mean(axis=1).argmax()
        stds_AAR = list(np.array(stds_AAR).transpose())
        min_std_para = np.array(means_AAR).mean(axis=1).min()
        min_arg_para = np.array(means_AAR).mean(axis=1).argmin()
        max_mean_sub = np.array(means_AAR).mean(axis=0).max()
        max_arg_sub = np.array(means_AAR).mean(axis=0).argmax()
        out_file.write('使用'+str(ch_dict[model])+'对数据集'+str(datasets.name)+'的'+str(datasets.n_subject)+'名被试进行实验，结果显示：')
        out_file.write('对' + str(datasets.n_subject) + '名被试做平均，'+'参数'+str(paras_AAR_sub[int(max_arg_para)]).rstrip('}').lstrip('{').replace(':','=').replace('\'','')+'下模型的表现最好，准确率达到'+str('{:.2f}').format(max_mean_para)+'。')
        out_file.write('参数'+str(paras_AAR_sub[int(min_arg_para)]).rstrip('}').lstrip('{').replace(':','=').replace('\'','')+'下模型的表现最稳定，模型准确率的标准差为'+str('{:.2f}').format(min_std_para)+'。')
        out_file.write('对所有参数做平均，被试'+str(max_arg_sub+1)+'的数据集训练得到的模型表现最好，模型的平均准确率为'+str('{:.2f}').format(max_mean_sub)+'。')
        out_file.write('\n')



        plot_module_selection(str(model) + '_selection for '+str(datasets.name), paras_AAR_sub, means_AAR, stds_AAR, width=0.8,
                              save=True)

def compare_AAR_results(datasets,out_file):
    ch_dict = {'AR_LDA':'使用线性判别分析的自回归模型','AR_SVC':'使用支持向量机的自回归模型','AR_KNN':'使用k最近邻分类器的自回归模型'}
    subject_datasets = []
    for i in range(datasets.n_subject):
        datasets_i = Subject_dataset(database=datasets, subject_id=i)
        subject_datasets.append(datasets_i)

    AAR_LDA_res = []
    AAR_SVC_res = []
    AAR_KNN_res = []

    cm0_all = 0
    cm1_all = 0
    cm2_all = 0
    for j, exp_dataset in enumerate(subject_datasets):
        train_data, test_data = split_dataset(exp_dataset, 0.8, 0)
        x_test, y_test = next(iter(DataLoader(test_data, len(test_data))))

        clf_0 = joblib.load('subject'+str(j)+'_'+'best_AAR_estimator_LDA.m')
        score = clf_0.score(x_test.numpy(), y_test.numpy())
        p0,cm0 = cm_plot(y=y_test.numpy(),yp=clf_0.predict(x_test.numpy()),classes=datasets.chi_names,sub=str(j+1),dataset=datasets.name,model='AAR_LDA',return_cm=True)
        p0.plot()
        cm0_all += cm0
        AAR_LDA_res.append(score)

        clf_1 = joblib.load('subject'+str(j)+'_'+'best_AAR_estimator_SVC.m')
        score = clf_1.score(x_test.numpy(), y_test.numpy())
        p1,cm1 = cm_plot(y=y_test.numpy(),yp=clf_1.predict(x_test.numpy()),classes=datasets.chi_names,sub=str(j+1),dataset=datasets.name,model='AAR_SVC',return_cm=True)
        p1.plot()
        cm1_all += cm1
        AAR_SVC_res.append(score)

        clf_2 = joblib.load('subject'+str(j)+'_'+'best_AAR_estimator_KNN.m')
        score = clf_2.score(x_test.numpy(), y_test.numpy())
        p2,cm2 = cm_plot(y=y_test.numpy(),yp=clf_2.predict(x_test.numpy()),classes=datasets.chi_names,sub=str(j+1),dataset=datasets.name,model='AAR_KNN',return_cm=True)
        p2.plot()
        cm2_all += cm2
        AAR_KNN_res.append(score)

    res = [AAR_LDA_res, AAR_SVC_res, AAR_KNN_res]
    models = ['AR_LDA', 'AR_SVC', 'AR_KNN']
    max_res_para = np.array(res).mean(axis=1).max()
    max_arg_para = np.array(res).mean(axis=1).argmax()
    max_res_sub = np.array(res).mean(axis=0).max()
    max_arg_sub = np.array(res).mean(axis=0).argmax()
    out_file.write('在数据集'+str(datasets.name)+'上评估自回归模型的泛化表现,结果显示：表现最佳的模型是'+str(ch_dict[models[max_arg_para]])+'，在'+str(datasets.n_subject)+'名被试中的平均准确率达到'+str('{:.2f}').format(max_res_para)+'。')
    out_file.write('第'+str(max_arg_sub+1)+'名被试表现最佳，平均准确率达到'+str('{:.2f}').format(max_res_sub)+'。\n')
    all_cm_plot(cm=cm0_all,dataset=datasets,classes=datasets.chi_names,model='AR_LDA')
    all_cm_plot(cm=cm1_all, dataset=datasets, classes=datasets.chi_names, model='AR_SVC')
    all_cm_plot(cm=cm2_all, dataset=datasets, classes=datasets.chi_names, model='AR_KNN')
    plot_acc_bars( datasets.name+'数据集的分类结果' , models , res, width=0.8,
                  save=True)

def fit_CSP_module(log_name,datasets,model,out_file,**kwargs):
    ch_dict = {'CSP_SVC': '共空间模式模型', 'CSSP_LDA': '共空间频谱模式', 'CSSSP_LDA': '共稀疏频谱空间模式', 'SBCSP_LDA': '子带共空间模式',
               'FBCSP_LDA': '滤波器组共空间模式', 'DFBCSP_FR_LDA': '基于费雪比的判别滤波器组共空间模式'}

    subject_datasets = []
    in_chan = datasets.signal_shape()[0]
    time_steps = datasets.signal_shape()[1]
    classes = datasets.n_classes
    fs = datasets.fs

    #default values
    f_l_csp = [6]
    f_h_csp = [37]
    tao_cssp = [5]
    T_csssp = [5]
    f_l_sbcsp = [6]
    f_h_sbcsp = [37]
    select_ratio_fbcsp = [0.15]
    select_ratio_DFBCSPFR = [0.1, 0.2]
    for key in kwargs:
        if key == 'f_l_csp':
            f_l_csp = kwargs[key]
        if key == 'f_h_csp':
            f_h_csp = kwargs[key]
        if key == 'tao_cssp':
            tao_cssp = kwargs[key]
        if key == 'T_csssp':
            T_csssp = kwargs[key]
        if key == 'T_sbcsp':
            T_sbcsp = kwargs[key]
        if key == 'select_ratio_fbcsp':
            select_ratio_fbcsp = kwargs[key]
        if key == 'f_l_sbcsp':
            f_l_sbcsp = kwargs[key]
        if key == 'f_h_sbcsp':
            f_h_sbcsp = kwargs[key]

    with open(log_name, 'w') as f:
        for i in range(datasets.n_subject):
            datasets_i = Subject_dataset(database=datasets, subject_id=i)
            subject_datasets.append(datasets_i)

        means_AAR = []
        stds_AAR = []
        paras_AAR = []

        for j, exp_dataset in enumerate(subject_datasets):
            train_data, test_data = split_dataset(exp_dataset, 0.8, 0)
            x_train, y_train = next(iter(DataLoader(train_data, len(train_data))))
            batch_size = 16

            if model == 'CSP_SVC':
                "1.CSP_estimator_SVC"
                f.write('estimate CSP_estimator_SVC for subject' + str(j) + '\n')
                # train and module selection
                params = {
                    'f_l': f_l_csp,
                    'f_h': f_h_csp
                }
                f.write('f_l:' + str(f_l_csp) + '\n')
                f.write('f_h:' + str(f_h_csp) + '\n')
                f.write('4-fold cv to select model' + '\n')
                gs = GridSearchCV(CSP_estimator_SVC(fs=fs), param_grid=params, cv=4)
                gs.fit(x_train.numpy(), y_train.numpy())
                f.write(str(gs.grid_scores_) + '\n')
                clf_0 = gs.best_estimator_
                joblib.dump(clf_0, 'subject'+str(j)+'_'+'best_CSP_estimator_SVC.m')

                # test
                x_test, y_test = next(iter(DataLoader(test_data, len(test_data))))
                score = clf_0.score(x_test.numpy(), y_test.numpy())
                # 0.68
                f.write('CSP_estimator_SVC: test score for subject' + str(i) + ':' + str(score) + '\n')
                f.write('\n')

            if model == 'CSSP_LDA':
                "2.CSSP_estimator_LDA"
                f.write('estimate CSSP_estimator_LDA for subject' + str(j) + '\n')
                # train and module selection
                params = {
                    'tao': tao_cssp
                }
                f.write('tao' + str(tao_cssp) + '\n')
                f.write('4-fold cv to select model' + '\n')
                gs = GridSearchCV(CSSP_estimator_LDA(fs=fs), param_grid=params, cv=4)
                gs.fit(x_train.numpy(), y_train.numpy())
                f.write(str(gs.grid_scores_) + '\n')
                clf_1 = gs.best_estimator_
                joblib.dump(clf_1, 'subject'+str(j)+'_'+'best_CSSP_estimator_LDA.m')
                # test
                x_test, y_test = next(iter(DataLoader(test_data, len(test_data))))
                score = clf_1.score(x_test.numpy(), y_test.numpy())
                # 0.68
                f.write('CSSP_estimator_LDA: test score for subject' + str(i) + ':' + str(score) + '\n')
                f.write('\n')

            if model == 'CSSSP_LDA':
                "3.CSSSP_estimator_LDA"
                f.write('estimate CSSSP_estimator_LDA for subject' + str(j) + '\n')
                # train and module selection
                params = {
                    'T': T_csssp
                }
                f.write('T' + str(T_csssp) + '\n')
                f.write('4-fold cv to select model' + '\n')
                gs = GridSearchCV(CSSSP_estimator_LDA(fs=fs), param_grid=params, cv=4)
                gs.fit(x_train.numpy(), y_train.numpy())
                f.write(str(gs.grid_scores_) + '\n')
                clf_2 = gs.best_estimator_
                joblib.dump(clf_2, 'subject'+str(j)+'_'+'best_CSSSP_estimator_LDA.m')

                # test
                x_test, y_test = next(iter(DataLoader(test_data, len(test_data))))
                score = clf_2.score(x_test.numpy(), y_test.numpy())
                # 0.68
                f.write('CSSSP_estimator_LDA: test score for subject' + str(i) + ':' + str(score) + '\n')
                f.write('\n')

            if model == 'SBCSP_LDA':
                "4.SBCSP_estimator_LDA"
                f.write('estimate SBSSP_estimator_LDA for subject' + str(j) + '\n')
                # train and module selection
                f.write('4-fold cv to select model' + '\n')

                params = {
                    'f_l': f_l_sbcsp,
                    'f_h':f_h_sbcsp
                }
                gs = GridSearchCV(SBSSP_estimator_LDA(fs=fs), param_grid=params, cv=4)
                gs.fit(x_train.numpy(), y_train.numpy())
                f.write(str(gs.grid_scores_) + '\n')
                clf_3 = gs.best_estimator_
                joblib.dump(clf_3, 'subject'+str(j)+'_'+'best_SBSSP_estimator_LDA.m')

                # test
                x_test, y_test = next(iter(DataLoader(test_data, len(test_data))))
                score = clf_3.score(x_test.numpy(), y_test.numpy())
                # 0.68
                f.write('SBSSP_estimator_LDA: test score for subject' + str(i) + ':' + str(score) + '\n')
                f.write('\n')

            if model == 'FBCSP_LDA':
                "5.FBCSP_estimator_LDA"
                f.write('estimate FBSSP_estimator_LDA for subject' + str(j) + '\n')
                # train and module selection
                f.write('4-fold cv to select model' + '\n')
                params = {
                    'select_ratio': select_ratio_fbcsp
                }
                f.write('select_ratio' + str(select_ratio_fbcsp) + '\n')
                gs = GridSearchCV(FBCSP_estimator_LDA(fs=fs), param_grid=params, cv=4)
                gs.fit(x_train.numpy(), y_train.numpy())
                f.write(str(gs.grid_scores_) + '\n')
                clf_4 = gs.best_estimator_
                joblib.dump(clf_4,'subject'+str(j)+'_'+ 'best_FBCSP_estimator_LDA.m')

                # test
                x_test, y_test = next(iter(DataLoader(test_data, len(test_data))))
                score = clf_4.score(x_test.numpy(), y_test.numpy())
                # 0.68
                f.write('FBCSP_estimator_LDA: test score for subject' + str(i) + ':' + str(score) + '\n')
                f.write('\n')

            if model == 'DFBCSP_FR_LDA':
                "6.DFBCSP_FR_estimator_LDA"
                f.write('estimate DFBSSP_FR_estimator_LDA for subject' + str(j) + '\n')
                # train and module selection
                f.write('4-fold cv to select model' + '\n')

                params = {
                    'select_ratio': select_ratio_DFBCSPFR
                }
                f.write('select_ratio' + str(select_ratio_DFBCSPFR) + '\n')

                gs = GridSearchCV(DFBCSP_FR_estimator_LDA(C3_idx=datasets.C3, C4_idx=datasets.C4, fs=fs), param_grid=params, cv=4)
                gs.fit(x_train.numpy(), y_train.numpy())
                f.write(str(gs.grid_scores_) + '\n')
                clf_5 = gs.best_estimator_
                joblib.dump(clf_5,'subject'+str(j)+'_'+'best_DFBCSP_FR_estimator_LDA.m')

                # test
                x_test, y_test = next(iter(DataLoader(test_data, len(test_data))))
                score = clf_5.score(x_test.numpy(), y_test.numpy())
                # 0.68
                f.write('DFBCSP_FR_estimator_LDA: test score for subject' + str(i) + ':' + str(score) + '\n')
                f.write('\n')

            means_AAR_sub = []
            stds_AAR_sub = []
            paras_AAR_sub = []
            for (para, mean, values) in gs.grid_scores_:
                means_AAR_sub.append(mean)
                stds_AAR_sub.append(values.std())
                paras_AAR_sub.append(para)

            means_AAR.append(means_AAR_sub)
            stds_AAR.append(stds_AAR_sub)

        means_AAR = list(np.array(means_AAR).transpose())
        max_mean_para = np.array(means_AAR).mean(axis=1).max()
        max_arg_para = np.array(means_AAR).mean(axis=1).argmax()
        stds_AAR = list(np.array(stds_AAR).transpose())
        min_std_para = np.array(means_AAR).mean(axis=1).min()
        min_arg_para = np.array(means_AAR).mean(axis=1).argmin()
        max_mean_sub = np.array(means_AAR).mean(axis=0).max()
        max_arg_sub = np.array(means_AAR).mean(axis=0).argmax()
        out_file.write('使用'+str(ch_dict[model])+'对数据集'+str(datasets.name)+'的'+str(datasets.n_subject)+'名被试进行实验，结果显示：')
        out_file.write('对' + str(datasets.n_subject) + '名被试做平均，'+'参数'+str(paras_AAR_sub[int(max_arg_para)]).rstrip('}').lstrip('{').replace(':','=').replace('\'','')+'下模型的表现最好，准确率达到'+str('{:.2f}').format(max_mean_para)+'。')
        out_file.write('参数'+str(paras_AAR_sub[int(min_arg_para)]).rstrip('}').lstrip('{').replace(':','=').replace('\'','')+'下模型的表现最稳定，模型准确率的标准差为'+str('{:.2f}').format(min_std_para)+'。')
        out_file.write('对所有参数做平均，被试'+str(max_arg_sub+1)+'的数据集训练得到的模型表现最好，模型的平均准确率为'+str('{:.2f}').format(max_mean_sub)+'。')
        out_file.write('\n')



        plot_module_selection(str(model) + '_selection for '+str(datasets.name), paras_AAR_sub, means_AAR, stds_AAR, width=0.8,
                              save=True)

def compare_CSP_dl_results(datasets,out_file):
    ch_dict = {'CSP_SVC': '共空间模式模型', 'CSSP_LDA': '共空间频谱模式模型', 'CSSSP_LDA': '共稀疏频谱空间模式模型', 'SBCSP_LDA': '子带共空间模式模型',
               'FBCSP_LDA': '滤波器组共空间模式模型', 'DFBCSP_FR_LDA': '基于费雪比的判别滤波器组共空间模式模型'}
    subject_datasets = []
    for i in range(datasets.n_subject):
        datasets_i = Subject_dataset(database=datasets, subject_id=i)
        subject_datasets.append(datasets_i)

    CSP_SVC_res = []
    CSSP_LDA_res = []
    CSSSP_LDA_res = []
    FBCSP_LDA_res = []
    SBCSP_LDA_res = []
    DFBCSP_FR_LDA_res = []
    FBCSPNet_res = []
    deep_FBCSPNet_res = []
    EEGNet_res = []


    cm0_all = 0
    cm1_all = 0
    cm2_all = 0
    cm3_all = 0
    cm4_all = 0
    cm5_all = 0
    for j, exp_dataset in enumerate(subject_datasets):
        train_data, test_data = split_dataset(exp_dataset, 0.2, 0)
        x_test,y_test = next(iter(DataLoader(test_data,len(test_data))))

        clf_0 = joblib.load('subject'+str(j)+'_'+'best_CSP_estimator_SVC.m')
        p0, cm0 = cm_plot(y=y_test.numpy(), yp=clf_0.predict(x_test.numpy()), classes=datasets.chi_names,
                          sub=str(j + 1), dataset=datasets.name, model='CSP_SVC', return_cm=True)
        p0.plot()
        cm0_all += cm0
        score = clf_0.score(x_test.numpy(), y_test.numpy())
        CSP_SVC_res.append(score)

        clf_1 = joblib.load('subject'+str(j)+'_'+'best_CSSP_estimator_LDA.m')
        p1, cm1 = cm_plot(y=y_test.numpy(), yp=clf_1.predict(x_test.numpy()), classes=datasets.chi_names,
                          sub=str(j + 1), dataset=datasets.name, model='CSSP_LDA', return_cm=True)
        p1.plot()
        cm1_all += cm1
        score = clf_1.score(x_test.numpy(), y_test.numpy())
        CSSP_LDA_res.append(score)

        clf_2 = joblib.load('subject'+str(j)+'_'+'best_CSSSP_estimator_LDA.m')
        p2, cm2 = cm_plot(y=y_test.numpy(), yp=clf_2.predict(x_test.numpy()), classes=datasets.chi_names,
                          sub=str(j + 1), dataset=datasets.name, model='CSSSP_LDA', return_cm=True)
        p2.plot()
        cm2_all += cm2
        score = clf_2.score(x_test.numpy(), y_test.numpy())
        CSSSP_LDA_res.append(score)

        clf_3 = joblib.load('subject'+str(j)+'_'+'best_SBSSP_estimator_LDA.m')
        p3, cm3 = cm_plot(y=y_test.numpy(), yp=clf_3.predict(x_test.numpy()), classes=datasets.chi_names,
                          sub=str(j + 1), dataset=datasets.name, model='SBCSP_LDA', return_cm=True)
        p3.plot()
        cm3_all += cm3
        score = clf_3.score(x_test.numpy(), y_test.numpy())
        FBCSP_LDA_res.append(score)

        clf_4 = joblib.load('subject'+str(j)+'_'+'best_FBCSP_estimator_LDA.m')
        p4, cm4 = cm_plot(y=y_test.numpy(), yp=clf_4.predict(x_test.numpy()), classes=datasets.chi_names,
                          sub=str(j + 1), dataset=datasets.name, model='FBCSP_LDA', return_cm=True)
        p4.plot()
        cm4_all += cm4
        score = clf_4.score(x_test.numpy(), y_test.numpy())
        SBCSP_LDA_res.append(score)

        clf_5 = joblib.load('subject'+str(j)+'_'+'best_DFBCSP_FR_estimator_LDA.m')
        p5, cm5 = cm_plot(y=y_test.numpy(), yp=clf_5.predict(x_test.numpy()), classes=datasets.chi_names,
                          sub=str(j + 1), dataset=datasets.name, model='DFBCSP_FR_LDA', return_cm=True)
        p5.plot()
        cm5_all += cm5
        score = clf_5.score(x_test.numpy(), y_test.numpy())
        DFBCSP_FR_LDA_res.append(score)

        model = torch.load('subject'+str(j)+'_'+'fbcspnet.pth')
        test_acc, _ = model.evaluate_train(test_dataset=test_data, batch_size=batch_size)
        FBCSPNet_res.append(test_acc)

        model = torch.load('subject'+str(j)+'_'+'deep_fbcspnet.pth')
        test_acc, _ = model.evaluate_train(test_dataset=test_data, batch_size=batch_size)
        deep_FBCSPNet_res.append(test_acc)

        model = torch.load('subject'+str(j)+'_'+'eegnet.pth')
        test_acc, _ = model.evaluate_train(test_dataset=test_data, batch_size=batch_size)
        EEGNet_res.append(test_acc)

    res = [CSP_SVC_res,CSSP_LDA_res,CSSSP_LDA_res,FBCSP_LDA_res,SBCSP_LDA_res,DFBCSP_FR_LDA_res,FBCSPNet_res,deep_FBCSPNet_res,EEGNet_res]

    models = ['CSP_SVC','CSSP_LDA','CSSSP_LDA','FBCSP_LDA','SBCSP_LDA','DFBCSP_FR_LDA','ShallowConv','DeepConv','EEGNet']
    max_res_para = np.array(res).mean(axis=1).max()
    max_arg_para = np.array(res).mean(axis=1).argmax()
    max_res_sub = np.array(res).mean(axis=0).max()
    max_arg_sub = np.array(res).mean(axis=0).argmax()
    out_file.write('在数据集'+str(datasets.name)+'上评估共空间模式模型和深度学习模型的泛化表现,结果显示：表现最佳的模型是'+str(ch_dict[models[max_arg_para]])+'，在'+str(datasets.n_subject)+'名被试中的平均准确率达到'+str('{:.2f}').format(max_res_para)+'。')
    out_file.write('第'+str(max_arg_sub+1)+'名被试表现最佳，平均准确率达到'+str('{:.2f}').format(max_res_sub)+'。\n')
    all_cm_plot(cm=cm0_all,dataset=datasets,classes=datasets.chi_names,model='CSP_SVC')
    all_cm_plot(cm=cm1_all, dataset=datasets, classes=datasets.chi_names, model='CSSP_LDA')
    all_cm_plot(cm=cm2_all, dataset=datasets, classes=datasets.chi_names, model='CSSSP_LDA')
    all_cm_plot(cm=cm3_all,dataset=datasets,classes=datasets.chi_names,model='FBCSP_LDA')
    all_cm_plot(cm=cm4_all, dataset=datasets, classes=datasets.chi_names, model='SBCSP_LDA')
    all_cm_plot(cm=cm5_all, dataset=datasets, classes=datasets.chi_names, model='DFBCSP_FR_LDA')
    plot_acc_bars('results for' + datasets.name, models , res, width=0.8,
                  save=True)

def compare_dl_results(datasets,out_file):
    subject_datasets = []
    for i in range(datasets.n_subject):
        datasets_i = Subject_dataset(database=datasets, subject_id=i)
        subject_datasets.append(datasets_i)

    FBCSPNet_res = []
    deep_FBCSPNet_res = []
    EEGNet_res = []

    cm0_all = 0
    cm1_all = 0
    cm2_all = 0
    for j, exp_dataset in enumerate(subject_datasets):
        train_data, test_data = split_dataset(exp_dataset, 0.8, 0)
        x_test,y_test = next(iter(DataLoader(test_data,len(test_data))))
        if y_test.min()==1:
            y_test -= 1
        model = torch.load('subject'+str(j)+'_'+'fbcspnet.pth')
        test_acc, _ = model.evaluate_train(test_dataset=test_data, batch_size=batch_size)
        FBCSPNet_res.append(test_acc)
        p0, cm0 = cm_plot(y=y_test.numpy(), yp=logits_to_pred(model=model,x=x_test).numpy(), classes=datasets.chi_names,
                          sub=str(j + 1), dataset=datasets.name, model='ShallowConvNet', return_cm=True)
        p0.plot()
        cm0_all += cm0

        model = torch.load('subject'+str(j)+'_'+'deep_fbcspnet.pth')
        test_acc, _ = model.evaluate_train(test_dataset=test_data, batch_size=batch_size)
        deep_FBCSPNet_res.append(test_acc)
        p1, cm1 = cm_plot(y=y_test.numpy(), yp=logits_to_pred(model=model, x=x_test).numpy(), classes=datasets.chi_names,
                          sub=str(j + 1), dataset=datasets.name, model='deepConvNet', return_cm=True)
        p1.plot()
        cm1_all += cm1

        model = torch.load('subject'+str(j)+'_'+'eegnet.pth')
        test_acc, _ = model.evaluate_train(test_dataset=test_data, batch_size=batch_size)
        EEGNet_res.append(test_acc)
        p2, cm2 = cm_plot(y=y_test.numpy(), yp=logits_to_pred(model=model, x=x_test).numpy(),
                          classes=datasets.chi_names,
                          sub=str(j + 1), dataset=datasets.name, model='EEGNet', return_cm=True)
        p2.plot()
        cm2_all += cm2
    res = [FBCSPNet_res,deep_FBCSPNet_res,EEGNet_res]

    all_cm_plot(cm=cm0_all,dataset=datasets,classes=datasets.chi_names,model='ShallowConvNet')
    all_cm_plot(cm=cm1_all, dataset=datasets, classes=datasets.chi_names, model='deepConvNet')
    all_cm_plot(cm=cm2_all, dataset=datasets, classes=datasets.chi_names, model='EEGNet')

    models = ['ShallowConv','DeepConv','EEGNet']
    max_res_para = np.array(res).mean(axis=1).max()
    max_arg_para = np.array(res).mean(axis=1).argmax()
    max_res_sub = np.array(res).mean(axis=0).max()
    max_arg_sub = np.array(res).mean(axis=0).argmax()
    out_file.write('第'+str(max_arg_sub+1)+'名被试表现最佳，平均准确率达到'+str('{:.2f}').format(max_res_sub)+'。\n')
    plot_acc_bars('results for' + datasets.name, models , res, width=0.8,
                  save=True)

def compare_CSP_results(datasets,out_file):
    ch_dict = {'CSP_SVC': '共空间模式模型', 'CSSP_LDA': '共空间频谱模式模型', 'CSSSP_LDA': '共稀疏频谱空间模式模型', 'SBCSP_LDA': '子带共空间模式模型',
               'FBCSP_LDA': '滤波器组共空间模式模型', 'DFBCSP_FR_LDA': '基于费雪比的判别滤波器组共空间模式模型'}
    subject_datasets = []
    for i in range(datasets.n_subject):
        datasets_i = Subject_dataset(database=datasets, subject_id=i)
        subject_datasets.append(datasets_i)

    CSP_SVC_res = []
    CSSP_LDA_res = []
    CSSSP_LDA_res = []
    FBCSP_LDA_res = []
    SBCSP_LDA_res = []
    DFBCSP_FR_LDA_res = []


    cm0_all = 0
    cm1_all = 0
    cm2_all = 0
    cm3_all = 0
    cm4_all = 0
    cm5_all = 0
    for j, exp_dataset in enumerate(subject_datasets):
        train_data, test_data = split_dataset(exp_dataset, 0.8, 0)
        x_test,y_test = next(iter(DataLoader(test_data,len(test_data))))

        clf_0 = joblib.load('subject'+str(j)+'_'+'best_CSP_estimator_SVC.m')
        p0, cm0 = cm_plot(y=y_test.numpy(), yp=clf_0.predict(x_test.numpy()), classes=datasets.chi_names,
                          sub=str(j + 1), dataset=datasets.name, model='CSP_SVC', return_cm=True)
        p0.plot()
        cm0_all += cm0
        score = clf_0.score(x_test.numpy(), y_test.numpy())
        CSP_SVC_res.append(score)

        clf_1 = joblib.load('subject'+str(j)+'_'+'best_CSSP_estimator_LDA.m')
        p1, cm1 = cm_plot(y=y_test.numpy(), yp=clf_1.predict(x_test.numpy()), classes=datasets.chi_names,
                          sub=str(j + 1), dataset=datasets.name, model='CSSP_LDA', return_cm=True)
        p1.plot()
        cm1_all += cm1
        score = clf_1.score(x_test.numpy(), y_test.numpy())
        CSSP_LDA_res.append(score)

        clf_2 = joblib.load('subject'+str(j)+'_'+'best_CSSSP_estimator_LDA.m')
        p2, cm2 = cm_plot(y=y_test.numpy(), yp=clf_2.predict(x_test.numpy()), classes=datasets.chi_names,
                          sub=str(j + 1), dataset=datasets.name, model='CSSSP_LDA', return_cm=True)
        p2.plot()
        cm2_all += cm2
        score = clf_2.score(x_test.numpy(), y_test.numpy())
        CSSSP_LDA_res.append(score)

        clf_3 = joblib.load('subject'+str(j)+'_'+'best_SBSSP_estimator_LDA.m')
        p3, cm3 = cm_plot(y=y_test.numpy(), yp=clf_3.predict(x_test.numpy()), classes=datasets.chi_names,
                          sub=str(j + 1), dataset=datasets.name, model='SBCSP_LDA', return_cm=True)
        p3.plot()
        cm3_all += cm3
        score = clf_3.score(x_test.numpy(), y_test.numpy())
        SBCSP_LDA_res.append(score)

        clf_4 = joblib.load('subject'+str(j)+'_'+'best_FBCSP_estimator_LDA.m')
        p4, cm4 = cm_plot(y=y_test.numpy(), yp=clf_4.predict(x_test.numpy()), classes=datasets.chi_names,
                          sub=str(j + 1), dataset=datasets.name, model='FBCSP_LDA', return_cm=True)
        p4.plot()
        cm4_all += cm4
        score = clf_4.score(x_test.numpy(), y_test.numpy())
        FBCSP_LDA_res.append(score)

        clf_5 = joblib.load('subject'+str(j)+'_'+'best_DFBCSP_FR_estimator_LDA.m')
        p5, cm5 = cm_plot(y=y_test.numpy(), yp=clf_5.predict(x_test.numpy()), classes=datasets.chi_names,
                          sub=str(j + 1), dataset=datasets.name, model='DFBCSP_FR_LDA', return_cm=True)
        p5.plot()
        cm5_all += cm5
        score = clf_5.score(x_test.numpy(), y_test.numpy())
        DFBCSP_FR_LDA_res.append(score)



    res = [CSP_SVC_res,CSSP_LDA_res,CSSSP_LDA_res,SBCSP_LDA_res,FBCSP_LDA_res,DFBCSP_FR_LDA_res]

    models = ['CSP_SVC','CSSP_LDA','CSSSP_LDA','SBCSP_LDA','FBCSP_LDA','DFBCSP_FR_LDA']
    max_res_para = np.array(res).mean(axis=1).max()
    max_arg_para = np.array(res).mean(axis=1).argmax()
    max_res_sub = np.array(res).mean(axis=0).max()
    max_arg_sub = np.array(res).mean(axis=0).argmax()
    out_file.write('在数据集'+str(datasets.name)+'上评估共空间模式模型和深度学习模型的泛化表现,结果显示：表现最佳的模型是'+str(ch_dict[models[max_arg_para]])+'，在'+str(datasets.n_subject)+'名被试中的平均准确率达到'+str('{:.2f}').format(max_res_para)+'。')
    out_file.write('第'+str(max_arg_sub+1)+'名被试表现最佳，平均准确率达到'+str('{:.2f}').format(max_res_sub)+'。\n')
    all_cm_plot(cm=cm0_all,dataset=datasets,classes=datasets.chi_names,model='CSP_SVC')
    all_cm_plot(cm=cm1_all, dataset=datasets, classes=datasets.chi_names, model='CSSP_LDA')
    all_cm_plot(cm=cm2_all, dataset=datasets, classes=datasets.chi_names, model='CSSSP_LDA')
    all_cm_plot(cm=cm3_all,dataset=datasets,classes=datasets.chi_names,model='SBCSP_LDA')
    all_cm_plot(cm=cm4_all, dataset=datasets, classes=datasets.chi_names, model='FBCSP_LDA')
    all_cm_plot(cm=cm5_all, dataset=datasets, classes=datasets.chi_names, model='DFBCSP_FR_LDA')
    plot_acc_bars('results for' + datasets.name, models , res, width=0.8,
                  save=True)

def fit_dl_module(log_name,datasets,model,max_epoch,batch_size,out_file,subjects=None):
    subject_datasets = []
    fs = datasets.fs
    if subjects==None:
        subjects = range(datasets.n_subject)
    in_chan = datasets.signal_shape()[0]
    time_steps = datasets.signal_shape()[1]
    fs = datasets.fs


    with open(log_name, 'w') as f:
        for i in range(datasets.n_subject):
            datasets_i = Subject_dataset(database=datasets, subject_id=i)
            subject_datasets.append(datasets_i)

        for j, exp_dataset in enumerate(subject_datasets):
            train_data, test_data = split_dataset(exp_dataset, 0.8, 0)
            x_train, y_train = next(iter(DataLoader(train_data, len(train_data))))


            vis_name = 'subject' + str(j)

            # "1.FBCSPNet"
            if model == 'FBCSPNet':

                paras = {'eps': [1e-3, 1e-5], 'linear_init_std': [0.1, 0.01, 0.001]}

                for params in get_one_para(paras):
                    for para in params:
                        if next(iter(para)) == 'eps':
                            eps = para['eps']
                        if next(iter(para)) == 'linear_init_std':
                            linear_init_std = para['linear_init_std']
                    f.write('estimate FBCSPNet for subject' + str(j) + '\n')
                    f.write('hyperparameter eps:' + str(eps) + '\n')
                    f.write('hyperparameter linear_init_std:' + str(linear_init_std) + '\n')
                    vis_name = model+ ' for '+datasets.name+':subject' + str(j) + 'eps:' + str(eps) + 'linear_init_std' + str(linear_init_std)
                    # train and module selection
                    model_fbcspnet = FBCSPNet(in_chans=in_chan, time_steps=time_steps, classes=datasets.n_classes, env=vis_name, eps=eps,
                                              linear_init_std=linear_init_std,fs=fs)
                    model_fbcspnet.fit(train_dataset=train_data, batch_size=batch_size, max_epoch=max_epoch,
                                       test_dataset=test_data, reg=1,save_name='subject'+str(j)+'_'+'fbcspnet.pth')
                    acc_fbcspnet, _ = model_fbcspnet.evaluate_train(test_dataset=test_data, batch_size=batch_size)
                    f.write('FBCSPNet: test score for subject' + str(i) + ':' + str(acc_fbcspnet) + '\n')
                    f.write('\n')
            # "2.deepFBCSPNet"
            elif model == 'deepFBCSPNet':
                paras = {'eps': [1e-3, 1e-5], 'linear_init_std': [0.1, 0.01, 0.001]}

                for params in get_one_para(paras):
                    for para in params:
                        if next(iter(para)) == 'eps':
                            eps = para['eps']
                        if next(iter(para)) == 'linear_init_std':
                            linear_init_std = para['linear_init_std']
                    f.write('estimate deep_FBCSPNet for subject' + str(j) + '\n')
                    f.write('hyperparameter eps:' + str(eps) + '\n')
                    f.write('hyperparameter linear_init_std:' + str(linear_init_std) + '\n')
                    vis_name = model + ' for ' + datasets.name + ':subject' + str(j) + 'eps:' + str(
                        eps) + 'linear_init_std' + str(linear_init_std)

                    # train and module selection
                    model_deep_fbcspnet = deep_FBCSPNet(in_chans=in_chan, time_steps=time_steps,
                                                        classes=datasets.n_classes, env=vis_name, eps=eps,
                                                        linear_init_std=linear_init_std,fs=fs)
                    model_deep_fbcspnet.fit(train_dataset=train_data, batch_size=batch_size, max_epoch=max_epoch,
                                            test_dataset=test_data,save_name='subject'+str(j)+'_'+'deep_fbcspnet.pth')
                    acc_deepfbcspnet, _ = model_deep_fbcspnet.evaluate_train(test_dataset=test_data,
                                                                             batch_size=batch_size)
                    f.write('deep_FBCSPNet: test score for subject' + str(i) + ':' + str(acc_deepfbcspnet) + '\n')
                    f.write('\n')
            # "3.EEGNet"
            elif model == 'EEGNet':
                paras = {'eps': [1e-3, 1e-5], 'linear_init_std': [0.1, 0.01, 0.001]}

                for params in get_one_para(paras):
                    for para in params:
                        if next(iter(para)) == 'eps':
                            eps = para['eps']
                        if next(iter(para)) == 'linear_init_std':
                            linear_init_std = para['linear_init_std']
                    f.write('estimate EEGNet for subject' + str(j) + '\n')
                    f.write('hyperparameter eps:' + str(eps) + '\n')
                    f.write('hyperparameter linear_init_std:' + str(linear_init_std) + '\n')
                    vis_name = model + ' for ' + datasets.name + ':subject' + str(j) + 'eps:' + str(
                        eps) + 'linear_init_std' + str(linear_init_std)

                    # train and module selection
                    model_eegnet = EEGNet(in_chans=in_chan, time_steps=time_steps, fs=fs,
                                          n_classes=datasets.n_classes, env=vis_name, eps=eps, linear_init_std=linear_init_std)
                    model_eegnet.fit(train_dataset=train_data, batch_size=batch_size, max_epoch=max_epoch,
                                     test_dataset=test_data,save_name='subject'+str(j)+'_'+'eegnet.pth')
                    acc_eegnet, _ = model_eegnet.evaluate_train(test_dataset=test_data, batch_size=batch_size)
                    f.write('EEGNet: test score for subject' + str(i) + ':' + str(acc_eegnet) + '\n')
                    f.write('\n')


def fit_rnn_module(log_name,datasets,model,max_epoch,batch_size,out_file):
    subject_datasets = []
    fs = datasets.fs

    in_chan = datasets.signal_shape()[0]
    time_steps = datasets.signal_shape()[1]
    fs = datasets.fs


    with open(log_name, 'w') as f:
        for i in range(datasets.n_subject):
            datasets_i = Subject_dataset(database=datasets, subject_id=i)
            subject_datasets.append(datasets_i)

        for j, exp_dataset in enumerate(subject_datasets):
            train_data, test_data = split_dataset(exp_dataset, 0.8, 0)
            x_train, y_train = next(iter(DataLoader(train_data, len(train_data))))


            vis_name = 'subject' + str(j)

            # "1.FBCSPNet"
            if model == 'FBCSPNet':

                paras = {'eps': [1e-3], 'linear_init_std': [0.01]}

                for params in get_one_para(paras):
                    for para in params:
                        if next(iter(para)) == 'eps':
                            eps = para['eps']
                        if next(iter(para)) == 'linear_init_std':
                            linear_init_std = para['linear_init_std']
                    f.write('estimate FBCSPNet for subject' + str(j) + '\n')
                    f.write('hyperparameter eps:' + str(eps) + '\n')
                    f.write('hyperparameter linear_init_std:' + str(linear_init_std) + '\n')
                    vis_name = model+ ' for '+datasets.name+':subject' + str(j) + 'eps:' + str(eps) + 'linear_init_std' + str(linear_init_std)
                    # train and module selection
                    model_fbcspnet = FBCSPNet(in_chans=in_chan, time_steps=time_steps, classes=datasets.n_classes, env=vis_name, eps=eps,
                                              linear_init_std=linear_init_std,fs=fs)
                    model_fbcspnet.fit(train_dataset=train_data, batch_size=batch_size, max_epoch=max_epoch,
                                       test_dataset=test_data, reg=1,save_name='subject'+str(j)+'_'+'fbcspnet.pth')
                    acc_fbcspnet, _ = model_fbcspnet.evaluate_train(test_dataset=test_data, batch_size=batch_size)
                    f.write('FBCSPNet: test score for subject' + str(i) + ':' + str(acc_fbcspnet) + '\n')
                    f.write('\n')
            # "2.deepFBCSPNet"
            elif model == 'deepFBCSPNet':
                paras = {'eps': [1e-3], 'linear_init_std': [0.01]}

                for params in get_one_para(paras):
                    for para in params:
                        if next(iter(para)) == 'eps':
                            eps = para['eps']
                        if next(iter(para)) == 'linear_init_std':
                            linear_init_std = para['linear_init_std']
                    f.write('estimate deep_FBCSPNet for subject' + str(j) + '\n')
                    f.write('hyperparameter eps:' + str(eps) + '\n')
                    f.write('hyperparameter linear_init_std:' + str(linear_init_std) + '\n')
                    vis_name = model + ' for ' + datasets.name + ':subject' + str(j) + 'eps:' + str(
                        eps) + 'linear_init_std' + str(linear_init_std)

                    # train and module selection
                    model_deep_fbcspnet = deep_FBCSPNet(in_chans=in_chan, time_steps=time_steps,
                                                        classes=datasets.n_classes, env=vis_name, eps=eps,
                                                        linear_init_std=linear_init_std,fs=fs)
                    model_deep_fbcspnet.fit(train_dataset=train_data, batch_size=batch_size, max_epoch=max_epoch,
                                            test_dataset=test_data,save_name='subject'+str(j)+'_'+'deep_fbcspnet.pth')
                    acc_deepfbcspnet, _ = model_deep_fbcspnet.evaluate_train(test_dataset=test_data,
                                                                             batch_size=batch_size)
                    f.write('deep_FBCSPNet: test score for subject' + str(i) + ':' + str(acc_deepfbcspnet) + '\n')
                    f.write('\n')
            # "3.EEGNet"
            elif model == 'EEGNet':
                paras = {'eps': [1e-3], 'linear_init_std': [0.01]}

                for params in get_one_para(paras):
                    for para in params:
                        if next(iter(para)) == 'eps':
                            eps = para['eps']
                        if next(iter(para)) == 'linear_init_std':
                            linear_init_std = para['linear_init_std']
                    f.write('estimate EEGNet for subject' + str(j) + '\n')
                    f.write('hyperparameter eps:' + str(eps) + '\n')
                    f.write('hyperparameter linear_init_std:' + str(linear_init_std) + '\n')
                    vis_name = model + ' for ' + datasets.name + ':subject' + str(j) + 'eps:' + str(
                        eps) + 'linear_init_std' + str(linear_init_std)

                    # train and module selection
                    model_eegnet = EEGNet(in_chans=in_chan, time_steps=time_steps, fs=fs,
                                          n_classes=datasets.n_classes, env=vis_name, eps=eps, linear_init_std=linear_init_std)
                    model_eegnet.fit(train_dataset=train_data, batch_size=batch_size, max_epoch=max_epoch,
                                     test_dataset=test_data,save_name='subject'+str(j)+'_'+'eegnet.pth')
                    acc_eegnet, _ = model_eegnet.evaluate_train(test_dataset=test_data, batch_size=batch_size)
                    f.write('EEGNet: test score for subject' + str(i) + ':' + str(acc_eegnet) + '\n')
                    f.write('\n')


def compare_all_results(datasets,out_file,ratio=0.5,subjects=None):
    ch_dict = {'AR_LDA':'基于LDA的自回归模型','AR_SVC':'基于SVM的自回归模型','AR_KNN':'基于kNN的自回归模型','CSP_SVC': '共空间模式模型', 'CSSP_LDA': '共空间频谱模式模型', 'CSSSP_LDA': '共稀疏频谱空间模式模型', 'SBCSP_LDA': '子带共空间模式模型',
               'FBCSP_LDA': '滤波器组共空间模式模型', 'DFBCSP_FR_LDA': '基于费雪比的判别滤波器组共空间模式模型','fbcspnet':'ShallowConvNets','deepfbcspnet':'DeepConvNet','eegnet':'EEGNet'}
    subject_datasets = []
    for i in range(datasets.n_subject):
        datasets_i = Subject_dataset(database=datasets, subject_id=i)
        subject_datasets.append(datasets_i)
    if subjects==None:
        subjects = range(datasets.n_subject)
    AAR_LDA_res = []
    AAR_SVC_res = []
    AAR_KNN_res = []
    CSP_SVC_res = []
    CSSP_LDA_res = []
    CSSSP_LDA_res = []
    FBCSP_LDA_res = []
    SBCSP_LDA_res = []
    DFBCSP_FR_LDA_res = []
    FBCSPNet_res = []
    deep_FBCSPNet_res = []
    EEGNet_res = []


    cm0_all = 0
    cm1_all = 0
    cm2_all = 0
    cm3_all = 0
    cm4_all = 0
    cm5_all = 0
    cm6_all = 0
    cm7_all = 0
    cm8_all = 0
    cm9_all = 0
    cm10_all = 0
    cm11_all = 0
    for j, exp_dataset in enumerate(subject_datasets):
        if j in subjects:
            train_data, test_data = split_dataset(exp_dataset, ratio, 0)
            x_test,y_test = next(iter(DataLoader(test_data,len(test_data))))

            clf_0 = joblib.load('subject' + str(j) + '_' + 'best_AAR_estimator_LDA.m')
            score = clf_0.score(x_test.numpy(), y_test.numpy())
            cm0 = get_cm(y=y_test.numpy(), yp=clf_0.predict(x_test.numpy()))
            cm0_all += cm0
            AAR_LDA_res.append(score)

            clf_1 = joblib.load('subject' + str(j) + '_' + 'best_AAR_estimator_SVC.m')
            score = clf_1.score(x_test.numpy(), y_test.numpy())
            cm1 = get_cm(y=y_test.numpy(), yp=clf_1.predict(x_test.numpy()))
            cm1_all += cm1
            AAR_SVC_res.append(score)

            clf_2 = joblib.load('subject' + str(j) + '_' + 'best_AAR_estimator_KNN.m')
            score = clf_2.score(x_test.numpy(), y_test.numpy())
            cm2 = get_cm(y=y_test.numpy(), yp=clf_2.predict(x_test.numpy()))
            cm2_all += cm2
            AAR_KNN_res.append(score)


            clf_3 = joblib.load('subject'+str(j)+'_'+'best_CSP_estimator_SVC.m')
            cm3 = get_cm(y=y_test.numpy(), yp=clf_3.predict(x_test.numpy()))
            cm3_all += cm3
            score = clf_3.score(x_test.numpy(), y_test.numpy())
            CSP_SVC_res.append(score)

            clf_4 = joblib.load('subject'+str(j)+'_'+'best_CSSP_estimator_LDA.m')
            cm4 = get_cm(y=y_test.numpy(), yp=clf_4.predict(x_test.numpy()))
            cm4_all += cm4
            score = clf_4.score(x_test.numpy(), y_test.numpy())
            CSSP_LDA_res.append(score)

            clf_5 = joblib.load('subject'+str(j)+'_'+'best_CSSSP_estimator_LDA.m')
            cm5 = get_cm(y=y_test.numpy(), yp=clf_5.predict(x_test.numpy()))
            cm5_all += cm5
            score = clf_5.score(x_test.numpy(), y_test.numpy())
            CSSSP_LDA_res.append(score)

            try:
                clf_6 = joblib.load('subject'+str(j)+'_'+'best_SBSSP_estimator_LDA.m')
                cm6 = get_cm(y=y_test.numpy(), yp=clf_6.predict(x_test.numpy()))
                cm6_all += cm6
                score = clf_6.score(x_test.numpy(), y_test.numpy())
                SBCSP_LDA_res.append(score)
            except:
                clf_6 = joblib.load('subject' + str(j) + '_' + 'best_FBCSP_estimator_LDA.m')
                cm6 = get_cm(y=y_test.numpy(), yp=clf_6.predict(x_test.numpy()))
                cm6_all += cm6
                score = clf_6.score(x_test.numpy(), y_test.numpy())
                SBCSP_LDA_res.append(score)

            clf_7 = joblib.load('subject'+str(j)+'_'+'best_FBCSP_estimator_LDA.m')
            cm7 = get_cm(y=y_test.numpy(), yp=clf_7.predict(x_test.numpy()))
            cm7_all += cm7
            score = clf_7.score(x_test.numpy(), y_test.numpy())
            FBCSP_LDA_res.append(score)

            clf_8 = joblib.load('subject'+str(j)+'_'+'best_DFBCSP_FR_estimator_LDA.m')
            cm8 = get_cm(y=y_test.numpy(), yp=clf_8.predict(x_test.numpy()))
            cm8_all += cm8
            score = clf_8.score(x_test.numpy(), y_test.numpy())
            DFBCSP_FR_LDA_res.append(score)

            if y_test.min()==1:
                y_test -= 1
            model = torch.load('subject'+str(j)+'_'+'fbcspnet.pth')
            test_acc, _ = model.evaluate_train(test_dataset=test_data, batch_size=batch_size)
            FBCSPNet_res.append(float(test_acc.numpy()))
            try:
                cm9 = get_cm(y=y_test.numpy(), yp=logits_to_pred(model=model,x=x_test).numpy())
                cm9_all += cm9
            except:
                pass

            model = torch.load('subject'+str(j)+'_'+'deep_fbcspnet.pth')
            test_acc, _ = model.evaluate_train(test_dataset=test_data, batch_size=batch_size)
            deep_FBCSPNet_res.append(float(test_acc.numpy()))
            try:
                cm10 = get_cm(y=y_test.numpy(), yp=logits_to_pred(model=model, x=x_test).numpy())
                cm10_all += cm10
            except:
                pass
            model = torch.load('subject'+str(j)+'_'+'eegnet.pth')
            test_acc, _ = model.evaluate_train(test_dataset=test_data, batch_size=batch_size)
            EEGNet_res.append(float(test_acc.numpy()))
            try:
                cm11 = get_cm(y=y_test.numpy(), yp=logits_to_pred(model=model,x=x_test).numpy())
                cm11_all += cm11
            except:
                pass

    res = [AAR_LDA_res, AAR_SVC_res, AAR_KNN_res,CSP_SVC_res,CSSP_LDA_res,CSSSP_LDA_res,SBCSP_LDA_res,FBCSP_LDA_res,DFBCSP_FR_LDA_res,FBCSPNet_res,deep_FBCSPNet_res,EEGNet_res]
    models = ['AR_LDA', 'AR_SVC', 'AR_KNN','CSP_SVC','CSSP_LDA','CSSSP_LDA','SBCSP_LDA','FBCSP_LDA','DFBCSP_FR_LDA','ShallowConv','DeepConv','EEGNet']
    max_res_para = np.array(res).mean(axis=1).max()
    max_arg_para = np.array(res).mean(axis=1).argmax()
    max_res_sub = np.array(res).mean(axis=0).max()
    max_arg_sub = np.array(res).mean(axis=0).argmax()
    for i,model_name in enumerate(models):
        out_file.write(str(model_name)+':')
        out_file.write(str(res[i])+'\n')
    all_cm_plot(cm=cm0_all, dataset=datasets, classes=datasets.chi_names, model='AR_LDA')
    all_cm_plot(cm=cm1_all, dataset=datasets, classes=datasets.chi_names, model='AR_SVC')
    all_cm_plot(cm=cm2_all, dataset=datasets, classes=datasets.chi_names, model='AR_KNN')
    all_cm_plot(cm=cm3_all,dataset=datasets,classes=datasets.chi_names,model='CSP_SVC')
    all_cm_plot(cm=cm4_all, dataset=datasets, classes=datasets.chi_names, model='CSSP_LDA')
    all_cm_plot(cm=cm5_all, dataset=datasets, classes=datasets.chi_names, model='CSSSP_LDA')
    all_cm_plot(cm=cm6_all, dataset=datasets, classes=datasets.chi_names, model='SBCSP_LDA')
    all_cm_plot(cm=cm7_all,dataset=datasets,classes=datasets.chi_names,model='FBCSP_LDA')
    all_cm_plot(cm=cm8_all, dataset=datasets, classes=datasets.chi_names, model='DFBCSP_FR_LDA')
    try:
        all_cm_plot(cm=cm9_all, dataset=datasets, classes=datasets.chi_names, model='ShallowConvNet')
        all_cm_plot(cm=cm10_all, dataset=datasets, classes=datasets.chi_names, model='DeepConvNet')
        all_cm_plot(cm=cm11_all, dataset=datasets, classes=datasets.chi_names, model='EEGNet')
    except:
        pass
    plot_acc_bars_no_anno('results for' + datasets.name, models , res, width=0.8,
                  save=True)


