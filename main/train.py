import sys
import os
import pickle

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from configuration import config_init
from frame import Learner


def SL_train(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.device)
    roc_datas, prc_datas = [], []

    # ToDo 两种的kmers的更改
    if config.model == 'FusionDNAbert':
        config.kmers = [3, 6]

    learner = Learner.Learner(config)
    learner.setIO()
    learner.setVisualization()
    learner.load_data()
    learner.init_model()
    learner.adjust_model()
    # learner.load_params()
    learner.init_optimizer()
    learner.def_loss_func()
    learner.train_model()


def SL_fintune():
    # config = config_SL.get_config()
    config = pickle.load(open('../result/jobID/config.pkl', 'rb'))
    config.path_params = '../result/jobID/DNAbert, MCC[0.64].pt'
    learner = Learner.Learner(config)
    learner.setIO()
    learner.setVisualization()
    learner.load_data()
    learner.init_model()
    learner.load_params()
    learner.init_optimizer()
    learner.def_loss_func()
    learner.train_model()
    learner.test_model()


def select_dataset():
    # DNA-MS
    # path_train_data = '../data/DNA_MS/tsv/5hmC/5hmC_H.sapiens/train.tsv'
    # path_test_data = '../data/DNA_MS/tsv/5hmC/5hmC_H.sapiens/test.tsv'
    # path_train_data = '../data/DNA_MS/tsv/5hmC/5hmC_M.musculus/train.tsv'
    # path_test_data = '../data/DNA_MS/tsv/5hmC/5hmC_M.musculus/test.tsv'
    path_train_data = '../data/DNA_MS/tsv/4mC/4mC_C.equisetifolia/train.tsv'
    path_test_data = '../data/DNA_MS/tsv/4mC/4mC_C.equisetifolia/test.tsv'
    # path_train_data = '../data/DNA_MS/tsv/4mC/4mC_F.vesca/train.tsv'
    # path_test_data = '../data/DNA_MS/tsv/4mC/4mC_F.vesca/test.tsv'
    # path_train_data = '../data/DNA_MS/tsv/4mC/4mC_S.cerevisiae/train.tsv'
    # path_test_data = '../data/DNA_MS/tsv/4mC/4mC_S.cerevisiae/test.tsv'
    # path_train_data = '../data/DNA_MS/tsv/4mC/4mC_Tolypocladium/train.tsv'
    # path_test_data = '../data/DNA_MS/tsv/4mC/4mC_Tolypocladium/test.tsv'
    # path_train_data = '../data/DNA_MS/tsv/6mA/6mA_A.thaliana/train.tsv'
    # path_test_data = '../data/DNA_MS/tsv/6mA/6mA_A.thaliana/test.tsv'
    # path_train_data = '../data/DNA_MS/tsv/6mA/6mA_C.elegans/train.tsv'
    # path_test_data = '../data/DNA_MS/tsv/6mA/6mA_C.elegans/test.tsv'
    # path_train_data = '../data/DNA_MS/tsv/6mA/6mA_C.equisetifolia/train.tsv'
    # path_test_data = '../data/DNA_MS/tsv/6mA/6mA_C.equisetifolia/test.tsv'
    # path_train_data = '../data/DNA_MS/tsv/6mA/6mA_D.melanogaster/train.tsv'
    # path_test_data = '../data/DNA_MS/tsv/6mA/6mA_D.melanogaster/test.tsv'
    # path_train_data = '../data/DNA_MS/tsv/6mA/6mA_F.vesca/train.tsv'
    # path_test_data = '../data/DNA_MS/tsv/6mA/6mA_F.vesca/test.tsv'
    # path_train_data = '../data/DNA_MS/tsv/6mA/6mA_H.sapiens/train.tsv'
    # path_test_data = '../data/DNA_MS/tsv/6mA/6mA_H.sapiens/test.tsv'
    # path_train_data = '../data/DNA_MS/tsv/6mA/6mA_R.chinensis/train.tsv'
    # path_test_data = '../data/DNA_MS/tsv/6mA/6mA_R.chinensis/test.tsv'
    # path_train_data = '../data/DNA_MS/tsv/6mA/6mA_S.cerevisiae/train.tsv'
    # path_test_data = '../data/DNA_MS/tsv/6mA/6mA_S.cerevisiae/test.tsv'
    # path_train_data = '../data/DNA_MS/tsv/6mA/6mA_T.thermophile/train.tsv'
    # path_test_data = '../data/DNA_MS/tsv/6mA/6mA_T.thermophile/test.tsv'
    # path_train_data = '../data/DNA_MS/tsv/6mA/6mA_Tolypocladium/train.tsv'
    # path_test_data = '../data/DNA_MS/tsv/6mA/6mA_Tolypocladium/test.tsv'
    # path_train_data = '../data/DNA_MS/tsv/6mA/6mA_Xoc BLS256/train.tsv'
    # path_test_data = '../data/DNA_MS/tsv/6mA/6mA_Xoc BLS256/test.tsv'
    # train_dict = {
    #     "4mCF": '../data/DNA_MS/tsv/4mC/4mC_F.vesca/train.tsv',
    #     "4mCS": '../data/DNA_MS/tsv/4mC/4mC_S.cerevisiae/train.tsv',
    #     "4mCC": '../data/DNA_MS/tsv/4mC/4mC_C.equisetifolia/train.tsv',
    #     "4mCT": '../data/DNA_MS/tsv/4mC/4mC_Tolypocladium/train.tsv',
    #     "5hmCH": '../data/DNA_MS/tsv/5hmC/5hmC_H.sapiens/train.tsv',
    #     "5hmCM": '../data/DNA_MS/tsv/5hmC/5hmC_M.musculus/train.tsv',
    #     "6mAA": '../data/DNA_MS/tsv/6mA/6mA_A.thaliana/train.tsv',
    #     "6mACEL": '../data/DNA_MS/tsv/6mA/6mA_C.elegans/train.tsv',
    #     "6mACEQ": '../data/DNA_MS/tsv/6mA/6mA_C.equisetifolia/train.tsv',
    #     "6mAD": '../data/DNA_MS/tsv/6mA/6mA_D.melanogaster/train.tsv',
    #     "6mAF": '../data/DNA_MS/tsv/6mA/6mA_F.vesca/train.tsv',
    #     "6mAH": '../data/DNA_MS/tsv/6mA/6mA_H.sapiens/train.tsv',
    #     "6mAR": '../data/DNA_MS/tsv/6mA/6mA_R.chinensis/train.tsv',
    #     "6mAS": '../data/DNA_MS/tsv/6mA/6mA_S.cerevisiae/train.tsv',
    #     "6mATT": '../data/DNA_MS/tsv/6mA/6mA_T.thermophile/train.tsv',
    #     "6mATO": '../data/DNA_MS/tsv/6mA/6mA_Tolypocladium/train.tsv',
    #     "6mAX": '../data/DNA_MS/tsv/6mA/6mA_Xoc BLS256/train.tsv',
    # }
    #
    # test_dict = {
    #     "4mCF": '../data/DNA_MS/tsv/4mC/4mC_F.vesca/test.tsv',
    #     "4mCS": '../data/DNA_MS/tsv/4mC/4mC_S.cerevisiae/test.tsv',
    #     "4mCC": '../data/DNA_MS/tsv/4mC/4mC_C.equisetifolia/test.tsv',
    #     "4mCT": '../data/DNA_MS/tsv/4mC/4mC_Tolypocladium/test.tsv',
    #     "5hmCH": '../data/DNA_MS/tsv/5hmC/5hmC_H.sapiens/test.tsv',
    #     "5hmCM": '../data/DNA_MS/tsv/5hmC/5hmC_M.musculus/test.tsv',
    #     "6mAA": '../data/DNA_MS/tsv/6mA/6mA_A.thaliana/test.tsv',
    #     "6mACEL": '../data/DNA_MS/tsv/6mA/6mA_C.elegans/test.tsv',
    #     "6mACEQ": '../data/DNA_MS/tsv/6mA/6mA_C.equisetifolia/test.tsv',
    #     "6mAD": '../data/DNA_MS/tsv/6mA/6mA_D.melanogaster/test.tsv',
    #     "6mAF": '../data/DNA_MS/tsv/6mA/6mA_F.vesca/test.tsv',
    #     "6mAH": '../data/DNA_MS/tsv/6mA/6mA_H.sapiens/test.tsv',
    #     "6mAR": '../data/DNA_MS/tsv/6mA/6mA_R.chinensis/test.tsv',
    #     "6mAS": '../data/DNA_MS/tsv/6mA/6mA_S.cerevisiae/test.tsv',
    #     "6mATT": '../data/DNA_MS/tsv/6mA/6mA_T.thermophile/test.tsv',
    #     "6mATO": '../data/DNA_MS/tsv/6mA/6mA_Tolypocladium/test.tsv',
    #     "6mAX": '../data/DNA_MS/tsv/6mA/6mA_Xoc BLS256/test.tsv',
    # }
    # print(sys.argv)
    # path_train_data = train_dict[sys.argv[2]]
    # path_test_data = test_dict[sys.argv[4]]

    print("train" + path_train_data, "test" + path_test_data)
    return path_train_data, path_test_data


if __name__ == '__main__':
    config = config_init.get_config()
    config.path_train_data, config.path_test_data = select_dataset()
    SL_train(config)
