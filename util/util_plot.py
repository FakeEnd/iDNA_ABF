import matplotlib.pyplot as plt
import seaborn as sns
import collections

colors = ["#8D5CDC", "#EA52BC", "#FF6691", "#FF946B","#FFC859", "#F9F871"]

def draw_ROC_PRC_curve(roc_datas, prc_datas, name, config):
    # roc_data = [FPR, TPR, AUC]
    # prc_data = [recall, precision, AP]

    sns.set(style="darkgrid")
    plt.figure(figsize=(16, 8))
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    lw = 2

    plt.subplot(1, 2, 1)
    for index, roc_data in enumerate(roc_datas):
            plt.plot(roc_data[0], roc_data[1], color=colors[index],lw=lw, label=name[index] + ' (AUC = %0.2f)' % roc_data[2])  ### 假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontdict={'weight': 'normal', 'size': 20})
    plt.ylabel('True Positive Rate', fontdict={'weight': 'normal', 'size': 20})
    plt.title('receiver operating characteristic curve', fontdict={'weight': 'normal', 'size': 23})
    plt.legend(loc="lower right", prop={'weight': 'normal', 'size': 16})

    plt.subplot(1, 2, 2)
    # plt.step(self.prc_data[0], self.prc_data[1], color='b', alpha=0.2,where='post')
    # plt.fill_between(prc_data[0], prc_data[1], step='post', alpha=0.2,color='b')
    for index, prc_data in enumerate(prc_datas):
        plt.plot(prc_data[0], prc_data[1], color=colors[index],
             lw=lw, label=name[index] + ' (AP = %0.2f)' % prc_data[2])  ### 假正率为横坐标，真正率为纵坐标做曲线

    plt.xlabel('Recall', fontdict={'weight': 'normal', 'size': 20})
    plt.ylabel('Precision', fontdict={'weight': 'normal', 'size': 20})
    plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision Recall curve', fontdict={'weight': 'normal', 'size': 23})
    plt.legend(loc="lower left", prop={'weight': 'normal', 'size': 16})

    plt.savefig(
        '{}/{}/{}.{}'.format(config.path_save , config.learn_name , 'ROC_PRC', config.save_figure_type))
    # plt.show()

def draw_statistics_bar(traindataset, testdataset, config):
    totaldataset = []
    totaldataset.extend(traindataset)
    totaldataset.extend(testdataset)

    plt.figure()

    if config.model == 'DNAbert':
        colors = ['#4DE199', '#F4E482', '#BAAD4E','#827919']
        statstic = [0,0,0,0] # A, C, T, G
        labels = ['A', 'C', 'T', 'G']
        for seq in totaldataset:
            for i in range(len(seq)):
                if seq[i] == 'A':
                    statstic[0] = statstic[0] + 1
                elif seq[i] == 'C':
                    statstic[1] = statstic[1] + 1
                elif seq[i] == 'T':
                    statstic[2] = statstic[2] + 1
                elif seq[i] == 'G':
                    statstic[3] = statstic[3] + 1
        # print(statstic)
        plt.bar(labels, statstic, color=colors)  # or `color=['r', 'g', 'b']`
    elif config.model == 'prot_bert_bfd' or config.model == 'prot_bert':
        colors = ['#e52d5d', '#e01b77', '#d31b92', '#bb2cad', '#983fc5', '#7b60e0', '#547bf3', '#0091ff', '#00b6ff', '#00d0da', '#00df83', '#a4e312' ]
        statstic = collections.defaultdict(int)
        for seq in totaldataset:
            for i in seq:
                statstic[i] += 1
        # print(statstic)
        labels = statstic.keys()
        plt.bar(statstic.keys(), statstic.values(), color=colors[:len(labels)])  # or `color=['r', 'g', 'b']`

    plt.savefig('{}/{}/{}.{}'.format(config.path_save, config.learn_name, 'statistics', config.save_figure_type))

    plt.show()

if __name__ == '__main__':
    c = ['DLIPTSSKLVV','DLIPTSSKLVV','DLIPTSSKLVV','AETCZAO','ABCDEFGHIJKMN']
    colors = ['#e52d5d', '#e01b77', '#d31b92', '#bb2cad', '#983fc5', '#7b60e0', '#547bf3', '#0091ff', '#00b6ff', '#00d0da', '#00df83', '#a4e312' ]
    statstic = collections.defaultdict(int)
    for seq in c:
        for i in seq:
            statstic[i] += 1
    # print(statstic)
    labels = statstic.keys()
    s = statstic.values()
    print(plt.cm.get_cmap('rainbow', len(labels)))
    plt.bar(labels, s, color=colors[:len(labels)])  # or `color=['r', 'g', 'b']`
    plt.show()