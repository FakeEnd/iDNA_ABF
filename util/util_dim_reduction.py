from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def t_sne(title, data, data_index, data_label, class_num, config):
    print('processing data')
    X_tsne = TSNE(n_components=2).fit_transform(data)  # [num_samples, n_components]
    print('processing data over')

    font = {"color": "darkred", "size": 13, "family": "serif"}
    # plt.style.use("dark_background")
    plt.style.use("default")

    plt.figure()
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=data_index, alpha=0.6, cmap=plt.cm.get_cmap('rainbow', class_num))
    if data_label:
        for i in range(len(X_tsne)):
            plt.annotate(data_label[i], xy=(X_tsne[:, 0][i], X_tsne[:, 1][i]),
                         xytext=(X_tsne[:, 0][i] + 1, X_tsne[:, 1][i] + 1))
    plt.title(title, fontdict=font)

    if data_label is None:
        cbar = plt.colorbar(ticks=range(class_num))
        cbar.set_label(label='digit value', fontdict=font)
        plt.clim(0 - 0.5, class_num - 0.5)
    # final_path = PATH + str(re.sub('/', '_', config.train_data)) + title + '.png'
    # plt.savefig(final_path)
    plt.show()