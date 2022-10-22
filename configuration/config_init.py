import argparse


def get_config():
    parse = argparse.ArgumentParser(description='common train config')

    # project parameters
    parse.add_argument('-learn-name', type=str, default='trainCross', help='本次训练名称')

    parse.add_argument('-path-save', type=str, default='../result/', help='保存字典的位置')
    parse.add_argument('-save-best', type=bool, default=False, help='当得到更好的准确度是否要保存')

    parse.add_argument('-threshold', type=float, default=0.95, help='准确率阈值')
    parse.add_argument('-cuda', type=bool, default=True)
    # parse.add_argument('-cuda', type=bool, default=False)
    parse.add_argument('-device', type=int, default=1)

    parse.add_argument('-num_workers', type=int, default=4)
    parse.add_argument('-num_class', type=int, default=2)

    # TODO  change kmer
    parse.add_argument('-kmer', type=int, default=5)
    # parse.add_argument('-adversarial', type=bool, default=True)
    parse.add_argument('-adversarial', type=bool, default=False)

    # save path
    parse.add_argument('-train-name', type=str, help='-train-name')
    parse.add_argument('-test-name', type=str, help='-test-name')

    parse.add_argument('-path_train_data', type=str, default='../data/test.txt', help='训练数据的位置')
    parse.add_argument('-path_test_data', type=str, default='../data/test.txt', help='训练数据的位置')

    parse.add_argument('-path-params', type=str, default=None, help='模型参数路径')
    parse.add_argument('-model-save-name', type=str, default='BERT', help='保存模型的命名')
    parse.add_argument('-save-figure-type', type=str, default='png', help='保存图片的文件类型')

    # training
    parse.add_argument('-mode', type=str, default='train-test', help='训练模式')

    # TODO modify model type
    parse.add_argument('-model', type=str, default='ClassificationDNAbert', help='训练模型名称')
    # parse.add_argument('-model', type=str, default='FusionDNAbert', help='训练模型名称')
    #
    parse.add_argument('-interval-log', type=int, default=10, help='经过多少batch记录一次训练状态')
    parse.add_argument('-interval-test', type=int, default=1, help='经过多少epoch对测试集进行测试')

    parse.add_argument('-epoch', type=int, default=10, help='迭代次数')
    # parse.add_argument('-optimizer', type=str, default='Adam', help='优化器名称')
    parse.add_argument('-optimizer', type=str, default='AdamW', help='优化器名称')
    parse.add_argument('-loss-func', type=str, default='CE', help='损失函数名称, CE/FL')

    # TODO change batchSize
    parse.add_argument('-batch-size', type=int, default=32)

    parse.add_argument('-lr', type=float, default=0.00005)
    parse.add_argument('-reg', type=float, default=0.003, help='正则化lambda')
    parse.add_argument('-b', type=float, default=0.06, help='flooding model')
    parse.add_argument('-gamma', type=float, default=2, help='gamma in Focal Loss')
    parse.add_argument('-alpha', type=float, default=0.5, help='alpha in Focal Lo\ss')

    config = parse.parse_args()
    return config
