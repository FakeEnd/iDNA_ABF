from configuration import config_init

def load_fasta(filename, skip_first=False):
    with open(filename, 'r') as file:
        content = file.read()
    content_split = content.split('\n')

    train_dataset = []
    train_label = []
    test_dataset = []
    test_label = []
    for index, record in enumerate(content_split):
        if index % 2 == 1:
            continue
        recordsplit = record.split('|')
        if recordsplit[-1] == 'training':
            train_label.append(int(recordsplit[-2]))
            train_dataset.append(content_split[index + 1])
        if recordsplit[-1] == 'testing':
            test_label.append(int(recordsplit[-2]))
            test_dataset.append(content_split[index + 1])
    return train_dataset, train_label, test_dataset, test_label

def load_tsv_format_data(filename, skip_head=True):
    sequences = []
    labels = []

    with open(filename, 'r') as file:
        if skip_head:
            next(file)
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            list = line.split('\t')
            sequences.append(list[2])
            labels.append(int(list[1]))

    return sequences, labels

if __name__ == '__main__':
    config = config_init.get_config()
    config.path_train_data = '../data/DNA_MS/tsv/6mA/6mA_A.thaliana/train.tsv'
    config.path_test_data = '../data/DNA_MS/tsv/6mA/6mA_A.thaliana/test.tsv'
