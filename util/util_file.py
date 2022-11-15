from configuration import config_init


def load_easy_fasta(filename, skip_first=False):
    with open(filename, 'r') as file:
        content = file.read()
    content_split = content.split('\n')

    seqs = []
    for index, record in enumerate(content_split):
        if index % 2 == 1:
            seqs.append(content_split[index])

    return seqs


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


def read_csv(filename, skip_head=True):
    sequences = []
    labels = []
    with open(filename, 'r') as file:
        if skip_head:
            next(file)
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            list = line.split(',')
            feature = []
            for i in range(len(list) - 1):
                feature.append(int(list[i]))
            sequences.append(feature)
            labels.append(int(list[-1]))

    return sequences, labels


def load_txt_data(filename, skip_head=True):
    sequences = []

    with open(filename, 'r') as file:
        if skip_head:
            next(file)
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            sequences.append(line)

    return sequences


def load_tsv_po_ne_data(filename, skip_head=True):
    negsequences = []
    possequences = []
    labels = []

    with open(filename, 'r') as file:
        if skip_head:
            next(file)
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            list = line.split('\t')
            if list[1] == '1':
                possequences.append(list[2])
            elif list[1] == '0':
                negsequences.append(list[2])

    return possequences, negsequences


if __name__ == '__main__':
    config = config_init.get_config()
    config.path_train_data = '../data/DNA_MS/tsv/6mA/6mA_A.thaliana/train.tsv'
    config.path_test_data = '../data/DNA_MS/tsv/6mA/6mA_A.thaliana/test.tsv'
