from gcn_ner import GCNNer

if __name__ == '__main__':
    GCNNer.train_and_save(dataset='./data/train.conll', saving_dir='./data/', epochs=31)
