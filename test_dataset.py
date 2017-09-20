from gcn_ner import GCNNer

if __name__ == '__main__':
    ner = GCNNer(ner_filename='./data/ner-gcn-10.tf', trans_prob_file='./data/trans_prob.pickle')
    ner.test('./data/test.conll')
