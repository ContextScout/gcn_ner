import numpy as np
import pickle
import random
import sys
import logging

import gcn_ner.utils as utils
from gcn_ner.ner_model import GCNNerModel


class GCNNer:
    _logger = logging.getLogger(__name__)

    def __init__(self, ner_filename, trans_prob_file):
        self._ner = GCNNerModel.load(ner_filename)
        self._trans_prob = pickle.load(open(trans_prob_file, "rb"))

    def get_entity_tuples_from_sentence(self, sentence):
        '''
        Given a sentence it returns the named entities within it.

        :param sentence: The input sentence
        :return: a list of triples of the form (<ENTITY>, <ENTITY_TYPE>, [<ENTITY_START_POS>, <ENTITY_END_POS>])
        '''
        try:
            words, embeddings, idx = utils.aux.get_words_embeddings_from_sentence(sentence)
            entities = utils.tuples.get_entities_from_tuple(words, embeddings, self._ner, self._trans_prob)
            return utils.tuples.clean_tuples(words, entities, idx)
        except:
            self._logger.warning('Cannot process the following sentence: ' + sentence)
            return []

    def get_entity_tuples_from_text(self, text):
        '''
        Given a sentence it returns the named entities within it.

        :param sentence: The input sentence
        :return: a list of triples of the form (<ENTITY>, <ENTITY_TYPE>, [<ENTITY_START_POS>, <ENTITY_END_POS>])
        '''

        text = utils.tuples.clean_text(text)
        sentences = utils.aux.get_words_embeddings_from_text(text)
        all_words = []
        all_entities = []
        all_idx = []
        for words, embeddings, idx in sentences:
            try:
                entities = utils.tuples.get_entities_from_tuple(words, embeddings, self._ner, self._trans_prob)
                all_words.extend(words)
                all_entities.extend(entities)
                all_idx.extend(idx)
            except:
                self._logger.warning('Cannot process the following sentence: ' + ' '.join(words))
        return utils.tuples.clean_tuples(all_words, all_entities, all_idx)

    def test(self, dataset):
        '''
        The system tests the current NER model against a text in the CONLL format.

        :param dataset: the filename of a text in the CONLL format
        :return: None, the function prints precision, recall and chunck F1
        '''

        sentences = utils.aux.get_all_sentences(dataset)
        data, _ = utils.aux.get_data_from_sentences(sentences)
        precision, recall, f1 = utils.testing.get_gcn_results(self._ner, data, self._trans_prob)
        print('precision:', precision)
        print('recall:', recall)
        print('F1:', f1)

    @staticmethod
    def train_and_save(dataset, saving_dir, epochs=20, bucket_size=10):
        '''

        :param dataset: A file in the CONLL format to use as a training.
        :param saving_dir: The directory where to save the results
        :param epochs: The number of epochs to use in the training
        :param bucket_size: The batch size of the training.
        :return: An instance of this class (GCNNer
        '''
        print('Training the system according to the dataset ', dataset)
        return utils.training.train_and_save(dataset, saving_dir, epochs, bucket_size)
