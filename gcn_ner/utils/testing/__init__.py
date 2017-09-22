import logging

_logger = logging.getLogger(__name__)


def get_gcn_results(gcn_model, data, trans_prob):
    import numpy as np
    import copy

    from ..aux import create_graph_from_sentence_and_word_vectors
    from ..aux import create_full_sentence
    from ..aux import tags

    TAGS = copy.deepcopy(tags)
    TAGS.append('UNK')

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    total_positive = 0
    total_negative = 0


    total_sentences = 0
    broken_sentences = 0
    for words, sentence, tag, classification in data:
        old_rhs = ''
        old_lhs = ''
        full_sentence = create_full_sentence(words)
        word_embeddings = sentence
        total_sentences += 1
        try:
            A_fw, A_bw, tags, X = create_graph_from_sentence_and_word_vectors(full_sentence, word_embeddings)
            prediction = gcn_model.predict_with_viterbi(A_fw, A_bw, X, tags, trans_prob)
        except Exception as e:
            _logger.warning('Cannot process the following sentence: ' + full_sentence)
            print([TAGS[np.argmax(t)] for t in tags])
            print(words)
            broken_sentences += 1
            continue

        open_rhs = False
        open_lhs = False
        has_entity = False

        for word, lhs, rhs in zip(words, prediction, classification):
            rhs_changed = False
            lhs_changed = False

            if old_rhs != rhs:
                open_rhs = not open_rhs
                rhs_changed = True

            if old_lhs != rhs:
                open_lhs = not open_lhs
                lhs_changed = True

            if not open_rhs and rhs_changed:
                total_positive += 1

            if open_rhs and rhs_changed and open_lhs and lhs_changed:
                if lhs == rhs:
                    has_entity = True

            if open_rhs and not open_lhs:
                if has_entity:
                    false_negative += 1
                has_entity = False

            if has_entity and not open_rhs and rhs_changed and not open_lhs and lhs_changed:
                true_positive += 1

            if rhs[-1] != 0.:
                if lhs == rhs:
                    true_negative += 1
                if lhs != rhs:
                    false_positive += 1
                total_negative += 1

            old_rhs = rhs
            old_lhs = lhs

    print('Total sentences', total_sentences)
    print('Broken sentences', broken_sentences)

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2. / (1. / precision + 1. / recall)

    return precision, recall, f1
