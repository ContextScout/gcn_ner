def get_gcn_results(gcn_model, data, trans_prob):
    from ..aux import create_graph_from_sentence_and_word_vectors

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    total_positive = 0
    total_negative = 0

    for words, sentence, tag, classification in data:
        old_rhs = ''
        old_lhs = ''
        full_sentence = ' '.join(words)
        word_embeddings = sentence
        try:
            A_fw, A_bw, X = create_graph_from_sentence_and_word_vectors(full_sentence, word_embeddings)
            prediction = gcn_model.predict_with_viterbi(A_fw, A_bw, X, tag, trans_prob)
        except:
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

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2. / (1. / precision + 1. / recall)

    return precision, recall, f1
