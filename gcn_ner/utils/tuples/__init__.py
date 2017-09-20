def clean_text(text):
    text = text.replace('\n', ' ')
    return text


def get_entities_from_tuple(words, embeddings, ner, trans_prob):
    import gcn_ner.utils as utils

    sentence = ' '.join(words)
    A_fw, A_bw, tags, X = utils.aux.create_graph_from_sentence_and_word_vectors(sentence, embeddings)
    predictions = ner.predict_with_viterbi(A_fw, A_bw, X, tags, trans_prob)
    entities = [utils.aux.get_entity_name(p) for p in predictions]
    return entities


def erase_non_entities(all_words, all_entities, all_idx):
    return [(w, e, i) for w, e, i in zip(all_words, all_entities, all_idx) if e and w != ' ']


def join_consecutive_tuples(tuples):
    for i in range(len(tuples) - 1):
        curr_type = tuples[i][1]
        curr_end_idx = tuples[i][2][1]
        next_type = tuples[i + 1][1]
        next_start_idx = tuples[i + 1][2][0]
        if curr_type == next_type and curr_end_idx == next_start_idx - 1:
            curr_word = tuples[i][0]
            next_word = tuples[i + 1][0]
            curr_start_idx = tuples[i][2][0]
            next_end_idx = tuples[i + 1][2][1]
            tuples[i + 1] = (curr_word + ' ' + next_word,
                             curr_type,
                             [curr_start_idx, next_end_idx])
            tuples[i] = ()
    tuples = [t for t in tuples if t]
    return tuples


def clean_tuples(all_words, all_entities, all_idx):
    tuples = erase_non_entities(all_words, all_entities, all_idx)
    tuples = join_consecutive_tuples(tuples)
    return tuples
