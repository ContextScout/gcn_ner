import spacy
import numpy as np

parser = spacy.load('en_core_web_md')

default_vector = parser('entity')[0].vector

tags = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS",
        "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP",
        "WP$", "WRB"]

classes = ["CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LAW", "LOC",  "MONEY", "NORP", "ORDINAL",
           "ORG", "PERCENT", "PERSON", "PRODUCT", "QUANTITY", "TIME", "WORK_OF_ART"]


def clean_word(word, tag):
    word = word
    if tag == '.':
        word = word.replace('/', '')
    return word


def vector_is_empty(vector):
    to_throw = 0
    for item in vector:
        if item == 0.0:
            to_throw += 1
    if to_throw == len(vector):
        return True
    return False


def get_clean_word_vector(word, tag):
    parsed = parser(clean_word(word, tag))
    try:
        vector = parsed[0].vector
        if vector_is_empty(vector):
            vector = default_vector
    except:
        vector = default_vector
    return np.array(vector, dtype=np.float64)


def get_tagging_vector(tag):
    vector = [0.] * (len(tags) + 1)
    index = len(tags)
    try:
        index = tags.index(tag)
    except:
        pass
    vector[index] = 1.
    return vector


def get_class_vector(class_name):
    vector = [0.] * (len(classes) + 1)
    index = len(classes)
    try:
        index = classes.index(class_name)
    except:
        pass
    vector[index] = 1.
    return vector


def get_entity_num(class_name):
    entity_num = len(classes)
    try:
        entity_num = classes.index(class_name)
    except:
        pass
    return entity_num


def _get_word_vectors_from_tokens(tokens):
    words = []
    vectors = []
    tags = []
    idx = []
    for token in tokens:
        word = token.orth_
        tag = token.pos_
        words.append(word)
        vectors.append(get_clean_word_vector(word, tag))
        tags.append(tag)
        idx.append([token.idx, token.idx + len(token.orth_)])
    return words, vectors, idx


def get_entity_name(prediction):
    index = np.argmax(prediction)
    try:
        return classes[index]
    except:
        return ''


def get_words_embeddings_from_sentence(sentence):
    tokens = parser(sentence)
    return _get_word_vectors_from_tokens(tokens)


def get_words_embeddings_from_text(text):
    doc = parser(text)
    sentences = []
    for sent in doc.sents:
        sentences.append(_get_word_vectors_from_tokens(sent))
    return sentences


def create_graph_from_sentence_and_word_vectors(sentence, word_vectors):
    if not isinstance(sentence, str):
        raise TypeError("String must be an argument")
    from igraph import Graph
    from .nl import SpacyTagger as Tagger, SpacyParser as Parser

    tagger = Tagger(sentence)
    parser = Parser(tagger)

    X = []
    for i in range(len(word_vectors)):
        X.append(word_vectors[i])
    X = np.array(X)

    nodes, edges, words, tags, types = parser.execute()
    g = Graph(directed=True)
    g.add_vertices(nodes)
    g.add_edges(edges)
    A_fw = np.array(g.get_adjacency().data)

    nodes, edges, words, tags, types = parser.execute_backward()
    g2 = Graph(directed=True)
    g2.add_vertices(nodes)
    g2.add_edges(edges)
    A_bw = np.array(g2.get_adjacency().data)

    tag_logits = [get_tagging_vector(tag) for tag in tags]
    return A_fw, A_bw, tag_logits, X


def get_all_sentences(filename):
    file = open(filename)
    sentences = []
    items = []
    old_entity = ''
    for line in file.readlines():
        if line[0] == '#':
            continue
        elements = line.split()
        if len(elements) < 5:
            if items != []:
                sentences.append(items)
            items = []
            continue
        word = elements[3].strip()
        tag = elements[4].strip()
        entity, old_entity = decide_entity(elements[10].strip(), old_entity)
        items.append((word, tag, entity))
    return sentences


def decide_entity(string, prior_entity):
    if string == '*)':
        return prior_entity, ''
    if string == '*':
        return prior_entity, prior_entity
    entity = ''
    for item in classes:
        if string.find(item) != -1:
            entity = item
    prior_entity = ''
    if string.find(')') == -1:
        prior_entity = entity
    return entity, prior_entity


def get_data_from_sentences(sentences):
    all_data = []
    A = np.zeros((len(classes) + 1, len(classes) + 1))
    total_tokens = 0
    for sentence in sentences:
        word_data = []
        class_data = []
        tag_data = []
        words = []
        prior_entity = len(classes)
        for word, tag, entity in sentence:
            words.append(word)
            word_vector = get_clean_word_vector(word, tag)
            tag_vector = get_tagging_vector(tag)
            tag_data.append(tag_vector)
            vector = word_vector
            class_vector = get_class_vector(entity)
            entity_num = get_entity_num(entity)
            word_data.append(vector)
            class_data.append(class_vector)
            A[prior_entity, entity_num] += 1
            prior_entity = entity_num
            total_tokens += 1
        all_data.append((words, word_data, tag_data, class_data))
    A /= total_tokens
    return all_data, A
