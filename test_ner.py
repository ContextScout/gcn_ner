_error_message = '''
Please provide a text as an input.
You can either provide the text as an argument: python test_ner.py Hard to believe this program was made in September 2017.
Or pipe the text from the command line: python test_ner.py < data/random_text.txt
'''

def _aggregate_sentence(args):
    return_str = ''
    for argument in args:
        return_str += argument + ' '
    return return_str


def _get_entity_tuples_from_sentence(sentence):
    from gcn_ner import GCNNer
    ner = GCNNer(ner_filename='./data/ner-gcn-21.tf', trans_prob_file='./data/trans_prob.pickle')
    entity_tuples = ner.get_entity_tuples_from_text(sentence)
    return entity_tuples


if __name__ == '__main__':
    import os
    import sys

    if len(sys.argv) > 1:
        sentence = _aggregate_sentence(sys.argv[1:])
        print(_get_entity_tuples_from_sentence(sentence))
    else:
        if os.isatty(0):
            print(_error_message)
            exit(0)
        sentence = sys.stdin.read().strip()
        if sentence:
            print(_get_entity_tuples_from_sentence(sentence))
