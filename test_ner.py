from gcn_ner import GCNNer

TEXT = '''
Google was started in early 1996 by Larry Page and Sergey Brin, two students at Stanford University, USA.
It used to be called Backrub.
they made it into a company, Google Inc., on September 7, 1998 at a friend's garage in Menlo Park, California.
In February 1999, the company moved to 165 University Ave., Palo Alto, California.
it moved to another place called the Googleplex.
'''

if __name__ == '__main__':
    ner = GCNNer(ner_filename='./data/ner-gcn-12.tf', trans_prob_file='./data/trans_prob.pickle')

    # Extract all the entities from text
    entity_tuples = ner.get_entity_tuples_from_text(TEXT)
    print(entity_tuples)
