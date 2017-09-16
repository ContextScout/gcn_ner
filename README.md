NER that uses Graph Conv Nets
=============================

This is an implementation of a named entity recognizer that uses Graph
Convolutional Networks. The reference article is [Graph convolutional
networks applied to NER](https://arxiv.org). This code has an F1 score
of 79.7 ± 0.3.


Installation
------------
```bash
git clone https://github.com/contextscout/gcn_ner.git

cd gcn_ner

virtualenv --python=/usr/bin/python3 .env

source .env/bin/activate

pip install -r requirements.txt

python -m spacy download en

python -m spacy download en_core_web_md
```
Test NER
--------
Execute the file
```python
python test_ner.py
```


Train NER
---------
You will need to put your 'train.conll' into the 'data/' directory,
then execute the file
```python
python train.py
```


Test NER
--------
You will need to put your 'test.conll' into the 'data/' directory,
then execute the file
```python
python test_dataset.py
```


