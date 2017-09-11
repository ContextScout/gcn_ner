NER that uses Graph Conv Nets
=============================

This is an implementation of a named entity recognizer that uses Graph
Convolutional Networks.

Installation
------------
```bash
git clone https://github.com/fractalego/gcn_ner.git

cd gcn_ner

virtualenv --python=/usr/bin/python3 .env

source ./env/bin/activate

pip3 install -r requirements.txt

python3 -m spacy download en

python3 -m spacy download en_core_web_md
```
Test NER
--------
Execute the file
```python
python3 test_ner.py
```


Train NER
---------
You will need to put your 'train.conll' into the 'data/' directory,
then execute the file
```python
python3 train.py
```


Test NER
--------
You will need to put your 'test.conll' into the 'data/' directory,
then execute the file
```python
python3 test_dataset.py
```


