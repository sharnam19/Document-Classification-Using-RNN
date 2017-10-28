import json
import nltk
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

f = open("data/trainingdata.txt","rb")
labels = []
text = []
for l in f:
    labels.append(int(l[0])-1)
    text.append(" ".join(l[2:].split(" ")[:-1]))
labels = labels[1:]
text = text[1:]
word_to_index = {}
word_count = {}
index = 1
int_texts = []
sent_len_grp = {}
ZERO = '<ZERO>'
word_to_index[ZERO]=0
word_count[ZERO]=0
for sent in text:
    tokens = sent.split(" ")
    int_text=""
    tokens = [w for w in tokens if not w in stop]
    if len(tokens)>=20:
        tokens=tokens[:20]
    elif len(tokens)<20:
        while len(tokens) !=20:
            tokens.append(ZERO)
    
    for token in tokens:
        if token not in word_to_index:
            word_to_index[token]=index
            word_count[token]=0
            index+=1
        word_count[token]+=1
        int_text+=str(word_to_index[token])+" "
    
    int_texts.append(int_text.strip())
# index = 0
# word_to_index2={}
# for key, value in reversed(sorted(word_count.iteritems(), key=lambda (k,v): (v,k))):
#     word_to_index2[key]=index
#     index+=1
data = {}
data['labels']= labels
data['text'] = int_texts
data['w2ix'] = word_to_index
print(int_texts)
json.dump(data,open("data/data.json","wb"))