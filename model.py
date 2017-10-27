import numpy as np
from sequential import *
from loss import *
from layer import *
import json

data = json.loads(open("data/data.json","rb").read())
X = data['text']
y = data['labels']
w2ix = data['w2ix']
vocab_size = len(w2ix.keys())
trainX = X[:5000]
trainY = y[:5000]
testX = X[5000:]
testY = y[5000:]
embed_dimension = 128

hidden_dimension = 256
output_dimension = 8

Wembed = np.random.normal(loc=0.0, scale=1.0,size=(vocab_size,embed_dimension))
Wx = np.random.normal(loc=0.0,scale=1.0,size=(embed_dimension,hidden_dimension))
Wh = np.random.normal(loc=0.0,scale=1.0,size=(hidden_dimension,hidden_dimension))
bh = np.zeros((hidden_dimension,))
Why = np.random.normal(loc=0.0,scale=1.0,size=(hidden_dimension,output_dimension))
by = np.zeros((output_dimension,))
epoch = 1000
                      
learning_rate = 1e-3
bptt = 4
forward= rnn_step
backward = rnn_step_backward
for e in range(epoch):
    for x in range(len(X)):
        index_sentence = trainX[x].strip().split(" ")
        hprev = np.zeros((1,hidden_dimension))
        index_sentence2 = map(str,index_sentence)
        for start_index in range(0,len(index_sentence2),bptt):
            sent = map(int,index_sentence2[start_index:min(start_index+bptt,len(index_sentence2))])
            caches = []
            wcaches = []
            for index in sent:
                wforward,wcache = word_embedding_forward(np.array([index]),Wembed)
                print(wforward.shape)
                hprev,tcache = forward(wforward,hprev,Wx,Wh,bh)
                caches.append(tcache)
                wcaches.append(wcache)
            
            #print(hprev.shape)
            out,tcache = affine_forward(hprev,Why,by)
            #print(out.shape)
            predicted,loss,dx = softmax_loss(out,trainY[x])
            print("Loss is: "+str(loss))
            dx,dWhy,dby = affine_backward(dx,tcache)
            Why -= learning_rate*dWhy
            by -= learning_rate*dby
            dWx = np.zeros_like(Wx)
            dWh = np.zeros_like(Wh)
            dbh = np.zeros_like(bh)
            dWembed = np.zeros_like(Wembed)
            tdh_prev = dx
            for pos in range(0,len(caches))[::-1]:
                dx,tdh_prev,tdWx,tdWh,tdbh = backward(tdh_prev,caches[pos])
                dWx+=tdWx
                dWh+=tdWh
                dbh+=tdbh
                tdWembed = word_embedding_backward(dx,wcaches[pos])
                dWembed+=tdWembed
            Wx -= learning_rate*dWx
            Wh -= learning_rate*dWh
            bh -= learning_rate*dbh
            Wembed -= learning_rate*dWembed