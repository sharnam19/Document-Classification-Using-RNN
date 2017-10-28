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
embed_dimension = 100

hidden_dimension = 128
output_dimension = 8

Wembed = np.random.normal(loc=0.0, scale=1.0,size=(vocab_size,embed_dimension))
Wx = np.random.normal(loc=0.0,scale=1.0,size=(embed_dimension,hidden_dimension))
Wh = np.random.normal(loc=0.0,scale=1.0,size=(hidden_dimension,hidden_dimension))
bh = np.zeros((hidden_dimension,))
Why = np.random.normal(loc=0.0,scale=1.0,size=(hidden_dimension,output_dimension))
by = np.zeros((output_dimension,))
epoch = 100

losses = []
learning_rate = 1e-3
bptt = 4
batch_size = 128
T = 20
forward= rnn_step
backward = rnn_step_backward
ix_sentences=[]
for x in range(0,len(X)):
    index_sentence = X[x].strip().split(" ")
    index_sentence = map(int,map(str,index_sentence))
    ix_sentences.append(index_sentence)

X = np.array(ix_sentences)
print(X.shape)
trainX = X[:5000]
testX = X[5000:]
trainY = np.array(y[:5000])
testY = np.array(y[5000:])
for e in range(epoch):
    totalLoss = 0.0
    for batch_start in range(0,trainX.shape[0],batch_size):
        batchX = X[batch_start:min(batch_start+batch_size,trainX.shape[0])]
        hprev = np.zeros((batchX.shape[0],hidden_dimension))
        for start_index in range(0,T,bptt):
            sent = batchX[:,start_index:min(start_index+bptt,T)]
            caches = []
            wcaches = []
            for t in range(0,sent.shape[1]):
                # print("Starting")
                # print("Sent")
                # print(sent[:,t].shape)
                wforward,wcache = word_embedding_forward(sent[:,t],Wembed)
                # print("Embed")
                # print(wforward.shape)
                hprev,tcache = forward(wforward,hprev,Wx,Wh,bh)
                # print("H")
                # print(hprev.shape)
                caches.append(tcache)
                wcaches.append(wcache)
                
                out,tcache = affine_forward(hprev,Why,by)
                # print("Out")
                # print(out.shape)
                predicted,loss,dx = softmax_loss(out,trainY[batch_start:min(batch_start+batch_size,X.shape[0])])
                totalLoss += loss
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
    losses.append(totalLoss/trainX.shape[0])
    print("Loss: "+str(losses[-1]))

summ = 0
for batch_start in range(0,testX.shape[0],batch_size):
    batchX = testX[batch_start:min(batch_start+batch_size,testX.shape[0])]
    hprev = np.zeros((batchX.shape[0],hidden_dimension))
    for start_index in range(0,T,bptt):
        sent = batchX[:,start_index:min(start_index+bptt,T)]
        for t in range(0,sent.shape[1]):
            wforward,_ = word_embedding_forward(sent[:,t],Wembed)
            hprev,_= forward(wforward,hprev,Wx,Wh,bh)

        out,_= affine_forward(hprev,Why,by)
        predicted= softmax_loss(out)
        summ+=np.sum(predicted == testY[batch_start:min(batch_start+batch_size,testX.shape[0])])

print(1.*summ/testY.shape[0])
data = {}
data['Wh']=Wh.tolist()
data['Wembed']=Wembed.tolist()
data['Wx']=Wx.tolist()
data['bh']=bh.tolist()
data['by']=by.tolist()
data['Why']=Why.tolist()
data['loss']=losses

json.dump(data,open("model.json","wb"))