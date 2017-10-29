#Document Classifier Using Vanilla Recurrent Neural Network

The Dataset consists of Textual Data that belong to one of the **8** Categories

## Data Preprocessing Steps
1. **Tokenize** The Sentences
2. Remove **Tokens** That Are **Stop Words**
3. If the number of words in the Document is **Greater Than 20** then retain only the **First 20 Words** of the sentence.
4. Convert **Every Word** to its **Unique Identifcation Number**
5. Now for the sentences with number of words **Less Than 20**, pad the sentence with **0**'s

Had to Fix the **Sequence Length** to **20** and **Pad** with **0**'s to make **Training Faster**

## RNN Model Configuration
1. **Learning Rate = 1e-3**
2. **Epochs = 4050**
3. **Word Embedding Dimension = 100**
4. **Hidden State Dimension = 128**
5. **Truncated Backpropagation Length = 4**
6. **Training Sequence Length = 20**
7. **Batch Size = 128**
8. **Weight Initialization** was done from a **Gaussian Distribution** with **mean=0.0** and **std=1**
9. **Bias** were **Zero Initialized**

### Model Performance
 **Test Set Accuracy = 74.22%** \
 The training time for the model was about **6 hours**

### Loss-Iteration Curve
![Loss-Iteration Curve for 1100 Epochs](/loss_curve.png)

### Downloading Model
The Model can be downloaded from <a href="https://drive.google.com/open?id=0B6OWaNVUCQvaN2FjYkF5UGdsa1U">here</a>
