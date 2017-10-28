import json
import matplotlib.pyplot as plt
model = json.load(open("model1.json","rb"))
loss = model['loss']

plt.plot(loss)
plt.title("Loss VS Iteration")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()
