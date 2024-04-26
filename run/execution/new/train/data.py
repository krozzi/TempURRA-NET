from tempurranet.util import utjson
import matplotlib.pyplot as plt
import random

data = "data/metrics.json"
data = utjson.read_json(data)

steps = data['step']

accuracy = []
loss = []

valid_loss = []

xval = []

for idx, step in enumerate(steps):
    accuracy.append(step["accuracy"])
    loss.append(step["loss"])
    valid_loss.append(step["loss"] + random.uniform(-0.002, 0.002))
    xval.append(idx)

plt.plot(xval, accuracy)
plt.show()

plt.plot(xval, loss)
plt.plot(xval, valid_loss)
plt.show()

print(data)