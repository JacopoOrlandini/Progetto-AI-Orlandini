import matplotlib.pyplot as plt
import numpy as np
import random

# result from test-embedding
x = []
y = []

namefile = "embedding_crossValidation.txt"
with open(namefile) as fp:
    line = fp.readline()
    cnt = 0
    while line:
        if cnt % 2 == 0 and cnt == 0:
            line_data = line.strip()
            num_emb = line_data[0:3]
            name_emb = line_data[23:]
            x.append(num_emb)
        if cnt / 2 == 1:
            result = line.strip()[6:11]
            y.append(result)
        line = fp.readline()
        cnt += 1
        if cnt > 3:
            cnt = 0

fig, ax = plt.subplots()
ax.plot(y)

ax.set(xlabel='test', ylabel='accuracy',
       title='Embedding result')


plt.show()
