#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

x = np.arange(1,10)
y = np.arange(1,10)
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_title('Scatter Plot')
plt.xlabel('X')
plt.ylabel('Y')
cValue = ['b','g','r','c','m','y']
ax1.scatter(x,y,c=cValue,marker='s')
plt.legend('x1')
plt.show()
