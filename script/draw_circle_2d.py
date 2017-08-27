#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Ellipse, Circle
from matplotlib.backends.backend_pdf import PdfPages

def main(x, r, file_path=None, cValue=None, params=None):
    c_map = ['b','g','r','c','m','y']
    n = len(x) # nx2
    if cValue is None:
        c_id = np.random.randint(0,6,size=n)
        cValue = [c_map[id] for id in c_id]
    #print cValue
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    # draw circle
    for i in xrange(n):
        ax.add_patch(Circle(xy=(x[i][0],x[i][1]), radius=r[i], fill=False, ec = cValue[i], alpha=1))
        #ax.add_patch(Circle(xy=(x[i][0],x[i][1]), radius=r[i], fill=False))

    # draw scatter
    ax.scatter(x[:, 0], x[:, 1], c = cValue, marker='x')
    plt.axis('scaled')
   # plt.axis([-10,10,-10,10])
    #plt.axis('auto')
    plt.show()
    if not file_path is None:
        pp = PdfPages(file_path)
        pp.savefig(fig)
        pp.close()
        print "saved!"

if __name__ == '__main__':
    x = np.random.randint(1,20,size=[6,2])
    r = np.random.randint(1,5,size=6)
    file_path = "/Users/wangyun/Desktop/test.pdf"
    main(x,r,file_path)

