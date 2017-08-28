import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

class CalFractal(object):
    @staticmethod
    def _unique(l_l, N_l):
    if len(N_l)<=1:
        return (l_l,N_l)

    N_last = N_l[-1]

    same_count = 0

    for i in range(2, len(N_l)):
        if(N_l[-i]==N_last):
            same_count+=1

    return ( l_l[:-same_count], N_l[:-same_count] )

    @staticmethod
    def rn_box_cover(l, nsample, d, nodes):
        xmin = np.amin(nodes, axis=0)
        xmax = np.amax(nodes, axis=0)
        count = np.size(nodes, 0)
        N_array = np.zeros(nsample)
        for i in xrange(nsample):
            cover = dict()
            x0 = xmin - rd.uniform(0, l, size=d)
            #print(x0)
            for j in xrange(count):
                foo = (nodes[j] - x0) / l
                cover[tuple(foo.astype(int))]=1
                #cover[ np.floor((nodes[j]-x0))/l ] = 1
            N_array[i] = len(cover)
        # print(cover)
        return min(N_array)

    @staticmethod
    def rn_fractal(nodes, d, params, save_path = None):
        xmin = np.amin(nodes, axis=0)
        xmax = np.amax(nodes, axis=0)
        count = np.size(nodes, 0)
        L = max(xmax-xmin)

        nsample = params["nsample"]

        l_list=[]
        N_list = []
        l = L * 2.0
        N = 1

        #while N < count:
        for k in range(30):
            N = CalFractal.rn_box_cover(l, nsample, d, nodes)
            l_list.append(l)
            N_list.append(N)
            l = l / 2


        (l_list, N_list) = CalFractal._unique(l_list, N_list)
        log_l = np.array([np.log2(item) for item in l_list])
        log_N = np.array([np.log2(item) for item in N_list])

        f = np.poly1d(np.polyfit(-log_l, log_N, 1))

        #print(l_list)

        #print(N_list)
        plt.plot(-log_l, log_N)
        if save_path is not None:
            plt.savefig("save_path")
        #print(f)
        return f(0)

def main():
    nodes = np.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]])
    params = {"nsample": 3000}
    print(CalFractal.rn_fractal(nodes, 3, params))

if __name__ == "__main__":
    main()
