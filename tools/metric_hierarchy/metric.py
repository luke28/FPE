import os
import sys
import math

class Metric(object):
    @staticmethod
    def get_pairwise_res(labels):
        dic = {}
        for i in xrange(len(labels)):
            if labels[i] not in dic:
                dic[labels[i]] = [i]
            else:
                dic[labels[i]].append(i)
        ret = set()
        for k in dic:
            items = dic[k]
            for i in xrange(len(items)):
                for item in items[i+1:]:
                    ret.add((items[i], item))
        return ret

    @staticmethod
    def get_abcd(pre, truth, n):
        a = b = c = d = 0.0
        for item in pre:
            if item in truth:
                a += 1
            else:
                b += 1
        for item in truth:
            if item not in truth:
                c += 1
        d = n - a - b -c
        return a, b, c, d

    @staticmethod
    def pairwise_init(pre, labels):
        pre_dic = Metric.get_pairwise_res(pre)
        label_dic = Metric.get_pairwise_res(labels)
        n = len(labels)
        n = n * (n - 1) / 2
        a, b, c, d = Metric.get_abcd(pre_dic, label_dic, n)
        return a, b, c, d

    @staticmethod
    def Jaccard_Coefficient(pre, labels):
        a,b,c,d = Metric.pairwise_init(pre, labels)
        print a,b,c,d
        return a / (a + b + c)

    @staticmethod
    def FM(pre, labels):
        a,b,c,d = Metric.pairwise_init(pre, labels)
        try:
            return math.sqrt(a / (a + b) + a / (a + c))
        except ZeroDivisionError:
            print "FM Divide zero"
            print a, b, c
            m1 = 1
            m2 = 1
            if a + b != 0:
                m1 = a + b
            if a + c != 0:
                m2 = a + c
            return math.sqrt(m1 + m2)


