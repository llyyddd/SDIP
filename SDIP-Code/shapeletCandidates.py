# Each time series of c  produces a subsequence as a shapelet candidate of c
import copy

import numpy as np

import time
from PLNN import *
from changeSeq import *

def kgram(k, ts):
    ''' Produce time series grams.'''

    # A sliding window with a step size of 1 in the range from max to min
    composition = [ts[i:i + k] for i in range(0, len(ts) - k + 1, k)]
    # print("sub-senquence number:")
    # print(len(composition))
    return composition


def weight_kgram(k, weights):
    ''' Procuce weight matrix grams, each one is  a sub-matrix.'''
    # Take out the weight corresponding to each dimension of the subsequence
    composition = [weights[:, i:i + k] for i in range(0, len(weights[0]) - k + 1, k)]
    # print("sub-wights")
    # print(composition)
    return composition




def calculate_product_1(w, seq, new_seq, b):
    # the L2 distance between Ax and Ax‘
    '''
      Calculate the product of a sub-sequence of one time series and it's coefficient sub-matrix, w_i*x_i+b_i.
      Return the number of result less zeros.
    '''

    product1 = np.matmul(w, seq) + b
    product2 = np.matmul(w, new_seq) + b
    difference=np.linalg.norm(product1 - product2, ord=2)
    return difference

def calculate_product_2(w, x, b):
    # the number of inequality signs Changes

    '''
      Calculate the product of a sub-sequence of one time series and it's coefficient sub-matrix, w_i*x_i+b_i.
      Return the number of result less zeros.
    '''

    product = np.matmul(w, x)+b
    #print(len(product[product<0]))
    return len(product[product<0])


def findSubseq(args, TSdata, data_index):
    '''
    find the interpretable subsequence of the given time series TSdata
    :param TSdata: a time series
    :param inequality_path: the file path to save the inequalities
    :param data_index: the index of TSdata
    :return: (start, end, score(the L2 distance), index)
    '''
    # define sliding window
    ratio = 0.1
    if args.datasize > 50:
        Lmin = 10
    else:
        Lmin = 6
    Lmax = args.datasize + 1

    length = len(TSdata)

    # Load the weights and biases of the trained model for calculation

    coefficients = interpret(args, TSdata)

    weights = coefficients[:, :-2]

    bias = coefficients[:, length:-1]

    dict1 = {}
    dict2 = {}

    # sliding window
    k = Lmax - 1
    while k >= Lmin:
        # print(k)
        start = 0

        while start <= TSdata.size - k:
            # add noise to the subsequence
            seq = incrementSeq(TSdata, start, start + k - 1, TSdata.size)
            new_ts = TSdata + seq

            diff1 = calculate_product_1(weights, TSdata.reshape(length, 1), new_ts.reshape(length, 1), bias)
            diff2 = calculate_product_2(weights, new_ts.reshape(length, 1), bias)

            dict1[(start, k)] = diff1
            dict2[(start, k)] = diff2

            if (ratio * k) < 1:
                start += 1
            start += int(ratio * k)
        k = int(k / 2)
    # print("The total create sub-sequeence:", len(dict2))

    f1 = zip(dict1.values(), dict1.keys())
    c1 = sorted(f1, reverse=True)

    f2 = zip(dict2.values(), dict2.keys())
    c2 = sorted(f2, reverse=True)


    maxnum = c2[0][0]
    threshold_value = 10
    diff = 0
    inter_seq = []
    for i in c2:
        if i[0] == maxnum:
            inter_seq.append(i)

    maxdiff1 = 0
    if inter_seq == len(dict2):
        print("NO shapelet")
        return 0, 0, 0, data_index
    for i in inter_seq:
        # print(i)
        if (dict1[i[1]] > maxdiff1):
            maxdiff1 = dict1[i[1]]
            S = i
            # print(S)
    # print("the interpret subseq of the time series:", S)
    # print(S[1][0], S[1][0] + S[1][1] - 1)


    return S[1][0], S[1][0] + S[1][1] - 1, maxdiff1, data_index




def generateCSC(args, class_label):
    '''
    find the shapelet candidates set of class c
    :param txt_path: the file path of training set
    :param inequality_path: the file path of the saved inequalities
    :param class_label: the current class
    :return:(score,shapelet candidate)
    '''

    test = pd.read_table(args.train_data_path, sep='  ', header=None, engine='python').astype(float)

    # label
    test_y = test.loc[:, 0].astype(int)
    # data
    test_x = test.loc[:, 1:]

    current_class_index = np.where(test_y == class_label)[0]

    # shapelet candidates set：each one contains start,end, score,index

    candidate=[]

    for i in current_class_index:
        one_ts = np.array(test_x.loc[i, :])
        one_ts_label = np.array(test_y.loc[i])
        one_ts_S=[]

        a, b, c, d = findSubseq(args, one_ts, i)
        one_ts_S.append(a)
        one_ts_S.append(b)
        one_ts_S.append(c)
        one_ts_S.append(d)

        one_ts_S=np.asarray(one_ts_S)


        seq_start = one_ts_S[0].astype('int64')
        seq_end = one_ts_S[1].astype('int64')
        seq_score = one_ts_S[2].astype('float')
        seq_index = one_ts_S[3].astype('int64')
        seq = list(test_x.loc[seq_index, seq_start:seq_end])
        seq.insert(0, seq_score*-1.)
        candidate.append(seq)

    return candidate

def generateShapeletCandidates(args):
    '''
    discover candidate time
    '''
    start = time.process_time()

    test = pd.read_table(args.train_data_path, sep='  ', header=None, engine='python').astype(float)

    test_y = test.loc[:, 0].astype(int)
    class_num = np.unique(test_y)
    for i in class_num:

        shapeletC = generateCSC(args, i)

    end = time.process_time()
    return end-start
