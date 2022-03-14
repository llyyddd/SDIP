# SDIP

import numpy as np
import argparse

import pandas as pd
from PLNN import *
from shapeletCandidates import *
from ClusterFinalShapelet import *

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PLNN Coffee Example')
    parser.add_argument('--batch-size', type=int, default=64,
                        metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=500,
                        metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        metavar='LR', help='learning rate default(: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=True, help='desables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50,
                        metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, help='For saving the current model')
    parser.add_argument('--train_data_path', type=str, default='', help='input data path for training')
    parser.add_argument('--test_data_path', type=str, default='', help='input data path for testing')
    parser.add_argument('--datasize', type=int, default='', help='input the data length of the dataset')
    parser.add_argument('--model_path', type=str, default='', help='input the path for save PLNN model')
    parser.add_argument('--label_format', type=int, default='', help='input the data format')
    parser.add_argument('--stage', type=str, default='', help='train for training a PLNN model and discover_shapelet for generate linear inequalities')
    parser.add_argument('--H1', type=int, default='4', help='Number of nodes for the first hidden layer')
    parser.add_argument('--H2', type=int, default='16', help='Number of nodes for the second hidden layer')
    parser.add_argument('--H3', type=int, default='2', help='Number of nodes for the third hidden layer')
    parser.add_argument('--result_path', type=str, default='output/res.txt', help='the path for final shapelet set')

    args = parser.parse_args()
    # print(args)
    stage = args.stage


    if stage == 'train':
        train(args)
    elif stage == 'discover_shapelet':
        generateFinalShapelet(args)
        # interpret(args, data)
        time = generateShapeletCandidates(args)
        print("The shapelets discovery time is : %.03f seconds" % time)
    else:
        print("Wrong Command!")

if __name__=='__main__':
    main()