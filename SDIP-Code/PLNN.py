import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pandas as pd
from data_Loader import MyCustomDataset



def active_state(x):
    '''Calculate hidden neurons' active states.
       - x: hidden neurons outputs
       - Return a list of zeros/ones of hidden neuron states
    '''
    x = x.detach().numpy()
    x = x.reshape(x.shape[1])
    states = x.copy()
    states[x > 0] = 1
    return list(states.astype(int))


class PLNN(nn.Module):
    def __init__(self, D_in, H1, H2, H3, D_out):
        super(PLNN, self).__init__()
        self.fc1 = nn.Linear(D_in, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, H3)
        self.fc4 = nn.Linear(H3, D_out)

    def forward(self, x,stage):
        if stage=='train':
            h1 = F.relu(self.fc1(x))
            h2 = F.relu(self.fc2(h1))
            h3 = F.relu(self.fc3(h2))
            out = self.fc4(h3)
            out = F.log_softmax(out, dim=1)
            return out
        states = {}
        h1 = F.relu(self.fc1(x))
        states['h1'] = active_state(h1)
        # print('Hidden Layer 1 active states: ', states['h1'])
        h2 = F.relu(self.fc2(h1))
        states['h2'] = active_state(h2)
        h3 = F.relu(self.fc3(h2))
        states['h3'] = active_state(h3)
        #
        out = self.fc4(h3)
        out = F.log_softmax(out, dim=1)
        return states, out


def calculate_inequality_cofficients(args, model, data, filename):

    states, output = model(data,args.stage)
    #    _, prediction = torch.max(output.data, 1)
    #   prediction = np.array(prediction)
    #  prediction = prediction.reshape(prediction.shape[0],1)
    #  print("prediction is ", prediction)
    w1, b1 = model.state_dict()['fc1.weight'], model.state_dict()['fc1.bias']
    w2, b2 = model.state_dict()['fc2.weight'], model.state_dict()['fc2.bias']
    w3, b3 = model.state_dict()['fc3.weight'], model.state_dict()['fc3.bias']
    w4, b4 = model.state_dict()['fc4.weight'], model.state_dict()['fc4.bias']

    diag_s1 = torch.diag(torch.tensor((states['h1']),
                                      dtype=torch.float32))
    w2_hat = torch.matmul(w2, torch.matmul(diag_s1, w1))
    b2_hat = torch.matmul(w2, torch.matmul(diag_s1, b1)) + b2

    diag_s2 = torch.diag(torch.tensor((states['h2']),
                                      dtype=torch.float32))

    w3_hat = torch.matmul(w3, torch.matmul(diag_s2, w2_hat))
    b3_hat = torch.matmul(w3, torch.matmul(diag_s2, b2_hat)) + b3
    #    print(w3_hat.size(), b3_hat.size())


    weights = torch.cat((w1, w2_hat, w3_hat)).numpy()
    bias = torch.cat((b1, b2_hat, b3_hat)).numpy()

    # bias = bias.reshape(22, 1)
    bias = bias.reshape(args.H1+args.H2+args.H3, 1)
    active_states = np.hstack((states['h1'], states['h2'],
                               states['h3'])).astype(int)
    # active_states = active_states.reshape(22, 1)
    active_states = active_states.reshape(args.H1+args.H2+args.H3, 1)

    weight_bias = np.append(weights, bias, axis=1)

    weight_bias_states = np.append(weight_bias, active_states, axis=1)

    #print(len(weight_bias_states))
    output_file = open(filename, 'wb')
    np.savetxt(output_file, weight_bias_states, delimiter=',')
    output_file.close()
    return filename


def change_inequality_signs(coefficient_filename):
    '''
       Change all the inequalities such that less and equal zeros into greate zeros by multiply -1 to both sides of them.
       That is, change  ax + by + c <= 0 into -ax-by-c > 0.
       '''
    weight_bias_states = np.loadtxt(coefficient_filename, delimiter=',')

    le_zeros = weight_bias_states[weight_bias_states[(slice(None), -1)] <= 0]  # 最后一个元素小于等于0，即状态为0
    g_zeros = weight_bias_states[weight_bias_states[(slice(None), -1)] > 0]    #即状态为1
    #print(le_zeros)
    #print(g_zeros)
    # transformed into inequalities greater than 0
    nle_zeros = -1 * le_zeros
    #print(nle_zeros)
    # make the right sides of all equalities to be 0.

    nle_zeros[:, -1] = 1
    g_zeros[:, -1] = 1
    # Put together in rows
    coefficients = np.concatenate((nle_zeros, g_zeros), axis=0)
    #print(coefficients)
    return coefficients


def check_inequality_coefficients(data, coefficient_filename):
    ''' Check '''
    # Check whether it has been transformed into an inequality that is all greater than 0
    coefficients = change_inequality_signs(coefficient_filename)
    weights = coefficients[:, :len(coefficients[0]) - 2]
    bias = coefficients[:, len(coefficients[0]) - 2:-1]

    x = data.numpy()
    x = x.reshape(-1, 1)
    #print(x.shape)
    #print(np.matmul(weights, x) + bias)  # all should be greate zeros


def train(args):
    use_cuda = not args.no_cuda and torch.cuda.is_avaiable()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # loading training data

    train_loader = torch.utils.data.DataLoader(
        MyCustomDataset(args.train_data_path, args.label_format),
        batch_size=args.batch_size, shuffle=True, **kwargs)


    test_loader = torch.utils.data.DataLoader(
        MyCustomDataset(args.test_data_path, args.label_format),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    #    print('length of test loader', len(test_loader))
    #   print('length of test loader dataset', len(test_loader.dataset))

    # define model
    # D_in, D_out = 24, 2
    D_in, D_out = args.datasize, 2
    # H1, H2, H3 = 16, 16, 16
    # training the model
    model = PLNN(D_in, args.H1, args.H2, args.H3, D_out).to(device)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        # print(epoch)
        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = data.to(device), target.to(device)
            #        data = data.view(-1, 2) # maybe should use this line for multiple dimension data

            optimizer.zero_grad()

            output = model(data, args.stage)
            loss = F.nll_loss(output, target)

            loss.backward()
            optimizer.step()

        # print("training loss", loss)
        if epoch % 100 == 0:
            #     print('########', batch_idx, len(data), len(train_loader.dataset), len(train_loader))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Training Loss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item()))

        #       testing
        model.eval()
        test_loss, correct = 0.0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data, args.stage)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)
            # print("testing loss", test_loss)
        if epoch % 100 == 0:
            print('\nTest set Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))

    if (args.save_model):
        # torch.save(model.state_dict(), 'TWOLeadECG_plnn.pt')
        torch.save(model.state_dict(), args.model_path)
    print("the trained PLNN model has been saved in {}!".format(args.model_path))


def interpret(args,data):
    # main()
    D_in, D_out = args.datasize, 2

    model = PLNN(D_in, args.H1, args.H2, args.H3, D_out)

    model.load_state_dict(torch.load(args.model_path))
    # print(model)
    test_loader = torch.utils.data.DataLoader(
        MyCustomDataset(args.train_data_path, args.label_format),
        batch_size=64, shuffle=True)

    model.eval()

    l = data.size

    data = data.reshape(-1, l)
    data = torch.from_numpy(data)
    data = data.type(torch.FloatTensor)
    states, output = model(data,args.stage)
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

    inequality_coefficient_filename = './inequality.txt'
    calculate_inequality_cofficients(args,model, data, inequality_coefficient_filename)
    coefficients = change_inequality_signs(inequality_coefficient_filename)
    check_inequality_coefficients(data, inequality_coefficient_filename)
    return coefficients








