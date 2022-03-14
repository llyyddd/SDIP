# Cluster the candidates of each class and find the shapelets of each class

import time
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from shapeletCandidates import *


#cluster number
cluster_num=3
def loadDataSet(shapeleCan_list):
    # Process the data set format and add a blank value to each list to store the class
    for SC in shapeleCan_list:
        SC.insert(len(SC), 0)
    return shapeleCan_list

#DTW distance
def distance(x,y):
    return fastdtw(x, y, dist=euclidean)


# Initialize the cluster centers
def choice_center(data, k):
     centers = []
     for i in np.random.choice(len(data), k):
         centers.append(data[i])

     return centers

def k_center(data_list,center,class_num):
     print("generating the shapelets for class {}".format(class_num))

     flag = True
     i = 0
     iter=0
     while flag:
         iter+=1
         print("the {}-th cluster".format(iter))

         flag = False

         for i in range(len(data_list)):
             min_index = -2
             min_dis = float('inf')
             for j in range(len(center)):
                 dis,path = distance(data_list[i][1:-1], center[j][1:-1])
                 if dis < min_dis:
                     min_dis = dis
                     min_index = j
             if data_list[i][-1] != min_index:
                 flag = True
             data_list[i][-1] = min_index

         for k in range(len(center)):

             current_k = []
             for i in range(len(data_list)):
                 if data_list[i][-1] == k:
                     current_k.append(data_list[i])
             # print(k, "：", current_k)

             old_dis = 0.0
             for i in range(len(current_k)):
                 old_dis += distance(current_k[i][1:-1], center[k][1:-1])[0]

             for m in range(len(current_k)):
                 new_dis = 0.0
                 for n in range(len(current_k)):
                     new_dis += distance(current_k[m][1:-1], current_k[n][1:-1])[0]
                 if new_dis < old_dis:
                     old_dis = new_dis
                     center[k][:] = current_k[m][:]
                     # flag = True

         # i +=1


     for i in range(len(data_list)):
         min_index = -2
         min_dis = float('inf')
         for j in range(len(center)):
             dis = distance(data_list[i][1:-1], center[j][1:-1])[0]
             if dis < min_dis:
                 min_dis = dis
                 min_index = j
         data_list[i][-1] = min_index

     # the number of each cluster
     dict={}
     for i in range(len(data_list)):
         cluster=data_list[i][-1]
         if cluster not in dict.keys():
             dict[cluster] = 1
         dict[cluster] += 1

     shapelet=[]
     threshold=sum(dict.values())/len(center)
     for key, value in dict.items():
         if (value >= threshold):
             # print("dddd",key,center[key])
             shapelet.append(center[key])
     print("done!")
     return shapelet

def write2Txt(shapelet, out_path):
    # write shapelet to out_path
    f = open(out_path, 'a+')
    length=len(shapelet)
    f.write(str(length))
    f.write('\n')
    for info in shapelet:
        f.write(str(info[0])+',')
        f.write(str(len(info)-2)+',')
        f.write(','.join(str(i) for i in info[1:-1]))
        f.write('\n')

    # f.write('\n')
    f.close()

def generateFinalShapelet(args):
    test = pd.read_table(args.train_data_path, sep='  ', header=None, engine='python').astype(float)

    test_y = test.loc[:, 0].astype(int)
    class_num = np.unique(test_y)
    for i in class_num:
        print("generating shapelet candidates for class {}....".format(i))
        # Find the shapelet candidates of each class
        shapeletC = generateCSC(args, i)
        # print(shapeletC)
        print("done!")
        data_list = loadDataSet(shapeletC)
        # print(data_list)
        centers = choice_center(data_list, cluster_num)
        shapelet = k_center(data_list, centers, i)

        write2Txt(shapelet, args.result_path)

    print("the shapelets has been generated in {}".format(args.result_path))

