***************************************************************
*****************      SDIP (2021)     *******************
**** Author :           MHT, LIU YD      *******************
**** please contact mahaitao@qhd.neu.edu.cn for any problems *******
***************************************************************

The code is associated with the following paper:
SDIP: A Fast Time Series Shapelet Discovery Method Based on the Interpretation of Piecewise Linear Neural Networks (PLNN)

Detailed usage is explained below, here we give sample usage
for the main methods of the papers above. The example dataset is Coffee dataset from UCR archive, 
which has 2 classes(0 and 1), 28 train data, 28 test data, the length of all time series is 286.  
---------------------------------------------------------------
1) $ python SDIP.py --stage train --train_data_path  ./data/Coffee_TRAIN.txt --test_data_path ./data/Coffee_TEST.txt --model_path ./Coffee_PLNN.pt --datasize 286 --label_format 0 
2) $ python SDIP.py --stage discover_shapelet --train_data_path ./data/Coffee_TRAIN.txt --test_data_path ./data/Coffee_TEST.txt --model_path ./Coffee_PLNN.pt --datasize 286 --label_format 0 --result_path ./output/shapelets.txt
 
---------------------------------------------------------------

The code is written in Python 3.8 and tested on a unbutu 16.04 LTS OS.

== How to run the code ==

1. Use "SDIP.py" to train a PLNN model by employing the data in the training set . The trained model will be saved in "*.pt".
2. Use " SDIP.py " to interpret the trained PLNN model, that is, generate a set of linear inequalities, will be saved in "inequality.txt"; then to discover shapelets by utilizing linear inequalities in "inequality.txt", the shapelets founded will be saved in " output/shapelets.txt ".

== Usage ==

1. train the PLNN model.
$ python SDIP.py --stage <train|discover_shapelet> -train_data_path  <traindata> -test_data_path <testdata> -model_path <modelfile> -datasize <datasize> -label_format <[0|1]> [-H1 <H1>] [-H2 <H2>] [-H3 <H3>] [-epochs <epochs>]


2. discover shapelets
$ python SDIP.py -stage <train|discover_shapelet> -train_data_path  <traindata> -test_data_path <testdata> -model_path <modelfile> -datasize <datasize> -label_format <[0|1]> [-H1 <H1>] [-H2 <H2>] [-H3 <H3>] [-epochs <epochs>] -result_path <resultfile>

Description: 
  
  -stage              
                         train :              train a PLNN model             
                         discover_shapelet :  generate the shapelets set
	 					 
  -train_data_path       set the filename of training dataset
  
  -test_data_path        set the filename of test dataset
  
  -model_path            set the modelfile name for saving model
  
  -datasize              set the lengh of an instance time series of training dataset
  
  -label_format          set the label format, 0 or 1
                         0: the class label of time series dataset is start from 0
                         1: the class label of time series dataset is start from 1
						 
  -H1                    set the number of neurons for the first hidden layer of PLNN model
  
  -H2                    set the number of neurons  for the second hidden layer of PLNN model
  
  -H3                    set the number of neurons  for the thrid hidden layer of PLNN model
  
  -epochs                set the epochs for training a PLNN model
  
  -result_path           set the ouput file for saving shapelets set founded
  

