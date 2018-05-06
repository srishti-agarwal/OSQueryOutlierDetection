# OSQueryOutlierDetection
This is a research project work for CSCE 665. 
It analyses OSQuery logs, trains a model in unsupervised fashion and predicts 
over a test set.

Code Details:
1. Libraries required:
   1. pandas
   2. numpy
   3. scikit-learn
   4. matplotlib


Steps to run:
Before executing any of the following commands, ‘train.log’, 'test.log' for windows or 'train_lin.log' , test_lin.log'
must be present in the ‘data’ directory.
   1. Create and save features
      1. Run the command:
         $python main.py create
      2. Next select from windows or linux, choose 1 for windows and 2 for linux 
      3. This will create and save features in csv format in the data directory 
   
   2. Generate predictions from the features created in step 1
      1. Run the command:
         $python main.py test
      2. It will prompt you to select the type of classifier. 
      3. After choosing an option, the selected model will be created.
      4. See anomalous.csv to analyse the predicted anomalies from the test data.

* Code is developed using python 2.7 and should be compatible for higher versions as well
