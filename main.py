import model
import sys

def create_model():
    model.create_and_save()

def predict():
    print('Choose an algorithm: ')
    algo = input('1:OCSVM\n2:Clustering\n-->')
    print (algo)
    model.predict_anomalies(algo)

if __name__ == '__main__':
    if (len(sys.argv) == 2):
        if(sys.argv[1] == 'create'):
            create_model()
        elif(sys.argv[1] == 'test'):
            predict()
    else:
        print ('Usage:\tmain.py <Create Model["create"] / Test on full test set["test"]>')
        sys.exit(0)