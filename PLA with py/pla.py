import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt    

## dataset[row][0]: x1
## dataset[row][1]: x2
## dataset[row][2]: label

def data_process(path):
    rawdata = open(path).readlines()
    size = len(rawdata)
    cols = len(rawdata[0].strip().split(' '))
    
    dataset = [[0.0]*cols for i in range(size)]
    idx = 0
    for line in rawdata:
        tmp = line.strip().split(' ')
        dataset[idx][0] = float(tmp[0])
        dataset[idx][1] = float(tmp[1])
        dataset[idx][2] = float(tmp[2])
        idx += 1
    return dataset

def rand_list(length): # this create random weight in [0, 1)
    rlist = []
    '''
    for i in range(length):
        rlist.append(random.random())
    '''
    for i in range(length):
        rlist.append(random.randrange(0, 100))
    
    return rlist

def siggn(y):
    sign = 0.0
    if y >= 0.0:
        sign = 1.0
    else:
        sign = 0.0
    return sign

def pla(dataset):
    iteration = 0
    lr = 0.01
    w = rand_list(3)
    while True:
        finished = True
        for i in range(0, len(dataset)):
            x = dataset[i][:-1]
            y = w[0] + w[1]*x[0] + w[2]*x[1]
            if siggn(y) == dataset[i][2]:
                continue
            else:
                finished = False
                w[0] += lr*(dataset[i][2] - siggn(y))
                w[1] += lr*(dataset[i][2] - siggn(y))*x[0]
                w[2] += lr*(dataset[i][2] - siggn(y))*x[1]
        iteration += 1
        if finished == True:  # work until there is no misclassification
            break
    return w, iteration

def check_test(testset, w):
    count = 0
    for i in range(0, len(testset)):
        y = w[0] + w[1]*testset[i][0] + w[2]*testset[i][1]
        if siggn(y) == testset[i][2]:
            continue
        else:
            count += 1
    return count

def main():    
    path_test = "./twoclassData/set.test"
    test = data_process(path_test)
    
    for i in range(1, 11):
        fig, ax=plt.subplots()
        path_train = "./twoclassData/set" + str(i) + ".train"
        train = data_process(path_train)
        
        # use pandas and numpy only for picture
        df = pd.DataFrame(test)
        feature = df.iloc[:,[0,1]]
        label = df[2]
        x_zero = feature[label == 0.0]
        x_one = feature[label == 1.0]
        
        ax.scatter(x_zero[0],x_zero[1],marker="o",label="y=0")
        ax.scatter(x_one[0],x_one[1],marker="x",label="y=1")
        ax.legend()
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title("Data with weights trained by set")
        
        final_w, iterations = pla(train)
        print('Iteration times: ', iterations)
        line_x1 = np.linspace(0, 10, 200)
        line_x2 = (-final_w[0] - final_w[1]*line_x1) / final_w[2]
        ax.plot(line_x1, line_x2, 'black')
        mis = check_test(test, final_w)
        print('Misclassification rate: ', mis / len(test))
    

if __name__ == '__main__':
    main()