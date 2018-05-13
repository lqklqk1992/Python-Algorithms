import numpy as np


def svm_sgd_train_HIGGS():
    #f(w,b)=0.5*lambda*|w|2+max(0,1-y(w*x+b)), lambda=1/c;
    w = np.zeros(28)
    b=0
    epochs = 10000
    maxerr=0.001
    c=100      #slack penalty
    t=1        #time step

    for epoch in range(1,epochs):
        w1=w
        with open("HIGGS") as file:
            for l in file:   #read sample one by one
                arr=l.split(' ')
                y=int(arr[0])
                if y==0:
                    y=-1
                x=np.zeros(28)
                for i in range(1,len(arr)):
                    index=int(arr[i][:arr[i].find(":")])
                    value=float(arr[i][arr[i].find(":")+1:])
                    x[index-1]=value
                
                eta=1/t                                               # learning rate keep decreasing
                if (y*(np.dot(x, w)+b)) < 1:                          # if sample is between + and - support vectors or misclassified
                    w = w + eta * ( (x * y) - ((1/c)* w) )            # wj=wj+eta*((xij*yi)-lambda*w)
                    b = b + eta * y                                   # b=b+eta*yi
                else:
                    w = w + eta * (-1  *(1/c)* w)                     # else, w=w+eta*(-lambda*w)
                t=t+1

        print("epoch: %d, change in w: %f"%(epoch,np.sum(np.abs(w-w1))))

        if np.sum(np.abs(w-w1)) < maxerr:           # if w converge then break
            break

        with open('weight', 'w') as fo:  # write weight to file: w0 w1 w2 ... w27 b
            for i in range(0,28):
                fo.write('%f '%(w[i]))
            fo.write('%f'%(b))

    return [w,b]

def load_weight():
    w = np.zeros(28)
    b=0
    file = open("weight", "r") 
    data=file.readline().split(' ')  
    for i in range(len(data)):
        if i<28:
            w[i]=float(data[i])
        else:
            b=float(data[i])
    file.close()
    return [w,b]


if __name__=='__main__':

    #svm_sgd_train_HIGGS()

    w,b=load_weight()  #load weight

    with open("events.txt") as file:      #read event file and classify the data
        fo = open("result.txt", "w")      #write predict label to result.txt
        for l in file:
            arr=l.split(' ')
            x=np.zeros(28)

            for i in range(len(arr)):
                if arr[i].find(":")!=-1:
                    index=int(arr[i][:arr[i].find(":")])
                    value=float(arr[i][arr[i].find(":")+1:])
                    x[index-1]=value

            if (np.dot(x, w)+b)>0:
                fo.write("1 ")
            else:
                fo.write("0 ")
        fo.close()

    print("Pls open result.txt to see the predict label.")

