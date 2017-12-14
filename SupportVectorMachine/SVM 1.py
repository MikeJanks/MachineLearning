import numpy as np
from matplotlib import pyplot as plt

def svm_sgd_plot(X, Y):
    #Initializing the weight vector to zero
    w = np.zeros(len(X[0]))
    #learning rate
    eta = 1
    #The number of iterations for training
    epochs = 100000
    #stores micalculations
    errors = []

    for epoch in range(1, epochs):
        error = 0
        for i, x in enumerate(X):
            if(Y[i]*np.dot(X[i], w)) < 1:
                #misclassified
                w = w + eta * ((X[i] * Y[i]) + (-2 * (1/epoch)* w))
                error = 1
            else:
                #correct
                w = w + eta * (-2 * (1/epoch) * w)
        errors.append(error)

    # plt.plot(errors, '|')
    # plt.ylim(0.5,1.5)
    # plt.axes().set_yticklabels([])
    # plt.xlabel('Epoch')
    # plt.ylabel('Misclassified')
    # plt.show()

    return w


def main():

    w = svm_sgd_plot(X, y)

    if(1 <= np.dot(Xtest, w)):
        print("Says 1")
        if(1 <= Ytest[0]):
            print("Correct")
        else:
            print("Wrong")
    else:
        print("Says -1")
        if(1 > Ytest[0]):
            print("Correct")
        else:
            print("Wrong")




if __name__ == "__main__":
    main()