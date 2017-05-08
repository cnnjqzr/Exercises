import numpy as np
from numpy import dot
from numpy.linalg import inv
import matplotlib.pyplot as plt
from functools import reduce
def mdot(*args):
    """Multi argument dot function. http://wiki.scipy.org/Cookbook/MultiDot"""
    return reduce(np.dot, args)
def prepend_one(X):
    """prepend a one vector to X."""
    return np.column_stack([np.ones(X.shape[0]), X])

def grid2d(start, end, num=50):
    """Create an 2D array where each row is a 2D coordinate.
    np.meshgrid is pretty annoying!
    """
    dom = np.linspace(start, end, num)
    X0, X1 = np.meshgrid(dom, dom)
    return np.column_stack([X0.flatten(), X1.flatten()])
###############################################################################
def Data_preprocessing(file_name):
    data = np.loadtxt(file_name)
    X, y = data[:, :2], data[:, 2]
    return X,y

def Quadratic_feature(x):
    temp = list(x.tolist())
    feature = list()
    for x in x:
        for t in temp:
            feature.append(x * t)
        temp.remove(x)
    return np.array(feature)

def Reg_function(X , y, Lambda , function):
    if function == "Quad":
        phi = list()
        for x in X :
            phi.append(Quadratic_feature(x))
        phi = np.array(phi)
    else:
        phi = X

    beta_ = mdot(inv(dot(phi.T, phi) + Lambda * np.identity(phi.shape[1])), phi.T, y)
    return beta_

def mean_std_plot(Lambda,mean,std):
    x = np.array(Lambda)
    y = np.array(mean)
    e = np.array(std)
    plt.errorbar(x, y, e, linestyle='None', marker='o')
    plt.show()

def square_error(phi,beta_, y):
    # square error
    s_e = 0
    for i in range(0,y.shape[0]):
        prediction = mdot(beta_ , phi[i, :])
        s_e += (y[i] - prediction) ** 2
    print ("Square error" , s_e)
    return s_e

def cross_validation(X,y,lambda_,feature,k):
    if k>1:
        block_size = X.shape[0]/int(k)

        blocks_X = list()
        blocks_y = list()

        remain_X = X
        remain_y = y
        # partition data into blocks_X
        for i in range(0,k,1):
            begin = 0
            end = block_size
            blocks_X.append(remain_X[begin:end,:])
            blocks_y.append(remain_y[begin:end])
        # cross validation
        square_error_list = list()
        beta_list = list()
        for i in range(0, k, 1):
            print ("\n%d-th cross validation------------------"%(i+1))
            validation_X = np.array(list(blocks_X).pop(i))
            validation_y = np.array(list(blocks_y).pop(i))

            temp_X = list(blocks_X)
            temp_y = list(blocks_y)
            del temp_X[i]
            del temp_y[i]

            train_X = []
            train_y = []
            for block in temp_X:
                train_X += block.tolist()
            for block in temp_y:
                train_y += block.tolist()
            train_X = np.array(train_X)
            train_y = np.array(train_y)

            beta_ = Reg_function(train_X, train_y, lambda_, feature)

            if feature == "Quad":
                phi = list()
                for x in validation_X:
                    phi.append(Quadratic_feature(x))
                phi = np.array(phi)
            else:
                phi = validation_X
            square_error_list.append(square_error(phi,beta_,validation_y))
            beta_list.append(beta_)


        mean = np.mean(square_error_list)
        var = np.var(square_error_list)
        print ("Mean squared error: ",mean)
        print ("Variance: ",var)
        return (beta_list,mean,var)

    elif k==1:
        beta_ = Reg_function(X, y, lambda_, feature)
        if feature == "Quad":
            phi = list()
            for x in X:
                phi.append(Quadratic_feature(x))
            phi = np.array(phi)
        else:
            phi = X
        mean =square_error(phi, beta_, y)
        print ("Mean squared error: ", mean)
        print ("Variance: ", 0)
        return ([beta_],mean,0)


def main():
    Lambda_list = [i * i for i in range(1, 200, 2)]
    mean_list = list()
    var_list = list()
    argv = "3"
    type = ""
    k = 1
    if argv is None:
        pass
    elif argv == "1":
        X, y = Data_preprocessing("dataLinReg2D.txt")
        X = prepend_one(X)
        type = "lin"
    elif argv == "2":
        X, y = Data_preprocessing("dataQuadReg2D.txt")
        X = prepend_one(X)
        type = "Quad"
    elif argv == "3":
        X, y = Data_preprocessing("dataQuadReg2D_noisy.txt")
        X = prepend_one(X)
        type = "Quad"
        k = 10
    for Lambda in Lambda_list:
        result_list = cross_validation(X, y, Lambda, type, k)
        print ("lambda ", Lambda)
        print ("\n")
        mean_list.append(result_list[1])
        var_list.append(result_list[2] ** (1 / 2.0))
    mean_std_plot(Lambda_list, mean_list, var_list)
if __name__ == "__main__":
    main()