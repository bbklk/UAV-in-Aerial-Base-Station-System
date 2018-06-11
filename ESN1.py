# -*- coding: utf-8 -*-
"""
Echo State Networks
http://minds.jacobs-university.de/mantas
http://organic.elis.ugent.be/
http://www.scholarpedia.org/article/Echo_state_network
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import xlrd
from scipy import linalg
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve, accuracy_score
from sklearn.linear_model import Ridge, LogisticRegression
import matplotlib.ticker as ticker
import datetime

np.random.seed(47)
      
class ESN(object):

    def __init__(self, resSize=500, rho=0.9, cr=0.05, leaking_rate=0.2, W=None):
        """
        :param resSize: reservoir size  是规模N，储备池的神经元个数
        :param rho: spectral radius   内部链接权谱半径，W的绝对值最大的的特征值
        :param cr: connectivity ratio。 稀松程度
        :param leaking_rate: leaking rate   不知道
        :param W: predefined ESN reservoir   储备池的权值矩阵
        """
        self.resSize = resSize
        self.leaking_rate = leaking_rate

        if W is None:
            # generate the ESN reservoir
            N = resSize * resSize
            W = np.random.rand(N) - 0.5   #随机生成W矩阵 -0.5～0.5
            zero_index = np.random.permutation(N)[int(N * cr * 1.0):] #新生成一个矩阵，重新排列W大小的矩阵，但是只有未连接的那些节点的个数
            W[zero_index] = 0 #将未连接的节点设为零，代表稀疏连接
            W = W.reshape((self.resSize, self.resSize))  #N*N的矩阵
            # Option 1 - direct scaling (quick&dirty, reservoir-specific):
            #self.W *= 0.135 
            # Option 2 - normalizing and setting spectral radius (correct, slow):
            print 'ESN init: Setting spectral radius...',
            rhoW = max(abs(linalg.eig(W)[0]))   #linalg.eig返回两个值，只要第一个，特征值
            print 'done.'
            W *= rho / rhoW    #之所以叫回声状态网络，是因为前面时刻输入的信息会通过W回荡在储备池中，就像回声一样。为了避免储备池状态爆炸，W的特征值必须要小于等于1。所以我认为这句是控制特征值小于一，毕竟W是随机的
        else:
            assert W.shape[0] == W.shape[1] == resSize, "reservoir size mismatch"
        self.W = W

    def __init_states__(self, X, initLen, reset_state=True):

        # allocate memory for the collected states matrix
        self.S = np.zeros((len(X) - initLen, 1 + self.inSize + self.resSize))  #初始设置x（0）= 0
        if reset_state:
            self.s = np.zeros(self.resSize)
        s = self.s.copy()  #可以理解为赋值，s是Wback

        # run the reservoir with the data and collect S
        for t, u in enumerate(X):   #t是u在X中的编号，u是X中的元素
            s = (1 - self.leaking_rate) * s + self.leaking_rate * np.tanh(np.dot(self.Win, np.hstack((1, u))) + np.dot(self.W, s))  #np.dot（）是矩阵真正的乘法  hstack 是水平(按列顺序)把数组给堆叠起来，应该是多了一列偏置量
            if t >= initLen:
                self.S[t-initLen] = np.hstack((1, u, s))
        if reset_state:
            self.s = s

    def fit(self, X, y, lmbd=1e-6, initLen=50, init_states=True):
        """
        :param X: 1- or 2-dimensional array-like, shape (t,) or (t, d), where
        :         t - length of time series, d - dimensionality.
        :param y : array-like, shape (t,). Target vector relative to X.            X是数据，y是标签
        :param lmbd: regularization lambda
        :param initLen: Number of samples to wash out the initial random state    数据的长度
        :param init_states: False allows skipping states initialization if
        :                   it was initialized before (with same X).
        :                   Useful in experiments with different targets.
        """
        assert len(X) == len(y), "input lengths mismatch."   #数据和标签数量不一致
        
        self.inSize =  1 if np.ndim(X) == 1 else X.shape[1]   #ndim是数组的维数
        #print np.ndim(X), X.shape[1], self.inSize
        if init_states:
            print("ESN fit_ridge: Initializing states..."),
            self.Win = (np.random.rand(self.resSize, 1 + self.inSize) - 0.5) * 1  #初始化一个随机矩阵，Win
            self.__init_states__(X, initLen)
            print("done.")
        self.ridge = Ridge(alpha=lmbd, fit_intercept=False, solver='svd', tol=1e-6)  #这是岭回归的参数
        self.ridge.fit(self.S, y[initLen:])
        return self
       
    def fit_proba(self, X, y, lmbd=1e-6, initLen=50, init_states=True):
        """
        :param X: 1- or 2-dimensional array-like, shape (t,) or (t, d)
        :param y : array-like, shape (t,). Target vector relative to X.
        :param lmbd: regularization lambda
        :param initLen: Number of samples to wash out the initial random state
        :param init_states: see above
        """
        assert len(X) == len(y), "input lengths mismatch."
        self.inSize =  1 if np.ndim(X) == 1 else X.shape[1]   #设置inSize=1（如果number of dimension=1）否则inSize=X的列数
        if init_states:        
            print("ESN fit_proba: Initializing states..."),
            self.Win = (np.random.rand(self.resSize, 1 + self.inSize) - 0.5) * 1
            self.__init_states__(X, initLen)
            print("done.")
        self.logreg = LogisticRegression(C=1/lmbd, penalty='l2', fit_intercept=False, solver='liblinear')   #逻辑回归，预测用户走向应该是岭回归
        self.logreg.fit(self.S, y[initLen:])
        return self
        
    def predict(self, X, init_states=True):
        """
        :param X: 1- or 2-dimensional array-like, shape (t) or (t, d)
        :param init_states: see above
        """
        if init_states:        
            # assume states initialized with training data and we continue from there.
            self.__init_states__(X, 0, reset_state=False)   #这里是false 默认是true
        y = self.ridge.predict(self.S)
        return y
       
    def predict_proba(self, X, init_states=True):
        """
        :param X: 1- or 2-dimensional array-like, shape (t) or (t, d)
        :param init_states: see above
        """
        if init_states:
            # assume states initialized with training data and we continue from there.
            self.__init_states__(X, 0, reset_state=False)   #这里是false 默认是true
        y = self.logreg.predict_proba(self.S)
        return y[:,1]

def accuracy(test_lab, predicted):
    assert len(test_lab) == len(predicted), "input lengths mismatch."
    test1 = test_lab*0.9
    test2 = test_lab*1.1
    j = 0
    for i in range(len(test_lab)):
        if predicted[i][0] > test1[i][0] and predicted[i][1] > test1[i][1] and predicted[i][0] < test2[i][0] and predicted[i][1] < test2[i][1]:
            j += 1
        else:
            j = j 
    acc = j/i
    return acc

if __name__ == '__main__':
    
    # load the data
    xls = xlrd.open_workbook("/Users/bbklk/Desktop/daydistance.xls")
    xls1 = xlrd.open_workbook("/Users/bbklk/Desktop/dayrand.xls")
    count = len(xls.sheets())
    mse = []
    mse1 = []
    flag = True
    for i in range(count):
        starttime = datetime.datetime.now()
        table = xls.sheets()[i]
        table1 = xls1.sheets()[i]
        Nrow = table.nrows
        data = np.zeros((Nrow,2))
        data1 = np.zeros((Nrow,2))
        colx = table.col_values(1)
        coly = table.col_values(2)
        colx1 = table1.col_values(1)
        coly1 = table1.col_values(2)
        for j in range(Nrow):
            data[j][0] = colx[j]
            data[j][1] = coly[j]
            data1[j][0] = colx1[j]
            data1[j][1] = coly1[j]
        #data = MinMaxScaler(feature_range=(-0.3, 0.3)).fit_transform(data.reshape(1, -1))   #MinMaxScaler将属性缩放到一个区间，以维持稀疏矩阵中为0的条目  fit_transform()的作用就是先拟合数据，然后转化它将其转化为标准形式（方差为1，均值为0） reshape将data重排为1行好多列的数组
        #std计算全局标准差  randn是从标准正态分布中返回值（因为上一行的操作，所以是一行n列的）
        #print(len(data))
        trainLen = int(0.8*len(data))
        testLen = len(data) - trainLen-1
        
        dif = 1  # prediction horizon >=1  预测46个单位时间后的值
        #data = data.reshape(-1, 1)
        X = data[0:trainLen]
        y = data[dif:trainLen+dif]
        #print X[1][0], y[1][0]
        #y_p = map(lambda x: 1 if x else 0, data[dif:trainLen+dif][1] > data[0:trainLen][0])   #以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表，即如果>成立，则y_p返回1，否则返回0
        Xtest = data[trainLen-1:len(data)]
        ytest = data[trainLen+dif-1:len(data)+dif]
        #print Xtest, ytest
        #ytest_p = map(lambda x: 1 if x else 0, data[trainLen+dif:trainLen+testLen+dif] > data[trainLen:trainLen+testLen])  #同上
    
        resSize = 100
        rho = 0.9  # spectral radius
        cr = 0.05 # connectivity ratio
        leaking_rate = 0.2 # leaking rate
        lmbd = 1 # regularization coefficient
        initLen = 1
        esn = ESN(resSize=resSize, rho=rho, cr=cr, leaking_rate=leaking_rate)
    
        esn.fit(X, y, initLen=initLen, lmbd=lmbd, init_states=True)
        #esn.fit_proba(X, y_p, initLen=initLen, lmbd=lmbd, init_states=False)
        y_predicted = esn.predict(Xtest, init_states=False)   #岭回归
        #y_predicted_p = esn.predict_proba(Xtest, init_states=False)   #逻辑回归
    
        #compute metrics
        errorLen = testLen
        #print errorLen, len(ytest[0:errorLen]), len(y_predicted[0:errorLen])
        mse.append(mean_squared_error(ytest[0:errorLen], y_predicted[0:errorLen]))   #求均方误差 参数是真实值，预测值
        mse1.append(mean_squared_error(data[:], data1[:]))
        #acc.append(accuracy(ytest[0:errorLen], y_predicted[0:errorLen]))  #计算正确率，真实值和预测值
        #auc = roc_auc_score(ytest_p[0:errorLen], y_predicted_p[0:errorLen])   #ROC曲线指接收器操作特性(receiver operating characteristic)曲线, 反映灵敏性和特效性连续变量,是用构图法揭示敏感性和特异性的相互关系，它通过将连续变量设定出多个不同的临界值，从而计算出一系列敏感性和特异性
                                                                          #AUC（Area Under Curve）被定义为ROC曲线下的面积，也可以认为是ROC曲线下面积占单位面积的比例，显然这个面积的数值不会大于1。又由于ROC曲线一般都处于y=x这条直线的上方，所以AUC的取值范围在0.5和1之间。
        #fpr, tpr, _ = roc_curve(ytest_p[0:errorLen], y_predicted_p[0:errorLen])   #求ROC曲线，真实的label（0，1）和目标分数，返回false positive rate和true positive rate
        #y_predicted_lab = np.zeros(len(y_predicted_p))
        #y_predicted_lab[ y_predicted_p >= 0.5] = 1
    
        print("Ridge regression MSE = {}     {}".format(mse[i],mse1[i]))
        for a in range(count):
            table = xls.sheets()[i]
            Nrow = table.nrows
            data = np.zeros((Nrow,2))
            colx = table.col_values(1)
            coly = table.col_values(2)
            for j in range(Nrow):
                data[j][0] = colx[j]
                data[j][1] = coly[j]
        data_predicted = esn.predict(data, init_states=False)
        t=[]
        groundx=[]
        groundy=[]
        predictx=[]
        predicty=[]
        if (len(data)>17) and (len(data_predicted)>17):
            for i in range(0,16):
                t.append(i*200)
                groundx.append(data[i+1][0])
                groundy.append(data[i+1][1])
                predictx.append(data_predicted[i][0])
                predicty.append(data_predicted[i][1])
        print "groundx", groundx
        print "groundy", groundy
        print "predictx", predictx
        print "predicty", predicty
        endtime = datetime.datetime.now()
        print "time",(endtime - starttime)
        plt.xlabel("Time (s)", fontsize=30)
        plt.ylabel("Distance (m)", fontsize=30)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.plot(t,groundx,'r,-',linewidth=4,label='Real Position X')
        plt.plot(t,groundy,'r,-.',linewidth=4, label='Real Position Y')
        plt.plot(t,predictx,'g,:',linewidth=3,label='Predicted Position X')
        plt.plot(t,predicty,'g,-.',linewidth=3, label='Predicted Position Y')
        plt.legend(bbox_to_anchor=(1.0, 1), loc=1,borderaxespad=0., fontsize=20)
        plt.show()
        
        #fig = plt.figure()
        #ax1 = fig.add_subplot(111)
        #plt.xlabel("Latitude", fontsize=25)
        #ax1.get_xaxis().get_major_formatter().set_useOffset(False)
        ######plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        #plt.ylabel("Longtitude", fontsize=25)
        #plt.xticks(fontsize=25)
        #plt.yticks(fontsize=25)
        #plt.grid(True)
        #ax1.plot(data[:,0],data[:,1],'b,-',linewidth=4,label='Ground_Truth')
        #ax1.legend(loc=1)
        #ax2 = ax1.twinx()
        #plt.yticks([], fontsize=25)
        #ax2.plot(data_predicted[:,0],data_predicted[:,1],'r,-',linewidth=4, label='Prediction')
        #ax2.legend(loc=3)
        #plt.title("The Comparation between Prediction and Reality", fontsize=25)
        #plt.show()

        print i

    avg = sum(mse)/len(mse)
    avg1 = sum(mse1)/len(mse1)
    print avg
    print avg1


    #plt.figure(1).clear()
    #x = np.linspace(0, len(mse), len(mse))
    #for i in range(count):
    #    mse[i] = mse[i]*1000000

    #plt.plot(x, mse, 'b', linewidth=4)
    #plt.xticks(fontsize=25)
    #plt.yticks(fontsize=25)
    ###plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.8f'))
    #plt.xlabel("Batch number", fontsize=25)
    ###plt.ylim(0,1e-7)
    #plt.ylabel('MSE(1e-6)', fontsize=25)
    #plt.show()
