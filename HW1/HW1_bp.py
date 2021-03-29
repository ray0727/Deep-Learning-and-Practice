#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

class backpropagation(object):
    def __init__(self):
        #parameters
        self.inputsize = 2
        self.hid1size = 10
        self.hid2size = 10 
        self.outputsize = 1
        self.x_train = []
        self.y_train = []
        self.x_train_1 = []
        self.y_train_1 = []

        #weights
        self.w1 = np.random.randn(self.inputsize, self.hid1size)
        self.w2 = np.random.randn(self.hid1size, self.hid2size)
        self.w3 = np.random.randn(self.hid2size, self.outputsize)

        self.loss_list = []
        #sigmoid
        self.epoch_linear = 12000
        self.epoch_xor = 12000
        self.lr_linear = 0.1
        self.lr_xor = 0.1

        #relu
        self.epoch_linear_rl = 30000
        self.epoch_xor_rl = 20000
        self.lr_linear_rl = 0.1
        self.lr_xor_rl = 0.1
         

    def generate_linear(self, n=100):
        pts = np.random.uniform(0, 1 , (n,2))
        inputs = []
        labels = []
        for pt in pts:
            inputs.append([pt[0], pt[1]])
            distance = (pt[0]-pt[1])/1.414
            if pt[0] > pt[1]:
                labels.append(0)
            else:
                labels.append(1)
        return np.array(inputs), np.array(labels).reshape(n, 1)

    def generate_XOR_easy(self):
        inputs = []
        labels = []
        for i in range(11):
            inputs.append([0.1*i, 0.1*i])
            labels.append(0)
            if 0.1*i == 0.5:
                continue
            inputs.append([0.1*i, 1-0.1*i])
            labels.append(1)
        return np.array(inputs), np.array(labels).reshape(21, 1)

    def generate_dataset(self):
        self.x_train, self.y_train = self.generate_linear(n=100)
        self.x_train_1, self.y_train_1 = self.generate_XOR_easy()

    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def derivative_sigmoid(self,x):
        return np.multiply(x, 1-x)

    def relu(self, x):
        return np.maximum(0, x)

    def derivative_relu(self, x):
        z = np.copy(x)
        z[z>0]=1
        z[z<=0]=0
        return z

    def mseloss(self, y_hat, y):
        #MSE
        return np.mean((y_hat - y)**2)

    def forward(self, x, act):
        
        if act == "sigmoid":
            self.x1 = np.dot(x, self.w1)
            self.a1 = self.sigmoid(self.x1)
            self.x2 = np.dot(self.a1, self.w2)
            self.a2 = self.sigmoid(self.x2)
            self.x3 = np.dot(self.a2, self.w3)
            self.y_pred = self.sigmoid(self.x3)
            return self.y_pred
        if act == "relu":
            self.x1 = np.dot(x, self.w1)
            self.a1 = self.relu(self.x1)
            self.x2 = np.dot(self.a1, self.w2)
            self.a2 = self.relu(self.x2)
            self.x3 = np.dot(self.a2, self.w3)
            self.y_pred = self.relu(self.x3)
            return self.y_pred
        # forward without activation
        # self.x1 = np.dot(x, self.w1)
        # self.x2 = np.dot(self.x1, self.w2)
        # self.x3 = np.dot(self.x2, self.w3)
        # self.y_pred = self.x3
        # return self.y_pred

    def train(self, x, y, epoch, act, lr):
        # x:training data y:ground truth
        for i in range(epoch):
            output = self.forward(x, act)
            loss = self.mseloss(output, y)
            self.loss_list.append(loss)
            #back propagation
            if act == "sigmoid":
                loss_grad = (output - y)
                w3_way = loss_grad *self.derivative_sigmoid(output) #(m,1) * (m,1)
                w3_grad = self.a2.T.dot(w3_way)
                w2_way = w3_way.dot(self.w3.T)*self.derivative_sigmoid(self.a2) #(m,1)(1,4), (m,4)
                w2_grad = self.a1.T.dot(w2_way) #(4,m)(m,4)
                w1_way = w2_way.dot(self.w2.T)*self.derivative_sigmoid(self.a1) #(m,4),(4,4)  *(m,4)
                w1_grad = x.T.dot(w1_way)      #(2,m),(m,4)

            if act == "relu":
                loss_grad = (output - y)
                w3_way = loss_grad *self.derivative_relu(output) #(m,1) * (m,1)
                w3_grad = self.a2.T.dot(w3_way)
                w2_way = w3_way.dot(self.w3.T)*self.derivative_relu(self.a2) #(m,1)(1,4), (m,4)
                w2_grad = self.a1.T.dot(w2_way) #(4,m)(m,4)
                w1_way = w2_way.dot(self.w2.T)*self.derivative_relu(self.a1) #(m,4),(4,4)  *(m,4)
                w1_grad = x.T.dot(w1_way)      #(2,m),(m,4)

            #back propagation without activation
            # loss_grad = (output - y)
            # w3_way = loss_grad  #(m,1) * (m,1)
            # w3_grad = self.x2.T.dot(w3_way)
            # w2_way = w3_way.dot(self.w3.T) #(m,1)(1,4), (m,4)
            # w2_grad = self.x1.T.dot(w2_way) #(4,m)(m,4)
            # w1_way = w2_way.dot(self.w2.T) #(m,4),(4,4)  *(m,4)
            # w1_grad = x.T.dot(w1_way)      #(2,m),(m,4)

            #update weights
            self.w1 = self.w1 - lr*w1_grad
            self.w2 = self.w2 - lr*w2_grad
            self.w3 = self.w3 - lr*w3_grad
            
            if(x.shape[0]==100):
                if (i+1) % 1000 == 0:
                    print("Epoch: {}| linear Loss: {}".format(i+1, loss))

            if(x.shape[0]==21):
                if (i+1) % 1000 == 0:
                    print("Epoch: {}| XOR Loss: {}".format(i+1, loss))

        print("Training End")
        #new weights for next training
        self.w1 = np.random.randn(self.inputsize, self.hid1size)
        self.w2 = np.random.randn(self.hid1size, self.hid2size)
        self.w3 = np.random.randn(self.hid2size, self.outputsize)

        if(x.shape[0]==100):
            print("linear predict:\n", output)
            plt.plot(self.loss_list)
            plt.title("Linear Loss curve with lr={}".format(lr))
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.show()
            self.loss_list = []
        else:
            print("xor predict:\n", output)
            plt.plot(self.loss_list)
            plt.title("XOR Loss curve with lr={}".format(lr))
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.show()
            # plt.savefig("./images/xor_loss_epoch.png")
            # plt.close
            self.loss_list = []

            
    def show_result(self, x, y, pred_y):
        count = 0
        plt.subplot(1,2,1)
        plt.title("Ground truth", fontsize=18)
        for i in range(x.shape[0]):
            if(y[i] == 0):
                plt.plot(x[i][0], x[i][1], "ro")
            else:
                plt.plot(x[i][0], x[i][1], "bo")

        plt.subplot(1,2,2)
        plt.title("Predict result", fontsize=18)
        for i in range(x.shape[0]):
            if pred_y[i] < 0.5:
                plt.plot(x[i][0], x[i][1], "ro")
            else:
                plt.plot(x[i][0], x[i][1], "bo")

        for i in range(x.shape[0]):
            if(pred_y[i]-y[i]<0.1 and pred_y[i]-y[i]>-0.1):
                count+=1
            else:
                print(pred_y[i])

        if(x.shape[0]==100):
            print("save linear")
            print("linear accuracy = ", count, "/", x.shape[0],"=", float(count/x.shape[0]))
            plt.show()
            
        else:
            print("save xor")
            print("xor accuracy = ", count, "/", x.shape[0],"=", float(count/x.shape[0]))
            plt.show()

if __name__ == "__main__":
    bp = backpropagation()
    bp.generate_dataset()
    bp.train(bp.x_train, bp.y_train, bp.epoch_linear, "sigmoid", bp.lr_linear)
    bp.show_result(bp.x_train, bp.y_train, bp.y_pred)
    bp.train(bp.x_train_1, bp.y_train_1, bp.epoch_xor, "sigmoid", bp.lr_xor)
    bp.show_result(bp.x_train_1,bp.y_train_1, bp.y_pred)

    # bp.train(bp.x_train, bp.y_train, bp.epoch_linear_rl, "relu", bp.lr_linear_rl)
    # bp.show_result(bp.x_train, bp.y_train, bp.y_pred)
    # bp.train(bp.x_train_1, bp.y_train_1, bp.epoch_xor_rl, "relu", bp.lr_xor_rl)
    # bp.show_result(bp.x_train_1,bp.y_train_1, bp.y_pred)
