import math  
import random  
import numpy as np  
import matplotlib.pyplot as plt    
random.seed(0)  
  
def rand(a, b):  
    return (b - a) * random.random() + a  
  
def make_matrix(m, n, fill=0.0):  
    mat = []  
    for i in range(m):  
        mat.append([fill] * n)  
    return mat  
  
def sigmoid(x):  
    return 1.0 / (1.0 + math.exp(-x))  
  
def sigmod_derivate(x):  
    return x * (1 - x)  
  
class BPNeuralNetwork:  
    def __init__(self):  
        self.input_n = 0  
        self.hidden_n = 0  
        self.hidden2_n = 0
        self.output_n = 0  
        self.input_cells = []  
        self.hidden_cells = []  
        self.hidden2_cells = []  
        self.output_cells = []  
        self.input_weights = [] 
        self.hidden_weights = []
        self.output_weights = []  
        self.input_correction = []  
        self.hidden_correction = []
        self.output_correction = []  
  
    def setup(self, ni, nh,nh2, no):  
        self.input_n = ni + 1  
        self.hidden_n = nh 
        self.hidden2_n = nh2 
        self.output_n = no  
        # 初始化 cells  
        self.input_cells = [1.0] * self.input_n  
        self.hidden_cells = [1.0] * self.hidden_n 
        self.hidden2_cells = [1.0] * self.hidden2_n 
        self.output_cells = [1.0] * self.output_n  
        # 初始化 weights  
        self.input_weights = make_matrix(self.input_n, self.hidden_n) 
        self.hidden_weights = make_matrix(self.hidden_n, self.hidden2_n) 
        self.output_weights = make_matrix(self.hidden2_n, self.output_n)  
        # 随机数赋值 
        for i in range(self.input_n):  
            for h in range(self.hidden_n):  
                self.input_weights[i][h] = rand(-0.2, 0.2)  
        for h in range(self.hidden_n):  
            for h2 in range(self.hidden2_n):  
                self.hidden_weights[h][h2] = rand(-0.5, 0.5)                 
        for h2 in range(self.hidden2_n):  
            for o in range(self.output_n):  
                self.output_weights[h2][o] = rand(-2.0, 2.0)  
        # 初始化 correction 
        self.input_correction = make_matrix(self.input_n, self.hidden_n) 
        self.hidden_correction = make_matrix(self.hidden_n, self.hidden2_n)
        self.output_correction = make_matrix(self.hidden2_n, self.output_n)  
  
    def predict(self, inputs):  
        # input layer 赋值  
        for i in range(self.input_n - 1):  
            self.input_cells[i] = inputs[i]  
        # hidden layer 赋值 
        for j in range(self.hidden_n):  
            total = 0.0  
            for i in range(self.input_n):  
                total += self.input_cells[i] * self.input_weights[i][j]  
            self.hidden_cells[j] = sigmoid(total)  
        # hidden2 layer 赋值 
        for j2 in range(self.hidden2_n):  
            total = 0.0  
            for j in range(self.hidden_n):  
                total += self.hidden_cells[j] * self.hidden_weights[j][j2]  
            self.hidden2_cells[j2] = sigmoid(total)                  
        # output layer 赋值 
        for k in range(self.output_n):  
            total = 0.0  
            for j2 in range(self.hidden2_n):  
                total += self.hidden2_cells[j2] * self.output_weights[j2][k]  
            self.output_cells[k] = sigmoid(total)  
        return self.output_cells[:]  
  
    def back_propagate(self, case, label, learn, correct):  
        
        self.predict(case)  
        # output layer 误差  
        output_deltas = [0.0] * self.output_n  
        for o in range(self.output_n):  
            error = label - self.output_cells[o]  
            output_deltas[o] = sigmod_derivate(self.output_cells[o]) * error 
        # hidden2 layer 误差  
        hidden2_deltas = [0.0] * self.hidden2_n  
        for h2 in range(self.hidden2_n):  
            error = 0.0  
            for o in range(self.output_n):  
                error += output_deltas[o] * self.output_weights[h2][o]  
            hidden2_deltas[h2] = sigmod_derivate(self.hidden2_cells[h2]) * error        
        # hidden layer 误差   
        hidden_deltas = [0.0] * self.hidden_n  
        for h in range(self.hidden_n):  
            error = 0.0  
            for h2 in range(self.hidden2_n):  
                error += hidden2_deltas[h2] * self.hidden_weights[h][h2]  
            hidden_deltas[h] = sigmod_derivate(self.hidden_cells[h]) * error  
        # 更新 output weights  
        for h2 in range(self.hidden2_n):  
            for o in range(self.output_n):  
                change = output_deltas[o] * self.hidden2_cells[h2]  
                self.output_weights[h2][o] += learn * change + correct * self.output_correction[h2][o]  
                self.output_correction[h2][o] = change  
        # 更新 hidden weights  
        for h in range(self.hidden_n):  
            for h2 in range(self.hidden2_n):  
                change = hidden2_deltas[h2] * self.hidden_cells[h]  
                self.hidden_weights[h][h2] += learn * change + correct * self.hidden_correction[h][h2]  
                self.hidden_correction[h][h2] = change    
        # 更新 input weights  
        for i in range(self.input_n):  
            for h in range(self.hidden_n):  
                change = hidden_deltas[h] * self.input_cells[i]  
                self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]  
                self.input_correction[i][h] = change  
        # 总误差 
        error = 0.0  
        error += 0.5 * (label - self.output_cells[o]) ** 2  
        return error  
  
    def train(self, cases, labels, limit, learn, correct):  
        error_results = []  
        nums = []  
        for j in range(limit):  
            error = 0.0  
            for i in range(len(cases)):  
                label = labels[i]  
                case = cases[i]  
                error += self.back_propagate(case, label, learn, correct)  
            if j%50==0 :  
                print (j , error) 
                error_results.append(error)  
                nums.append(j)  
                
        plt.xlabel("迭代次数",fontproperties='SimHei')    
        plt.ylabel("误差",fontproperties='SimHei')    
        plt.title("双隐层BP神经网络",fontproperties='SimHei') 
        
        plt.xlim(0.0, 4000.0)
        plt.ylim(0.0, 20.)  
          
        plt.plot(nums, error_results)
        plt.show()
  
    def test(self):  
        dataset = np.loadtxt('cancer.csv', delimiter=",")  
        cases = dataset[:,1:10]  
        labels = dataset[:,10]  
        self.setup(8,20,5,2)
        self.train(cases[:400], labels, 4000, 0.05, 0.1)  
        count = [0,0]  
        i=0  
        for case in cases[400:]:  
            
            if abs(self.predict(case)[0]-labels[400+i])<0.1:  
                count[0]=count[0]+1  
            else:   
                count[1]=count[1]+1  
            i=i+1  
        print ('result =',count)  
        print ('error =',float(count[1])/(count[0]+count[1]))  
if __name__ == '__main__':  
    nn = BPNeuralNetwork()  
    nn.test()  