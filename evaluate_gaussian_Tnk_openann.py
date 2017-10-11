
K = [2] #,3,4,5] #,2,3,4,5] #,4] #,3] #,4,5,6,7,8] #,9,10]
H = [1,2,3,4,5,6,7,8] #,32,128,512]
max_l = 10
max_d = 1
max_r = 1

import itertools
import numpy
import random
from sklearn.neural_network import MLPClassifier
from openann import *

results = []
print("n","k","h","successful classifications", "rate")
for k in K:
  numpy.random.seed(0)
  # print data
  for h in H:
    numpy.random.seed(0)
    N = (h*(k+1)+h+1)*3
    for n in range(N):
      n += 1
      data_results = []
      l_len = min(n-1,max_l-1)
      for r_data in range(max_d):
        numpy.random.seed(r_data)
        data = numpy.random.normal(size=[N,k])
        numpy.random.seed(0)
        true_results = 0
        for label_int in range(2**l_len):
          index = label_int
          if max_l < n:
            label_int = random.randint(0, 2**(n-1))
          labels = [int(i) for i in bin(label_int * 2 + 2**(N+2))[-n:]]
          d = data[:n]
          converged = False
          for r_mlp in range(max_r):
            dataset = DataSet(d, numpy.array([labels]))
            
            # Create network
            net = Net()
            net.input_layer(k)
            net.fully_connected_layer(h, Activation.RECTIFIER)
            net.output_layer(1, Activation.RECTIFIER)
            print("net created")
            # Train network
            stop_dict = {"minimal_value_differences" : 1e-10}
            lbfgs = LBFGS(stop_dict)
            lbfgs.optimize(net, dataset)
            
            # Use network
            converged = True
            for i in range(n):
                y = net.predict(d[i])
                if (y<0.5 and labels[i]>=0.5) or (y>=0.5 and labels[i]<0.5):
                  converged = False
                  break
            if converged:
              true_results += 1
              break
        data_results.append(true_results)
      true_results = max(data_results)
      print(n, k, h, true_results, true_results*1.0/2**l_len)
      results.append(true_results*1.0/2**l_len)
      if true_results == 0:
        break
    print()
    print(results)
    print()
    results = []