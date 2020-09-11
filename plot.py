import torch
import numpy as np
import matplotlib.pyplot as plt

Jacobian = torch.load('Jacobian_no_rot.pt')
Jacobian = Jacobian.view(10000, 3*32*32, 10)


num = 10000
single_value = torch.zeros(num, 10)
for i in range(num):
    u, s, v = torch.svd(Jacobian[i, :, :])
    single_value[i, :] = s/torch.max(s)
  
single_value_mean = torch.mean(single_value, axis=0)
# plt.plot(single_value_mean)
# plt.show()
print(single_value_mean)



'''
Jacobian_no_rot = torch.load('Jacobian_no_rot.pt')
Jacobian_no_rot = Jacobian_no_rot.view(10000, 3*32*32, 10)
single_value_no_rot = torch.zeros(num, 10)
for i in range(num):
    u, s, v = torch.svd(Jacobian_no_rot[i, :, :])
    single_value_no_rot[i, :] = s/torch.max(s)
    
single_value_mean_no_rot = torch.mean(single_value_no_rot, axis=0)



plt.plot(single_value_mean_rot)
'''
