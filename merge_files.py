import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../SKF')
import load_save

dir = "results/"
LS_123, alpha_list_123, L_list_123 = load_save.my_load(dir + "LS_low_rank_regression_(139, 144)")
LS_123 = [list(LS_123[:, nn]) for nn in range(3)]
LS_134, alpha_list_134, L_list_134 = load_save.my_load(dir + "LS_low_rank_regression_(139, 144)_19.01.2022_23h45")

LS_all = LS_134.copy()
LS_all.insert(1, LS_123[1])
alpha_list_all = alpha_list_134.copy()
alpha_list_all.insert(1, alpha_list_123) # alpha for L=2

LS_all[0] = LS_123[0] + LS_all[0][1:]  # merge L=1
alpha_list_all[0] = alpha_list_123 + alpha_list_all[0][1:] # merge L=1
LS_all[2] = LS_all[2] + LS_123[2][1:]    # merge L=3
alpha_list_all[2] = alpha_list_all[2] + alpha_list_123[1:]    # merge L=3

LS_45, alpha_list_45, L_list_45 = load_save.my_load(dir+"LS_low_rank_regression_(139, 144)_20.01.2022_17h10")

alpha_list_all[3] = alpha_list_all[3][:4] + alpha_list_45[0] + alpha_list_all[3][4:]    # merge L=4
LS_all[3] = LS_all[3][:4] + LS_45[0] + LS_all[3][4:]    # merge L=4
LS_all.append(LS_45[1])   # merge L=5
alpha_list_all.append(alpha_list_45[1])   # merge L=5

LS_6, alpha_list_6, L_list_6 = load_save.my_load(dir+"LS_low_rank_regression_(139, 144)_22.01.2022_14h57")
LS_all.append(LS_6[0])      # merge L=6
alpha_list_all.append(alpha_list_6[0])   # merge L=6

L_list_all = [1, 2, 3, 4, 5, 6]

MuMv = (139, 144)
for nn in range(len(L_list_all)):
    plt.semilogx(alpha_list_all[nn], LS_all[nn])
    save_file = "LS_low_rank_regression_" + str(MuMv) + "_" + "L=" + str(L_list_all[nn])
    load_save.my_save(dir + save_file, LS_all[nn], alpha_list_all[nn], L_list_all[nn])

LS_conv, alpha_conv = load_save.my_load(dir+"LS_conv_regression")
plt.semilogx(alpha_conv, LS_conv)

p1 = 180.0/295
ls_0 = -p1*np.log(p1)-(1-p1)*np.log(1-p1)
plt.semilogx([10*3, 10**6], [ls_0, ls_0], linestyle="--", color="black")

plt.grid()
plt.axis([10**3, 10**6, 0.66, 0.68 ])
a = 1
None