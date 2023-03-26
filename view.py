import numpy as np
import os

trmm = []
count = 0
trmm_path = 'G:/new_try/gpm_data'
trmm_list = os.listdir(trmm_path)
for i in range(len(trmm_list)):
    trmm.append(
        os.path.join(trmm_path + '/', trmm_list[i])
    )
for i in range(len(trmm_list)):
    pre = np.load(trmm[i])
    pre_m = np.max(pre)
    print(pre_m)
    if pre_m > 10:
        count = count + 1
print(count)
