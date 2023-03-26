import numpy as np
import scipy.ndimage
import os


x = np.arange(100).reshape(10,10)

print('Original array:')
print (x)

print ('Resampled by a factor of 2 with nearest interpolation:')
print (scipy.ndimage.zoom(x, 2, order=0))


print ('Resampled by a factor of 2 with bilinear interpolation:')
print (scipy.ndimage.zoom(x, 2, order=1))


print ('Resampled by a factor of 2 with cubic interpolation:')
print (scipy.ndimage.zoom(x, 2, order=3))

print ('Downsampled by a factor of 0.5 with default interpolation:')
print(scipy.ndimage.zoom(x, 0.4))

gpm_path = 'G:/new_try/gpm_data'
save_era_path = 'G:/downscale/gpm_0.25_5'
era_u = os.listdir(gpm_path)

era_and_gpm = []

for i in range(len(era_u)):
    era_and_gpm.append(
        (os.path.join(gpm_path + '/', era_u[i]))
    )

for i in range(len(era_u)):
    pre = np.load(era_and_gpm[i])
    lr = scipy.ndimage.zoom(pre, 0.4, order=5)
    np.save(os.path.join(save_era_path + '/', era_u[i]), lr)
    print(lr.shape)