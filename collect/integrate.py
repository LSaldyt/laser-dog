from ahrs.filters import Madgwick
from pyquaternion import Quaternion
import numpy as np

gyro_data = np.array([[0, 0, 0]])
acc_data  = np.array([[1, 0, 0]])
madgwick = Madgwick(gyr=gyro_data, acc=acc_data)
qw = Quaternion(madgwick.Q.T)

print(qw)
diff = qw * qw.conjugate
theta = 2 * np.arccos(diff[0])
print(theta)
