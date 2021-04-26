from ahrs.filters import Madgwick
from pyquaternion import Quaternion
import numpy as np
from pprint import pprint
from math import degrees, atan2
from datetime import datetime
import os

def from_file(fname):
    gyr = []
    acc = []
    t   = []
    d   = []
    l   = []
    with open(fname, 'r') as infile:
        headers = next(infile)
        for line in infile:
            cells = line.split(',')
            if len(cells) != 9:
                print(f'Bad (len {len(cells)}): ', cells)
                continue
            try:
                g = [float(c) for c in cells[2:5]]
                a = [float(c) for c in cells[5:8]]
                gyr.append(g)
                acc.append(a)
                t.append(int(float(cells[0])))
                d.append(float(cells[1]))
                l.append(int(cells[-1]))
            except ValueError as e:
                print(e)
                pass
    return gyr, acc, t, d, l

def get_qw(gyr, acc):
    gyro_data = np.array(gyr)
    acc_data  = np.array(acc)
    madgwick = Madgwick(gyr=gyro_data, acc=acc_data)
    return Quaternion(madgwick.Q.T[:,-1:])

def null(n=1):
    return get_qw([[0,0,0]] * n, [[0,0,0]] * n)

def get_theta(gyr, acc, d):
    reset = 0
    theta = []
    for n in range(1, len(gyr) + 1):
        print(f'Integrating: {n}', end='\r')
        dist = d[n-1]
        if dist < 10.0 and dist > 0.0:
            print(f'Reset at: {n} ' + ' ' * 60)
            reset = n - 1
        try:
            nl = null()
            q = get_qw(gyr[reset:n], acc[reset:n])
            qw, qx, qy, qz = q.elements
            yaw = atan2(2.0*(qy*qz + qw*qx), qw*qw - qx*qx - qy*qy + qz*qz)
            theta.append(degrees(yaw))
        except ValueError:
            pass
    print('Done')
    return theta

def time_to_index(t_event, times):
    for i, ti in enumerate(times):
        if ti > t_event:
            return i
    return -1

def parse_time(filename):
    datestr, i, ext = filename.replace('capture_', '').split('.')
    return datetime.strptime(datestr, '%Y-%m-%d_%H_%M_%S'), int(i)

def main():
    gyr, acc, t, d, l = from_file('raw_data.csv')
    d = [200.0 if di < 0.000001 else di for di in d]
    theta = get_theta(gyr, acc, d)

    first_i = l.index(0)
    t_laser = t[first_i]
    t_im_laser = parse_time('capture_2021-04-26_05_33_44.8.jpg')[0].timestamp()
    time_offset = (t_im_laser - t_laser)
    theta_init = theta[first_i]

    images = [(im, parse_time(im)) for im in os.listdir('images') if im.count('.') == 2]
    with open('labels.csv', 'w') as outfile:
        outfile.write('filename, time, theta, distance, laser\n')
        for image, (date, i) in sorted(images, key=lambda t : t[1]):
            im_time = date.timestamp() - time_offset
            im_index = time_to_index(im_time, t)
            im_dist  = d[im_index]
            if im_dist < 10.0: # Reset
                theta_init = theta[im_index]
                print(f'Reset at {im_index}')
            im_theta = theta[im_index] - theta_init
            im_laser = 1 - l[im_index]
            outfile.write(f'{image:<20}, {im_time:.2f}, {im_theta:.2f}, {im_dist:.2f}, {im_laser:<1}\n')

if __name__ == '__main__':
    main()
