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

def get_theta(gyr, acc):
    theta = []
    for n in range(1, len(gyr) + 1):
        try:
            nl = null()
            q = get_qw(gyr[:n], acc[:n])
            qw, qx, qy, qz = q.elements
            yaw = atan2(2.0*(qy*qz + qw*qx), qw*qw - qx*qx - qy*qy + qz*qz)
            theta.append(degrees(yaw))
        except ValueError:
            pass
    return theta

def time_to_index(t_event, times):
    for i, ti in enumerate(times):
        if ti > t_event:
            return i
    return -1

def parse_time(filename):
    datestr, i, ext = filename.replace('capture_', '').split('.')
    return datetime.strptime(datestr, '%Y-%M-%d_%H-%M-%S'), int(i)

def main():
    THETA_INIT = 77

    gyr, acc, t, d, l = from_file('pointer_data.csv')
    theta = get_theta(gyr, acc)

    first_i = l.index(0)
    t_laser = t[first_i]
    diff    = t_laser - t[0]
    t_epoch = t_laser - DELTA * FIRST
    t_start = t_epoch - t[0]

    images = [im, parse_time(im) for im in os.listdir('images')]

    print(f'Calibrating sample data based on first laser image being image {FIRST}')
    print(f'This accounts for {diff} seconds of collection pre-laser')
    print(f'Or  roughly {diff*10}/{len(images)} image samples')
    print(f'And roughly {first_i}/{len(t)} labeller telemetry points')
    print(f'In terms of labeller telemetry, this starts at {t_start}')

    with open('labels.csv', 'w') as outfile:
        outfile.write('filename, time, theta, distance, laser\n')
        for i, image in enumerate(sorted(images, key=lambda t : t[1])):
            im_time = t_epoch + DELTA * i
            im_index = time_to_index(im_time, t)

            im_theta = theta[im_index] - THETA_INIT
            im_dist  = d[im_index]
            im_laser = 1 - l[im_index]

            outfile.write(f'{image:<20}, {im_time:.2f}, {im_theta:.2f}, {im_dist:.2f}, {im_laser:<1}\n')

if __name__ == '__main__':
    main()
