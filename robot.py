from iiwaPy import sunrisePy
import time
import numpy as np

ip = '172.31.1.148'
iiwa = sunrisePy(ip)
iiwa.setBlueOn()
time.sleep(2)
iiwa.setBlueOff()

from math import pi

relVel = 0.1
vel = 10

calib_pos = [4.29982580e+02, -7.16138496e-02,  400,  pi, -0.6, pi]
iiwa.movePTPLineEEF(calib_pos, vel)
calib_pos[0] += 50
iiwa.movePTPLineEEF(calib_pos, vel)

iiwa.close()
