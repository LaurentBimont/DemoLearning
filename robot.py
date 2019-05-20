from iiwaPy.sunrisePy import sunrisePy
import time
import numpy as np
from gripper import RobotiqGripper
from math import pi

class Robot:
    def __init__(self):
        ip = '172.31.1.148'
        self.iiwa = sunrisePy(ip)
        self.grip = RobotiqGripper("/dev/ttyUSB0")
        self.iiwa.setBlueOn()
        self.grip.reset()
        self.grip.activate()
        time.sleep(2)
        self.iiwa.setBlueOff()
        self.relVel = 0.1
        self.vel = 10

        # Define Flange
        self.iiwa.attachToolToFlange([-1.5, 1.54, 152.8, 0, 0, 0])

    def getCart(self):
        return self.iiwa.getEEFCartesianPosition()

    def checkTCPJoint(self):
        cartPos = self.iiwa.getEEFCartesianPosition()
        vel = 1
        self.iiwa.movePTPLineEEF(cartPos, vel)
        cartPos[5] += np.pi/8
        self.iiwa.movePTPLineEEF(cartPos, vel)
        cartPos[5] -= np.pi/8
        cartPos[4] += np.pi/8
        self.iiwa.movePTPLineEEF(cartPos, vel)
        cartPos[4] -= np.pi/8
        cartPos[3] += np.pi/8
        self.iiwa.movePTPLineEEF(cartPos, vel)
        cartPos[3] -= np.pi/8
        self.iiwa.movePTPLineEEF(cartPos, vel)

    def grasp(self, pos, ang, speed=40):
        grasp_above = [pos[0], pos[1], pos[2]+100, ang, 0, -np.pi]
        self.iiwa.movePTPLineEEF(grasp_above, speed)
        grasp = [pos[0], pos[1], pos[2], ang, 0, -np.pi]
        self.iiwa.movePTPLineEEF(grasp, speed)
        self.grip.closeGripper()

        print(self.grip.isObjectDetected())
        self.grip.openGripper()

if __name__=="__main__":
    try:
        rob = Robot()
        pos_grasp = [4.96389713e+02, 1.14319867e+02, 3.75044594e+02]
        ang_grasp = -np.pi
        rob.grasp(pos_grasp, ang_grasp)
    except:
        pass
    rob.iiwa.close()
