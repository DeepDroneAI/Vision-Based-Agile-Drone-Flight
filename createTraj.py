import numpy as np
from scipy.spatial.transform import Rotation
from geom_utils import *
from airsimdroneracingvae.types import Pose, Vector3r, Quaternionr


class Trajectory:
    def __init__(self, gateNum, r):
        self.gateNum = gateNum
        self.r = r
        self.track = []
        self.circleAngle = 360.0
        self.startAngle = 90.0
        self.gateHeight = 2
        self.yawTrack = None

    def buildSimpleTrajectory(self):
        angleBetween = self.circleAngle / self.gateNum
        yawTrack = []
        for i in range(self.gateNum):
            currAngle = self.startAngle - angleBetween*i
            yawTrack.append(currAngle*np.pi/180)
            vectorOfGate = twoDPolarTranslation(self.r, currAngle, -self.gateHeight)
            rotationQuat = Rotation.from_euler('ZYX', [currAngle, 0., 0.], degrees=True).as_quat()
            mainQuaternion = Quaternionr(rotationQuat[0], rotationQuat[1], rotationQuat[2], rotationQuat[3])
            self.track.append(Pose(vectorOfGate, mainQuaternion))
        self.yawTrack = np.array(yawTrack)


if __name__ == '__main__':
    traj = Trajectory(24, 20)
    traj.buildSimpleTrajectory()
    for t in traj.yawTrack:
        print(t)

