import numpy as np
from scipy.spatial.transform import Rotation
from geom_utils import *
from airsimdroneracingvae.types import Pose, Vector3r, Quaternionr


class Trajectory:
    def __init__(self, gateNum, r, clockwise=True):
        self.gateNum = gateNum
        self.r = r
        self.clockwise = clockwise
        self.track = []
        self.circleAngle = 360.0
        self.startAngle = 90.0
        self.droneStart = 0.0
        self.droneInit = None
        self.gateHeight = 2
        self.yawTrack = None
        self.possibleGateHeights = [-1., -0.5, 0.0, 0.5, 1.]
        self.upBound = 1.
        self.lowBound = - self.upBound

    def initDronePose(self):
        dronePoseRadius = self.r * 21/20.0
        droneDiffAngle = self.circleAngle / (self.gateNum*2)
        if self.clockwise:
            droneSpawnAngle = self.startAngle + droneDiffAngle
        else:
            droneSpawnAngle = self.startAngle - droneDiffAngle
        vectorOfDrone = twoDPolarTranslation(dronePoseRadius, droneSpawnAngle, -self.gateHeight)
        droneQuat = Rotation.from_euler('ZYX', [0., 0., 0.], degrees=True).as_quat()
        droneQuaternion = Quaternionr(droneQuat[0], droneQuat[1], droneQuat[2], droneQuat[3])
        self.droneInit = Pose(vectorOfDrone, droneQuaternion)

    def buildSimpleTrajectory(self):
        angleBetween = self.circleAngle / self.gateNum
        yawTrack = []
        for i in range(self.gateNum):
            if self.clockwise:
                currAngle = self.startAngle - angleBetween*i
            else:
                currAngle = self.startAngle + angleBetween*i
            yawTrack.append(currAngle*np.pi/180)
            vectorOfGate = twoDPolarTranslation(self.r, currAngle, -self.gateHeight)
            rotationQuat = Rotation.from_euler('ZYX', [currAngle, 0., 0.], degrees=True).as_quat()
            mainQuaternion = Quaternionr(rotationQuat[0], rotationQuat[1], rotationQuat[2], rotationQuat[3])
            self.track.append(Pose(vectorOfGate, mainQuaternion))
        self.yawTrack = np.array(yawTrack)

    def buildHeightDiffTrajectory(self):
        angleBetween = self.circleAngle / self.gateNum
        yawTrack = []
        for i in range(self.gateNum):
            if self.clockwise:
                currAngle = self.startAngle - angleBetween * i
            else:
                currAngle = self.startAngle + angleBetween * i
            yawTrack.append(currAngle*np.pi/180)
            randInt = np.random.randint(len(self.possibleGateHeights))
            vectorOfGate = twoDPolarTranslation(self.r, currAngle, -(self.gateHeight + self.possibleGateHeights[randInt]))
            rotationQuat = Rotation.from_euler('ZYX', [currAngle, 0., 0.], degrees=True).as_quat()
            mainQuaternion = Quaternionr(rotationQuat[0], rotationQuat[1], rotationQuat[2], rotationQuat[3])
            self.track.append(Pose(vectorOfGate, mainQuaternion))
        self.yawTrack = np.array(yawTrack)

    def randomPosedTrajectory(self):
        angleBetween = self.circleAngle / self.gateNum
        yawTrack = []
        for i in range(self.gateNum):
            if self.clockwise:
                currAngle = self.startAngle - angleBetween * i
            else:
                currAngle = self.startAngle + angleBetween * i
            yawTrack.append(currAngle * np.pi / 180)
            randomZPick = np.random.random() * (self.upBound - self.lowBound) + self.lowBound
            vectorOfGate = twoDPolarTranslation(self.r, currAngle,
                                                -(self.gateHeight + randomZPick))
            randomXPick = np.random.random() * (self.upBound - self.lowBound) + self.lowBound
            vectorOfGate.x_val += randomXPick * np.cos(np.radians(currAngle))
            vectorOfGate.y_val += randomXPick * np.sin(np.radians(currAngle))
            rotationQuat = Rotation.from_euler('ZYX', [currAngle, 0., 0.], degrees=True).as_quat()
            mainQuaternion = Quaternionr(rotationQuat[0], rotationQuat[1], rotationQuat[2], rotationQuat[3])
            self.track.append(Pose(vectorOfGate, mainQuaternion))
        self.yawTrack = np.array(yawTrack)

    def pickRandomGatePose(self, indexOfGate):
        if self.clockwise:
            angleOfGate = self.startAngle - (self.circleAngle / self.gateNum) * indexOfGate
        else:
            angleOfGate = self.startAngle + (self.circleAngle / self.gateNum) * indexOfGate
        randomXPick = np.random.random() * (self.upBound - self.lowBound) + self.lowBound
        randomZPick = np.random.random() * (self.upBound - self.lowBound) + self.lowBound
        rotationQuat = Rotation.from_euler('ZYX', [angleOfGate, 0., 0.], degrees=True).as_quat()
        qtG = Quaternionr(rotationQuat[0], rotationQuat[1], rotationQuat[2], rotationQuat[3])
        vectorOfGate = twoDPolarTranslation(self.r, angleOfGate,
                                            -(self.gateHeight + randomZPick))
        vectorOfGate.x_val += randomXPick * np.cos(np.radians(angleOfGate))
        vectorOfGate.y_val += randomXPick * np.sin(np.radians(angleOfGate))
        return Pose(vectorOfGate, qtG)


if __name__ == '__main__':
    traj = Trajectory(24, 20)
    traj.buildHeightDiffTrajectory()
    for t in traj.track:
        print(t, '\n')

