import numpy as np
import os
import sys
from os.path import isfile, join

import airsimdroneracingvae as airsim
# print(os.path.abspath(airsim.__file__))
from airsimdroneracingvae.types import Pose, Vector3r, Quaternionr
from scipy.spatial.transform import Rotation
import time


#Extras for Trajectory and Control
from quadrotor import *
from geom_utils import QuadPose, dist3dp
from traj_planner import Traj_Planner
# changed 
from pose_sampler import * 
from controller import Controller
from quad_model import Model
from mpc_class import MPC_controller
from createTraj import *




class PoseSampler:
    def __init__(self,v_avg=5, with_gate=True):
        self.num_samples = 1
        self.curr_idx = 0
        self.current_gate = 0
        self.with_gate = with_gate
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.simLoadLevel('Soccer_Field_Easy')
        time.sleep(1)        
        #self.client = airsim.MultirotorClient()
        self.configureEnvironment()
        self.total_cost=0
        self.dtau=1e-2

        self.v_avg=v_avg
        self.traj=Traj_Planner()
        self.state=np.zeros(12)

        self.quad=Model()
        self.controller=Controller()
        self.mpc_controller=MPC_controller()
        self.gateNumber = 18
        self.radius = 20

        self.path = Trajectory(self.gateNumber, self.radius)
        self.path.randomPosedTrajectory()
        self.path.initDronePose()
        self.yaw_track = self.path.yawTrack
        self.track = self.path.track
        self.drone_init = self.path.droneInit
        self.saveCnt = 0
        self.loopCnt = 0
        self.distanceToGate = 1
        self.saveChanged = True

        self.track = self.track  # for circle trajectory change this with circle_track
        self.drone_init = self.drone_init  # for circle trajectory change this with drone_init_circle
        self.state = np.array(
            [self.drone_init.position.x_val, self.drone_init.position.y_val, self.drone_init.position.z_val, 0, 0,
             self.yaw_track[0] - np.pi / 2, 0, 0, 0, 0, 0, 0])

        self.transformation = self.transformation = transforms.Compose([
            transforms.Resize([200, 200]),
            # transforms.Lambda(self.gaussian_blur),
            # transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation),
            transforms.ToTensor()])

    def conf_u(self,u):
        for i in range(3):
            u[i]=(u[i]-5)/10

        return u

    def fly_through_gates(self):
        
        self.client.simSetVehiclePose(QuadPose(self.state[[0,1,2,3,4,5]]), True)
        self.quad.reset(x=self.state)

        index = 0
        v_des=0
        gTruths = []
        iNames = []
        maxTurn = self.gateNumber * 10
        vel_statu=False
        while(True):
            if index==maxTurn:
                break
            if vel_statu==False:
                v_des=min(v_des+1,self.v_avg)
            else:
                v_des=min(v_des-1,self.v_avg)
            if v_des==self.v_avg:
                vel_statu=True
            elif v_des==1:
                vel_statu=False
            for i in range(1):

                # Trajectory generate
                waypoint_world = np.array([self.track[index % self.gateNumber].position.x_val,
                                           self.track[index % self.gateNumber].position.y_val,
                                           self.track[index % self.gateNumber].position.z_val])
                posf = waypoint_world
                yawf = self.yaw_track[index % self.gateNumber] - np.pi / 2 - (int(index / self.gateNumber)) * 2 * np.pi

                pos0 = [self.state[0], self.state[1], self.state[2]]
                vel0 = [self.state[3], self.state[4], self.state[5]]
                ang_vel0 = [self.state[9], self.state[10], self.state[11]]
                yaw0 = self.state[8]

                
                

                velf=[v_des*np.cos(yawf),v_des*np.sin(yawf),0,0]

                x_initial=[pos0[0],pos0[1],pos0[2],yaw0]
                x_final=[posf[0],posf[1],posf[2],yawf]
                vel_initial=[vel0[0]*np.cos(yaw0)-vel0[1]*np.sin(yaw0),vel0[1]*np.cos(yaw0)+vel0[0]*np.sin(yaw0),vel0[2],ang_vel0[2]]
                vel_final=velf
                a_initial=[0,0,0,0]
                a_final=[0,0,0,0]

                pose_err=0
                for j in range(3):
                    pose_err+=pow(x_final[j]-pos0[j],2)

                pose_err=np.sqrt(pose_err)
                T=pose_err/(v_des)
                N=int(T/self.dtau)
                t=np.linspace(0,T,N)
                self.traj.find_traj(x_initial=x_initial,x_final=x_final,v_initial=vel_initial,v_final=vel_final,T=T)

                # Creating the path if not exist
                if not os.path.isdir('images/images-0'):
                    os.mkdir(os.getcwd() + '/images/' + 'images-{}'.format(self.loopCnt))
                    gTruths = []

                # Creating the path if not exist
                if not os.path.isdir('images/images-0'):
                    os.mkdir(os.getcwd() + '/images/' + 'images-{}'.format(self.loopCnt))
                gTruths = []
                iNames = []
                t_current=0.0
                for k in range(len(t)):
                    t_current=t[k] 

                    sName = self.getInstantaneousImg(save=True)
                    rho, phi, theta = cartesian_to_spherical(self.state, waypoint_world)
                    yawDif = yawf - self.state[5]
                    # print("wp: ", cartesian_to_spherical(self.state, waypoint_world), 'yaw: ', yawf - self.state[5])
                    gTruths.append([str(rho), str(phi), str(theta), str(yawDif - np.pi / 2)])  # yawDif - np.pi/2
                    iNames.append(sName)
                    # time.sleep(.01)
                    # print("own    position: ", quad_pose[:3])
                    # print("target position: ", waypoint_world)
                    
                    # print("distance: ", self.distanceToGate)


                    target=self.traj.get_target(t_current)
                    vel_target=self.traj.get_vel(t_current)

                    x_t=[vel_target[0]*np.cos(target[3])+vel_target[1]*np.sin(target[3]),vel_target[1]*np.cos(target[3])-vel_target[0]*np.sin(target[3]),vel_target[2],target[3]]
                    
                    u_nn=self.controller.run_controller(x=self.state[3:12],x_t=x_t)
                    self.state=self.quad.run_model(self.conf_u(u_nn))

                    vel_total=np.sqrt(pow(self.state[3],2)+pow(self.state[4],2)+pow(self.state[5],2))
                    print("Total_vel:{}".format(vel_total))



                    quad_pose = [self.state[0], self.state[1], self.state[2], 0, 0, self.state[8]]
                    #vel_target=[vel_target[0], vel_target[1], vel_target[2], 0, 0, vel_target[3]]
                    #self.total_cost+=abs(np.sqrt(pow(quad_pose[0],2)+pow(quad_pose[1],2))-10)
                    #self.state=np.array([target[0],target[1],target[2],0,0,target[3],vel_target[0],vel_target[1],vel_target[2],0,0,vel_target[3]])
                    self.client.simSetVehiclePose(QuadPose(quad_pose), True)
                    #time.sleep(.01)
                    self.distanceToGate = round(dist3dp(quad_pose, waypoint_world), 2)
                    #print("Distance:{}".format(self.distanceToGate))

                    if self.distanceToGate < 1.0 and self.saveChanged :
                        
                        #self.loopCnt = min(maxTurn - 1, self.loopCnt)
                        """print('gate_' + str((index - 1) % self.gateNumber))
                        self.destroyAndSpawnGate('gate_' + str((index - 1) % self.gateNumber))"""
                        self.saveChanged = False
                        # print("PATHFOLDER CHANGED")
                        if not os.path.isdir('images/images-{}'.format(self.loopCnt+1)):
                            os.mkdir(os.getcwd() + '/images/' + 'images-{}'.format(self.loopCnt+1))
                        break

                index += 1
                self.saveChanged = True
            self.writeGroundTruthVals(gTruths, iNames)
            self.loopCnt += 1

    def destroyAndSpawnGate(self, gateName):
        print("current gate Names: ")
        for gate_object in self.client.simListSceneObjects(".*[Gg]ate.*"):
            print(gate_object)
        # destroy the object
        #self.client.simDestroyObject(gateName)
        print("object destroyed: {}".format(gateName))
        # Spawn the gate again with different pose
        gateIndex = int(gateName.split('_')[1])
        poseVector = self.path.pickRandomGatePose(gateIndex)
        #tgt_name = self.client.simSpawnObject(gateName, "RedGate16x16",
                                              #Pose(position_val=Vector3r(0, 0, 15)), 0.75)
        self.client.simSetObjectPose(self.tgt_name[gateIndex], poseVector, True)
        print("object spawned: {} \nx:{}\ny:{}\nz:{}".format(gateName, poseVector.position.x_val,
                                                           poseVector.position.y_val,
                                                           poseVector.position.z_val))
        self.track[gateIndex] = poseVector  # change it, so flight do not confuse

    def writeGroundTruthVals(self, gTruths, imageNames):
        cwd = os.getcwd()
        if not os.path.isfile(cwd + '/images/images-{}/image_labels-{}.txt'.format(self.loopCnt, self.loopCnt)):
            open(cwd + '/images/images-{}/image_labels-{}.txt'.format(self.loopCnt, self.loopCnt), 'w')
        txt1Name = cwd + '/images/images-{}/image_labels-{}.txt'.format(self.loopCnt, self.loopCnt)
        with open(txt1Name, "w") as writtenTxt:
            for line in gTruths:
                writtenTxt.write(" ".join(line) + "\n")

        if not os.path.isfile(cwd + '/images/images-{}/image_names-{}.txt'.format(self.loopCnt, self.loopCnt)):
            open(cwd + '/images/images-{}/image_names-{}.txt'.format(self.loopCnt, self.loopCnt), 'w')
        txt2Name = cwd + '/images/images-{}/image_names-{}.txt'.format(self.loopCnt, self.loopCnt)
        with open(txt2Name, "w") as writtenTxt:
            for line in imageNames:
                writtenTxt.write("".join(line) + "\n")

    def getInstantaneousImg(self, save=False):
        #self.client.simSetVehiclePose(QuadPose(self.state[[0, 1, 2, 3, 4, 5]]), True)
        image_response = self.client.simGetImages([airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)])[0]
        # if len(image_response.image_data_uint8) == image_response.width * image_response.height * 3:
        img1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)  # get numpy array
        img_rgb = img1d.reshape(image_response.height, image_response.width,
                                3)  # reshape array to 4 channel image array H X W X 3
        # img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        # anyGate = self.isThereAnyGate(img_rgb)
        cwd = os.getcwd()
        if save:
            # saveName = cwd + '/images/images-{}'.format(self.loopCnt) + "/frame000" + str(self.saveCnt) + '.png'
            saveName = cwd + '/images/images-{}'.format(self.loopCnt) + "/frame000" + str(self.saveCnt) + '.png'
            cv2.imwrite(os.path.join(saveName), img_rgb)
            # print(os.path.join(cwd + '/images/images-{}'.format(self.loopCnt) + "/frame000" + str(self.saveCnt)) + 'dist->' + str(self.distanceToGate) + '.png')
            self.saveCnt += 1
            # img =  Image.fromarray(img_rgb)
        # image = self.transformation(img)
        # return saveName
        return "/images/images-{}/frame000".format(self.loopCnt) + str(self.saveCnt - 1) + '.png'


    def update(self, mode):
        '''
        convetion of names:
        p_a_b: pose of frame b relative to frame a
        t_a_b: translation vector from a to b
        q_a_b: rotation quaternion from a to b
        o: origin
        b: UAV body frame
        g: gate frame
        '''
        
        #self.client.simSetObjectPose(self.tgt_name, p_o_g_new, True)
        #min_vel, min_acc, min_jerk, pos_waypoint_interp, min_acc_stop, min_jerk_full_stop
        MP_list = ["min_acc", "min_jerk", "min_jerk_full_stop", "min_vel"]
        #MP_list = ["min_vel"]

        if self.with_gate:
            # gate_name = "gate_0"
            # self.tgt_name = self.client.simSpawnObject(gate_name, "RedGate16x16", Pose(position_val=Vector3r(0,0,15)), 0.75)
            # self.client.simSetObjectPose(self.tgt_name, self.track[0], True)
            for i, gate in enumerate(self.track):
                #print ("gate: ", gate)
                gate_name = "gate_" + str(i)
                self.tgt_name = self.client.simSpawnObject(gate_name, "RedGate16x16", Pose(position_val=Vector3r(0,0,15)), 0.75)
                self.client.simSetObjectPose(self.tgt_name, gate, True)
        # request quad img from AirSim
        time.sleep(0.001)

        if mode == "FLY":
            self.fly_through_gates()
        print("trajectory_cost:{}".format(self.total_cost))
        

                    
    def configureEnvironment(self):
        for gate_object in self.client.simListSceneObjects(".*[Gg]ate.*"):
            self.client.simDestroyObject(gate_object)
            time.sleep(0.05)
        if self.with_gate:
            self.tgt_name = self.client.simSpawnObject("gate", "RedGate16x16", Pose(position_val=Vector3r(0,0,15)), 0.75)
        else:
            self.tgt_name = "empty_target"


