import numpy as np
import os
import sys
from os.path import isfile, join

import airsimdroneracingvae as airsim
#import airsim as Airsim
# print(os.path.abspath(airsim.__file__))
from airsimdroneracingvae.types import Pose, Vector3r, Quaternionr
from scipy.spatial.transform import Rotation
import time

#Extras for Trajectory and Control
from quadrotor import *
from geom_utils import QuadPose
from traj_planner import Traj_Planner
from controller import Controller
from quad_model import Model

# Extras for Perception
import torch
from torchvision import models, transforms
from torchvision.transforms.functional import normalize, resize, to_tensor
from PIL import Image
import cv2
from createTraj import *
import Dronet
import lstmf
import threading
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation

class PoseSampler:
    def __init__(self,v_avg=5, with_gate=True):
        self.base_path= os.getcwd()
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
        self.client2 = airsim.MultirotorClient()
        self.client2.confirmConnection()

        self.iteration = 0
        # Dronet
        self.device = torch.device("cpu")
        self.Dronet =  Dronet.ResNet(Dronet.BasicBlock, [1,1,1,1], num_classes = 4)
        self.Dronet.to(self.device)
        #print("Dronet Model:", self.Dronet)
        #self.Dronet.load_state_dict(torch.load(self.base_path+'/weights/2_0.0001_39_loss_0.0246_PG.pth',map_location=torch.device('cpu')))
        self.Dronet.load_state_dict(torch.load(self.base_path+'/weights/Dronet_new.pth',map_location=torch.device('cpu')))    
        self.Dronet.eval()

        # LstmR
        input_size = 4
        output_size = 4
        lstmR_hidden_size = 16
        lstmR_num_layers = 1
        self.lstmR = lstmf.LstmNet(input_size, output_size, lstmR_hidden_size, lstmR_num_layers)
        self.lstmR.to(self.device)
        #print("lstmR Model:", self.lstmR)
        self.lstmR.load_state_dict(torch.load(self.base_path+'/weights/R_2.pth',map_location=torch.device('cpu')))   
        self.lstmR.eval()

        # covNet
        self.device = torch.device("cpu")
        self.covNet =  Dronet.ResNet(Dronet.BasicBlock, [1,1,1,1], num_classes = 1)
        self.covNet.to(self.device)
        #print("Dronet Model:", self.Dronet)
        self.covNet.load_state_dict(torch.load(self.base_path+'/weights/16_0.001_65_loss_0.0120_PG.pth',map_location=torch.device('cpu')))   
        self.covNet.eval() 

        self.loop_ration=5 
        self.vel_des=0

        self.quad=Model()
        self.controller=Controller()

        self.period_denum = 30.

        self.v_avg=v_avg
        self.traj=Traj_Planner()
        self.state=np.zeros(12)
        self.gateNumber=18
        self.radius=20
        
        self.saveCnt = 0
        self.loopCnt = 0
        self.distanceToGate = 1
        self.saveChanged = True        
        
        quat0 = R.from_euler('ZYX',[90.,0.,0.],degrees=True).as_quat()
        quat1 = R.from_euler('ZYX',[80.,0.,0.],degrees=True).as_quat()
        quat2 = R.from_euler('ZYX',[70.,0.,0.],degrees=True).as_quat()
        self.yaw_track=np.array([90])*np.pi/180

        self.track = [Pose(Vector3r(0.,2*10.,-2.) , Quaternionr(quat0[0],quat0[1],quat0[2],quat0[3])),
                      Pose(Vector3r(10.,20.5,-2.) , Quaternionr(quat1[0],quat1[1],quat1[2],quat1[3])),
                      Pose(Vector3r(20.,19.5,-2.) , Quaternionr(quat2[0],quat2[1],quat2[2],quat2[3]))]
            
        quat_drone = R.from_euler('ZYX',[90.,0.,0.],degrees=True).as_quat()
        self.drone_init = Pose(Vector3r(-10.,2*10.,-2), Quaternionr(quat_drone[0],quat_drone[1],quat_drone[2],quat_drone[3]))

        self.state=np.array([self.drone_init.position.x_val,self.drone_init.position.y_val,-self.drone_init.position.z_val,0,0,self.yaw_track[0]-np.pi/2,0,0,0,0,0,0])
                
        self.posf=[self.track[0].position.x_val, self.track[0].position.y_val, self.track[0].position.z_val]  
        self.yawf=self.yaw_track[0]-np.pi/2    
        self.index=0
        self.yaw0 = self.state[8]
    
        #-----------------------------------------------------------------------        

    def check_completion(self, index, quad_pose,waypoint_world, eps=1):
        x, y, z = quad_pose[0], quad_pose[1], quad_pose[2]

        """ xd = self.track[index].position.x_val
        yd = self.track[index].position.y_val
        zd = self.track[index].position.z_val
        psid = Rotation.from_quat([self.track[index].orientation.x_val, self.track[index].orientation.y_val, 
                                   self.track[index].orientation.z_val, self.track[index].orientation.w_val]).as_euler('ZYX',degrees=False)[0] """

        #target = [xd, yd, zd, psid] 
        check_arrival = False

        diff = quad_pose[0] - waypoint_world[0]

        if (-eps <= diff <= eps):
            #self.quad.calculate_cost(target=target, final_calculation=True)
            check_arrival = True

        return check_arrival

    def conf_u(self,u):
        for i in range(3):
            u[i]=(u[i]-5)/10

        return u

    def calculate_covariance(self):

        #print("Gate x,y,z,yaw: ", self.waypoint_world_real[0], self.waypoint_world_real[1], -self.waypoint_world_real[2], self.yaw_track[0]-np.pi/2)
        #print("Dronet x,y,z,yaw: ", self.posf[0], self.posf[1], self.posf[2], self.yawf)

        x_error = (self.posf[0] - self.waypoint_world_real[0])*(self.posf[0] - self.waypoint_world_real[0])
        y_error = (self.posf[1] - self.waypoint_world_real[1])*(self.posf[1] - self.waypoint_world_real[1])
        z_error = (self.posf[2] + self.waypoint_world_real[2])*(self.posf[2] + self.waypoint_world_real[2])
        yaw_error = self.yawf * self.yawf

        #print("x,y,z error: ", x_error, y_error, z_error)
        #print("Pos Error: ", x_error + y_error + z_error)        
        #print("Yaw error: ", yaw_error)
        #print("########")

        return ((x_error + y_error + z_error)/100) + (yaw_error/(2*np.pi))

    def add_noise(self,img):
        
        gauss_noise = max(0.2,np.random.normal(1,0.5))

        self.brightness = (gauss_noise,gauss_noise)
        self.contrast = (1,1)
        self.saturation = (1,1)
        self.hue = (0,0)

        #if probability < 0.8:
        self.transformation = transforms.Compose([
                            transforms.Resize([200, 200]),
                            transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation, hue = self.hue),
                            transforms.ToTensor()])
        
        self.transformation1 = transforms.Compose([
                            transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation, hue = self.hue)])
        
        image_dronet = self.transformation(img)
        image_noisy = cv2.cvtColor(np.array(self.transformation1(img)),cv2.COLOR_BGR2RGB)
        
        return image_dronet,image_noisy
    
    def run_dronet(self,image_response):
        
        img1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)  # get numpy array
        img_rgb = img1d.reshape(image_response.height, image_response.width, 3)  # reshape array to 4 channel image array H X W X 3
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        
        img =  Image.fromarray(img_rgb)
        image_dronet, image_noisy = self.add_noise(img)  # Adds noise to the image with 1/3 probability
        
        pose_gate_body = self.Dronet(image_dronet)
        covariance = float(self.covNet(image_dronet))

        self.posf,self.yawf=self.dronet_to_body(pose_gate_body)

        real_error = self.calculate_covariance()

        self.list_covnet.append(covariance)
        self.list_difference.append(covariance-real_error)
        self.list_realerror.append(real_error)
    
        print("Real Error: ", real_error)
        print("Covnet Error: ",covariance)
        print("Difference: ", covariance-real_error)
        print("##########################")

        self.iteration = self.iteration + 1


    def dronet_to_body(self,pose_gate_body):
        drone_pose=[self.state[0], self.state[1], -self.state[2], 0, 0, self.state[8]]
        pose_gate_body = pose_gate_body.numpy().reshape(-1,1)
        waypoint_world = spherical_to_cartesian(drone_pose, pose_gate_body)
        yaw_diff = pose_gate_body[3][0]
        posf = [waypoint_world[0], waypoint_world[1], -waypoint_world[2]]
        yawf = drone_pose[5]+yaw_diff+np.pi/2

        return posf,yawf

    def update_target(self):

        #distortionParams = self.CameraDistortionParameters()
        #self.client2.simSetDistortionParams('0',distortionParams)
        image_response = self.client2.simGetImages([airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)])[0]
 
        self.run_dronet(image_response)

    def CameraDistortionParameters(self):
        HorzWaveContrib  = np.random.normal(0,0.03)
        HorzWaveStrength = np.random.normal(0,2)
        HorzWaveVertSize = np.random.normal(0,10)

        output = {"HorzWaveContrib": HorzWaveContrib ,"HorzWaveStrength": HorzWaveStrength, "HorzWaveVertSize":HorzWaveVertSize}

        return output

    def update_target_loop(self):
        
        while(True):
            with torch.no_grad():
                self.update_target()
                #self.run_traj_planner()

    def run_traj_planner(self):
        posf=self.posf
        yawf=self.yawf

        pos0 = [self.state[0], self.state[1], self.state[2]]
        vel0 = [self.state[3], self.state[4], self.state[5]]
        ang_vel0 = [self.state[9], self.state[10], self.state[11]]
        yaw0 = self.state[8] 

        #vel_des=min((self.index+1)*2,self.v_avg)
        vel_des = self.v_avg

        velf=[vel_des*np.cos(yawf),vel_des*np.sin(yawf),0,0]

        x_initial=[pos0[0],pos0[1],pos0[2],yaw0]
        x_final=[posf[0],posf[1],posf[2],yawf]
        vel_initial=[vel0[0]*np.cos(yaw0)-vel0[1]*np.sin(yaw0),vel0[1]*np.cos(yaw0)+vel0[0]*np.sin(yaw0),vel0[2],ang_vel0[2]]
        vel_final=velf

        pose_err=0
        for j in range(3):
            pose_err+=pow(x_final[j]-pos0[j],2)

        pose_err=np.sqrt(pose_err)
        T=pose_err/(vel_des)
        N=int(T/self.dtau)
        t=np.linspace(0,T,N)
        #print(len(t))
        self.traj.find_traj(x_initial=x_initial,x_final=x_final,v_initial=vel_initial,v_final=vel_final,T=T) 

        

    def fly_through_gates(self):
        quad_pose = [self.state[0], self.state[1], -self.state[2], 0, 0, self.state[8]]
        self.client.simSetVehiclePose(QuadPose(quad_pose), True)
        time.sleep(.01)
        self.quad.reset(x=self.state)

        index = 0
        vel_des=0

        self.list_covnet = []
        self.list_difference = []
        self.list_realerror = []

        t=threading.Thread(target=self.update_target_loop)
        t.start()

        

        while(True):
            with torch.no_grad():
                #self.update_target()
                self.run_traj_planner()
                t_current=0.0
                n=10

                target=self.traj.get_target(self.dtau*n)
                    
                vel_target=self.traj.get_vel(self.dtau*n) 
                yaw0=self.state[8]

                waypoint_world = np.array([self.track[index].position.x_val, self.track[index].position.y_val, self.track[index].position.z_val])
                self.waypoint_world_real = np.array([self.track[index].position.x_val, self.track[index].position.y_val, self.track[index].position.z_val])

                for k in range(n):
                    t_current=t_current+self.dtau 
                    
                    yaw_target=yaw0+(k+1)*(target[3]-self.state[8])/n

                    x_t=[vel_target[0]*np.cos(yaw_target)+vel_target[1]*np.sin(yaw_target),vel_target[1]*np.cos(yaw_target)-vel_target[0]*np.sin(yaw_target),vel_target[2],yaw_target]
                    
                    u_nn=self.controller.run_controller(x=self.state[3:12],x_t=x_t)
                    self.state=self.quad.run_model(self.conf_u(u_nn))

                    vel_total=np.sqrt(pow(self.state[3],2)+pow(self.state[4],2)+pow(self.state[5],2))
                    #print("Total_vel:{}".format(vel_total))
  
                    quad_pose = [self.state[0], self.state[1], -self.state[2], 0, 0, self.state[8]]

                    self.total_cost+=abs(np.sqrt(pow(quad_pose[0],2)+pow(quad_pose[1],2))-10)
                    self.client.simSetVehiclePose(QuadPose(quad_pose), True)
                    time.sleep(.01)


                    check_arrival = self.check_completion(index, quad_pose,waypoint_world)

                    if check_arrival: # drone arrives to the gate  
                        gate_completed = True
                        #self.v_avg = np.random.uniform(5,10)
                        #self.state = np.array([self.drone_init.position.x_val,self.drone_init.position.y_val, -self.drone_init.position.z_val,0,0,self.yaw_track[0]-np.pi/2,0,0,0,0,0,0])
                        #quad_pose = [self.state[0], self.state[1], -self.state[2], 0, 0, self.state[8]]
                        #self.client.simSetVehiclePose(QuadPose(quad_pose), True)
                        #time.sleep(.01)
                        #self.quad.reset(x=self.state)

                        if index == 2:
                            plt.figure()
                            plt.plot(self.list_covnet, label = "CovNet")
                            plt.plot(self.list_realerror, label = "Real Error")
                            #plt.plot(self.list_difference, label = "Error Difference")
                            plt.legend(['CovNet', 'Real Error'])

                            plt.xlabel('Iteration')
                            plt.ylabel('Covariance')
                            plt.grid('on')
                            plt.title('Iteration vs Covariance Graph')
                            
                            plt.savefig("covariance.png")

                            plt.show()

                        index += 1
                        print("Drone has gone through the {0}. gate.".format(index))
                        break

                 

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


    # writes your data to file
    def writePosToFile(self, r, theta, psi, phi_rel):
        data_string = '{0} {1} {2} {3}\n'.format(r, theta, psi, phi_rel)
        self.file.write(data_string)


