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

from DQN_solver import *


class PoseSampler:
    def __init__(self,v_avg=5, with_gate=True):
        self.base_path="/home/drone-ai/Documents/Github/Vision-Based-Agile-Drone-Flight"
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

        # Dronet
        self.device = torch.device("cpu")
        self.Dronet =  Dronet.ResNet(Dronet.BasicBlock, [1,1,1,1], num_classes = 4)
        self.Dronet.to(self.device)
        #print("Dronet Model:", self.Dronet)
        self.Dronet.load_state_dict(torch.load(self.base_path+'/weights/16_0.001_2_loss_0.0101_PG.pth',map_location=torch.device('cpu')))   
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
        self.covNet.load_state_dict(torch.load(self.base_path+'/weights/covNet.pth',map_location=torch.device('cpu')))   
        self.covNet.eval() 
        self.covariance=0

        self.n_states=14
        self.n_actions=5

        self.dqn_agent=DQNAgent(state_space=self.n_states,action_space=self.n_actions,max_memory_size=100000,batch_size=256,gamma=.99,lr=.003,exploration_max=0,exploration_min=0,exploration_decay=0)
        self.dqn_agent.load_model("model-100")
        self.rl_states=np.zeros(self.n_states)
        self.rl_next_states=np.zeros(self.n_states)
        self.rl_reward=0
        self.rl_action=0
        self.done=False


        self.episode_number=1000

        self.loop_ration=5 
        self.vel_des=0

        self.crash=False
        self.current_velocity=0


        self.quad=Model()
        self.controller=Controller()


        self.brightness = 0.
        self.contrast = 0.
        self.saturation = 0.
        self.period_denum = 30.

        self.transformation = transforms.Compose([
                            transforms.Resize([200, 200]),
                            #transforms.Lambda(self.gaussian_blur),
                            #transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation),
                            transforms.ToTensor()])



        self.v_avg=v_avg
        self.traj=Traj_Planner()
        self.state=np.zeros(12)
        self.gateNumber=18
        self.radius=20

        self.path = Trajectory(self.gateNumber, self.radius)
        self.path.buildSimpleTrajectory()
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
            
        quat_drone = R.from_euler('ZYX',[0.,0.,0.],degrees=True).as_quat()
        self.drone_init = Pose(Vector3r(-5.,2*10.,-2), Quaternionr(quat_drone[0],quat_drone[1],quat_drone[2],quat_drone[3])) 
        self.posf=[self.track[0].position.x_val, self.track[0].position.y_val, self.track[0].position.z_val]  
        self.yawf=self.yaw_track[0]-np.pi/2    
        self.index=0
        self.yaw0 = self.state[8]

        self.line_list=[]

        self.track = self.track # for circle trajectory change this with circle_track
        self.drone_init = self.drone_init # for circle trajectory change this with drone_init_circle
        self.state=np.array([self.drone_init.position.x_val,self.drone_init.position.y_val,-self.drone_init.position.z_val,0,0,self.yaw_track[0]-np.pi/2,0,0,0,0,0,0])
        self.find_gate_edges()
        #-----------------------------------------------------------------------             

    def find_gate_distances(self):
        gate_1 = self.track[0]
        init_to_gate = np.linalg.norm([gate_1.position.x_val-self.drone_init.position.x_val, gate_1.position.y_val-self.drone_init.position.y_val, gate_1.position.z_val-self.drone_init.position.z_val])
        self.gate_gate_distances.append(init_to_gate)
        for i in range(len(self.track)-1):
            gate_1 = self.track[i]
            gate_2 = self.track[i+1]
            gate_to_gate = np.linalg.norm([gate_1.position.x_val-gate_2.position.x_val, gate_1.position.y_val-gate_2.position.y_val, gate_1.position.z_val-gate_2.position.z_val])
            self.gate_gate_distances.append(gate_to_gate)


    def check_on_road(self):
        gate_drone_distances = []
        for i in range(len(self.track)):
            drone_x, drone_y, drone_z = self.quad.state[0], self.quad.state[1], self.quad.state[2]
            gate_x, gate_y, gate_z = self.track[i].position.x_val, self.track[i].position.y_val, self.track[i].position.z_val
            drone_to_center = np.linalg.norm([gate_x-drone_x, gate_y-drone_y, gate_z-drone_z])
            gate_drone_distances.append(drone_to_center)
            
        max_gate_to_gate = np.max(self.gate_gate_distances)
        min_drone_to_gate = np.min(gate_drone_distances)

        if min_drone_to_gate > 1.1 * max_gate_to_gate:
            return False

        return True


    def find_gate_edges(self):
        for i in range(len(self.track)):
            rot_matrix = Rotation.from_quat([self.track[i].orientation.x_val, self.track[i].orientation.y_val, 
                                      self.track[i].orientation.z_val, self.track[i].orientation.w_val]).as_dcm().reshape(3,3)
            gate_x_range = [.75, -.75]
            gate_z_range = [.75, -.75]
            edge_ind = 0
            #print "\nGate Ind: {0}, Gate x={1:.3}, y={2:.3}, z={3:.3}".format(i+1, self.track[i].position.x_val, self.track[i].position.y_val, self.track[i].position.z_val)
            gate_pos = np.array([self.track[i].position.x_val, self.track[i].position.y_val, self.track[i].position.z_val])
            
            check_list = []
            gate_edge_list = []
            # print ""
            for x_rng in gate_x_range:
                for z_rng in gate_z_range:
                    gate_edge_range = np.array([x_rng, 0., z_rng])
                    gate_edge_world = np.dot(rot_matrix, gate_edge_range.reshape(-1,1)).ravel()
                    gate_edge_point = np.array([gate_pos[0]+gate_edge_world[0], gate_pos[1]+gate_edge_world[1], gate_pos[2]+gate_edge_world[2]])
                    edge_ind += 1
                    # print "Index: {0}, Edge x={1:.3}, y={2:.3}, z={3:.3}".format(edge_ind, gate_edge_point[0], gate_edge_point[1], gate_edge_point[2])
                    # quad_pose = [gate_edge_point[0], gate_edge_point[1], gate_edge_point[2], 0., 0., 0.]
                    # self.client.simSetVehiclePose(QuadPose(quad_pose), True)
                    # time.sleep(3)
                    gate_edge_list.append([gate_edge_point[0], gate_edge_point[1], gate_edge_point[2]])

            ind = 0
            # print "\nFor Gate: " + str(i)
            # print "They are on the same line"
            for i in range(len(gate_edge_list)):
                for j in range(len(gate_edge_list)):
                    edge_i = np.array(gate_edge_list[i])
                    edge_j = np.array(gate_edge_list[j])
                    if i != j and (i+j) != 3 and [i,j] not in check_list and [j,i] not in check_list:
                        # print "Index: " + str(ind) + " - " + str(i) + "/" + str(j)
                        # print "edge_i: " + str(edge_i) + " edge_j: " + str(edge_j)
                        u_v = abs(edge_i - edge_j)
                        current_list = [edge_i, edge_j, u_v]
                        self.line_list.append(current_list)
                        check_list.append([i,j])
                        check_list.append([j,i])
                        ind += 1

                        # print "Edge_i"
                        # quad_pose = [edge_i[0], edge_i[1], edge_i[2], -0., -0., 0.]
                        # self.client.simSetVehiclePose(QuadPose(quad_pose), True)
                        # time.sleep(3)
                        # print "Edge_j"
                        # quad_pose = [edge_j[0], edge_j[1], edge_j[2], -0., -0., 0.]
                        # self.client.simSetVehiclePose(QuadPose(quad_pose), True)
                        # time.sleep(3)



    def check_collision(self, max_distance = 0.15):
        distance_list = []
        for i in range(len(self.track)):
            gate = self.track[i]
            distance = np.linalg.norm([self.state[0] - gate.position.x_val, self.state[1] - gate.position.y_val, -self.state[2] - gate.position.z_val])
            distance_list.append(distance)        

        distance_min = np.min(distance_list) # this is the distance of drone's center point to the closest gate 
        #print "Minimum distance: {0:.3}".format(distance_min)

        if distance_min < 1.: # if this value less than threshold, collision check should be done
            drone_x_range = [.1, -.1]
            drone_y_range = [.1, -.1]
            drone_z_range = [.025, -.025]
            rot_matrix = R.from_euler('ZYX',[self.state[8], self.state[7], self.state[6]],degrees=False).as_dcm().reshape(3,3)
            drone_pos = np.array([self.state[0], self.state[1], -self.state[2]])
            edge_ind = 0

            #Collision check for drone's centroid
            # for i, line in enumerate(self.line_list):
            #     edge_i, edge_j, u_v = line
            #     # p1, p2, p3 = Point3D(edge_i[0], edge_i[1], edge_i[2]), Point3D(edge_j[0], edge_j[1], edge_j[2]), Point3D(drone_pos[0], drone_pos[1], drone_pos[2])
            #     # l1 = Line3D(p1, p2) 
            #     # distance = l1.distance(p3).evalf()
            #     distance_from_center = edge_i - drone_pos
            #     distance = np.linalg.norm(np.cross(distance_from_center, u_v)) / np.linalg.norm(u_v)
                
            #     #print "Edge: {0}, (Numeric) Distance from the center: {1:.3}".format(i, distance) 
            #     if distance < max_distance:
            #         print "Collision detected!"
            #         print "Index: {0}, Drone center x={1:.3}, y={2:.3}, z={3:.3}".format(i, drone_pos[0], drone_pos[1], drone_pos[2])

            #         return True

            # Collision check for Drone's corner points
            for x_rng in drone_x_range:
                for y_rng in drone_y_range:
                    for z_rng in drone_z_range:
                        drone_range = np.array([x_rng, y_rng, z_rng])
                        drone_range_world = np.dot(rot_matrix.T, drone_range.reshape(-1,1)).ravel()
                        drone_edge_point = np.array([drone_pos[0]+drone_range_world[0], drone_pos[1]+drone_range_world[1], drone_pos[2]+drone_range_world[2]])
                        edge_ind += 1
                        
                        
                        for i, line in enumerate(self.line_list):
                            edge_i, edge_j, u_v = line
                            distance_from_center = edge_i - drone_edge_point
                            distance = np.linalg.norm(np.cross(distance_from_center, u_v)) / np.linalg.norm(u_v)
                            #print "Edge: {0}, (Numeric) Distance from the center: {1:.3}".format(i, distance) 
                            if distance < max_distance:

                                return True
            
            # print "No Collision!"
            return False


        else:
            return False

       

    def test_collision(self, gate_index):
        phi = np.random.uniform(-np.pi/6, np.pi/6)
        theta =  np.random.uniform(-np.pi/6, np.pi/6)
        psi = np.random.uniform(-np.pi/6, np.pi/6)
        quad_pose = [self.track[gate_index].position.x_val, self.track[gate_index].position.y_val, self.track[gate_index].position.z_val, -phi, -theta, psi]
        self.quad.state = [quad_pose[0], quad_pose[1], quad_pose[2], phi, theta, psi, 0., 0., 0., 0., 0., 0.]
        self.client.simSetVehiclePose(QuadPose(quad_pose), True)
        self.check_collision()
        time.sleep(5)
        

        rot_matrix = Rotation.from_quat([self.track[gate_index].orientation.x_val, self.track[gate_index].orientation.y_val, 
                                      self.track[gate_index].orientation.z_val, self.track[gate_index].orientation.w_val]).as_dcm().reshape(3,3)
        gate_x_range = [np.random.uniform(0.6, 1.0), -np.random.uniform(0.6, 1.0)]
        gate_z_range = [np.random.uniform(0.6, 1.0), -np.random.uniform(0.6, 1.0)]
        edge_ind = 0
        #print "\nGate Ind: {0}, Gate x={1:.3}, y={2:.3}, z={3:.3}".format(i+1, self.track[i].position.x_val, self.track[i].position.y_val, self.track[i].position.z_val)
        gate_pos = np.array([self.track[gate_index].position.x_val, self.track[gate_index].position.y_val, self.track[gate_index].position.z_val])
        gate_edge_list = []
        for x_rng in gate_x_range:
            gate_edge_range = np.array([x_rng/1.5, 0., 0.25*np.random.uniform(-1,1)])
            gate_edge_world = np.dot(rot_matrix, gate_edge_range.reshape(-1,1)).ravel()
            gate_edge_point = np.array([gate_pos[0]+gate_edge_world[0], gate_pos[1]+gate_edge_world[1], gate_pos[2]+gate_edge_world[2]])
            self.quad.state = [gate_edge_point[0], gate_edge_point[1], gate_edge_point[2], phi, theta, psi, 0., 0., 0., 0., 0., 0.]
            quad_pose = [gate_edge_point[0], gate_edge_point[1], gate_edge_point[2], -phi, -theta, psi]
            self.client.simSetVehiclePose(QuadPose(quad_pose), True)
            self.check_collision()
            time.sleep(5)
            

        for z_rng in gate_z_range:
            gate_edge_range = np.array([0.25*np.random.uniform(-1,1), 0., z_rng/1.5])
            gate_edge_world = np.dot(rot_matrix, gate_edge_range.reshape(-1,1)).ravel()
            gate_edge_point = np.array([gate_pos[0]+gate_edge_world[0], gate_pos[1]+gate_edge_world[1], gate_pos[2]+gate_edge_world[2]])
            edge_ind += 1
            self.quad.state = [gate_edge_point[0], gate_edge_point[1], gate_edge_point[2], phi, theta, psi, 0., 0., 0., 0., 0., 0.]
            quad_pose = [gate_edge_point[0], gate_edge_point[1], gate_edge_point[2], -phi, -theta, psi]
            self.client.simSetVehiclePose(QuadPose(quad_pose), True)
            self.check_collision()
            time.sleep(5) 


    def check_completion(self, index, quad_pose, eps=0.45):
        x, y, z = quad_pose[0], quad_pose[1], quad_pose[2]

        xd = self.track[index].position.x_val
        yd = self.track[index].position.y_val
        zd = self.track[index].position.z_val
        psid = Rotation.from_quat([self.track[index].orientation.x_val, self.track[index].orientation.y_val, 
                                   self.track[index].orientation.z_val, self.track[index].orientation.w_val]).as_euler('ZYX',degrees=False)[0]

        target = [xd, yd, zd, psid] 
        check_arrival = False


        if ( (abs(abs(xd)-abs(x)) <= eps) and (abs(abs(yd)-abs(y)) <= eps) and (abs(abs(zd)-abs(z)) <= eps)):
            #self.quad.calculate_cost(target=target, final_calculation=True)
            check_arrival = True

        return check_arrival

    def conf_u(self,u):
        for i in range(3):
            u[i]=(u[i]-5)/10

        return u

    def run_dronet(self,image_response):
        img1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)  # get numpy array
        img_rgb = img1d.reshape(image_response.height, image_response.width, 3)  # reshape array to 4 channel image array H X W X 3
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        # anyGate = self.isThereAnyGate(img_rgb)
        #cv2.imwrite(os.path.join(self.base_path, 'images', "frame" + str(self.curr_idx).zfill(len(str(self.num_samples))) + '.png'), img_rgb)
        img =  Image.fromarray(img_rgb)
        image = self.transformation(img)  
        pose_gate_body = self.Dronet(image)

        return pose_gate_body 

    def dronet_to_body(self,pose_gate_body):
        drone_pose=[self.state[0], self.state[1], -self.state[2], 0, 0, self.state[8]]
        pose_gate_body = pose_gate_body.numpy().reshape(-1,1)
        waypoint_world = spherical_to_cartesian(drone_pose, pose_gate_body)
        yaw_diff = pose_gate_body[3][0]
        posf = [waypoint_world[0], waypoint_world[1], -waypoint_world[2]]
        yawf = drone_pose[5]+yaw_diff+np.pi/2

        return posf,yawf

    def update_target(self):
        image_response = self.client2.simGetImages([airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)])[0]
 
        pose_gate_body=self.run_dronet(image_response)
        self.posf,self.yawf=self.dronet_to_body(pose_gate_body) 

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

        vel_des=min((self.index+1)*2,self.v_avg+1)

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

    def get_rl_state(self):
        states=np.zeros(14)
        for i in range(9):
            states[i]=self.state[i]
        for i in range(3):
            states[9+i]=self.posf[i]
        states[12]=self.yawf
        states[13]=self.covariance

        return states
    
    def get_rl_reward(self):
        reward=0
        if self.crash or self.get_distance_to_center_check():
            reward-=1000000
        reward+=self.current_velocity
        if self.index==18:
            reward+=1000

        return reward

    def reset_drone(self):
        self.state=np.array([self.drone_init.position.x_val,self.drone_init.position.y_val,-self.drone_init.position.z_val,0,0,self.yaw_track[0]-np.pi/2,0,0,0,0,0,0])

    def get_current_velocity(self):
        self.current_velocity=np.sqrt(pow(self.state[3],2)+pow(self.state[4],2)+pow(self.state[5],2))

    def get_distance_to_center_check(self):
        distance=np.sqrt(pow(self.state[0],2)+pow(self.state[1],2)+pow(self.state[2]-2,2))

        if distance>self.radius+1 or distance<self.radius-1:
            return True
        return False

    def get_done_statu(self):
        if self.get_distance_to_center_check() or self.crash or self.index==18:
            return True
        return False



    def fly_through_gates(self):
        for epp in range(self.episode_number):
            self.reset_drone()
            print("Episode {} running...".format(epp))
            quad_pose = [self.state[0], self.state[1], -self.state[2], 0, 0, self.state[8]]
            self.client.simSetVehiclePose(QuadPose(quad_pose), True)
            self.quad.reset(x=self.state)
            self.crash=False
            self.index=0
            rl_index=0
            while(True):
                with torch.no_grad():

                    self.update_target()
                    self.get_current_velocity()

                    if self.get_done_statu():
                        break

                    
                    self.rl_states=torch.Tensor(np.array(self.get_rl_state()))
                    self.rl_action=self.dqn_agent.act(self.rl_states)
                    rl_index+=1

                    self.v_avg=self.rl_action.item()
                    print("Des_Vel={}".format(self.v_avg+1))

                    
                    self.run_traj_planner()

                    t_current=0.0
                    n=10

                    target=self.traj.get_target(self.dtau*n)
                        
                    vel_target=self.traj.get_vel(self.dtau*n) 
                    yaw0=self.state[8]               

                    for k in range(n):
                        t_current=t_current+self.dtau 
                        
                        yaw_target=yaw0+(k+1)*(target[3]-self.state[8])/n

                        x_t=[vel_target[0]*np.cos(yaw_target)+vel_target[1]*np.sin(yaw_target),vel_target[1]*np.cos(yaw_target)-vel_target[0]*np.sin(yaw_target),vel_target[2],yaw_target]
                        
                        u_nn=self.controller.run_controller(x=self.state[3:12],x_t=x_t)
                        self.state=self.quad.run_model(self.conf_u(u_nn))


                        quad_pose = [self.state[0], self.state[1], -self.state[2], 0, 0, self.state[8]]

                        self.total_cost+=abs(np.sqrt(pow(quad_pose[0],2)+pow(quad_pose[1],2))-10)
                        self.client.simSetVehiclePose(QuadPose(quad_pose), True)
                        #time.sleep(.001)

                        check_arrival = self.check_completion(self.index%self.gateNumber, quad_pose)

                        self.crash=self.check_collision()

                        if self.crash:
                            print("Drone crashed")
                            break

                        if check_arrival: # drone arrives to the gate
                            gate_completed = True
                            self.index += 1
                            print("Drone has gone through the {0}. gate.".format(self.index))
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

