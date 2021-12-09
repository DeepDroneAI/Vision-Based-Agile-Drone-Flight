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

        quat0 = R.from_euler('ZYX',[90.,0.,0.],degrees=True).as_quat()
        quat1 = R.from_euler('ZYX',[60.,0.,0.],degrees=True).as_quat()
        quat2 = R.from_euler('ZYX',[30.,0.,0.],degrees=True).as_quat()
        quat3 = R.from_euler('ZYX',[0.,0.,0.],degrees=True).as_quat()
        quat4 = R.from_euler('ZYX',[-30.,0.,0.],degrees=True).as_quat() 
        quat5 = R.from_euler('ZYX',[-60.,0.,0.],degrees=True).as_quat() 
        quat6 = R.from_euler('ZYX',[-90.,0.,0.],degrees=True).as_quat()
        quat7 = R.from_euler('ZYX',[-120.,0.,0.],degrees=True).as_quat()
        quat8 = R.from_euler('ZYX',[-150.,0.,0.],degrees=True).as_quat()
        quat9 = R.from_euler('ZYX',[-180.,0.,0.],degrees=True).as_quat()
        quat10 = R.from_euler('ZYX',[-210.,0.,0.],degrees=True).as_quat()
        quat11 = R.from_euler('ZYX',[-250.,0.,0.],degrees=True).as_quat()
        self.yaw_track=np.array([90,60,30,0,-30,-60,-90,-120,-150,-180,-210,-250])*np.pi/180
        self.track = [Pose(Vector3r(0.,2*10.,-2.) , Quaternionr(quat0[0],quat0[1],quat0[2],quat0[3])),
                    Pose(Vector3r(2*5.,2*8.66,-2) , Quaternionr(quat1[0],quat1[1],quat1[2],quat1[3])),
                    Pose(Vector3r(2*8.66,2*5.,-2) , Quaternionr(quat2[0],quat2[1],quat2[2],quat2[3])),
                    Pose(Vector3r(2*10.,0.,-2) , Quaternionr(quat3[0],quat3[1],quat3[2],quat3[3])),
                    Pose(Vector3r(2*8.66,2*-5.,-2) , Quaternionr(quat4[0],quat4[1],quat4[2],quat4[3])),
                    Pose(Vector3r(2*5.,2*-8.66,-2) , Quaternionr(quat5[0],quat5[1],quat5[2],quat5[3])), 
                    Pose(Vector3r(0.,2*-10.,-2) , Quaternionr(quat6[0],quat6[1],quat6[2],quat6[3])),
                    Pose(Vector3r(2*-5.,2*-8.66,-2) , Quaternionr(quat7[0],quat7[1],quat7[2],quat7[3])),
                    Pose(Vector3r(2*-8.66,2*-5,-2) , Quaternionr(quat8[0],quat8[1],quat8[2],quat8[3])),
                    Pose(Vector3r(2*-10.,0,-2) , Quaternionr(quat9[0],quat9[1],quat9[2],quat9[3])),
                    Pose(Vector3r(2*-8.66,2*5.,-2) , Quaternionr(quat10[0],quat10[1],quat10[2],quat10[3])),
                    Pose(Vector3r(2*-5.,2*8.66,-2) , Quaternionr(quat11[0],quat11[1],quat11[2],quat11[3]))]
            
        quat_drone = R.from_euler('ZYX',[0.,0.,0.],degrees=True).as_quat()
        self.drone_init = Pose(Vector3r(-5.,2*10.,-2), Quaternionr(quat_drone[0],quat_drone[1],quat_drone[2],quat_drone[3]))

        self.track = self.track # for circle trajectory change this with circle_track
        self.drone_init = self.drone_init # for circle trajectory change this with drone_init_circle
        self.state=np.array([self.drone_init.position.x_val,self.drone_init.position.y_val,self.drone_init.position.z_val,0,0,self.yaw_track[0]-np.pi/2,0,0,0,0,0,0])

    def conf_u(self,u):
        for i in range(3):
            u[i]=(u[i]-5)/10

        return u

    def fly_through_gates(self):
        
        self.client.simSetVehiclePose(QuadPose(self.state[[0,1,2,3,4,5]]), True)
        self.quad.reset(x=self.state)

        index = 0
        v_des=0
        while(True):
            if index==12*10:
                break


            for i in range(1):

                # Trajectory generate
                waypoint_world = np.array([self.track[index%12].position.x_val, self.track[index%12].position.y_val, self.track[index%12].position.z_val])
                
                posf=waypoint_world
                yawf=self.yaw_track[index%12]-np.pi/2-(int(index/12))*2*np.pi

                pos0 = [self.state[0], self.state[1], self.state[2]]
                vel0 = [self.state[3], self.state[4], self.state[5]]
                ang_vel0 = [self.state[9], self.state[10], self.state[11]]
                yaw0 = self.state[8]

                v_des=min(v_des+2,self.v_avg)
                

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
                
                t_current=0.0
                for k in range(len(t)):
                    t_current=t[k] 
                    target=self.traj.get_target(t_current)
                    vel_target=self.traj.get_vel(t_current)

                    x_t=[vel_target[0]*np.cos(target[3])+vel_target[1]*np.sin(target[3]),vel_target[1]*np.cos(target[3])-vel_target[0]*np.sin(target[3]),vel_target[2],target[3]]
                    
                    u_nn=self.controller.run_controller(x=self.state[3:12],x_t=x_t)
                    self.state=self.quad.run_model(self.conf_u(u_nn))

                    if index<100:
                        roll=self.state[6]
                        pitch=self.state[7]
                    else:
                        roll=self.state[6]*np.pi/45
                        pitch=self.state[7]*np.pi/45

                    vel_total=np.sqrt(pow(self.state[3],2)+pow(self.state[4],2)+pow(self.state[5],2))
                    print("Total_vel:{}".format(vel_total))



                    quad_pose = [self.state[0], self.state[1], self.state[2], 0, 0, self.state[8]]
                    #vel_target=[vel_target[0], vel_target[1], vel_target[2], 0, 0, vel_target[3]]
                    #self.total_cost+=abs(np.sqrt(pow(quad_pose[0],2)+pow(quad_pose[1],2))-10)
                    #self.state=np.array([target[0],target[1],target[2],0,0,target[3],vel_target[0],vel_target[1],vel_target[2],0,0,vel_target[3]])
                    self.client.simSetVehiclePose(QuadPose(quad_pose), True)
                    time.sleep(.01)


                index += 1


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


