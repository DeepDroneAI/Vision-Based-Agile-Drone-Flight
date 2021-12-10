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
from traj_planner_min_jerk import Traj
# changed 
from pose_sampler import * 
from createTraj import *


class PoseSampler:
    def __init__(self,v_avg=5, with_gate=True, velInc=False):
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
        self.gTruths = []

        self.velInc=velInc
        self.v_avg=v_avg
        self.vAvgValues = [3., 6., 9.]
        self.traj=Traj_Planner()
        self.state=np.zeros(12)
        self.traj2=Traj()
        self.gateNumber = 24
        self.radius = 20

        traj = Trajectory(self.gateNumber, self.gateNumber)
        traj.buildSimpleTrajectory()
        traj.initDronePose()
        self.yaw_track = traj.yawTrack
        self.track = traj.track
        self.drone_init = traj.droneInit
        self.saveCnt = 0 
        self.loopCnt = 0 
        self.distanceToGate = 1 
        self.saveChanged = True 

        self.track = self.track # for circle trajectory change this with circle_track
        self.drone_init = self.drone_init # for circle trajectory change this with drone_init_circle
        self.state=np.array([self.drone_init.position.x_val,self.drone_init.position.y_val,self.drone_init.position.z_val,0,0,self.yaw_track[0]-np.pi/2,0,0,0,0,0,0])
        #-----------------------------------------------------------------------             
        self.transformation = self.transformation = transforms.Compose([
                              transforms.Resize([200, 200]),
                              #transforms.Lambda(self.gaussian_blur),
                              #transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation),
                              transforms.ToTensor()])

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
            distance = np.linalg.norm([self.quad.state[0] - gate.position.x_val, self.quad.state[1] - gate.position.y_val, self.quad.state[2] - gate.position.z_val])
            distance_list.append(distance)        

        distance_min = np.min(distance_list) # this is the distance of drone's center point to the closest gate 
        #print "Minimum distance: {0:.3}".format(distance_min)

        if distance_min < 1.: # if this value less than threshold, collision check should be done
            drone_x_range = [.1, -.1]
            drone_y_range = [.1, -.1]
            drone_z_range = [.025, -.025]
            rot_matrix = R.from_euler('ZYX',[self.quad.state[5], self.quad.state[4], self.quad.state[3]],degrees=False).as_dcm().reshape(3,3)
            drone_pos = np.array([self.quad.state[0], self.quad.state[1], self.quad.state[2]])
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
            self.quad.calculate_cost(target=target, final_calculation=True)
            check_arrival = True

        return check_arrival

    def fly_through_gates(self):

        self.client.simSetVehiclePose(QuadPose(self.state[[0, 1, 2, 3, 4, 5]]), True)

        index = 0
        turnCount = len(self.vAvgValues)
        maxTurn = self.gateNumber * turnCount
        gTruths = []
        iNames = []
        while (True):
            if index == maxTurn:
                break

            for i in range(1):

                # Trajectory generate
                waypoint_world = np.array([self.track[index % self.gateNumber].position.x_val,
                                           self.track[index % self.gateNumber].position.y_val,
                                           self.track[index % self.gateNumber].position.z_val])
                posf = waypoint_world
                yawf = self.yaw_track[index % self.gateNumber] - np.pi / 2 - (int(index / self.gateNumber)) * 2 * np.pi

                pos0 = [self.state[0], self.state[1], self.state[2]]
                vel0 = [self.state[6], self.state[7], self.state[8]]
                ang_vel0 = [self.state[9], self.state[10], self.state[11]]
                yaw0 = self.state[5]
                if self.velInc:
                    self.v_avg = self.vAvgValues[int(index/self.gateNumber)]

                velf = [self.v_avg * np.cos(yawf), self.v_avg * np.sin(yawf), 0, 0]

                x_initial = [pos0[0], pos0[1], pos0[2], yaw0]
                x_final = [posf[0], posf[1], posf[2], yawf]
                vel_initial = [vel0[0], vel0[1], vel0[2], ang_vel0[2]]
                vel_final = velf
                a_initial = [0, 0, 0, 0]
                a_final = [0, 0, 0, 0]
                pose_err = 0
                for j in range(3):
                    pose_err += pow(x_final[j] - pos0[j], 2)

                pose_err = np.sqrt(pose_err)
                T = pose_err / (self.v_avg)
                N = int(T / self.dtau)
                t = np.linspace(0, T, N)
                self.traj.find_traj(x_initial=x_initial, x_final=x_final, v_initial=vel_initial, v_final=vel_final, T=T)
                # self.traj.find_traj(x_initial=x_initial,x_final=x_final,v_initial=vel_initial,v_final=vel_final,a_initial=a_initial,a_final=a_final,T=T)

                # Creating the path if not exist
                if not os.path.isdir('images/images-0'):
                    os.mkdir(os.getcwd() + '/images/' + 'images-{}'.format(self.loopCnt))
                    gTruths = []

                # Creating the path if not exist
                if not os.path.isdir('images/images-0'):
                    os.mkdir(os.getcwd() + '/images/' + 'images-{}'.format(self.loopCnt))

                t_current = 0.0
                for k in range(len(t)):
                    t_current = t[k]
                    sName = self.getInstantaneousImg(save=True)  # getting img dataset
                    target = self.traj.get_target(t_current)
                    vel_target = self.traj.get_vel(t_current)
                    quad_pose = [target[0], target[1], target[2], 0, 0, target[3]]
                    vel_target = [vel_target[0], vel_target[1], vel_target[2], 0, 0, vel_target[3]]
                    rho, phi, theta = cartesian_to_spherical(self.state, waypoint_world)
                    yawDif = yawf - self.state[5]
                    # print("wp: ", cartesian_to_spherical(self.state, waypoint_world), 'yaw: ', yawf - self.state[5])
                    gTruths.append([str(rho), str(phi), str(theta), str(yawDif - np.pi / 2)])  # yawDif - np.pi/2
                    iNames.append(sName)
                    self.total_cost += abs(np.sqrt(pow(quad_pose[0], 2) + pow(quad_pose[1], 2)) - 10)
                    self.state = np.array(
                        [target[0], target[1], target[2], 0, 0, target[3], vel_target[0], vel_target[1], vel_target[2],
                         0, 0, vel_target[3]])
                    self.client.simSetVehiclePose(QuadPose(quad_pose), True)
                    # time.sleep(.01)
                    # print("own    position: ", quad_pose[:3])
                    # print("target position: ", waypoint_world)
                    self.distanceToGate = round(dist3dp(quad_pose, waypoint_world), 2)
                    # print("distance: ", self.distanceToGate)
                    if self.distanceToGate < 1.0 and self.saveChanged:
                        # gTruths = []
                        # iNames = []
                        # self.loopCnt += 1
                        self.loopCnt = min(maxTurn - 1, self.loopCnt)
                        self.saveChanged = False
                        # print("PATHFOLDER CHANGED")
                        if not os.path.isdir('images/images-{}'.format(self.loopCnt)):
                            os.mkdir(os.getcwd() + '/images/' + 'images-{}'.format(self.loopCnt))
                        break

                index += 1
                self.saveChanged = True
            #    if index > 10:
            #        break
            #print("yaz")
            self.writeGroundTruthVals(gTruths, iNames)

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

        # self.client.simSetObjectPose(self.tgt_name, p_o_g_new, True)
        # min_vel, min_acc, min_jerk, pos_waypoint_interp, min_acc_stop, min_jerk_full_stop
        MP_list = ["min_acc", "min_jerk", "min_jerk_full_stop", "min_vel"]
        # MP_list = ["min_vel"]

        if self.with_gate:
            # gate_name = "gate_0"
            # self.tgt_name = self.client.simSpawnObject(gate_name, "RedGate16x16", Pose(position_val=Vector3r(0,0,15)), 0.75)
            # self.client.simSetObjectPose(self.tgt_name, self.track[0], True)
            for i, gate in enumerate(self.track):
                # print ("gate: ", gate)
                gate_name = "gate_" + str(i)
                self.tgt_name = self.client.simSpawnObject(gate_name, "RedGate16x16",
                                                           Pose(position_val=Vector3r(0, 0, 15)), 0.75)
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
            self.tgt_name = self.client.simSpawnObject("gate", "RedGate16x16", Pose(position_val=Vector3r(0, 0, 15)),
                                                       0.75)
        else:
            self.tgt_name = "empty_target"

    # writes your data to file
    def writePosToFile(self, r, theta, psi, phi_rel):
        data_string = '{0} {1} {2} {3}\n'.format(r, theta, psi, phi_rel)
        self.file.write(data_string)

    def getInstantaneousImg(self, save=False):
        self.client.simSetVehiclePose(QuadPose(self.state[[0, 1, 2, 3, 4, 5]]), True)
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