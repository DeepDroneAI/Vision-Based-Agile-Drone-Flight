import numpy as np
from controller import Controller
from quad_model import Model
import matplotlib.pyplot as plt
from mpc_class import MPC_controller
from traj_planner import Traj_Planner



quad=Model()
quad2=Model()
controller=Controller()
mpc_controller=MPC_controller()

def to_dataset(x,u):
    x_data=np.zeros(13)
    for i in range(9):
        x_data[i]=x[i]
    for i in range(4):
        x_data[i+9]=u[i]
    return x_data

def conf_u(u):
    for i in range(3):
        u[i]=(u[i]-5)/10

    return u
def pose_err(x,xt):
    err=0
    for i in range(3):
        err+=pow(xt[i]-x[i],2)
    return np.sqrt(err)





traj=Traj_Planner()


Gate=np.array([[0,10,-2,0*np.pi/180],[5,8.66,-2,-30*np.pi/180],[8.66,5,-2,-60*np.pi/180],[10,0,-2,-90*np.pi/180],
               [8.66,-5,-2,-120*np.pi/180],[5,-8.66,-2,-150*np.pi/180],[0,-10,-2,-180*np.pi/180],
               [-5,-8.66,-2,-210*np.pi/180],[-8.66,-5,-2,-240*np.pi/180],[-10,0,-2,-270*np.pi/180],
               [-8.66,5,-2,-300*np.pi/180],[-5,8.66,-2,-330*np.pi/180],[0,10,-2,-360*np.pi/180]])

initial_pos=[-5,20,-2,0]
initial_vel=[0,0,0,0]

foward_vel=10

current_pose=initial_pos
current_vel=initial_vel


traj_arr=[]
pose_arr=[]
vel_arr=[]


mpc_controller=MPC_controller()
controller=Controller()

quad=Model()
quad.reset(x=[initial_vel[0],initial_pos[1],initial_pos[2],0,0,0,0,0,0,0,0,0])
T_sum=0
N_sum=0
des_vel=0
for i in range(int(len(Gate[:,0])/1)):
    #print("Gate {}".format(i+1))
    target_pose=[Gate[i][0]*2,Gate[i][1]*2,Gate[i][2],Gate[i][3]]
    des_vel=min(des_vel+100,foward_vel)

    target_vel=[des_vel*np.cos(target_pose[3]),des_vel*np.sin(target_pose[3]),0,0]
    T=pose_err(current_pose,target_pose)/(foward_vel/2 if i==0 else foward_vel)
    #T=T*3/4
    print("i:{}  T:{}".format(i,T))
    traj.find_traj(x_initial=current_pose,x_final=target_pose,v_initial=current_vel,v_final=target_vel,T=T)
    N=int(T/.01)
    t=np.linspace(0,T,N)
    T_sum+=T
    N_sum+=N

    for j in range(N):
        t_current=t[j]
        t_pose=traj.get_target(t_current)
        traj_arr.append(t_pose)
        t_vel=traj.get_vel(t_current)

        tb_vel=[t_vel[0]*np.cos(t_pose[3])+t_vel[1]*np.sin(t_pose[3]),t_vel[1]*np.cos(t_pose[3])-t_vel[0]*np.sin(t_pose[3]),t_vel[2],t_vel[3]]
        vel_arr.append(tb_vel)
        u_mpc=controller.run_controller(x=quad.x[3:12],x_t=[tb_vel[0],tb_vel[1],tb_vel[2],t_pose[3]])
        next_state=quad.run_model(conf_u(u_mpc))
        pose_arr.append(np.array(quad.x).tolist())
        #print(quad.x)

    current_pose=[quad.x[0],quad.x[1],quad.x[2],quad.x[8]]
    current_vel=[quad.x[3]*np.cos(quad.x[8])-quad.x[4]*np.sin(quad.x[8]),quad.x[4]*np.cos(quad.x[8])+quad.x[3]*np.sin(quad.x[8]),quad.x[5],quad.x[11]]






#print(pose_arr)
traj_arr=np.array(traj_arr)
pose_arr=np.array(pose_arr)
vel_arr=np.array(vel_arr)
t=np.linspace(0,T_sum,int(N_sum))

theta = np.linspace(0, 2*np.pi, 100)

radius = 20

a = radius*np.cos(theta)
b = radius*np.sin(theta)


np.savetxt("Traj_1/traj_arr.txt",traj_arr)
np.savetxt("Traj_1/pose_arr.txt",pose_arr)
np.savetxt("Traj_1/vel_arr.txt",vel_arr)
np.savetxt("Traj_1/t_arr.txt",t)




plt.plot(traj_arr[:,0],traj_arr[:,1])
plt.plot(pose_arr[:,0],pose_arr[:,1])
plt.plot(a,b)
plt.legend(["traj","mpc","optimal"])
plt.show()


plt.plot(t,vel_arr[:,0])
plt.plot(t,vel_arr[:,1])
plt.plot(t,pose_arr[:,3])
plt.plot(t,pose_arr[:,4])
plt.legend(["traj_vel_x","traj_vel_y","mpc_vel_x","mpc_vel_y"])

plt.show()

    

