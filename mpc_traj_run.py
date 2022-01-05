import numpy as np
from controller import Controller
from quad_model import Model
import matplotlib.pyplot as plt
from mpc_traj_follow import MPC_controller
from traj_planner import Traj_Planner



quad=Model()
mpc_controller=MPC_controller()

def conf_u(u):
    for i in range(3):
        u[i]=(u[i]-5)/10
    return u
def pose_err(x,xt):
    err=0
    for i in range(3):
        err+=pow(xt[i]-x[i],2)
    return np.sqrt(err)

def conf_traj(traj):
    pose=traj[0].tolist()
    vel=traj[1].tolist()

    vel=[vel[0]*np.cos(pose[3])+vel[1]*np.sin(pose[3]),vel[1]*np.cos(pose[3])-vel[0]*np.sin(pose[3]),vel[2],vel[3]]

    traj=[pose,vel]

    return np.array(traj)

traj=Traj_Planner()

mpc_controller=MPC_controller()
controller=Controller()

initial_pos=[0,0,0,0]
final_pos=[5,2,3,np.pi/3]
initial_vel=[0,0,0,0]
final_vel=[0,0,0,0]





quad=Model()
quad.reset(x=[initial_pos[0],initial_pos[1],initial_pos[2],initial_vel[0],initial_vel[1],initial_vel[2],0,0,0,0,0,0])

v_avg=5
pose_err=pose_err(initial_pos,final_pos)
T=pose_err/v_avg
traj.find_traj(initial_pos,final_pos,initial_vel,final_vel,T)
dt=0.01
N=int(T/dt)

t=np.linspace(0,T,N)

pose_ref=[]
vel_ref=[]
pose=[]
vel=[]


for t_current in t:
    curent_traj=traj.get_traj(t_current)
    curent_traj=conf_traj(curent_traj)
    x=quad.x
    pose.append([x[0],x[1],x[2],x[8]])
    vel.append([x[3],x[4],x[5],x[11]])
    u=mpc_controller.run_controller(x=x,traj=curent_traj)
    pose_ref.append(curent_traj[0].tolist())
    vel_ref.append(curent_traj[1].tolist())
    u=conf_u(u)
    quad.run_model(u=u)


pose_ref=np.array(pose_ref)
vel_ref=np.array(vel_ref)
pose=np.array(pose)
vel=np.array(vel)

plt.subplot(2, 2, 1)
plt.plot(t,pose_ref[:,0])
plt.plot(t,pose[:,0])
plt.legend(['Referance x','x'])

plt.subplot(2, 2, 2)
plt.plot(t,pose_ref[:,1])
plt.plot(t,pose[:,1])
plt.legend(['Referance y','y'])

plt.subplot(2, 2, 3)
plt.plot(t,pose_ref[:,2])
plt.plot(t,pose[:,2])
plt.legend(['Referance z','z'])

plt.subplot(2, 2, 4)
plt.plot(t,pose_ref[:,3])
plt.plot(t,pose[:,3])
plt.legend(['Referance psi','psi'])
plt.show()



plt.subplot(2, 2, 1)
plt.plot(t,vel_ref[:,0])
plt.plot(t,vel[:,0])
plt.legend(['Referance vx','vx'])

plt.subplot(2, 2, 2)
plt.plot(t,vel_ref[:,1])
plt.plot(t,vel[:,1])
plt.legend(['Referance vy','vy'])

plt.subplot(2, 2, 3)
plt.plot(t,vel_ref[:,2])
plt.plot(t,vel[:,2])
plt.legend(['Referance vz','vz'])

plt.subplot(2, 2, 4)
plt.plot(t,vel_ref[:,3])
plt.plot(t,vel[:,3])
plt.legend(['Referance r','r'])
plt.show()

