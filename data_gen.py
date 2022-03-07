import numpy as np
from controller import Controller
from quad_model import Model
import matplotlib.pyplot as plt
from mpc_class import MPC_controller
import argparse
import _pickle as cPickle

parser = argparse.ArgumentParser(description='MPC')
parser.add_argument('--ref', default="12.0,1.0,1.0,0.5" , type=str,
                    help='target vel')
parser.add_argument('--idx', default=0 , type=int,
                    help='index')

args = parser.parse_args()

file_idx=args.idx
ref_vel_arr=args.ref.split(",")
ref_vel=[]



for pose in ref_vel_arr:
    ref_vel.append(float(pose))



quad=Model()
mpc_controller=MPC_controller()

def conf_inputs(ref,x):
    x_nn=np.zeros(9)
    x_nn[0]=ref[0]-x[0]
    x_nn[1]=ref[1]-x[1]
    x_nn[2]=ref[2]-x[2]
    x_nn[3]=x[3]
    x_nn[4]=x[4]
    x_nn[5]=ref[3]-x[5]
    x_nn[6]=x[6]
    x_nn[7]=x[7]
    x_nn[8]=x[8]
    return x_nn


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

x0=quad.x

T=2
dt=1e-2
N=int(T/dt)
t=np.linspace(0,T,N)

xd=ref_vel[0]*np.ones(N)
yd=ref_vel[1]*np.ones(N)
zd=ref_vel[2]*np.ones(N)
psid=ref_vel[3]*np.ones(N)

state_arr=np.zeros((N,9))
state_arr_mpc=np.zeros((N,9))
u_list_mpc=np.zeros((N,4))


mse_arr=np.zeros(N)
index=0
Data_arr=[]
for i in range(N):
    xt=[xd[i],yd[i],zd[i],psid[i]]
    x0=x0[3:12]
    u_mpc=mpc_controller.run_controller(x=x0,x_t=xt)
    u_list_mpc[i,:]=u_mpc
    Data_arr.append(to_dataset(x=conf_inputs(xt,x0),u=u_mpc))       
    x0=quad.run_model(conf_u(u_mpc))
    state_arr[i,:]=x0[3:12]

    

f = open('Dataset/Data1/D0_data.pkl', 'wb')


pickler = cPickle.Pickler(f)

pickler.dump(np.array(Data_arr))



