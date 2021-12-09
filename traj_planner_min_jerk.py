import numpy as np


class Traj:
    def __init__(self):
        self.c=np.zeros((6,4))
        

    def __calculate_A(self,T):
        A=np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[1,T,pow(T,2),pow(T,3),pow(T,4),pow(T,5)],
        [0,1,2*T,3*pow(T,2),4*pow(T,3),5*pow(T,4)],[0,0,2,6*T,12*pow(T,2),20*pow(T,3)]])
        return A

    def __calculate_xm(self,x_initial,x_final,v_initial,v_final,a_initial,a_final):
        xm=np.array([x_initial,x_final,v_initial,v_final,a_initial,a_final])
        return xm

    def __calculate_c(self,xm,A):
        c=np.zeros((6,4))
        for i in range(4):
            c[:,i]=np.matmul(np.linalg.inv(A),xm[:,i].reshape((6,1))).reshape((1,6))
        return c

    def find_traj(self,x_initial,x_final,v_initial,v_final,a_initial,a_final,T):
        A=self.__calculate_A(T)
        xm=self.__calculate_xm(x_initial,x_final,v_initial,v_final,a_initial,a_final)
        self.c=self.__calculate_c(xm,A)
    
    def __calculate_ref(self,c,t):
        s=0
        for i in range(6):
            s+=c[i]*t[i]
        return s

    def get_traj(self,t):
        t_arr_1=[1,t,pow(t,2),pow(t,3),pow(t,4),pow(t,5)]
        t_arr_2=[0,1,2*t,3*pow(t,2),4*pow(t,3),5*pow(t,4)]
        t_arr_3=[0,0,2,6*t,12*pow(t,2),20*pow(t,3)]

        c1=self.c[:,0]
        c2=self.c[:,1]
        c3=self.c[:,2]
        c4=self.c[:,3]

        xt=self.__calculate_ref(c1,t_arr_1)
        vxt=self.__calculate_ref(c1,t_arr_2)
        axt=self.__calculate_ref(c1,t_arr_3)
        yt=self.__calculate_ref(c2,t_arr_1)
        vyt=self.__calculate_ref(c2,t_arr_2)
        ayt=self.__calculate_ref(c2,t_arr_3)
        zt=self.__calculate_ref(c3,t_arr_1)
        vzt=self.__calculate_ref(c3,t_arr_2)
        azt=self.__calculate_ref(c3,t_arr_3)
        psit=self.__calculate_ref(c4,t_arr_1)
        rt=self.__calculate_ref(c4,t_arr_2)
        rdt=self.__calculate_ref(c4,t_arr_3)

        return [[xt,yt,zt,psit],[vxt,vyt,vzt,rt],[axt,ayt,azt,rdt]]

traj=Traj()
traj.find_traj([0,0,0,0],[5,0,0,0],[0,0,0,0],[2,0,0,0],[0,0,0,0],[0,0,0,0],3)
print(traj.get_traj(0.0)[1])
print(traj.get_traj(1.0)[1])
print(traj.get_traj(2.0)[1])
print(traj.get_traj(3.0)[1])

