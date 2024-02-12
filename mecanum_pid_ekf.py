import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.random import randn
from pid_controller import PID_controller
from EKF import Extended_kalmanFilter
from kinematic_mecanum import mecanum

dt=0
rr=2
w=np.pi/20
yaw=(np.pi)*dt
sim_time =430
sampling_time = 0.01
lx = 0.165
ly = 0.225 
r = 0.076
# Simulatio parameters
x_start = 0.0
y_start = 2.0
theta_start = 0.0
x_end = 5.0
y_end = 5.0
yaw_end =0.785
process_noise = 0.001*randn(3,1)
measurement_noise = 0.5*randn(3,1)
# Gain PID of x
kp_x = 8.0
ki_x = 0.02
kd_x = 0.04

# Gain PID of y
kp_y = 8.0
ki_y = 0.02
kd_y = 0.04

# Gain PID of theta
kp_yaw = 5.0
ki_yaw = 0.01
kd_yaw = 0.04
mec=mecanum(r,lx,ly)

#Gain EKF
Q = 1*np.diag([0.1, 0.1, 0.1])**2
R = 20*np.diag([1, 1, 1])**2
P = np.diag([0.1,0.1,0.1])
A = np.diag([1,1,1])
## Create PID for each x, y, yaw
alpha = 0.7
integral_min = -3.0
integral_max = 3.0
output_min = -3.0
output_max = 3.0
pid_x=PID_controller(kp_x,ki_x,kd_x,sampling_time,alpha,integral_min,integral_max,output_min,output_max)
pid_y=PID_controller(kp_y,ki_y,kd_y,sampling_time,alpha,integral_min,integral_max,output_min,output_max)
pid_yaw=PID_controller(kp_yaw,ki_yaw,kd_yaw,sampling_time,alpha,integral_min,integral_max,output_min,output_max)
# EKF
EKF = Extended_kalmanFilter(P,Q,R,A)
#Test trajectory normal
ref_path = np.array([x_end, y_end, yaw_end], dtype=np.float32)


def plot_arrow(x, y, yaw, length=0.025, width=0.3, fc="b", ec="k"):
    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * np.cos(yaw), length * np.sin(yaw),
            fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)
def calc_Circular_curve(r,t,w,yaw):
    x=r*np.sin(w*t)
    y=r*np.cos(w*t)
    theta=yaw
    return x,y,theta

current_x = x_start
current_y = y_start
current_yaw = theta_start
#calculate circle
x_ref,y_ref,theta_ref=calc_Circular_curve(rr,dt,w,yaw)
error_x = [x_ref-current_x]
error_y = [y_ref-current_y]
error_yaw = [theta_ref-current_yaw]
ax=rr*np.sin(np.linspace(0, 2*np.pi, 100))
ay=rr*np.cos(np.linspace(0, 2*np.pi, 100))
# Save history

hist_x = [current_x]
hist_y = [current_y]
hist_yaw = [current_yaw]
#calculate error
# error_x=[ref_path[0]-current_x]
# error_y=[ref_path[1]-current_y]
# error_yaw=[ref_path[2]-current_yaw]
# noise
noise_x = [current_x]
noise_y = [current_y]
noise_yaw = [current_yaw]

if __name__ == "__main__":
    for t in range(sim_time):
        # Generate path
        x_ref,y_ref,theta_ref=calc_Circular_curve(rr,dt,w,yaw)

        # Calculate error
        error_x.append(x_ref-current_x)
        error_y.append(y_ref-current_y)
        error_yaw.append(theta_ref-current_yaw)

        ## Apply PID
        output_vx = pid_x.calculate_PID(error_x)
        output_vy = pid_y.calculate_PID(error_y)
        output_omega = pid_yaw.calculate_PID(error_yaw)

        # 
        V = np.sqrt(output_vx**2+output_vy**2)
        w1,w2,w3,w4 = mec.inverse_kinematic(output_vx,output_vy,output_omega)
        x, y, theta = mec.discrete_state(current_x,current_y,current_yaw,w1,w2,w3,w4,sampling_time)
        
        # EKF
        measurement = np.array([[x],[y],[theta]])
        measurement_no = measurement + measurement_noise
        state = np.array([[current_x],[current_y],[current_yaw]])
        input = np.array([[w1],[w2],[w3],[w4]])
        State_pred,P_pred = EKF.ekf_predicted(state,input,sampling_time,process_noise)
        State_Update,P_update = EKF.ekf_update(State_pred,P_pred,measurement,measurement_noise)
        current_x = State_Update[0,0]
        current_y = State_Update[1,0]
        current_yaw = State_Update[2,0]

        # Update
        P = P_update
        state = State_Update
        noise_x.append(measurement_no[0,0])
        noise_y.append(measurement_no[1,0])
        noise_yaw.append(measurement_no[2,0])

        plt.clf()
        plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
        plot_arrow(current_x, current_y, current_yaw)
        plt.plot(ax, ay, marker="x", color="blue", label="Input Trajectory")
        plt.plot(hist_x, hist_y, color="red", label="PID Track")
        plt.plot(noise_x, noise_y,'.g', label="Noise")
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title("Velocity of robot [m/sec]:" + str(round(V,2)))
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.pause(0.0001)
        dt+=0.1
