import numpy as np
import math

class Extended_kalmanFilter:
    def __init__(self,P,Q,R,A):
        self.A = A
        self.H = np.eye(3)
        self.P = P
        self.Q = Q
        self.R = R
        self.r=0.076
        self.lx = 0.165
        self.ly = 0.225
    
    def getB_Matrix(self,theta,type=None):
        rot = np.array([
            [ math.cos(theta),  math.sin(theta),   0],
            [-math.sin(theta),  math.cos(theta),   0],
            [0             ,              0,       1]
        ], dtype=np.float32)
        if (type == "mecanum"):
            J = (self.r/4)*np.array([[1, 1, 1, 1],
                                [-1, 1, 1, -1],
                                [-1/(self.lx+self.ly), 1/(self.lx+self.ly), -1/(self.lx+self.ly), 1/(self.lx+self.ly)]])
        if (type == "omni"):
            J = (self.r/2)*np.array([[-np.sin(theta+np.pi/4),-np.sin(theta+3*np.pi/4),-np.sin(theta+5*np.pi/4),-np.sin(theta+7*np.pi/4)],
                          [np.cos(theta+np.pi/4),np.cos(self.theta+3*np.pi/4),np.cos(theta+5*np.pi/4),np.cos(theta+7*np.pi/4)],
                          [1/(self.lx+self.ly), 1/(self.lx+self.ly), 1/(self.lx+self.ly), 1/(self.lx+self.ly)]])
        B = rot @ J
        return B
    def ekf_predicted(self,state,input,dt,process_noise):
        # State estimate
        B = self.getB_Matrix(state[2],"mecanum")
        # state_pred = np.array([[x],[y],[yaw]]) + process_noise
        state_est = self.A @ state + dt*B @ input + process_noise
        # Predicted estimate
        P_pred = self.A @ self.P @ self.A.T + self.Q

        return state_est,P_pred
    
    def ekf_update(self,state_pred,P_pred,measurement,measurement_noise):

        # Observation model
        Y = self.H @ state_pred + measurement_noise

        #Calculate error from sensor
        Y_error = measurement - Y
        
        # Inovation Covariance
        S = self.H @ P_pred @ self.H.T + self.R

        # Kalman Gain
        K = P_pred @ self.H.T @ (np.linalg.pinv(S))

        # State Update
        State_up = state_pred + K @ Y_error

        # Predicted update
        P_update = (np.identity(3) - (K @ self.H)) @ P_pred

        return State_up, P_update


if __name__ == "__main__":
    EKF = Extended_kalmanFilter()



        
        
        