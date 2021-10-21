"""
Written by Dr. Elishai Ezra Tsur 
@ The Neuro-Biomorphoc Engineering Lab (NBEL-lab.com)
@ The Open University of Israel
@ 15.9.2020
"""

from enum import Enum
import sympy as sp
import numpy as np

ARM_ACCURACY_IN_METER = .001

class Optimizer(Enum):
    """ Designation of an optimization method for inverse kinematic
    
    We support two optimization methods for inverse kinematic:
    1. Standard resolved motion (STD): Based on Pseudo-inversed jacobian
    2. Dampened least squares method (DLS) or the Levenberg–Marquardt algorithm: 
        see https://en.wikipedia.org/wiki/Levenberg–Marquardt_algorithm for a detailed description
    """

    STD = 1 
    DLS = 2 

class viper300:
    """ Describe the Viperx200 6DOF robotic arm by Trossen Robotic
    
    The class provides the properties, transformation matrices and jacobian of the ViperX 300.
    The arm is described in: https://www.trossenrobotics.com/viperx-300-robot-arm-6dof.aspx
    """
    
    def __init__ (self):
        
        # Robots' joints
        self.n_joints = 5
        self.q0 = sp.Symbol('q0') 
        self.q1 = sp.Symbol('q1') 
        self.q2 = sp.Symbol('q2') 
        self.q3 = sp.Symbol('q3')
        self.q4 = sp.Symbol('q4')
        
        # length of the robots' links
        self.l1 = (127-9)      * 1e-3
        self.l2 = (427-127)    * 1e-3
        self.l3 = (60)         * 1e-3
        self.l4 = (253-60)     * 1e-3
        self.l5 = (359-253)    * 1e-3
        self.l6 = (567-359)    * 1e-3
        
        
        # Calculate the transformation matrix for base to EE in operational space
        self.T = self.calculate_Tx().subs([('l1', self.l1), 
                                           ('l2', self.l2), 
                                           ('l3', self.l3), 
                                           ('l4', self.l4), 
                                           ('l5', self.l5),
                                           ('l6', self.l6)])
        
        # Calculate the Jacobian matrix for the EE
        self.J = self.calculate_J().subs([('l1', self.l1), 
                                          ('l2', self.l2), 
                                          ('l3', self.l3), 
                                          ('l4', self.l4), 
                                          ('l5', self.l5), 
                                          ('l6', self.l6)])
    
    def calculate_Tx(self):
        """ Calculate the transformation matrix for base to EE in operational space """
        
        q0 = self.q0
        q1 = self.q1
        q2 = self.q2
        q3 = self.q3
        q4 = self.q4
        
        l1 = sp.Symbol('l1')
        l2 = sp.Symbol('l2')
        l3 = sp.Symbol('l3')
        l4 = sp.Symbol('l4')
        l5 = sp.Symbol('l5')
        l6 = sp.Symbol('l6')
        
        T01 = sp.Matrix([[sp.cos(q0),  0, sp.sin(q0), 0],
                         [0,           1, 0,          0],
                         [-sp.sin(q0), 0, sp.cos(q0), 0],
                         [0,           0, 0,          1]])

        T12 = sp.Matrix([[sp.cos(q1), -sp.sin(q1), 0, 0 ],
                         [sp.sin(q1),  sp.cos(q1), 0, l1],
                         [0,           0         , 1, 0 ],
                         [0,           0         , 0, 1 ]])

        T23 = sp.Matrix([[sp.cos(q2), -sp.sin(q2), 0, l3 ],
                         [sp.sin(q2),  sp.cos(q2), 0, l2],
                         [0,           0         , 1, 0 ],
                         [0,           0         , 0, 1 ]])
        
        T34 = sp.Matrix([[1, 0,           0,          l4 ],
                         [0, sp.cos(q3), -sp.sin(q3), 0],
                         [0, sp.sin(q3),  sp.cos(q3), 0],
                         [0, 0,           0,          1 ]])

        T45 = sp.Matrix([[sp.cos(q4), -sp.sin(q4), 0, l5],
                         [sp.sin(q4),  sp.cos(q4), 0, 0 ],
                         [0,           0         , 1, 0 ],
                         [0,           0         , 0, 1 ]])

        T56 = sp.Matrix([[1, 0, 0, l6],
                         [0, 1, 0, 0 ],
                         [0, 0, 1, 0 ],
                         [0, 0, 0, 1 ]])

        T = T01 * T12 * T23 * T34 * T45 * T56
        x = sp.Matrix([0, 0, 0, 1])
        Tx = T * x
        
        return Tx
    
    def calculate_J(self):
        """ Calculate the Jacobian matrix for the EE """
    
        q = [self.q0, self.q1, self.q2, self.q3, self.q4]
        J = sp.Matrix.ones(3, 5)
        for i in range(3):     # x, y, z
            for j in range(5): # Five joints
                # Differentiate and simplify
                J[i, j] = sp.simplify(self.T[i].diff(q[j]))
                
        return J
    
    def get_xyz_symbolic(self, q):
        """ Calculate EE location in operational space by solving the for Tx symbolically """
        
        return np.array(self.T.subs([('q0', q[0]), 
                                     ('q1', q[1]), 
                                     ('q2', q[2]), 
                                     ('q3', q[3]),
                                     ('q4', q[4])]), dtype='float')

    def calc_J_symbolic(self, q): 
        """ Calculate the jacobian symbolically """
        return np.array(self.J.subs([('q0', q[0]), 
                                     ('q1', q[1]), 
                                     ('q2', q[2]), 
                                     ('q3', q[3]),
                                     ('q4', q[4])]), dtype='float')
    
    def get_xyz_numeric(self, q):

        c0 = np.cos(q[0])
        c1 = np.cos(q[1])
        c2 = np.cos(q[2])
        c3 = np.cos(q[3])
        c4 = np.cos(q[4])

        s0 = np.sin(q[0])
        s1 = np.sin(q[1])
        s2 = np.sin(q[2])
        s3 = np.sin(q[3])
        s4 = np.sin(q[4])

        return np.array([[0.208*((-s1*c0*c2 - s2*c0*c1)*c3 + s0*s3)*s4 + 
            0.208*(-s1*s2*c0 + c0*c1*c2)*c4 - 0.299*s1*s2*c0 - 
            0.3*s1*c0 + 0.299*c0*c1*c2 + 0.06*c0*c1],
            [0.208*(-s1*s2 + c1*c2)*s4*c3 + 
            0.208*(s1*c2 + s2*c1)*c4 + 
            0.299*s1*c2 + 0.06*s1 + 0.299*s2*c1 + 0.3*c1 + 0.118],
            [0.208*((s0*s1*c2 + s0*s2*c1)*c3 + s3*c0)*s4 + 
            0.208*(s0*s1*s2 - s0*c1*c2)*c4 + 0.299*s0*s1*s2 + 
            0.3*s0*s1 - 0.299*s0*c1*c2 - 0.06*s0*c1], [1]], dtype='float')
    
    def calc_J_numeric(self, q):
        """ Calculate the Jacobian for q symbolically
         
         Equation was derived symbolically and was then written here manually.
         Nuerical evaluation works faster then symbolically. 
         """
        
        c0 = np.cos(q[0])
        c1 = np.cos(q[1])
        c2 = np.cos(q[2])
        c3 = np.cos(q[3])
        c4 = np.cos(q[4])
        
        s0 = np.sin(q[0])
        s1 = np.sin(q[1])
        s2 = np.sin(q[2])
        s3 = np.sin(q[3])
        s4 = np.sin(q[4])

        s12  = np.sin(q[1] + q[2])
        c12  = np.cos(q[1] + q[2])

        
        return np.array([[0.3*s0*s1 + 0.208*s0*s4*s12*c3 - 0.06*s0*c1 - 0.208*s0*c4*c12 - 
                          0.299*s0*c12 + 0.208*s3*s4*c0, -(0.06*s1 + 0.208*s4*c3*c12 + 
                          0.208*s12*c4 + 0.299*s12 + 0.3*c1)*c0,
                          -(0.208*s4*c3*c12 + 0.208*s12*c4 + 0.299*s12)*c0,
                          0.208*(s0*c3 + s3*s12*c0)*s4, 0.208*(s0*s3 - s12*c0*c3)*c4 - 
                          0.208*s4*c0*c12],
                         [0,-0.3*s1 - 0.208*s4*s12*c3 + 0.06*c1 + 0.208*c4*c12 + 0.299*c12,
                          -0.208*s4*s12*c3 + 0.208*c4*c12 + 0.299*c12,-0.208*s3*s4*c12,
                          -0.208*s4*s12 + 0.208*c3*c4*c12],
                         [-0.208*s0*s3*s4 + 0.3*s1*c0 + 0.208*s4*s12*c0*c3 - 0.06*c0*c1 - 
                          0.208*c0*c4*c12 -0.299*c0*c12,(0.06*s1 + 0.208*s4*c3*c12 + 0.208*s12*c4 + 
                          0.299*s12 + 0.3*c1)*s0,(0.208*s4*c3*c12 + 0.208*s12*c4 + 
                          0.299*s12)*s0, -0.208*(s0*s3*s12 - c0*c3)*s4,
                          0.208*(s0*s12*c3 + s3*c0)*c4 + 0.208*s0*s4*c12]], dtype='float')
    
class widow200:
    """ Describe the WidowX 200 5DOF robotic arm by Trossen Robotic
    
    The class provides the properties, transformation matrices and jacobian of the WidowX 200.
    The arm is described in: https://www.trossenrobotics.com/widowx-200-robot-arm.aspx
    """
    
    def __init__ (self):
        
        self.n_joints = 4
        
        self.l1 = (110.25-9)      * 1e-3
        self.l2 = (310.81-110.25) * 1e-3
        self.l3 = (50-14.1-15)    * 1e-3
        self.l4 = (200)           * 1e-3
        self.l5 = (422.43-250)    * 1e-3
        
        self.q0 = sp.Symbol('q0') 
        self.q1 = sp.Symbol('q1') 
        self.q2 = sp.Symbol('q2') 
        self.q3 = sp.Symbol('q3')
        
        self.T = self.calculate_Tx().subs([('l1', self.l1), 
                                           ('l2', self.l2), 
                                           ('l3', self.l3), 
                                           ('l4', self.l4), 
                                           ('l5', self.l5)])

        self.J = self.calculate_J().subs([('l1', self.l1), 
                                          ('l2', self.l2), 
                                          ('l3', self.l3), 
                                          ('l4', self.l4), 
                                          ('l5', self.l5)])
    
    def calculate_Tx(self):
        
        q0 = self.q0
        q1 = self.q1
        q2 = self.q2
        q3 = self.q3
        
        l1 = sp.Symbol('l1')
        l2 = sp.Symbol('l2')
        l3 = sp.Symbol('l3')
        l4 = sp.Symbol('l4')
        l5 = sp.Symbol('l5')
        
        T01 = sp.Matrix([[sp.cos(q0),  0, sp.sin(q0), 0],
                         [0,           1, 0,          0],
                         [-sp.sin(q0), 0, sp.cos(q0), 0],
                         [0,           0, 0,          1]])

        T12 = sp.Matrix([[sp.cos(q1), -sp.sin(q1), 0, 0 ],
                         [sp.sin(q1),  sp.cos(q1), 0, l1],
                         [0,           0         , 1, 0 ],
                         [0,           0         , 0, 1 ]])

        T23 = sp.Matrix([[sp.cos(q2), -sp.sin(q2), 0, l3 ],
                         [sp.sin(q2),  sp.cos(q2), 0, l2],
                         [0,           0         , 1, 0 ],
                         [0,           0         , 0, 1 ]])

        T34 = sp.Matrix([[sp.cos(q3), -sp.sin(q3), 0, l4],
                         [sp.sin(q3),  sp.cos(q3), 0, 0 ],
                         [0,           0         , 1, 0 ],
                         [0,           0         , 0, 1 ]])

        T45 = sp.Matrix([[1, 0, 0, l5],
                         [0, 1, 0, 0 ],
                         [0, 0, 1, 0 ],
                         [0, 0, 0, 1 ]])

        T = T01 * T12 * T23 * T34 * T45
        x = sp.Matrix([0, 0, 0, 1])
        Tx = T * x
        
        return Tx
    
    def calculate_J(self):
    
        q = [self.q0, self.q1, self.q2, self.q3]
        J = sp.Matrix.ones(3, 4)
        for i in range(3):     # x, y, z
            for j in range(4): # Four joints
                J[i, j] = sp.simplify(self.T[i].diff(q[j]))
                
        return J
        
    def get_xyz_symbolic(self, q):
        
        return np.array(self.T.subs([('q0', q[0]), 
                                     ('q1', q[1]), 
                                     ('q2', q[2]), 
                                     ('q3', q[3])]), dtype='float')

    def calc_J_symbolic(self, q): 
        
        return np.array(J_eval.subs([('q0', q[0]), 
                                     ('q1', q[1]), 
                                     ('q2', q[2]), 
                                     ('q3', q[3])]), dtype='float')
    
    def get_xyz_numeric(self, q):

        c0 = np.cos(q[0])
        c1 = np.cos(q[1])
        c2 = np.cos(q[2])
        c3 = np.cos(q[3])
        s0 = np.sin(q[0])
        s1 = np.sin(q[1])
        s2 = np.sin(q[2])
        s3 = np.sin(q[3])

        return np.array([[0.17243*(-s1*s2*c0 + c0*c1*c2*c3  + 0.17243*(-s1*c0*c2 - s2*c0*c1))*s3 - 
                          0.2*s1*s2*c0 - 0.20056*s1*c0 + 0.2*c0*c1*c2 + 0.0209*c0*c1        ],
                         [0.17243*(-s1*s2    + c1*c2)*s3    + 0.17243*(s1*c2 + s2*c1)*c3         + 
                          0.2*s1*c2    + 0.0209*s1     + 0.2*s2*c1    + 0.20056*c1 + 0.10125],
                         [0.17243*(s0*s1*s2  - s0*c1*c2)*c3 + 0.17243*(s0*s1*c2 + s0*s2*c1)*s3   + 
                          0.2*s0*s1*s2 + 0.20056*s0*s1 - 0.2*s0*c1*c2 - 0.0209*s0*c1        ],
                         [1]], dtype='float')

    def calc_J_numeric(self, q):

        c0 = np.cos(q[0])
        c1 = np.cos(q[1])
        c2 = np.cos(q[2])
        c3 = np.cos(q[3])
        s0 = np.sin(q[0])
        s1 = np.sin(q[1])
        s2 = np.sin(q[2])
        s3 = np.sin(q[3])

        s12  = np.sin(q[1] + q[2])
        c12  = np.cos(q[1] + q[2])
        s123 = np.sin(q[1] + q[2] + q[3])
        c123 = np.cos(q[1] + q[2] + q[3])

        return np.array([[(0.20056*s1 - 0.0209*c1 - 0.2*c12    - 0.17243*c123)*s0, 
                          -(0.0209*s1 + 0.2*s12 + 0.17243*s123  + 0.20056*c1)  *c0,
                          -(0.2*s12 + 0.17243*s123)*c0, -0.17243*s123*c0],
                         [ 0, -0.20056*s1 + 0.0209*c1 + 0.2*c12 + 0.17243*c123,
                           0.2*c12 + 0.17243*c123,0.17243*c123],
                         [(0.20056*s1 - 0.0209*c1 - 0.2*c12 - 0.17243*c123)*c0,
                          (0.0209*s1 + 0.2*s12 + 0.17243*s123 + 0.20056*c1)*s0,
                          (0.2*s12 + 0.17243*s123)*s0, 0.17243*s0*s123]], dtype='float')

def goto_target (arm, target, optimizer = Optimizer.DLS):
    """ Giving arm object, a target and optimizer, provides the required set of control signals 
    
    Returns the optimizing trajectory, error trace and arm configurazion to achieve the target.
    Target is defined in relative to the EE null position
    """
    
    result = False
    q = np.array([[0]*arm.n_joints], dtype='float').T # Zero configuration of the arm
    xyz_c = (arm.get_xyz_numeric(q))[:-1]             # Current operational position of the arm 
    #xyz_t = xyz_c + target                            # Target operational position
    xyz_t =  target                                    # Target operational position

    count = 0
    trajectory  = []
    error_tract = []


    while count < 200:

        xyz_c = (arm.get_xyz_numeric(q))[:-1]      # Get current EE position
        trajectory.append(xyz_c)                   # Store to track trajectory
        xyz_d = xyz_t - xyz_c                      # Get vector to target
        error = np.sqrt(np.sum(xyz_d**2))          # Distance to target
        error_tract.append(error)                  # Store distance to track error

        if  error < ARM_ACCURACY_IN_METER:
            result = True
            break        

        kp = 0.1                                   # Proportional gain term
        ux = xyz_d * kp                            # direction of movement
        J_x = arm.calc_J_numeric(q)                # Calculate the jacobian

        # Solve inverse kinematics accorting to the designated optimizaer
        if optimizer is Optimizer.STD: # Standard resolved motion
            u = np.dot(np.linalg.pinv(J_x), ux)

        elif optimizer is Optimizer.DLS: # Dampened least squares method
            u = np.dot(J_x.T, np.linalg.solve(np.dot(J_x, J_x.T) + np.eye(3) * 0.001, ux))

        q += u 
        count += 1
    #end       
        # Stop when within 1mm accurancy (arm mechanical accuracy limit)
        # if  error < ARM_ACCURACY_IN_METER:
        #     result = True
        #     break
    
#    print('Arm config: {}, with error: {}, achieved @ step: {}, RESULT: {}'.format(
#        np.rad2deg(q.T).astype(int), error, count, result))

    # !!!!! Shift all angles to be positive and then between 0-2pi !!!!!
    q = q + 1000*np.pi
    q = q % (2*np.pi)

    return q, trajectory, error_tract, xyz_t, result, count

###############################################################
# Forward Kinematics using angles in Q
###############################################################    
def FK(arm, q):
    #print(q)
    xyz_c = (arm.get_xyz_numeric(q))[:-1]             # Current operational position of the arm 
    return xyz_c

###############################################################
# Main
############################################################### 
"""
arm   = viper300()
q, trajectory, error_tract, xyz_t, result = goto_target(arm, np.array([[-0.25, 0.25, 0.25]]).T, optimizer = Optimizer.STD)

if result == False:
    print('Damn')

dbg =1
"""