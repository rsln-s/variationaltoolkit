# QAOA parameter optimization using a Coordinate Search Based Method (Compass Search)
from scipy.optimize import minimize
import numpy as np
import sys
import copy
from qiskit.aqua.components.optimizers import Optimizer
from variationaltoolkit.utils import allclose_periodical

#Requirements:
#params['n_iter'] = number of function evaluation
#
#Sweep through angles randomly
#For each angle k,
#    evaluate the circuit 3 times
#        using initial angle set
#        using initial angle set, but change angle[k] to angle[k] + perturbation
#        using initial angle set, but change angle[k] to angle[k] - perturbation
#    using these evaluations
#        fit a function f(theta) = a1*cos(theta - a2) + a3
#        theta_new = argmin f(theta)
#    if angleset & theta_new improve objective
#        angle[k] = theta_new


class SequentialOptimizer(Optimizer):

    CONFIGURATION = {
        'name': 'SequentialOptimizer',
        'description': 'SequentialOptimizer Optimizer',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'sequentialoptimizer_schema',
            'type': 'object',
            'properties': {
                'maxiter': {
                    'type': 'integer',
                    'default': 1000
                },
                'disp': {
                    'type': 'boolean',
                    'default': False
                },
            },
            'additionalProperties': False
        },
        'support_level': {
            'gradient': Optimizer.SupportLevel.ignored,
            'bounds': Optimizer.SupportLevel.ignored,
            'initial_point': Optimizer.SupportLevel.required
        },
        'options': ['maxiter', 'disp'],
        'optimizer': ['local']
    }


    def __init__(self, maxiter = 1000, disp=False, atol=1e-5):
        """
        Constructor.

        Args:
            maxiter: Maximum number of function evaluations.
            disp: Set to True to print convergence messages.
            atol: absolute tolerance on parameter change
        """

        super().__init__()
        self.coef = np.zeros((3,))
        self.maxeval = maxiter 
        self.evalused = 0 
        self.disp = disp
        self.objV = sys.maxsize
        self.atol = atol


    def get_support_level(self):
        return {
            'gradient': Optimizer.SupportLevel.ignored,
            'bounds': Optimizer.SupportLevel.ignored,
            'initial_point': Optimizer.SupportLevel.required
        }
    # Main Optimization Function
    # Performs Compass Search / Coordinate Search Method
    def optimize(self, num_parameters, objective, gradient_function=None,
                 variable_bounds=None, initial_point = None):

        #initialize the angles
        angles = initial_point 

        while True:
            #iterate over parameters, in a random order
            oldAngles = copy.deepcopy(angles)
            for p in np.random.permutation(num_parameters):

                #add/subtract perturbation on parameter p
                #pert = np.random.uniform(-np.pi + 0.25, np.pi - 0.25, 1)[0]
                pert = np.pi / 2 
                angleP_add_pert = np.remainder(np.asarray([angles[p]+pert]), np.pi*2)
                angleP_sub_pert = np.remainder(np.asarray([angles[p]-pert]), np.pi*2)

                #EVALUATE CIRCUIT AT THREE POINTS, with ANGLE[P]
                #updated as: ANGLE[P], ANGLE[P]+Perturbation, ANGLE[P]-Perturbation
                angles_1 = angles
                angles_2 = np.concatenate((angles[0:p],angleP_add_pert,angles[p+1:num_parameters]),axis = 0)
                angles_3 = np.concatenate((angles[0:p],angleP_sub_pert,angles[p+1:num_parameters]),axis = 0)

                f1 = objective(angles_1)
                f2 = objective(angles_2)
                f3 = objective(angles_3)
                #We have done three evaluations to estimate a1,a2,a3
                self.evalused += 3

                #Using the three evaluations, we fit a function of the form f(theta) = a1*cos(theta-a2)+a3
                #and min_theta f(theta)
                newAngle = self.fit_minimize([f1,f2,f3],[angles[p],angles_2[p],angles_3[p]])

                #if process succeeded
                if (newAngle != None):
                    newAngle = np.remainder(newAngle, np.pi*2)
                    newAngles = np.concatenate((angles[0:p],newAngle,angles[p+1:num_parameters]),axis = 0)
                    newObjV = objective(newAngles)
                    #We spent one evaluation to check if we have improved the obj funtion value
                    self.evalused += 1
                    if (newObjV<self.objV):
                        #set angle[p] to new theta because this update has reduced the objective value
                        angles[p]=newAngle
                        self.objV=newObjV

                if self.disp:
                    print(self.objV)
                if self.maxeval - self.evalused <= 0:
                    break

            #if sweeping through the parameters did not update any of them, terminate optimization
            if allclose_periodical(angles,oldAngles, 0, 2*np.pi, atol=self.atol):
                break
        return angles, self.objV, self.evalused 


    #this function finds the sum of squared loss
    # \sigma_i (f_i - a1*cos(x_i - a2)+a3)**2
    def squared_loss_model(self,F,X,a1,a2,a3):
        temp = np.asarray(F) - self.cosine_model(X,a1,a2,a3)
        return np.dot(temp,temp)

    #the sinusoidal model we fit
    def cosine_model(self,X,a1,a2,a3):
        return a1*np.cos(np.asarray(X)-a2) + a3

    #the sinuosoidal model evaluated at theta, and optimal a1,a2,a3
    def cosine_function(self,theta):
        return self.cosine_model(theta,self.coef[0],self.coef[1],self.coef[2])


    def fit_minimize(self,F, X):
        #F is a 3x1 array with three evaluations of the circuit at different anlge_P
        #We will fit a function of the form f(x) = a1*cos(x-a2)+a3
        #And then we will minimize f(x) returning argmin_x f(x)
        def coef_loss(A):
            return self.squared_loss_model(F,X,A[0],A[1],A[2])

        #randomly select initial values for a1,a2,a3
        x0 = np.random.uniform(-np.pi + 0.25, np.pi - 0.25, 3)
        minProb1 = minimize(coef_loss, x0, method='BFGS', tol=1e-6)

        if minProb1.success:
            self.coef = minProb1.x


        #having fitted the function, we now have a1*cos(x-a1)+a3, which we minimize w.r.t x
        ang0 = np.random.uniform(-np.pi + 0.25, np.pi - 0.25, 1)
        minProb2 = minimize(self.cosine_function, ang0, method='BFGS', tol=1e-6)

        if minProb2.success:
            ang_opt = minProb2.x
        else:
            ang_opt = None 

        return ang_opt
