# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://nrel.github.io/wind-hybrid-open-controller for documentation

import numpy as np
import casadi as cs

from whoc.controllers.controller_base import ControllerBase

POWER_SETPOINT_DEFAULT = 1e9

class WindFarmPowerDistributingController(ControllerBase):
    """
    Evenly distributes wind farm power reference between turbines without 
    feedback on current power generation.
    """
    def __init__(self, interface, input_dict, verbose=False):
        super().__init__(interface, verbose=verbose)

        self.dt = input_dict["dt"]  # Won't be needed here, but generally good to have
        self.n_turbines = input_dict["controller"]["num_turbines"]
        self.turbines = range(self.n_turbines)

        # Set initial conditions
        self.controls_dict = {"power_setpoints": [POWER_SETPOINT_DEFAULT] * self.n_turbines}

        # For startup


    def compute_controls(self):
        if "wind_power_reference" in self.measurements_dict:
            farm_power_reference = self.measurements_dict["wind_power_reference"]
        else:
            farm_power_reference = POWER_SETPOINT_DEFAULT
        
        self.turbine_power_references(farm_power_reference=farm_power_reference)

    def turbine_power_references(self, farm_power_reference=POWER_SETPOINT_DEFAULT):
        """
        Compute turbine-level power setpoints based on farm-level power
        reference signal.
        Inputs:
        - farm_power_reference: float, farm-level power reference signal
        Outputs:
        - None (sets self.controls_dict)
        """

        # Split farm power reference among turbines and set "no value" for yaw angles (Floris not
        # compatible with both power_setpoints and yaw_angles).
        self.controls_dict = {
            "power_setpoints": [farm_power_reference/self.n_turbines]*self.n_turbines,
            "yaw_angles": [-1000]*self.n_turbines
        }

        return None

class WindFarmPowerTrackingController(WindFarmPowerDistributingController):
    """
    Based on controller developed under A2e2g project.

    Inherits from WindFarmPowerDistributingController.
    """

    def __init__(self, interface, input_dict, proportional_gain=1, verbose=False):
        super().__init__(interface, input_dict, verbose=verbose)

        # No integral action for now. beta and omega_n not used.
        beta=0.7
        omega_n=0.01
        integral_gain=0 

        self.K_p = proportional_gain * 1/self.n_turbines
        self.K_i = integral_gain *(4*beta*omega_n)

        # Initialize controller (only used for integral action)
        self.e_prev = 0
        self.u_prev = 0
        self.u_i_prev = 0
        self.ai_prev = [0.33]*self.n_turbines # TODO: different method for anti-windup?
        self.n_saturated = 0 

    def turbine_power_references(self, farm_power_reference=POWER_SETPOINT_DEFAULT):
        """
        Compute turbine-level power setpoints based on farm-level power
        reference signal.
        Inputs:
        - farm_power_reference: float, farm-level power reference signal
        Outputs:
        - None (sets self.controls_dict)
        """
        
        turbine_current_powers = self.measurements_dict["turbine_powers"]
        farm_current_power = np.sum(turbine_current_powers)
        farm_current_error = farm_power_reference - farm_current_power

        self.n_saturated = 0 # TODO: determine whether to use gain scheduling
        if self.n_saturated < self.n_turbines:
            # with self.n_saturated = 0, gain_adjustment = 1
            gain_adjustment = self.n_turbines/(self.n_turbines-self.n_saturated)
        else:
            gain_adjustment = self.n_turbines
        K_p_gs = gain_adjustment*self.K_p
        K_i_gs = gain_adjustment*self.K_i

        # Discretize and apply difference equation (trapezoid rule)
        u_p = K_p_gs*farm_current_error
        u_i = self.dt/2*K_i_gs * (farm_current_error + self.e_prev) + self.u_i_prev

        # Apply integral anti-windup
        eps = 0.0001 # Threshold for anti-windup
        if (np.array(self.ai_prev) > 1/3-eps).all() or \
           (np.array(self.ai_prev) < 0+eps).all():
           u_i = 0
        
        u = u_p + u_i
        delta_P_ref = u

        turbine_power_setpoints = np.array(turbine_current_powers) + delta_P_ref
        
        # set "no value" for yaw angles (Floris not compatible with both 
        # power_setpoints and yaw_angles)
        self.controls_dict = {
            "power_setpoints": list(turbine_power_setpoints),
            "yaw_angles": [-1000]*self.n_turbines
        }

        # Store error, control
        self.e_prev = farm_current_error
        self.u_prev = u
        self.u_i_prev = u_i

        return None

class WindPowerSupervisoryControl(WindFarmPowerTrackingController):
    def __init__(self, interface, input_dict, proportional_gain=1, verbose=False):
        super().__init__(interface, input_dict, proportional_gain=1, verbose=False)

        self.nu = 1
        self.ny = 1

        self.setup_supervisory_control()

    def setup_supervisory_control(self):

        # Define outer-loop input-output mapping
        A = np.array([[1.0]])
        #H = lambda u, k, A=A, d=d: A @ u + d(k)  # Input-output mapping
        nablaH = lambda u, k, A=A: A.T  # Gradient of input-output mapping

        # Define cost function
        Cv = lambda y, u, k, yd: 0.5 * (y - yd) ** 2
        nablaC = lambda y, u, k, yd: nablaH(u, k) @ (y - yd)


        # Output Constraints: (h for equality constraints, g for inequality constraints)
        h0 = lambda y, u, k, yd: y - yd
        nablah0 = lambda y, u, k, yd: nablaH(u,k)@np.eye(self.ny)

        # input constraints: (h for equality constraints, g for inequality constraints)
        umax = np.array([[3500.0]])
        umin = np.array([[0.0]])
        g0 = lambda y, u, k, w, umax=umax: u - umax
        g1 = lambda y, u, k, w, umin=umin: umin - u
        nablag0 = lambda y, u, k, w: np.eye(self.nu)
        nablag1 = lambda y, u, k, w: -np.eye(self.nu)

        # Setup constraints
        equal_constraints = {}
        #equal_constraints[0] = h0

        grad_equal_constraints = {}
        #grad_equal_constraints[0] = nablah0

        inequal_constraints = {}
        inequal_constraints[0] = g0
        inequal_constraints[1] = g1

        grad_inequal_constraints = {}
        grad_inequal_constraints[0] = nablag0
        grad_inequal_constraints[1] = nablag1

        constraints = {'equal': equal_constraints, 'inequal': inequal_constraints}
        grad_constraints = {'equal': grad_equal_constraints, 'inequal': grad_inequal_constraints}

        # Setup gains and integrator
        gains = {'eta': 0.3, 'beta': 1.0}
        dt = 0.5
        integrator = Integrator(dt=dt)

        # Define initial condition
        #turbine_current_powers = self.measurements_dict["turbine_powers"]
        u0 = np.array([[3000.0]]) #np.array([np.sum(turbine_current_powers)])

        self.fo_control = FeedbackOptimization(nablaH=nablaH, grad_cost=nablaC,  constraints=constraints,
                                             grad_constraints=grad_constraints, gains=gains, integrator=integrator,
                                             u0=u0, ny=self.ny)

    def get_reference(self, y, ref):
        print('y = '+str(y))
        print('ref = '+str(ref))
        if y > 0.0:
            return self.fo_control(y=np.array([[y]]), k=0, w=np.array([[ref]]) )[0]
        else:
            return self.fo_control.uk[0]
    def turbine_power_references(self, farm_power_reference=POWER_SETPOINT_DEFAULT):
        """
        Compute turbine-level power setpoints based on farm-level power
        reference signal.
        Inputs:
        - farm_power_reference: float, farm-level power reference signal
        Outputs:
        - None (sets self.controls_dict)
        """

        turbine_current_powers = self.measurements_dict["turbine_powers"]
        print(self.measurements_dict["wind_speed"])
        farm_current_power = np.sum(turbine_current_powers)
        supervisory_reference = self.get_reference(y=farm_current_power, ref=farm_power_reference )
        #supervisory_reference = farm_power_reference
        farm_current_error = supervisory_reference - farm_current_power

        self.n_saturated = 0  # TODO: determine whether to use gain scheduling
        if self.n_saturated < self.n_turbines:
            # with self.n_saturated = 0, gain_adjustment = 1
            gain_adjustment = self.n_turbines / (self.n_turbines - self.n_saturated)
        else:
            gain_adjustment = self.n_turbines
        K_p_gs = gain_adjustment * self.K_p
        K_i_gs = gain_adjustment * self.K_i

        # Discretize and apply difference equation (trapezoid rule)
        u_p = K_p_gs * farm_current_error
        u_i = self.dt / 2 * K_i_gs * (farm_current_error + self.e_prev) + self.u_i_prev

        # Apply integral anti-windup
        eps = 0.0001  # Threshold for anti-windup
        if (np.array(self.ai_prev) > 1 / 3 - eps).all() or \
                (np.array(self.ai_prev) < 0 + eps).all():
            u_i = 0

        u = u_p + u_i
        delta_P_ref = u

        turbine_power_setpoints = np.array(turbine_current_powers) + delta_P_ref

        # set "no value" for yaw angles (Floris not compatible with both
        # power_setpoints and yaw_angles)
        self.controls_dict = {
            "power_setpoints": list(turbine_power_setpoints),
            "yaw_angles": [-1000] * self.n_turbines
        }

        # Store error, control
        self.e_prev = farm_current_error
        self.u_prev = u
        self.u_i_prev = u_i

        return None


class Integrator:

    def __init__(self, dt, integrator_type='Euler'):
        '''Defines forward-euler integrator

        dt: (float) sampling time
        integrator_type: (str) currently only accepts Euler and cont

        '''

        self.dt = dt
        self.integrator_type = integrator_type



    def __call__(self, f, x):
        '''Integrate system f at time k by one time step with x and u and time k using integrator_type method
        f: function evaluated at x to integrate
        x: state
        '''


        if self.integrator_type == 'Euler':
            # Standard Euler integration
            x_next = x + self.dt * f

        if self.integrator_type == 'cont':
            # Treat system as 'continuous'
            x_next = f

        return x_next


class FeedbackOptimization:

    def __init__(self, nablaH, grad_cost, constraints, grad_constraints, gains, integrator, u0, ny):
        ''' Defines the feedback optimization control law '''

        self.nablaH = nablaH
        self.constraints = constraints
        self.grad_constraints = grad_constraints
        self.grad_cost = grad_cost
        self.gains = gains
        self.integrator = integrator
        self.u0 = u0
        self.nu = len(u0)
        self.ny = ny

        # Initialize optimization controller
        self.uk = self.u0
        self.u_window = []
        self.y_window = []


    def setup_optimization(self, k=0, w=0):

        # Define casadi optimization object
        self.opti = cs.Opti()
        self.Theta = self.opti.variable(self.nu, 1)
        self.U = self.opti.parameter(self.nu, 1)
        self.Y = self.opti.parameter(self.ny, 1)

        # Define constraints, equality and inequality
        beta = self.gains['beta']
        for ii in range(len(self.grad_constraints['equal'])):
            constrainti = self.constraints['equal'][ii]
            grad_constrainti = self.grad_constraints['equal'][ii]
            self.opti.subject_to( grad_constrainti(self.Y, self.U, k, w).T@self.Theta == -beta*constrainti(self.Y, self.U, k, w) )

        for ii in range(len(self.grad_constraints['inequal'])):
            constrainti = self.constraints['inequal'][ii]
            grad_constrainti = self.grad_constraints['inequal'][ii]
            self.opti.subject_to( grad_constrainti(self.Y, self.U, k, w).T@self.Theta <= -beta * constrainti(self.Y, self.U, k, w) )

        # Define cost function
        cost_vec = self.Theta + self.grad_cost(self.Y, self.U, k, w)
        self.opti.minimize( cost_vec.T@cost_vec )

        # Define casadi optimization parameters
        p_opts = {"expand": True, "print_time": False, "verbose": False}
        s_opts = {'max_iter': 300,
                  'print_level': 1,
                  'warm_start_init_point': 'yes',
                  'tol': 1e-8,
                  'constr_viol_tol': 1e-8,
                  "compl_inf_tol": 1e-8,
                  "acceptable_tol": 1e-8,
                  "acceptable_constr_viol_tol": 1e-8,
                  "acceptable_dual_inf_tol": 1e-8,
                  "acceptable_compl_inf_tol": 1e-8,
                  }
        self.opti.solver("ipopt", p_opts, s_opts)

    def __call__(self, y, k=0, w=0.0):
        '''Evalute the feedback optimization control law at given state and time

        y = current state
        k = current time

        '''

        # Setup optimization
        self.setup_optimization(k=k, w=w)

        # Initialize optimizer
        self.opti.set_value(self.Y, y)
        eta = self.gains['eta']
        self.y_window.append( y )

        # Compute change in control
        self.opti.set_value(self.U, self.uk)
        try:
            sol = self.opti.solve()
        except:
            print('Failed debug value:')
            print('y = '+str(y))
            print('u = '+str(self.uk))
            print(self.opti.debug.value(self.U))
            raise(SystemExit)
        F = np.array(sol.value(self.Theta)).flatten()
        du = eta*F

        # Integrate to get next point
        self.uk = self.integrator(du.reshape((self.nu, 1)), self.uk)
        self.u_window.append( self.uk )

        return self.uk









