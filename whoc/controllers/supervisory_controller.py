import casadi as cs
import numpy as np
from whoc.controllers.wind_farm_power_tracking_controller import WindFarmPowerDistributingController

class HybridController(WindFarmPowerDistributingController):

    def __init__(self, interface, input_dict, verbose=False):
        super().__init__(interface, input_dict, verbose=verbose)

        # TO DO grab info from input_dict
        self.ny = 1
        self.nu = 1
        self.dt = 0.5
        self.gains = {'eta': 0.4, 'beta': 1.0}
        self.umax = np.array([3500.0]).reshape((self.nu,1))
        self.umin = np.array([0.0]).reshape((self.nu,1))

        self.k = 0

        self.setup_control()

    def setup_control(self):

        # Define outer-loop input-output mapping
        A = np.array([[1.0]])
        # H = lambda u, k, A=A, d=d: A @ u + d(k)  # Input-output mapping
        nablaH = lambda u, k, A=A: A.T  # Gradient of input-output mapping

        # Define cost function
        Cv = lambda y, u, k, yd: 0.5 * (y - yd) ** 2
        nablaC = lambda y, u, k, yd: nablaH(u, k) @ (y - yd)

        # Output Constraints: (h for equality constraints, g for inequality constraints)
        h0 = lambda y, u, k, yd: y - yd
        nablah0 = lambda y, u, k, yd: nablaH(u, k) @ np.eye(self.ny)

        # input constraints: (h for equality constraints, g for inequality constraints)
        g0 = lambda y, u, k, w, umax=self.umax: u - umax
        g1 = lambda y, u, k, w, umin=self.umin: umin - u
        nablag0 = lambda y, u, k, w: np.eye(self.nu)
        nablag1 = lambda y, u, k, w: -np.eye(self.nu)

        # Setup constraints
        equal_constraints = {}
        # equal_constraints[0] = h0

        grad_equal_constraints = {}
        # grad_equal_constraints[0] = nablah0

        inequal_constraints = {}
        inequal_constraints[0] = g0
        inequal_constraints[1] = g1

        grad_inequal_constraints = {}
        grad_inequal_constraints[0] = nablag0
        grad_inequal_constraints[1] = nablag1

        constraints = {'equal': equal_constraints, 'inequal': inequal_constraints}
        grad_constraints = {'equal': grad_equal_constraints, 'inequal': grad_inequal_constraints}

        # Setup gains and integrator

        integrator = Integrator(dt=self.dt)

        # Define initial condition
        # turbine_current_powers = self.measurements_dict["turbine_powers"]
        u0 = np.array([[3000.0]])  # np.array([np.sum(turbine_current_powers)])

        self.fo_control = FeedbackOptimization(nablaH=nablaH, grad_cost=nablaC, constraints=constraints,
                                               grad_constraints=grad_constraints, gains=self.gains, integrator=integrator,
                                               u0=u0, ny=self.ny)

    def compute_controls(self):

        # Get power and windspeed measurements
        turbine_current_powers = self.measurements_dict["turbine_powers"]
        farm_current_power = np.sum(turbine_current_powers)
        wind_speed = self.measurements_dict["wind_speed"]

        # TO DO: Replace with desired load reference
        ref = self.measurements_dict["wind_power_reference"]

        # Apply hybrid control after initialization
        if self.k > 2.0:
            supervisory_reference = self.fo_control(y=np.array([[farm_current_power]]), k=self.k, w=np.array([[ref]]) )
        else:
            supervisory_reference = self.fo_control.uk
        self.k += self.dt

        print('farm power ref = '+str(supervisory_reference.flatten()[0]))
        self.turbine_power_references(farm_power_reference=supervisory_reference.flatten()[0])



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

    def __call__(self, y, k, w):
        '''Evalute the feedback optimization control law at given state and time

        y = current state
        k = current time
        w = reference load power

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