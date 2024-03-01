import casadi as cs
import numpy as np
#from whoc.controllers.wind_farm_power_tracking_controller import WindFarmPowerDistributingController
from whoc.controllers.wind_farm_power_tracking_controller import WindFarmPowerTrackingController

class HybridController(WindFarmPowerTrackingController): #HybridController(WindFarmPowerDistributingController):

    def __init__(self, interface, input_dict, verbose=False):
        super().__init__(interface, input_dict, verbose=verbose)

        # TO DO grab info from input_dict
        self.ny = 1
        self.nu = len(input_dict['hybrid_controller']['components'])
        self.components = input_dict['hybrid_controller']['components']
        self.dt = input_dict['dt']
        self.gains = input_dict['hybrid_controller']['gains']
        umax = []
        umin = []
        for component in self.components:
            umax.append(input_dict['hybrid_controller']['max_reference'][component])
            umin.append(0.0)
        self.umax = np.array(umax).reshape((self.nu,1))
        self.umin = np.array(umin).reshape((self.nu,1))
        # self.umax = np.array([3500.0]).reshape((self.nu, 1))
        # self.umin = np.array([0.0]).reshape((self.nu, 1))
        self.num_turbines = input_dict['controller']['num_turbines']

        # Initialize time
        self.k = 0

        self.params = {}
        if 'wind' in self.components:
            self.params['wind_speed'] = 0.0
            self.params['load_ref'] = 0.0
            wind_power_vals = input_dict['hybrid_controller']['power_curve']['wind']
            self.wind_power_curve = lambda v, vals=wind_power_vals: self.num_turbines * np.array([np.polyval(vals, v)]).reshape((1, 1))
            a = input_dict['hybrid_controller']['set_point_to_power']['wind']['a']
            b = input_dict['hybrid_controller']['set_point_to_power']['wind']['b']
            self.params['dhw_dPwd'] = {'a': a, 'b': b}

        # Setup hybrid control
        self.setup_control()

    def setup_control(self):

        if 'wind' in self.components:
            dhw_dPwd = lambda x, a=self.params['dhw_dPwd']['a'], b=self.params['dhw_dPwd']['b']: np.exp(a*(x-b))/( 1.0 + np.exp(a*(x-b)))

        # Define outer-loop input-output mapping
        A = lambda w: np.array(dhw_dPwd(w)).reshape((self.nu, 1))
        # H = lambda u, k, A=A, d=d: A @ u + d(k)  # Input-output mapping
        nablaH = lambda u, k, params, A=A: A(params['wind_speed']).T  # Gradient of input-output mapping

        self.test = nablaH

        # Define cost function
        Cv = lambda y, u, k, params: 0.5 * (y - params['load_ref']) ** 2
        nablaC = lambda y, u, k, params: nablaH(u, k, params) @ (y - params['load_ref'])

        # input constraints: (h for equality constraints, g for inequality constraints)
        g0 = lambda y, u, k, params, umax=self.umax: u - umax
        g1 = lambda y, u, k, params, umin=self.umin: umin - u

        g2 = lambda y, u, k, params, power_curve=self.wind_power_curve: u - min(self.umax, power_curve(params['wind_speed']))
        nablag0 = lambda y, u, k, params: np.eye(self.nu)
        nablag1 = lambda y, u, k, params: -np.eye(self.nu)
        nablag2 = lambda y, u, k, params: np.eye(self.nu)

        # Setup constraints
        equal_constraints = {}
        # equal_constraints[0] = h0

        grad_equal_constraints = {}
        # grad_equal_constraints[0] = nablah0

        inequal_constraints = {}
        inequal_constraints[0] = g1
        inequal_constraints[1] = g2
        #inequal_constraints[2] = g0

        grad_inequal_constraints = {}
        grad_inequal_constraints[0] = nablag1
        grad_inequal_constraints[1] = nablag2
        #grad_inequal_constraints[2] = nablag0

        constraints = {'equal': equal_constraints, 'inequal': inequal_constraints}
        grad_constraints = {'equal': grad_equal_constraints, 'inequal': grad_inequal_constraints}

        # Setup gains and integrator

        integrator = Integrator(dt=self.dt)

        # Define initial condition
        # turbine_current_powers = self.measurements_dict["turbine_powers"]
        u0 = self.umax  # np.array([np.sum(turbine_current_powers)])

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
        self.params['load_ref'] = ref
        self.params['wind_speed'] = wind_speed
        print('wind_speed = '+str(wind_speed))
        print('max power =' +str(self.wind_power_curve(wind_speed)))

        print('dhw_dPwd = '+str(self.test(0,0,self.params)))

        # Apply hybrid control after initialization
        if self.k > 6.0:
            supervisory_reference = self.fo_control(y=np.array([[farm_current_power]]), k=self.k, w=self.params )
            print('farm power ref = ' + str(supervisory_reference.flatten()[0]))
            self.turbine_power_references(farm_power_reference=supervisory_reference.flatten()[0])
        else:
            for ii in range(self.nu):
                self.fo_control.uk[ii] = farm_current_power
            self.turbine_power_references()
        self.k += self.dt





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