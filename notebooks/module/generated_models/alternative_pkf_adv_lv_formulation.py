###########################################
#
#  Code automaticaly rendered from ...
#
#
#
###########################################
from sympkf.model import Model
import numpy as np

class alternative_pkf_adv_lv(Model):

    # Prognostic functions (sympy functions):
    prognostic_functions = (
            'A',    # Write comments on the function here
            'B',    # Write comments on the function here
            'V_A',    # Write comments on the function here
            'V_B',    # Write comments on the function here
            'V_AB',    # Write comments on the function here
            's_A_xx',    # Write comments on the function here
            's_B_xx',    # Write comments on the function here
        )

    
    
    # Spatial coordinates
    coordinates = (
            'x',    # Write comments on the coordinate here
        )

    
    # Set constant functions
    constant_functions = (
            'u',    # Writes comment on the constant function here
        )
    

    
    # Set constants
    constants = (
            'k_2',    # Writes comment on the constant here
            'k_3',    # Writes comment on the constant here
            'k_1',    # Writes comment on the constant here
        )
    

    def __init__(self, shape=None, lengths=None, **kwargs):

        super().__init__() # Time scheme is set from Model.__init__()
                
        #---------------------------------
        # Set index array from coordinates
        #---------------------------------
        
        # a) Set shape
        shape = len(self.coordinates)*(100,) if shape is None else shape 
        if len(shape)!=len(self.coordinates):
            raise ValueError(f"len(shape) {len(shape)} is different from len(coordinates) {len(self.coordinates)}")
        else:
            self.shape = shape
        
        # b) Set lengths
        lengths = len(self.coordinates)*(1.0,) if lengths is None else lengths
        if len(lengths)!=len(self.coordinates):
            raise ValueError(f"len(lengths) {len(lengths)} is different from len(coordinates) {len(self.coordinates)}")        
        else:
            self.lengths = lengths
        
        # c) Set indexes
        self._index = {}
        for k,coord in enumerate(self.coordinates):
            self._index[(coord,0)] = np.arange(self.shape[k], dtype=int)
                
        # Set x/dx
        #-------------
        self.dx = tuple(length/shape for length, shape in zip(self.lengths, self.shape))
        self.x = tuple(self.index(coord,0)*dx for coord, dx in zip(self.coordinates, self.dx))
        self.X = np.meshgrid(*self.x)

        
        #-----------------------
        # Set constant functions
        #-----------------------
          
        # Set a default nan value for constants
        self.u = np.nan # @@ set constant value @@
        
                
        # Set constant function values from external **kwargs (when provided)
        for key in kwargs:
            if key in self.constant_functions:
                setattr(self, key, kwargs[key])
        
        # Alert when a constant is np.nan
        for function in self.constant_functions:
            if getattr(self, function) is np.nan:
                print(f"Warning: function `{function}` has to be set")
        

        
        #---------------------------
        # Set constants of the model
        #---------------------------
          
        # Set a default nan value for constants
        self.k_2 = np.nan # @@ set constant value @@
        self.k_3 = np.nan # @@ set constant value @@
        self.k_1 = np.nan # @@ set constant value @@
        
                
        # Set constant values from external **kwargs (when provided)
        for key in kwargs:
            if key in self.constants:
                setattr(self, key, kwargs[key])
        
        # Alert when a constant is np.nan
        for constant in self.constants:
            if getattr(self, constant) is np.nan:
                print(f"Warning: constant `{constant}` has to be set")
        
    
    def index(self, coord, step:int):
        """ Return int array of shift index associated with coordinate `coord` for shift `step` """
        # In this implementation, indexes are memory saved in a dictionary, feed at runtime 
        if (coord,step) not in self._index:
            self._index[(coord,step)] = (self._index[(coord,0)]+step)%self.shape[self.coordinates.index(coord)]
        return self._index[(coord,step)]

    def trend(self, t, state):
        """ Trend of the dynamics """

        # Init output state with pointer on data
        #-------------------------------------------

        #   a) Set the output array
        dstate = np.zeros(state.shape)

        #   b) Set pointers on output array `dstate` for the computation of the physical trend (alias only).
        dA = dstate[0]
        dB = dstate[1]
        dV_A = dstate[2]
        dV_B = dstate[3]
        dV_AB = dstate[4]
        ds_A_xx = dstate[5]
        ds_B_xx = dstate[6]
        

        # Load physical functions from state
        #------------------------------------
        A = state[0]
        B = state[1]
        V_A = state[2]
        V_B = state[3]
        V_AB = state[4]
        s_A_xx = state[5]
        s_B_xx = state[6]
          
        
        
        # Alias for constant functions
        #-----------------------------
        u = self.u
                 
                
        
        # Alias for constants
        #--------------------
        k_2 = self.k_2
        k_3 = self.k_3
        k_1 = self.k_1
                 
                
         
        # Compute derivative
        #-----------------------
        #
        #  Warning: might be modified to fit appropriate boundary conditions. 
        #
        DB_x_o1 = (-B[np.ix_(self.index('x',-1))] + B[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        Ds_B_xx_x_o1 = (-s_B_xx[np.ix_(self.index('x',-1))] + s_B_xx[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DV_B_x_o1 = (-V_B[np.ix_(self.index('x',-1))] + V_B[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        Ds_A_xx_x_o1 = (-s_A_xx[np.ix_(self.index('x',-1))] + s_A_xx[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DV_A_x_o1 = (-V_A[np.ix_(self.index('x',-1))] + V_A[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DV_AB_x_o1 = (-V_AB[np.ix_(self.index('x',-1))] + V_AB[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DA_x_o1 = (-A[np.ix_(self.index('x',-1))] + A[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        Du_x_o1 = (-u[np.ix_(self.index('x',-1))] + u[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        
        
        # Implementation of the trend
        #--------------------------------
        dA[:] = -A*B*k_2 - A*Du_x_o1 + A*k_1 - DA_x_o1*u - V_AB*k_2
        dB[:] = A*B*k_2 - B*Du_x_o1 - B*k_3 - DB_x_o1*u + V_AB*k_2
        dV_A[:] = -2*A*V_AB*k_2 - 2*B*V_A*k_2 - DV_A_x_o1*u - 2*Du_x_o1*V_A + 2*V_A*k_1
        dV_B[:] = 2*A*V_B*k_2 + 2*B*V_AB*k_2 - DV_B_x_o1*u - 2*Du_x_o1*V_B - 2*V_B*k_3
        dV_AB[:] = A*V_AB*k_2 - A*V_B*k_2 + B*V_A*k_2 - B*V_AB*k_2 - DV_AB_x_o1*u - 2*Du_x_o1*V_AB + V_AB*k_1 - V_AB*k_3
        ds_A_xx[:] = -Ds_A_xx_x_o1*u + 2*Du_x_o1*s_A_xx  -2*k_2*A*V_AB*s_A_xx/V_A  + 2*k_2*A* V_B**.5 *s_A_xx**2 * (V_AB/ (V_A*V_B)**.5 * 2 / (s_A_xx + s_B_xx)) / V_A**.5
        ds_B_xx[:] = -Ds_B_xx_x_o1*u + 2*Du_x_o1*s_B_xx  +2*k_2*B*V_AB*s_B_xx/V_B  - 2*k_2*B* V_A**.5 *s_B_xx**2 * (V_AB /(V_A*V_B)**.5 * 2 / (s_A_xx + s_B_xx)) / V_B**.5
        
        
        return dstate        

    
