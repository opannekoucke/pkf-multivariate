import numpy as np

min_to_hour = 60 

def k3(t):
    return np.exp( -np.abs((t%24.0)-12)**3/10**2) * 0.6239692319730302 * min_to_hour

def k1(t): return 0.00152 * k3(t)




k2 = 12.3* min_to_hour
k4 = 0.275* min_to_hour
k5 = 10.2* min_to_hour
k6 = 0.12* min_to_hour



day_to_hour = 1./24

eROC = 0.0235 * day_to_hour 
eNO = .243 * day_to_hour
eNO2 = .027 * day_to_hour
l = .02 * day_to_hour


###########################################
#
#  Code automaticaly rendered from ...
#
#
#
###########################################
from sympkf.model import Model


class adv_grs(Model):

    # Prognostic functions (sympy functions):
    prognostic_functions = (
            'ROC',    # Write comments on the function here
            'RP',    # Write comments on the function here
            'NO',    # Write comments on the function here
            'NO2',    # Write comments on the function here
            'O3',    # Write comments on the function here
            'SNGN',    # Write comments on the function here
        )

    
    # Exogenous functions (sympy functions)
    exogenous_functions = (
            'k_3',    # Write comments on the exogenous function here
            'k_1',    # Write comments on the exogenous function here
        )
    
    
    # Spatial coordinates
    coordinates = (
            'x',    # Write comments on the coordinate here
        )

    
    # Set constant functions
    constant_functions = (
            'E_NO',    # Writes comment on the constant function here
            'u',    # Writes comment on the constant function here
            'E_NO2',    # Writes comment on the constant function here
            'E_ROC',    # Writes comment on the constant function here
        )
    

    
    # Set constants
    constants = (
            'k_5',    # Writes comment on the constant here
            'k_4',    # Writes comment on the constant here
            'k_6',    # Writes comment on the constant here
            'l',    # Writes comment on the constant here
            'k_2',    # Writes comment on the constant here
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

        land = np.load('module/generated_models/cadastre.npy')        
        self.E_ROC = eROC * land # @@ set constant value @@
        self.E_NO2 = eNO2 * land # @@ set constant value @@
        self.E_NO = eNO * land # @@ set constant value @@
        
                
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
          
        self.k_5 = k5
        self.k_4 = k4
        self.k_6 = k6
        self.l = l
        self.k_2 = k2
        
                
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
        dROC = dstate[0]
        dRP = dstate[1]
        dNO = dstate[2]
        dNO2 = dstate[3]
        dO3 = dstate[4]
        dSNGN = dstate[5]
        

        # Load physical functions from state
        #------------------------------------
        ROC = state[0]
        RP = state[1]
        NO = state[2]
        NO2 = state[3]
        O3 = state[4]
        SNGN = state[5]
          
        
        
        # Alias for constant functions
        #-----------------------------
        E_NO = self.E_NO
        u = self.u
        E_NO2 = self.E_NO2
        E_ROC = self.E_ROC
                 
                
        
        # Alias for constants
        #--------------------
        k_5 = self.k_5
        k_4 = self.k_4
        k_6 = self.k_6
        l = self.l
        k_2 = self.k_2
                 
                
         
        
        # Compute exogenous functions
        #-----------------------------
        exogenous = self.compute_exogenous(t, state) # None if no exogenous function exists
        k_3 = exogenous[0]
        k_1 = exogenous[1]
         
        # Compute derivative
        #-----------------------
        #
        #  Warning: might be modified to fit appropriate boundary conditions. 
        #
        DNO_x_o1 = (-NO[np.ix_(self.index('x',-1))] + NO[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DO3_x_o1 = (-O3[np.ix_(self.index('x',-1))] + O3[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DRP_x_o1 = (-RP[np.ix_(self.index('x',-1))] + RP[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        Du_x_o1 = (-u[np.ix_(self.index('x',-1))] + u[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DSNGN_x_o1 = (-SNGN[np.ix_(self.index('x',-1))] + SNGN[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DNO2_x_o1 = (-NO2[np.ix_(self.index('x',-1))] + NO2[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DROC_x_o1 = (-ROC[np.ix_(self.index('x',-1))] + ROC[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        
        
        # Implementation of the trend
        #--------------------------------
        dROC[:] = -DROC_x_o1*u - Du_x_o1*ROC + E_ROC - ROC*l
        dRP[:] = -DRP_x_o1*u - Du_x_o1*RP + ROC*k_1 - RP*l - RP*(NO*k_2 + 2*NO2*k_6 + RP*k_5)
        dNO[:] = -DNO_x_o1*u - Du_x_o1*NO + E_NO - NO*l - NO*(O3*k_4 + RP*k_2) + NO2*k_3
        dNO2[:] = -DNO2_x_o1*u - Du_x_o1*NO2 + E_NO2 + NO*O3*k_4 + NO*RP*k_2 - NO2*l - NO2*(2*RP*k_6 + k_3)
        dO3[:] = -DO3_x_o1*u - Du_x_o1*O3 - NO*O3*k_4 + NO2*k_3 - O3*l
        dSNGN[:] = -DSNGN_x_o1*u - Du_x_o1*SNGN + 2*NO2*RP*k_6 - SNGN*l
        
        
        return dstate        

    
    def compute_exogenous(self, t, state):
        """ Computation of exogenous functions """
        return [k3(t),k1(t)]
    
