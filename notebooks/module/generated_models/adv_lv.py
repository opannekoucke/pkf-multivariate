###########################################
#
#  Code automaticaly rendered from ...
#
#
#
###########################################
from sympkf.model import Model
import numpy as np

class adv_lv(Model):

    # Prognostic functions (sympy functions):
    prognostic_functions = (
            'A',    # Write comments on the function here
            'B',    # Write comments on the function here
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
            'k_3',    # Writes comment on the constant here
            'k_1',    # Writes comment on the constant here
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
        self.k_3 = np.nan # @@ set constant value @@
        self.k_1 = np.nan # @@ set constant value @@
        self.k_2 = np.nan # @@ set constant value @@
        
                
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
        

        # Load physical functions from state
        #------------------------------------
        A = state[0]
        B = state[1]
          
        
        
        # Alias for constant functions
        #-----------------------------
        u = self.u
                 
                
        
        # Alias for constants
        #--------------------
        k_3 = self.k_3
        k_1 = self.k_1
        k_2 = self.k_2
                 
                
         
        # Compute derivative
        #-----------------------
        #
        #  Warning: might be modified to fit appropriate boundary conditions. 
        #
        DA_x_o1 = (-A[np.ix_(self.index('x',-1))] + A[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        Du_x_o1 = (-u[np.ix_(self.index('x',-1))] + u[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DB_x_o1 = (-B[np.ix_(self.index('x',-1))] + B[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        
        
        # Implementation of the trend
        #--------------------------------
        dA[:] = -A*B*k_2 - A*Du_x_o1 + A*k_1 - DA_x_o1*u
        dB[:] = A*B*k_2 - B*Du_x_o1 - B*k_3 - DB_x_o1*u
        
        
        return dstate        

    
