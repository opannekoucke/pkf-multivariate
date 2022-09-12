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

class pkf_adv_grs(Model):

    # Prognostic functions (sympy functions):
    prognostic_functions = (
            'ROC',    # Write comments on the function here
            'RP',    # Write comments on the function here
            'NO',    # Write comments on the function here
            'NO2',    # Write comments on the function here
            'O3',    # Write comments on the function here
            'SNGN',    # Write comments on the function here
            'V_ROC',    # Write comments on the function here
            'V_RP',    # Write comments on the function here
            'V_NO',    # Write comments on the function here
            'V_NO2',    # Write comments on the function here
            'V_O3',    # Write comments on the function here
            'V_SNGN',    # Write comments on the function here
            'V_ROCRP',    # Write comments on the function here
            'V_ROCNO',    # Write comments on the function here
            'V_ROCNO2',    # Write comments on the function here
            'V_ROCO3',    # Write comments on the function here
            'V_ROCSNGN',    # Write comments on the function here
            'V_RPNO',    # Write comments on the function here
            'V_RPNO2',    # Write comments on the function here
            'V_RPO3',    # Write comments on the function here
            'V_RPSNGN',    # Write comments on the function here
            'V_NONO2',    # Write comments on the function here
            'V_NOO3',    # Write comments on the function here
            'V_NOSNGN',    # Write comments on the function here
            'V_NO2O3',    # Write comments on the function here
            'V_NO2SNGN',    # Write comments on the function here
            'V_O3SNGN',    # Write comments on the function here
            's_ROC_xx',    # Write comments on the function here
            's_RP_xx',    # Write comments on the function here
            's_NO_xx',    # Write comments on the function here
            's_NO2_xx',    # Write comments on the function here
            's_O3_xx',    # Write comments on the function here
            's_SNGN_xx',    # Write comments on the function here
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
            'u',    # Writes comment on the constant function here
            'E_ROC',    # Writes comment on the constant function here
            'E_NO2',    # Writes comment on the constant function here
            'E_NO',    # Writes comment on the constant function here
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
          
        # Set a default nan value for constants

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
        dV_ROC = dstate[6]
        dV_RP = dstate[7]
        dV_NO = dstate[8]
        dV_NO2 = dstate[9]
        dV_O3 = dstate[10]
        dV_SNGN = dstate[11]
        dV_ROCRP = dstate[12]
        dV_ROCNO = dstate[13]
        dV_ROCNO2 = dstate[14]
        dV_ROCO3 = dstate[15]
        dV_ROCSNGN = dstate[16]
        dV_RPNO = dstate[17]
        dV_RPNO2 = dstate[18]
        dV_RPO3 = dstate[19]
        dV_RPSNGN = dstate[20]
        dV_NONO2 = dstate[21]
        dV_NOO3 = dstate[22]
        dV_NOSNGN = dstate[23]
        dV_NO2O3 = dstate[24]
        dV_NO2SNGN = dstate[25]
        dV_O3SNGN = dstate[26]
        ds_ROC_xx = dstate[27]
        ds_RP_xx = dstate[28]
        ds_NO_xx = dstate[29]
        ds_NO2_xx = dstate[30]
        ds_O3_xx = dstate[31]
        ds_SNGN_xx = dstate[32]
        

        # Load physical functions from state
        #------------------------------------
        ROC = state[0]
        RP = state[1]
        NO = state[2]
        NO2 = state[3]
        O3 = state[4]
        SNGN = state[5]
        V_ROC = state[6]
        V_RP = state[7]
        V_NO = state[8]
        V_NO2 = state[9]
        V_O3 = state[10]
        V_SNGN = state[11]
        V_ROCRP = state[12]
        V_ROCNO = state[13]
        V_ROCNO2 = state[14]
        V_ROCO3 = state[15]
        V_ROCSNGN = state[16]
        V_RPNO = state[17]
        V_RPNO2 = state[18]
        V_RPO3 = state[19]
        V_RPSNGN = state[20]
        V_NONO2 = state[21]
        V_NOO3 = state[22]
        V_NOSNGN = state[23]
        V_NO2O3 = state[24]
        V_NO2SNGN = state[25]
        V_O3SNGN = state[26]
        s_ROC_xx = state[27]
        s_RP_xx = state[28]
        s_NO_xx = state[29]
        s_NO2_xx = state[30]
        s_O3_xx = state[31]
        s_SNGN_xx = state[32]
          
        
        
        # Alias for constant functions
        #-----------------------------
        u = self.u
        E_ROC = self.E_ROC
        E_NO2 = self.E_NO2
        E_NO = self.E_NO
                 
                
        
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
        DV_RPNO2_x_o1 = (-V_RPNO2[np.ix_(self.index('x',-1))] + V_RPNO2[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DV_ROC_x_o1 = (-V_ROC[np.ix_(self.index('x',-1))] + V_ROC[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DV_RP_x_o1 = (-V_RP[np.ix_(self.index('x',-1))] + V_RP[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DV_ROCRP_x_o1 = (-V_ROCRP[np.ix_(self.index('x',-1))] + V_ROCRP[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DV_ROCO3_x_o1 = (-V_ROCO3[np.ix_(self.index('x',-1))] + V_ROCO3[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DRP_x_o1 = (-RP[np.ix_(self.index('x',-1))] + RP[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        Ds_NO_xx_x_o1 = (-s_NO_xx[np.ix_(self.index('x',-1))] + s_NO_xx[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DNO2_x_o1 = (-NO2[np.ix_(self.index('x',-1))] + NO2[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DV_NO2SNGN_x_o1 = (-V_NO2SNGN[np.ix_(self.index('x',-1))] + V_NO2SNGN[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        Ds_RP_xx_x_o1 = (-s_RP_xx[np.ix_(self.index('x',-1))] + s_RP_xx[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DV_O3_x_o1 = (-V_O3[np.ix_(self.index('x',-1))] + V_O3[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DV_RPSNGN_x_o1 = (-V_RPSNGN[np.ix_(self.index('x',-1))] + V_RPSNGN[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DV_NO_x_o1 = (-V_NO[np.ix_(self.index('x',-1))] + V_NO[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        Ds_SNGN_xx_x_o1 = (-s_SNGN_xx[np.ix_(self.index('x',-1))] + s_SNGN_xx[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DO3_x_o1 = (-O3[np.ix_(self.index('x',-1))] + O3[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        Ds_NO2_xx_x_o1 = (-s_NO2_xx[np.ix_(self.index('x',-1))] + s_NO2_xx[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DSNGN_x_o1 = (-SNGN[np.ix_(self.index('x',-1))] + SNGN[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        Du_x_o1 = (-u[np.ix_(self.index('x',-1))] + u[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DV_SNGN_x_o1 = (-V_SNGN[np.ix_(self.index('x',-1))] + V_SNGN[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DV_RPNO_x_o1 = (-V_RPNO[np.ix_(self.index('x',-1))] + V_RPNO[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DROC_x_o1 = (-ROC[np.ix_(self.index('x',-1))] + ROC[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DV_ROCNO2_x_o1 = (-V_ROCNO2[np.ix_(self.index('x',-1))] + V_ROCNO2[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DV_NONO2_x_o1 = (-V_NONO2[np.ix_(self.index('x',-1))] + V_NONO2[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DV_NO2O3_x_o1 = (-V_NO2O3[np.ix_(self.index('x',-1))] + V_NO2O3[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        Ds_ROC_xx_x_o1 = (-s_ROC_xx[np.ix_(self.index('x',-1))] + s_ROC_xx[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        Ds_O3_xx_x_o1 = (-s_O3_xx[np.ix_(self.index('x',-1))] + s_O3_xx[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DV_NOSNGN_x_o1 = (-V_NOSNGN[np.ix_(self.index('x',-1))] + V_NOSNGN[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DNO_x_o1 = (-NO[np.ix_(self.index('x',-1))] + NO[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DV_NO2_x_o1 = (-V_NO2[np.ix_(self.index('x',-1))] + V_NO2[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DV_ROCSNGN_x_o1 = (-V_ROCSNGN[np.ix_(self.index('x',-1))] + V_ROCSNGN[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DV_RPO3_x_o1 = (-V_RPO3[np.ix_(self.index('x',-1))] + V_RPO3[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DV_ROCNO_x_o1 = (-V_ROCNO[np.ix_(self.index('x',-1))] + V_ROCNO[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DV_O3SNGN_x_o1 = (-V_O3SNGN[np.ix_(self.index('x',-1))] + V_O3SNGN[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        DV_NOO3_x_o1 = (-V_NOO3[np.ix_(self.index('x',-1))] + V_NOO3[np.ix_(self.index('x',1))])/(2*self.dx[self.coordinates.index('x')])
        
        
        # Implementation of the trend
        #--------------------------------
        dROC[:] = -DROC_x_o1*u - Du_x_o1*ROC + E_ROC - ROC*l
        dRP[:] = -DRP_x_o1*u - Du_x_o1*RP - NO*RP*k_2 - 2*NO2*RP*k_6 + ROC*k_1 - RP**2*k_5 - RP*l - V_RP*k_5 - V_RPNO*k_2 - 2*V_RPNO2*k_6
        dNO[:] = -DNO_x_o1*u - Du_x_o1*NO + E_NO - NO*O3*k_4 - NO*RP*k_2 - NO*l + NO2*k_3 - V_NOO3*k_4 - V_RPNO*k_2
        dNO2[:] = -DNO2_x_o1*u - Du_x_o1*NO2 + E_NO2 + NO*O3*k_4 + NO*RP*k_2 - 2*NO2*RP*k_6 - NO2*k_3 - NO2*l + V_NOO3*k_4 + V_RPNO*k_2 - 2*V_RPNO2*k_6
        dO3[:] = -DO3_x_o1*u - Du_x_o1*O3 - NO*O3*k_4 + NO2*k_3 - O3*l - V_NOO3*k_4
        dSNGN[:] = -DSNGN_x_o1*u - Du_x_o1*SNGN + 2*NO2*RP*k_6 - SNGN*l + 2*V_RPNO2*k_6
        dV_ROC[:] = -DV_ROC_x_o1*u - 2*Du_x_o1*V_ROC - 2*V_ROC*l
        dV_RP[:] = -DV_RP_x_o1*u - 2*Du_x_o1*V_RP - 2*NO*V_RP*k_2 - 4*NO2*V_RP*k_6 - 4*RP*V_RP*k_5 - 2*RP*V_RPNO*k_2 - 4*RP*V_RPNO2*k_6 + 2*V_ROCRP*k_1 - 2*V_RP*l
        dV_NO[:] = -DV_NO_x_o1*u - 2*Du_x_o1*V_NO - 2*NO*V_NOO3*k_4 - 2*NO*V_RPNO*k_2 - 2*O3*V_NO*k_4 - 2*RP*V_NO*k_2 - 2*V_NO*l + 2*V_NONO2*k_3
        dV_NO2[:] = -DV_NO2_x_o1*u - 2*Du_x_o1*V_NO2 + 2*NO*V_NO2O3*k_4 + 2*NO*V_RPNO2*k_2 - 4*NO2*V_RPNO2*k_6 + 2*O3*V_NONO2*k_4 - 4*RP*V_NO2*k_6 + 2*RP*V_NONO2*k_2 - 2*V_NO2*k_3 - 2*V_NO2*l
        dV_O3[:] = -DV_O3_x_o1*u - 2*Du_x_o1*V_O3 - 2*NO*V_O3*k_4 - 2*O3*V_NOO3*k_4 + 2*V_NO2O3*k_3 - 2*V_O3*l
        dV_SNGN[:] = -DV_SNGN_x_o1*u - 2*Du_x_o1*V_SNGN + 4*NO2*V_RPSNGN*k_6 + 4*RP*V_NO2SNGN*k_6 - 2*V_SNGN*l
        dV_ROCRP[:] = -DV_ROCRP_x_o1*u - 2*Du_x_o1*V_ROCRP - NO*V_ROCRP*k_2 - 2*NO2*V_ROCRP*k_6 - RP*V_ROCNO*k_2 - 2*RP*V_ROCNO2*k_6 - 2*RP*V_ROCRP*k_5 + V_ROC*k_1 - 2*V_ROCRP*l
        dV_ROCNO[:] = -DV_ROCNO_x_o1*u - 2*Du_x_o1*V_ROCNO - NO*V_ROCO3*k_4 - NO*V_ROCRP*k_2 - O3*V_ROCNO*k_4 - RP*V_ROCNO*k_2 - 2*V_ROCNO*l + V_ROCNO2*k_3
        dV_ROCNO2[:] = -DV_ROCNO2_x_o1*u - 2*Du_x_o1*V_ROCNO2 + NO*V_ROCO3*k_4 + NO*V_ROCRP*k_2 - 2*NO2*V_ROCRP*k_6 + O3*V_ROCNO*k_4 + RP*V_ROCNO*k_2 - 2*RP*V_ROCNO2*k_6 - V_ROCNO2*k_3 - 2*V_ROCNO2*l
        dV_ROCO3[:] = -DV_ROCO3_x_o1*u - 2*Du_x_o1*V_ROCO3 - NO*V_ROCO3*k_4 - O3*V_ROCNO*k_4 + V_ROCNO2*k_3 - 2*V_ROCO3*l
        dV_ROCSNGN[:] = -DV_ROCSNGN_x_o1*u - 2*Du_x_o1*V_ROCSNGN + 2*NO2*V_ROCRP*k_6 + 2*RP*V_ROCNO2*k_6 - 2*V_ROCSNGN*l
        dV_RPNO[:] = -DV_RPNO_x_o1*u - 2*Du_x_o1*V_RPNO - NO*V_RP*k_2 - NO*V_RPNO*k_2 - NO*V_RPO3*k_4 - 2*NO2*V_RPNO*k_6 - O3*V_RPNO*k_4 - RP*V_NO*k_2 - 2*RP*V_NONO2*k_6 - RP*V_RPNO*k_2 - 2*RP*V_RPNO*k_5 + V_ROCNO*k_1 - 2*V_RPNO*l + V_RPNO2*k_3
        dV_RPNO2[:] = -DV_RPNO2_x_o1*u - 2*Du_x_o1*V_RPNO2 + NO*V_RP*k_2 - NO*V_RPNO2*k_2 + NO*V_RPO3*k_4 - 2*NO2*V_RP*k_6 - 2*NO2*V_RPNO2*k_6 + O3*V_RPNO*k_4 - 2*RP*V_NO2*k_6 - RP*V_NONO2*k_2 + RP*V_RPNO*k_2 - 2*RP*V_RPNO2*k_5 - 2*RP*V_RPNO2*k_6 + V_ROCNO2*k_1 - V_RPNO2*k_3 - 2*V_RPNO2*l
        dV_RPO3[:] = -DV_RPO3_x_o1*u - 2*Du_x_o1*V_RPO3 - NO*V_RPO3*k_2 - NO*V_RPO3*k_4 - 2*NO2*V_RPO3*k_6 - O3*V_RPNO*k_4 - 2*RP*V_NO2O3*k_6 - RP*V_NOO3*k_2 - 2*RP*V_RPO3*k_5 + V_ROCO3*k_1 + V_RPNO2*k_3 - 2*V_RPO3*l
        dV_RPSNGN[:] = -DV_RPSNGN_x_o1*u - 2*Du_x_o1*V_RPSNGN - NO*V_RPSNGN*k_2 + 2*NO2*V_RP*k_6 - 2*NO2*V_RPSNGN*k_6 - 2*RP*V_NO2SNGN*k_6 - RP*V_NOSNGN*k_2 + 2*RP*V_RPNO2*k_6 - 2*RP*V_RPSNGN*k_5 + V_ROCSNGN*k_1 - 2*V_RPSNGN*l
        dV_NONO2[:] = -DV_NONO2_x_o1*u - 2*Du_x_o1*V_NONO2 - NO*V_NO2O3*k_4 + NO*V_NOO3*k_4 + NO*V_RPNO*k_2 - NO*V_RPNO2*k_2 - 2*NO2*V_RPNO*k_6 + O3*V_NO*k_4 - O3*V_NONO2*k_4 + RP*V_NO*k_2 - RP*V_NONO2*k_2 - 2*RP*V_NONO2*k_6 + V_NO2*k_3 - V_NONO2*k_3 - 2*V_NONO2*l
        dV_NOO3[:] = -DV_NOO3_x_o1*u - 2*Du_x_o1*V_NOO3 - NO*V_NOO3*k_4 - NO*V_O3*k_4 - NO*V_RPO3*k_2 - O3*V_NO*k_4 - O3*V_NOO3*k_4 - RP*V_NOO3*k_2 + V_NO2O3*k_3 + V_NONO2*k_3 - 2*V_NOO3*l
        dV_NOSNGN[:] = -DV_NOSNGN_x_o1*u - 2*Du_x_o1*V_NOSNGN - NO*V_O3SNGN*k_4 - NO*V_RPSNGN*k_2 + 2*NO2*V_RPNO*k_6 - O3*V_NOSNGN*k_4 + 2*RP*V_NONO2*k_6 - RP*V_NOSNGN*k_2 + V_NO2SNGN*k_3 - 2*V_NOSNGN*l
        dV_NO2O3[:] = -DV_NO2O3_x_o1*u - 2*Du_x_o1*V_NO2O3 - NO*V_NO2O3*k_4 + NO*V_O3*k_4 + NO*V_RPO3*k_2 - 2*NO2*V_RPO3*k_6 - O3*V_NONO2*k_4 + O3*V_NOO3*k_4 - 2*RP*V_NO2O3*k_6 + RP*V_NOO3*k_2 + V_NO2*k_3 - V_NO2O3*k_3 - 2*V_NO2O3*l
        dV_NO2SNGN[:] = -DV_NO2SNGN_x_o1*u - 2*Du_x_o1*V_NO2SNGN + NO*V_O3SNGN*k_4 + NO*V_RPSNGN*k_2 + 2*NO2*V_RPNO2*k_6 - 2*NO2*V_RPSNGN*k_6 + O3*V_NOSNGN*k_4 + 2*RP*V_NO2*k_6 - 2*RP*V_NO2SNGN*k_6 + RP*V_NOSNGN*k_2 - V_NO2SNGN*k_3 - 2*V_NO2SNGN*l
        dV_O3SNGN[:] = -DV_O3SNGN_x_o1*u - 2*Du_x_o1*V_O3SNGN - NO*V_O3SNGN*k_4 + 2*NO2*V_RPO3*k_6 - O3*V_NOSNGN*k_4 + 2*RP*V_NO2O3*k_6 + V_NO2SNGN*k_3 - 2*V_O3SNGN*l
        ds_ROC_xx[:] = -Ds_ROC_xx_x_o1*u + 2*Du_x_o1*s_ROC_xx
        ds_RP_xx[:] = -Ds_RP_xx_x_o1*u + 2*Du_x_o1*s_RP_xx
        ds_NO_xx[:] = -Ds_NO_xx_x_o1*u + 2*Du_x_o1*s_NO_xx
        ds_NO2_xx[:] = -Ds_NO2_xx_x_o1*u + 2*Du_x_o1*s_NO2_xx
        ds_O3_xx[:] = -Ds_O3_xx_x_o1*u + 2*Du_x_o1*s_O3_xx
        ds_SNGN_xx[:] = -Ds_SNGN_xx_x_o1*u + 2*Du_x_o1*s_SNGN_xx
        
        
        return dstate        

    
    
    def compute_exogenous(self, t, state):
        """ Computation of exogenous functions """
        return [k3(t),k1(t)]
