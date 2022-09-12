from model import AdvectionGRS
import numpy as np
from module.tools import make_samples

from module.generated_models import adv_grs, pkf_adv_grs
from module.multivariate import *

def make_forecast(ENSEMBLE_SIZE, INDEX):
	species_name = ['ROC','RP','NO','NO2','O3','SN(G)N']
	Nx = 241

	class EnKF_Advection_GRS(MultivariateEnKF1D, adv_grs):
	    def __init__(self, **kwargs):
		adv_grs.__init__(self, **kwargs)
		MultivariateEnKF1D.__init__(self, species_name) 
		
	class PKF_Advection_GRS(MultivariatePKF1D, pkf_adv_grs):
	    def __init__(self, **kwargs):
		pkf_adv_grs.__init__(self, **kwargs)
		MultivariatePKF1D.__init__(self, species_name)
		
		
		
	D = 1000  #km
	u_amp = 15 #kmh
	u_mean = 35  #kmh
	domain = enkf ; dx = domain.dx[0]
	u = (u_mean + u_amp*np.cos(2*np.pi*domain.x[0]))/D
	enkf.u = u 


	import pickle
	state0 = pickle.load(open('grs_data/grs_state0.pkl','rb'))
	species_name = ['ROC','RP','NO','NO2','O3','SN(G)N']

	initial_settings = {
		'Mean concentration' : {},
		'Std' : {},
		'Length-scale' : {}
	}

	lh0 = 15*dx
	lh0_2 = 12*dx

	for i,field in enumerate(species_name) :
	    std = 0.15 if field not in ['RP','NO'] else 0
	    initial_settings['Std'][field] = state0[i,150] * std + u*0
	    initial_settings['Mean concentration'][field] = state0[i,150] + make_samples(1,domain.x[0],lh0_2)[0] * \
		                initial_settings['Std'][field]
	    initial_settings['Length-scale'][field] = lh0 +u*0
	      

	from copy import deepcopy
	normalization = deepcopy(initial_settings)
	for i,field in enumerate(species_name) : 
	    normalization['Mean concentration'][field] = 1
	    normalization['Std'][field] = 1
	    normalization['Length-scale'][field] = dx
	    
	unities = deepcopy(initial_settings)
	for i,field in enumerate(species_name) : 
	    unities['Mean concentration'][field] = ' (ppb)'
	    unities['Std'][field] = ' (ppb)'
	    unities['Length-scale'][field] = r' ($\Delta x)$'
	    
	unities['Mean concentration']['ROC'] = ' (ppC)'
	unities['Std']['ROC'] = ' (ppC)'





	from sympy import Function, Derivative, Eq, symbols
	from sympkf import SymbolicPKF, t, FDModelBuilder
	x, kappa = symbols("x kappa")
	y= Function('y')(t,x)
	diff_eq = [Eq(Derivative(y,t),kappa*Derivative(y,x,2))]

	exec(FDModelBuilder(diff_eq, class_name='Diffusion1D').code)

	diff = Diffusion1D(shape=(Nx,),kappa=1e-4*5)
	diff.set_dt(1e-3)
	times = diff.window(1)
	saved_times= [times[-1]]
	y0 = np.array([(domain.x[0] > 0.45)*1.0])
	cadastre = diff.forecast(times,y0,saved_times)[1][0]




	ensemble_size = ENSEMBLE_SIZE

	ensemble0 = np.array([ initial_settings['Mean concentration'][field] +
		              make_samples(ensemble_size,
		                           domain.x[0],
		                           initial_settings['Length-scale'][field])
		              * initial_settings['Std'][field] for field in species_name])
		              

	enkf = AdvectionGRS(u, cadastre)
	enkf.dt = 1e-4


	time_saving_interval = 1.0/ 6
	traj_enkf = enkf.forecast(ensemble0, 72.0, time_saving_interval )

	import pickle
	pickle.dump(traj_enkf,open('grs_data/traj_enkf{}.pkl'.format(INDEX),'wb'))

                      
                      


