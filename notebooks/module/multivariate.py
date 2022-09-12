import numpy as np


class MultivariateEnKF1D(object):   
    def __init__(self, species_name):
        self.species_name = species_name
        self.n_species = len(species_name)

        self.mapping = {}
        for i, specie in enumerate(self.species_name) : self.mapping[specie] = i
        
    def derive(self, state,axis=0):
        return (np.roll(state,-1,axis=axis) - np.roll(state,1,axis=axis) )/(2*self.dx[0])
        
        
    def _correlation_matrix(self, ensemble):
        tmp = np.hstack([ensemble[:,i] for i in range(ensemble.shape[1])]).T
        return np.corrcoef(tmp)

    def _covariance_matrix(self, ensemble):
	    tmp = np.hstack([ensemble[:,i] for i in range(ensemble.shape[1])]).T
	    return np.cov(tmp)

    def _formated_correlation_matrix(self, ensemble):
        cor_mat = self._correlation_matrix(ensemble)
        out = {}
        Nx = self.shape[0]
        for i, specie in enumerate(self.species_name):
            out[specie] = {}
            for j, specie2 in enumerate(self.species_name):
                out[specie][specie2] = cor_mat[i*Nx:(i+1)*Nx, j*Nx:(j+1)*Nx]
        return out
    
    def _formated_covariance_matrix(self, ensemble):
        cov_mat = self._covariance_matrix(ensemble)
        out = {}
        Nx = self.shape[0]
        for i, specie in enumerate(self.species_name):
            out[specie] = {}
            for j, specie2 in enumerate(self.species_name):
                out[specie][specie2] = cov_mat[i*Nx:(i+1)*Nx, j*Nx:(j+1)*Nx]
        return out
    	    
        
    def assimilate(self, enkf_state, obs_network, obs_species, obs_values, obs_variances):
        if isinstance(enkf_state, list):
            enkf_state = np.array(enkf_state)
	
        state_f = np.hstack([enkf_state[:,i].mean(axis=0) for i in range(self.n_species) ])
        error_f = self._covariance_matrix(enkf_state)
        

        #observation operator : 
        H = np.zeros((len(obs_network),self.shape[0]*self.n_species))
        
        
        for i, (obs_loc, obs_specie) in enumerate(zip(obs_network, obs_species)):
        	index_specie = self.mapping[obs_specie]
        	
        	H[i, obs_loc + index_specie * self.shape[0]] = 1
        	

        # observation error covariance matrix
        R = np.diag(obs_variances)

        # kalman gain
        K = error_f @ H.T @ np.linalg.inv(H @ error_f @ H.T + R)

        #analysis step :
        error_a = (np.eye(self.n_species*self.shape[0]) - K @ H) @ error_f
        state_a = state_f + K @ (obs_values - H @ state_f)
        
        
        #new analysis ensemble :
        ensemble_size = enkf_state.shape[0]
        samples_a = np.random.multivariate_normal(state_a, error_a, ensemble_size)
        ensemble_a = []
        for i in range(self.n_species):
            ensemble_a.append(samples_a[:,i*self.shape[0]:(i+1)*self.shape[0]])
	      
        ensemble_a = np.array(ensemble_a).swapaxes(0,1)

        return ensemble_a
        

    
    def diagnosis(self, traj, cor_func_locations= [], cor_func_species=[]):
        diagnosis = {}
        times = traj.keys()
        for time in times:
            traj[time] = np.array(traj[time])
            diagnosis[time]= {'Mean concentration':{}, 'Std':{}, 'Length-scale':{}, 'Correlation': {} }
            for i,species in enumerate(self.species_name):
                diagnosis[time]['Mean concentration'][species] = traj[time][:,i].mean(axis=0)
                diagnosis[time]['Std'][species] = traj[time][:,i].std(axis=0)
                errors = traj[time][:,i] - diagnosis[time]['Mean concentration'][species]
                normalized_errors = errors/diagnosis[time]['Std'][species]

                dx_normalized_errors = self.derive(normalized_errors,axis=1)
                diagnosis[time]['Length-scale'][species] = (1/np.mean(dx_normalized_errors**2,axis=0))**.5
                diagnosis[time]['Correlation'][species] = {}                
                
            for i,species in enumerate(self.species_name):
                for j in range(i+1, self.n_species):
                    errors = traj[time][:,i] - traj[time][:,i].mean(axis=0)
                    errors_j = traj[time][:,j] - traj[time][:,j].mean(axis=0)
                    species_j = self.species_name[j]
                    diagnosis[time]['Correlation'][species][species_j] = np.mean(errors*errors_j,axis=0)/errors.std(axis=0)/errors_j.std(axis=0)
                    diagnosis[time]['Correlation'][species_j][species] = np.mean(errors*errors_j,axis=0)/errors.std(axis=0)/errors_j.std(axis=0)
                    
            diagnosis[time]['Correlation functions'] = {}
            diagnosis[time]['Covariance functions'] = {}
            cor_mat = self._formated_correlation_matrix(traj[time])
            cov_mat = self._formated_covariance_matrix(traj[time])
            for loc, specie in zip(cor_func_locations, cor_func_species):
                diagnosis[time]['Correlation functions'][loc] = {}
                diagnosis[time]['Covariance functions'][loc] = {}
                for specie2 in self.species_name:
                    diagnosis[time]['Correlation functions'][loc][specie+'/'+specie2] = cor_mat[specie][specie2][loc]
                    diagnosis[time]['Covariance functions'][loc][specie+'/'+specie2] = cov_mat[specie][specie2][loc]
                
                
        return diagnosis
        
        
import numpy as np

class MultivariatePKF1D(object):   
    def __init__(self, species_name):
        self.species_name = species_name
        self.n_species = len(species_name)	
        
        self.mapping = {}
        self.mapping['Mean concentration'] = {}
        self.mapping['Std'] = {}
        self.mapping['Length-scale'] = {}
        self.mapping['Covariance'] = {}
        total_n_fields = int(self.n_species * 3 + (self.n_species-1)*self.n_species / 2)
        cov_index = list(range(int((self.n_species-1)*self.n_species / 2)))
        for i, specie in enumerate(species_name):
            self.mapping['Mean concentration'][specie] = i
            self.mapping['Std'][specie] = self.n_species + i
            self.mapping['Length-scale'][specie] = int(2*self.n_species + (self.n_species-1)*self.n_species / 2  + i)
            self.mapping['Covariance'][specie]= {}
        for i, specie in enumerate(species_name):
            for j, specie2 in enumerate(self.species_name):
                if j>i:
                    self.mapping['Covariance'][specie][specie2] = 2*self.n_species + cov_index[0]
                    self.mapping['Covariance'][specie2][specie] = 2*self.n_species + cov_index[0]
                    cov_index.pop(0)
    def derive(self, state,axis=0):
        return (np.roll(state,-1,axis=axis) - np.roll(state,1,axis=axis) )/(2*self.dx[0])
        
    def _autocorrelation_model(self, s, loc):
        dist = np.abs(self.x[0] - self.x[0][loc])
        dist = np.minimum(dist, 0.5- (dist -0.5))
        output = (s[loc]*s)**.25/(0.5*(s[loc]+s))**.5 * \
            np.exp(- dist**2/ (s[loc]+s))
        return output
    
    def _autocovariance_model(self, V, s, loc):
        dist = np.abs(self.x[0] - self.x[0][loc])
        dist = np.minimum(dist, 0.5- (dist -0.5))
        output = (V[loc]*V)**.5 *\
            (s[loc]*s)**.25/(0.5*(s[loc]+s))**.5 * \
            np.exp(- dist**2/ (s[loc]+s))
        return output
    
    def _crosscorrelation_model(self, Vab,Va,Vb,sA,sB, loc):
        dist = np.abs(self.x[0] - self.x[0][loc])
        dist = np.minimum(dist, 0.5- (dist -0.5))
        output = 0.5 * (Vab[loc]/(Va[loc]*Vb[loc])**.5 + Vab/(Va*Vb)**.5) * \
                np.exp(-dist**2/( 0.5*(sA[loc]+sA +sB[loc]+sB) ))
        return output
        

    def _crosscovariance_model(self, Vab,Va,Vb,sA,sB, loc):
        dist = np.abs(self.x[0] - self.x[0][loc])
        dist = np.minimum(dist, 0.5- (dist -0.5))
        output = 0.5 * (Vab[loc]/(Va[loc]*Vb[loc])**.5 + Vab/(Va*Vb)**.5) * \
                np.exp(-dist**2/( 0.5*(sA[loc]+sA+sB[loc]+sB) )) * \
                np.sqrt(Va[loc]*Vb)
        return output
    
    def _get_covariance_func(self, pkf_state, loc, specie):
        funcs = {}
        for i, specie2 in enumerate(self.species_name):
            if specie2 == specie :
                func = self._autocovariance_model(pkf_state[self.mapping["Std"][specie]],
                                                  pkf_state[self.mapping["Length-scale"][specie]], loc)
            else :
                func = self._crosscovariance_model(
                pkf_state[self.mapping['Covariance'][specie][specie2]],
                pkf_state[self.mapping['Std'][specie]],
                pkf_state[self.mapping['Std'][specie2]],
                pkf_state[self.mapping['Length-scale'][specie]],
                pkf_state[self.mapping['Length-scale'][specie2]],
                loc)
            funcs[specie + '/'+specie2] = func
        return funcs
    
    def _get_correlation_func(self, pkf_state, loc, specie):
        funcs = {}
        for i, specie2 in enumerate(self.species_name):
            if specie2 == specie :
                func = self._autocorrelation_model(pkf_state[self.mapping["Length-scale"][specie]], loc)
            else :
                func = self._crosscorrelation_model(
                pkf_state[self.mapping['Covariance'][specie][specie2]],
                pkf_state[self.mapping['Std'][specie]],
                pkf_state[self.mapping['Std'][specie2]],
                pkf_state[self.mapping['Length-scale'][specie]],
                pkf_state[self.mapping['Length-scale'][specie2]],
                loc)
            funcs[specie + '/'+specie2] = func
        return funcs
            
                        
    def render_traj(self, traj, cor_func_locations = [], cor_func_species=[]):
        rendered_traj = {}
        times = traj.keys()
        for time in times :
            rendered_traj[time]= {'Mean concentration':{}, 'Std':{}, 'Length-scale':{}, 'Correlation': {} }
            for i, specie in enumerate(self.species_name) :       
                rendered_traj[time]['Mean concentration'][specie] = traj[time][self.mapping['Mean concentration'][specie]]
                rendered_traj[time]['Std'][specie] = traj[time][self.mapping['Std'][specie]]**.5
                rendered_traj[time]['Length-scale'][specie] = traj[time][self.mapping['Length-scale'][specie]]**.5   
                rendered_traj[time]['Correlation'][specie] = {}
                
            for i, specie in enumerate(self.species_name) :
                for j, specie2 in enumerate(self.species_name):
                    if j > i :
                        rendered_traj[time]['Correlation'][specie][specie2] = traj[time][self.mapping['Covariance'][specie][specie2]]/traj[time][self.mapping['Std'][specie]]**.5/traj[time][self.mapping['Std'][specie2]]**.5
                        rendered_traj[time]['Correlation'][specie2][specie] = traj[time][self.mapping['Covariance'][specie][specie2]]/traj[time][self.mapping['Std'][specie]]**.5/traj[time][self.mapping['Std'][specie2]]**.5
                            
                rendered_traj[time]['Correlation functions'] = {}
                rendered_traj[time]['Covariance functions'] = {}
                for loc, specie in zip(cor_func_locations, cor_func_species):
                    rendered_traj[time]['Correlation functions'][loc] = self._get_correlation_func(traj[time], loc, specie)
                    rendered_traj[time]['Covariance functions'][loc] = self._get_covariance_func(traj[time], loc, specie)
        
        return rendered_traj
    
    
    def multivariate_assimilation(self, pkf_state, obs_values, obs_locations,
                                  obs_species, obs_variances, method='O2'):
        Nx = self.shape[0]
        
        for obs_value, obs_loc, obs_specie, obs_var in zip(obs_values, obs_locations,
                                  obs_species, obs_variances):
            
            pkf_state_a = np.zeros_like(pkf_state)
            
            x_f = np.hstack([pkf_state[self.mapping['Mean concentration'][sp]] for sp in self.species_name])
            v_f = np.hstack([pkf_state[self.mapping['Std'][sp]] for sp in self.species_name])
            g_f = 1/np.hstack([pkf_state[self.mapping['Length-scale'][sp]] for sp in self.species_name])
            d_v_f = np.hstack([self.derive(pkf_state[self.mapping['Std'][sp]]) for sp in self.species_name])

            cor_funcs = []
            d_cor_funcs = []

            x_f_loc = pkf_state[self.mapping['Mean concentration'][obs_specie]][obs_loc]
            v_f_loc = pkf_state[self.mapping['Std'][obs_specie]][obs_loc]
            for k, specie in enumerate(self.species_name):
                if specie == obs_specie:
                    func = self._autocorrelation_model(pkf_state[self.mapping['Length-scale'][specie]],
                                                       obs_loc)
                    auto_var_func = pkf_state[k+self.n_species]
                    auto_cov_func = self._autocovariance_model(pkf_state[self.mapping['Std'][specie]],
                                                                pkf_state[self.mapping['Length-scale'][specie]],
                                           obs_loc)
                else :
                    func = self._crosscorrelation_model(
                        pkf_state[self.mapping['Covariance'][obs_specie][specie]],
                        pkf_state[self.mapping['Std'][obs_specie]],
                        pkf_state[self.mapping['Std'][specie]],
                        pkf_state[self.mapping['Length-scale'][obs_specie]],
                        pkf_state[self.mapping['Length-scale'][specie]],
                        obs_loc)
                    
                cor_funcs.append(func)
                d_cor_funcs.append(self.derive(func))
            cor_funcs = np.hstack(cor_funcs);
            d_cor_funcs = np.hstack(d_cor_funcs)
                              
            # state update :
            x_a = x_f + v_f**.5 * cor_funcs * v_f_loc**.5/ (v_f_loc + obs_var) *(obs_value - x_f_loc)
            
            #variance update :
            v_a = v_f * (1 - cor_funcs**2 * v_f_loc / (v_f_loc + obs_var))
            
            # metric tensor update :
            if method == 'O1':
                g_a = v_f/v_a * g_f 
            
            if method == 'O2':
                tmp  = np.hstack([self.derive((v_f**.5 * cor_funcs)[i*Nx:(i+1)*Nx]) for i in range(self.n_species)])
                tmp2 = np.hstack([self.derive((v_a)[i*Nx:(i+1)*Nx]) for i in range(self.n_species)])
                g_a = v_f/v_a * g_f + 1/(4*v_f*v_a) * (d_v_f)**2 \
                    - 1/v_a *tmp**2* v_f_loc/(v_f_loc + obs_var) \
                    - 1/(4*v_a**2) * tmp2**2
            s_a = 1/g_a
            
            # covariance update :
            for k1, sp1 in enumerate(self.species_name):
                for k2, sp2 in enumerate(self.species_name) :
                    if k2 > k1 :
                        v_ab_f = pkf_state[self.mapping['Covariance'][sp1][sp2]]
                        if sp1 == obs_specie :
                            cov_func_a_l = auto_cov_func
                        else :                    	
                            cov_func_a_l = self._crosscovariance_model(
                                    pkf_state[self.mapping['Covariance'][obs_specie][sp1]],
                                    pkf_state[self.mapping['Std'][obs_specie]],
                                    pkf_state[self.mapping['Std'][sp1]],
                                    pkf_state[self.mapping['Length-scale'][obs_specie]],
                                    pkf_state[self.mapping['Length-scale'][sp1]],
                                    obs_loc)
                        if sp2 == obs_specie :
                            cov_func_b_l = auto_cov_func
                        else :
                            cov_func_b_l = self._crosscovariance_model(
                                    pkf_state[self.mapping['Covariance'][obs_specie][sp2]],
                                    pkf_state[self.mapping['Std'][obs_specie]],
                                    pkf_state[self.mapping['Std'][sp2]],
                                    pkf_state[self.mapping['Length-scale'][obs_specie]],
                                    pkf_state[self.mapping['Length-scale'][sp2]],
                                    obs_loc)
                                    
                        v_ab_a = v_ab_f - 1/(v_f_loc + obs_var) * cov_func_a_l * cov_func_b_l
                        pkf_state_a[self.mapping['Covariance'][sp1][sp2]] = v_ab_a

            for k, specie in enumerate(self.species_name):
                pkf_state_a[self.mapping['Mean concentration'][specie]] = x_a[k*Nx:(k+1)*Nx]
                pkf_state_a[self.mapping['Std'][specie]] = v_a[k*Nx:(k+1)*Nx]
                pkf_state_a[self.mapping['Length-scale'][specie]] = s_a[k*Nx:(k+1)*Nx]
                
            
            pkf_state = pkf_state_a
            
        return pkf_state
