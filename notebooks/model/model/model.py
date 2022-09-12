import numpy as np
import matplotlib.pyplot as plt


from tqdm import tqdm

class Scheme:
    "Available schemes : Explicit Euler, RK2, RK4"
    def __init__(self, scheme_name):
        "Available schemes : Explicit Euler, RK2, RK4"
        schemes  = ['EE','RK2','RK4']  #list of all available schemes
        if scheme_name not in schemes :
            raise ValueError("This scheme is not defined.")
        self.scheme_name = scheme_name
        self.scheme = self._make_scheme()

    def _make_scheme(self):
        def EE_scheme(f,x,t,dt):
            x_next = x + dt*f(x,t)
            return x_next
        
        def RK2_scheme(f,x,t,dt):
            k1 = x + dt/2 * f(x,t)
            k2 = f(k1, t + dt/2)
            x_next = x + dt * k2
            return x_next

        def RK4_scheme(f,x,t,dt):
            k1 = f(x,t)
            k2 = f(x + dt/2 * k1, t + dt/2)
            k3 = f(x + dt/2 * k2, t + dt/2)
            k4 = f(x + dt   * k3, t + dt  )
            x_next = x + dt/6 *(k1 + 2*k2 + 2*k3 + k4)
            return x_next
        

        if self.scheme_name =='EE': return EE_scheme
        if self.scheme_name =='RK2': return RK2_scheme
        if self.scheme_name =='RK4': return RK4_scheme


    def step(self,f,x,t,dt):
        return self.scheme(f,x,t,dt)


from tqdm import tqdm
import pickle
class Model:
    def __init__(self, dt = 1e-1, scheme = 'EE', name= 'Undefined model name'):
        self.dt = dt
        self.scheme = Scheme(scheme)
        self.name = name
    
    def window(self, t_end, start=0.):
        times = list(np.arange(start, t_end, self.dt))
        if t_end not in times : times = times + [t_end]
        return times
    
    def trend(self, state, t):
        raise NotImplementedError
    
    
    def forecast(self, window, u0, saved_times = None, save_path = None , saved_times2=None):
        if saved_times is None:
            saved_times = window
            
        trajectory = {}
        pbar = tqdm(initial=0,total=len(window)-1,position=0)
            
        for time, next_time in tqdm(zip(window[:-1],window[1:]),position=0,leave=True):
            if time in saved_times : trajectory[time] = u0
            if save_path is not None and time in saved_times2 : pickle.dump(trajectory, open(save_path,'wb'))
            dt = next_time - time
            u1 = self.scheme.step(self.trend, u0, time, dt)
            u0 = u1
            pbar.update(1)
                
        pbar.close();
        
        time = window[-1]
        if time in saved_times : trajectory[time] = u0
        return trajectory
        #could be usefull :
        self.x = x
        self.t = t
        
        return trajectory
