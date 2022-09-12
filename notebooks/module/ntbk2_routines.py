import matplotlib.pyplot as plt
import numpy as np
def univariate_plot_bis(times,
                        x_hr, x_lr,
                        enkf_traj, pkf_traj,
                        normalizations, ensemble_size):
    
    fig, ax = plt.subplots(1,3,figsize=(16,5))
    fig.subplots_adjust(hspace=.25,wspace=.25)
    panel_index = [f'({c}) ' for c in 'abc']
    keys = ['Mean concentration','Std','Length-scale']
    fields = ['Mean concentration','Standard-deviation','Length-scale']
    normalization_labels=[r'$A/A^0$', r'$\sigma_x/\sigma^0$',r'$L_x/l_h^0$']

    alphas = np.linspace(0,1,len(times))
    Nx = len(x_lr)

    for k, field in enumerate(fields):
        ax[k].plot([0,1],[1,1],c='grey',alpha=.3)
        ax[k].set_title(panel_index[k] + field,fontsize=15)
        for i, time in enumerate(times):
            alpha = alphas[i]**1
            linewidth = 2 if time in [times[0],times[-1]] else 1
            label = fr'{ensemble_size}-Ensemble; $N_x=${3*Nx}'  if time == times[-1] else None
            ax[k].set_ylabel(normalization_labels[k],fontsize=15)
            ax[k].plot(x_hr, enkf_traj[time][keys[k]]['X']/normalizations[k],c='b',
                     linewidth=linewidth,label=label,linestyle='--',alpha=alpha)

            c = 'r' #if i in [0,n_stops-1] else 'lightsalmon'
            
            label = fr'PKF; $N_x=${Nx}' if time == times[-1] else None
            ax[k].plot(x_lr, pkf_traj[time][keys[k]]['X']/normalizations[k],c=c,
                     linewidth=linewidth,linestyle='-',label=label,zorder=3,alpha=alpha)
            ax[k].set_xlabel(r'$x/D$',fontsize=15)
            if k==0 and time == times[-1] : ax[k].legend(); ax[k].text(.73,1.53,r'$t=3 \tau_{adv}$',fontsize=14)
                
    return fig


def univariate_plot(times,x, enkf_traj, pkf_traj, normalizations, ensemble_size,label_enkf, **kwargs):
    fig, ax = plt.subplots(1,3,figsize=(16,5))
    fig.subplots_adjust(hspace=.25,wspace=.25)
    panel_index = [f'({c}) ' for c in 'abc']
    keys = ['Mean concentration','Std','Length-scale']
    fields = ['Mean concentration','Standard-deviation','Length-scale']
    normalization_labels=[r'$A/A^0$', r'$\sigma_x/\sigma^0$',r'$L_x/l_h^0$']

    alphas = np.linspace(0,1,len(times))
    Nx = len(x)
    for k, field in enumerate(fields):
        
        ax[k].set_title(panel_index[k] + field,fontsize=15)
        ax[k].plot([0,1],[1,1],c='grey',alpha=.3)
        for i, time in enumerate(times):
            
            alpha = alphas[i]**1.2
            linewidth = 2 if time in [times[-1]] else 1
            label = label_enkf if time == times[-1] else None
            ax[k].set_ylabel(normalization_labels[k],fontsize=15)
            ax[k].plot(x, enkf_traj[time][keys[k]]['X']/normalizations[k],
                       linewidth=linewidth,label=label,alpha=alpha,
                       **kwargs)

            c = 'r' #if i in [0,n_stops-1] else 'lightsalmon'
            label = fr'PKF; $N_x=${Nx}' if time == times[-1] else None
            ax[k].plot(x, pkf_traj[time][keys[k]]['X']/normalizations[k],c=c,
                     linewidth=linewidth,linestyle='-',label=label,zorder=3,alpha=alpha)
            ax[k].set_xlabel(r'$x/D$',fontsize=15)
            if k==0 and time == times[-1] : ax[k].legend(); ax[k].text(.73,1.53,r'$t=3 \tau_{adv}$',fontsize=14)
                
    return fig

def plot_gradient(ax, xx, yy , col1, col2, **kwargs):
    _yy = (yy-yy.min())/(yy.max()- yy.min())
    gradients = (col1 * _yy[:,np.newaxis] +  col2 * (1-_yy)[:,np.newaxis] )
    
    for x0,x1,y0,y1,col in zip(xx[:-1],xx[1:],yy[:-1],yy[1:],gradients[:-1]):
        ax.plot([x0,x1],[y0,y1],c=col,**kwargs)
    return ax
