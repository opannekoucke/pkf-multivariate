
import numpy as np

def diag_with_closures(traj_enkf):
    diag = {}
    
    for t, _ensemble in traj_enkf.items() : 
        ensemble = np.swapaxes(np.array(_ensemble),0,1)
        a = ensemble[0].mean(axis=0)
        b = ensemble[1].mean(axis=0)
        dx = 1/ ensemble.shape[-1]
        

        
        std_a = ensemble[0].std(axis=0)
        std_b = ensemble[1].std(axis=0)
        
        errors_A = ensemble[0] - a 
        errors_B = ensemble[1] - b 
        
        vab = (errors_A*errors_B).mean(axis=0)
        norm_errors_A = errors_A/std_a
        norm_errors_B = errors_B/std_b

        dx_norm_errors_A =   (np.roll(norm_errors_A,-1,axis=1) - np.roll(norm_errors_A,1,axis=1))/2/dx
        dx_norm_errors_B =   (np.roll(norm_errors_B,-1,axis=1) - np.roll(norm_errors_B,1,axis=1))/2/dx        
        
        sa = 1/(dx_norm_errors_A**2).mean(axis=0)
        sb = 1/(dx_norm_errors_B**2).mean(axis=0)
        
        eb_dx_ea = (norm_errors_B*dx_norm_errors_A).mean(axis=0)
        ea_dx_eb = (dx_norm_errors_B*norm_errors_A).mean(axis=0)
        dx_ea_dx_eb = (dx_norm_errors_B*dx_norm_errors_A).mean(axis=0)
        
        diag[t] = {}
        diag[t]['A'] = a; diag[t]['B'] = b;
        diag[t]['Va'] = std_a**2; diag[t]['Vb'] = std_b**2;
        diag[t]['Vab'] = vab
        diag[t]['sA'] = sa; diag[t]['sB'] = sb;
        diag[t]['eb_dx_ea'] = eb_dx_ea
        diag[t]['ea_dx_eb'] = ea_dx_eb
        diag[t]['dx_ea_dx_eb'] = dx_ea_dx_eb
    
    return diag
    
    
    
import numpy as np
import matplotlib.pyplot as plt

def Comparison_Anisotropies_LV_HO(diag_enkf, initial_settings,k,title='', c1 = 'fuchsia', c2 = "lawngreen"):
    times = np.array(list(diag_enkf.keys()))
    
    stdA = np.array([diag_enkf[t]['Va'] for t in times]).mean(axis=1)**.5
    stdB = np.array([diag_enkf[t]['Vb'] for t in times]).mean(axis=1)**.5
    la = np.array([diag_enkf[t]['sA'] for t in times]).mean(axis=1)**.5
    lb = np.array([diag_enkf[t]['sB'] for t in times]).mean(axis=1)**.5
    Vab = np.array([diag_enkf[t]['Vab'] for t in times]).mean(axis=1)
    dx_ea_dx_eb = np.array([diag_enkf[t]['dx_ea_dx_eb'] for t in times]).mean(axis=1)
    cAB = Vab/stdA/stdB
    
    fig, ax = plt.subplots(4,2,figsize=(16,8),sharex='all')

    fig.suptitle(title,fontsize=15)
    
    ax[0,0].plot(times,2*stdA/(stdA[0]+stdB[0]),label='$2 \sigma_A/(\sigma_A^0+ \sigma_B^0)$',c=c1)
    ax[0,0].plot(times,2*stdB/(stdA[0]+stdB[0]),label='$2 \sigma_B/(\sigma_A^0+ \sigma_B^0)$',c=c2)
    
    ax[1,0].plot(times,cAB,label=r'$corr(A, B)$',c='gray')
    ax[1,0].set_ylim(-1,1)

    ax[2,0].plot(times,2*la/(la[0]+lb[0]),label='$2 L_A/(L_A^0+ L_B^0)$',c=c1)
    ax[2,0].plot(times,2*lb/(la[0]+lb[0]),label='$2 L_B/(L_A^0+ L_B^0)$',c=c2)
    
    ax[3,0].plot(times,dx_ea_dx_eb,label=r'$\overline{\partial_x \tilde{\varepsilon}_A\partial_x \tilde{\varepsilon}_B}$',c='gray')
    ax[3,0].set_xlabel("$t$",fontsize=14)
    
    time_window = np.array(times)
    
    LA0 = initial_settings['Length-scale A']
    LB0 = initial_settings['Length-scale B']
    
    VA0 = initial_settings['std A']**2
    VB0 = initial_settings['std B']**2
    
    gA0, gB0 = 1/LA0**2, 1/LB0**2
    
    # Calcul des séries temporelles des statistiques
    ct, st = np.cos(k*time_window), np.sin(k*time_window)
    
    VA = ct**2 *VA0 + st**2*VB0
    VB = ct**2 *VB0 + st**2*VA0
    VAB = ct*st*(VA0-VB0) 
    
    stdA, stdB = np.sqrt(VA), np.sqrt(VB) 
    
    gA = (ct**2 *VA0 *gA0 + st**2*VB0 *gB0)/VA
    gB = (st**2 *VA0 *gA0 + ct**2*VB0 *gB0)/VB
    
    LA = 1/np.sqrt(gA)
    LB = 1/np.sqrt(gB)

    EAB = ct*st/(stdA*stdB)*(VA0*gA0-VB0*gB0)
    
    
    ax[0,1].plot(times,2*stdA/(stdA[0]+stdB[0]),label='$2 \sigma_A/(\sigma_A^0+ \sigma_B^0)$',c=c1,linestyle='--')
    ax[0,1].plot(times,2*stdB/(stdA[0]+stdB[0]),label='$2 \sigma_B/(\sigma_A^0+ \sigma_B^0)$',c=c2,linestyle='--')
    
    
    ax[1,1].plot(times,VAB/stdA/stdB,label=r'$corr(A, B)$',c='gray',linestyle='--')
    ax[1,1].set_ylim(-1,1)
    
    ax[2,1].plot(times,2*LA/(LA[0]+LB[0]),label='$2 L_A/(L_A^0+ L_B^0)$',c=c1,linestyle='--')
    ax[2,1].plot(times,2*LB/(LA[0]+LB[0]),label='$2 L_B/(L_A^0+ L_B^0)$',c=c2,linestyle='--')

    ax[2,0].set_ylim(0.8,1.2)
    ax[2,1].set_ylim(0.8,1.2)
    
    ax[3,1].plot(times,EAB,label=r'$\overline{\partial_x \tilde{\varepsilon}_A\partial_x \tilde{\varepsilon}_B}$',c='gray',linestyle='--')
    ax[3,1].set_xlabel("$t$",fontsize=14)
    ax[3,1].axhline(0,c='grey',alpha=0.3)
    ax[3,0].axhline(0,c='grey',alpha=0.3)
    ax[1,0].axhline(0,c='grey',alpha=0.3)
    ax[1,1].axhline(0,c='grey',alpha=0.3)
    
    fields = ['normalized std','cross-correlation', 'normalized length-scales',r'$\overline{\partial_x \tilde{\varepsilon}_A\partial_x \tilde{\varepsilon}_B}$']
    system = ['in the LV dyn.','in the HO dyn.']
    
    for i, (panel, index) in enumerate(zip(ax.flat, 'abcdefghij')):
        panel.legend();
        panel.set_title('('+index+') '+ fields[i//2] +' ' + system[i%2],fontsize=14);
    
    fig.tight_layout();
    return fig;
    
    
    
def quantify_process(enkf_curves,u,k1,k2,k3,characteristic_time, title='',text_height=60):
    times = list(enkf_curves.keys())
    Nx  = len(enkf_curves[times[0]]['A'])
    dx = 1./Nx
    def derive(state,axis=0):
        return (np.roll(state,-1,axis=axis) -  np.roll(state,1,axis=axis))/2/dx
        
    

    A = np.array([enkf_curves[t]['A'] for t in times])
    B = np.array([enkf_curves[t]['B'] for t in times])
    Va = np.array([enkf_curves[t]['Va'] for t in times])
    Vb = np.array([enkf_curves[t]['Vb'] for t in times])
    sa = np.array([enkf_curves[t]['sA'] for t in times])
    sb = np.array([enkf_curves[t]['sB'] for t in times])
    Vab = np.array([enkf_curves[t]['Vab'] for t in times])
    cl1 = np.array([enkf_curves[t]['dx_ea_dx_eb'] for t in times])
    cl2 = np.array([enkf_curves[t]['ea_dx_eb'] for t in times])
    cl3 = np.array([enkf_curves[t]['eb_dx_ea'] for t in times])
    

    fig, ax = plt.subplots(2,2,figsize=(16,8),sharex= 'all',sharey='row')
    fig.suptitle(title,fontsize=15)
    
    term1 = -2*k2*A*Vab*sa/Va
    term2 = 2*k2*A*Vb**.5*sa**2*cl1/Va**.5
    term3 = k2*A*sa**2*cl3 * derive(Vb,axis=1) /(Va*Vb)**.5
    term4 = -k2*A*Vb**.5*sa**2*cl3*derive(Va,axis=1) / Va**1.5
    term5 = 2*k2*Vb**.5*sa**2*cl3*derive(A,axis=1)/Va**.5
    term6 = - u *derive(sa,axis=1)
    term7 = + 2* sa * derive(u)

    terms = np.array([term1, term2, term3, term4, term5, term6, term7])

    chemistry = term1 + term2+ term3 + term4 + term5
    advection = term6 + term7

    names = [fr'$W^A_{j}$' for j in range(1,8)]
    names = [r'$W^A_{chem-1}$', r'$W^A_{chem-2}$', r'$W^A_{chem-3}$', r'$W^A_{chem-4}$', r'$W^A_{chem-5}$',r'$W^A_{adv-1}$',r'$W^A_{adv-2}$'  ]
    colors_terms = ['darkolivegreen','lime','darkgoldenrod','y','g', 'm','plum']
    alphas = [0.2,0.4,0.6,.8,1,.5,1]
    for i, (term,col,alpha,name) in enumerate(zip(terms,colors_terms,alphas, names)):
        qty  = np.abs(term) / np.sum(np.abs(terms),axis=0) *100
        ax[0,0].plot(np.array(times)/characteristic_time, qty.mean(axis=1),label= name,c= col,alpha=1)

    ax[0,0].set_ylabel('%',fontsize=14)
    ax[0,0].legend(loc='right');


    #Decomposition into two processes : chemistry and advection")
    groups = [chemistry, advection]
    names = [r'$W_{chem}^A$',r'$W_{adv}^A$']
    colors_group = ['forestgreen','violet']
    ax[1,0].set_ylim(-5,105)
    for i, (nam, term,col) in enumerate(zip(names,groups,colors_group)):
        qty  = np.abs(term) / (np.abs(chemistry) + np.abs(advection))  * 100
        ax[1,0].plot(np.array(times)/characteristic_time, qty.mean(axis=1),label=nam + ', mean=' + f'{qty.mean():.1f}%'  ,c=col)
        ax[1,0].axhline(qty.mean(),linestyle='--',alpha=0.4,c=col)
#         print('Importance of process ' + nam + f' : {qty.mean()*100:.2f}%')
    ax[1,0].set_xlabel(r"$t/\tau_{adv}$",fontsize=14)
    ax[1,0].set_ylabel('%',fontsize=14)
    ax[1,0].legend();

    ax[0,1].set_title("$B$", fontsize=15)
    
    term1 = 2*k2*B*Vab*sb/Vb
    term2 = -2*k2*B*Va**.5*sb**2*cl1/Vb**.5
    term3 = + k2*B*Va**.5*sb**2*cl2*derive(Vb,axis=1) / Vb**1.5
    term4 = -k2*B*sb**2*cl2 * derive(Va,axis=1) /(Va*Vb)**.5
    term5 = -2*k2*Va**.5*sb**2*cl2*derive(B,axis=1)/Vb**.5
    term6 = - u *derive(sb,axis=1)
    term7 = + 2* sb * derive(u)

    terms = np.array([term1, term2, term3, term4, term5, term6, term7])

    chemistry = term1 + term2+ term3 + term4 + term5
    advection = term6 + term7

    

#     names = [fr'$W^B_{j}$' for j in range(1,8)]
    names = [r'$W^B_{chem-1}$', r'$W^B_{chem-2}$', r'$W^B_{chem-3}$', r'$W^B_{chem-4}$', r'$W^B_{chem-5}$',r'$W^B_{adv-1}$',r'$W^B_{adv-2}$'  ]
    alphas = [0.2,0.4,0.6,.8,1,.5,1]
    for i, (term,col,alpha,name) in enumerate(zip(terms,colors_terms,alphas,names)):
        qty  = np.abs(term) / np.sum(np.abs(terms),axis=0) *100
        ax[0,1].plot(np.array(times)/characteristic_time, qty.mean(axis=1),label= name,c= col,alpha=1)
#         print('Term ' + str(i+1) +f' : {qty.mean()*100:.3f}%')
        
#     ax[0,1].xlabel("$t/T$",fontsize=14)
    ax[0,1].legend(loc='right');


#     print("\nDecomposition into two processes : chemistry and advection")
    groups = [chemistry, advection]
    names = [r'$W_{chem}^B$',r'$W_{adv}^B$']
    colors = ['blue','red']
    ax[1,1].set_ylim(-5,105)
    for i, (nam, term,col) in enumerate(zip(names,groups,colors_group)):
        qty  = np.abs(term) / (np.abs(chemistry) + np.abs(advection))*100
        ax[1,1].plot(np.array(times)/characteristic_time, qty.mean(axis=1),label=nam + ', mean=' + f'{qty.mean():.1f}%'  ,c=col)
        ax[1,1].axhline(qty.mean(),linestyle='--',alpha=0.4,c=col)
    ax[1,1].set_xlabel(r"$t/\tau_{adv}$",fontsize=14)
    ax[1,1].legend();
    
    ax[0,0].set_title("(a) Relative contribution by term for $A$", fontsize=13)
    ax[0,1].set_title("(b) Relative contribution by term for $B$", fontsize=13)
    ax[1,0].set_title("(c) Relative contribution by process for $A$", fontsize=13)
    ax[1,1].set_title("(d) Relative contribution by process for $B$", fontsize=13)
    
    ax[0,0].text(0,text_height, 'advection',c='magenta',alpha=0.5,fontsize=25)
    ax[0,1].text(0,text_height, 'advection',c='magenta',alpha=0.5,fontsize=25)
    ax[0,0].text(0,5, 'chemistry',c='olive',alpha=0.5,fontsize=25)
    ax[0,1].text(0,5, 'chemistry',c='olive',alpha=0.5,fontsize=25)
    
    fig.tight_layout(h_pad= 1, w_pad=2);
    return fig

