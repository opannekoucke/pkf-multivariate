import numpy as np


def make_samples(ensemble_size, domain, lh=.06):
    gauss = lambda dx : np.exp(-0.5*dx**2/lh**2) # lh has been previously specified 
    correlation = gauss(domain-domain[domain.shape[0]//2])
    spectrum = np.abs(np.fft.fft(correlation))

    std_spectrum= np.sqrt(spectrum)
    samples = []
    for i in range(ensemble_size):
        zeta = np.random.normal(size=domain.shape)
        zeta = np.fft.fft(zeta)
        ef = np.fft.ifft(std_spectrum * zeta)
        ef = np.real(ef)
        samples.append(ef)
    samples = np.vstack(samples)
    return samples



def make_bivariate_ensemble(size, initial_setting, xx):
    # defining initial state :
    A0 = np.ones(len(xx)) * initial_setting['A']
    B0 = np.ones(len(xx)) * initial_setting['B']

    # generating initial error fields :
    perturbationsA = make_samples(size, xx, initial_setting['Length-scale A'])\
                        * initial_setting['std A']
    perturbationsB = make_samples(size, xx, initial_setting['Length-scale B'])\
                        * initial_setting['std B']

    As = A0 + perturbationsA
    Bs = B0 + perturbationsB
    ensemble0 = np.swapaxes(np.array([As,Bs]),0,1)
    return ensemble0


def draw_covariance_matrix(mean, covariance_matrix, radius=1,**args):
    """
    Draw the ellispe corresponding to the covariance matrix M at location the mean and of radius radius
    """
    Lambda, P = np.linalg.eigh(covariance_matrix)
    theta = np.linspace(0,2*np.pi,200)
    y1 = np.sqrt(Lambda[0]) * np.cos(theta)
    y2 = np.sqrt(Lambda[1]) * np.sin(theta)
    y = np.array([y1,y2])
    x1, x2 = np.dot(P,y) * radius
    x1 += mean[0]
    x2 += mean[1]
    plt.plot(x1,x2,**args)
    plt.scatter(mean[0],mean[1],marker='o',zorder=3,**args)
