import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
import random
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import time
import random

df = pd.read_csv('nacas.csv')
NACAS = ['4415','4412','6412','2411']
df = df.loc[(df['naca'] == '4415') | (df['naca'] == '4412') | (df['naca'] == '6412') | (df['naca'] == '2411') | (df['naca'] == 'clarky')]

def Cx_data(naca, reynolds):
    return np.array(df.loc[(df['naca']==naca) & (df['reynolds'] == reynolds)]['CD'])

def Cz_data(naca, reynolds):
    return np.array(df.loc[(df['naca']==naca) & (df['reynolds'] == reynolds)]['CL'])

def Alpha_data(naca, reynolds):
    return np.array(df.loc[(df['naca']==naca) & (df['reynolds'] == reynolds)]['alpha'])

def Cx_spline(naca, reynolds):
    x = Alpha_data(naca, reynolds)
    y = Cx_data(naca,reynolds)
    spl = InterpolatedUnivariateSpline(x, y, k=5, ext = 3)
    return spl

def Cz_spline(naca, reynolds):
    x = Alpha_data(naca, reynolds)
    y = Cz_data(naca,reynolds)
    spl = InterpolatedUnivariateSpline(x, y, k=5, ext = 3)
    return spl

def simu(naca = '4412', N_elements = 50, r_min = 0.01, r_max = 0.07):

    # Créations des fonctions continues Cx, Cz
    func_Cx = dict()
    func_Cz = dict()
    for reynolds in [50000, 100000, 200000, 500000, 1000000]:
        func_Cx[str(reynolds)] = Cx_spline(naca, reynolds)
        func_Cz[str(reynolds)] = Cz_spline(naca, reynolds)

    # Discrétisation des rayons
    rayons = np.linspace(r_min, r_max, N_elements)

    # Calcul du pas associé
    delta = rayons[1]-rayons[0]

    n = 10000/60

    # Vitesse de flux d'air en vol stat définie par Froude : (sqrt(mg/8rhoS) pour une hélice entière)

    v_froude = np.sqrt(0.5 * 9.81 / (8 * (r_max**2-r_min**2)* np.pi))

    # Angle dû à la vitesse "froude"

    def beta(r):
        angle_rad = np.arctan(v_froude/(2*np.pi*n*r))
        angle_deg = angle_rad/np.pi*180
        return angle_deg

    # Fonction de vitesse
    def speed(r):
        v_t = 2*np.pi*r*n
        return np.sqrt(v_froude**2+v_t**2)

    # Fonction de surface d'un trapèze
    def area(l1, l2):
        return (np.abs(l1)+np.abs(l2))/2 * delta

    # Plus proche voisin d'un élément dans une liste

    def NN(element, L):
        l = len(L)
        cand = 0
        tol = np.abs(L[cand])+np.abs(element)
        for k in range(l):
            if np.abs(L[k]-element)<tol:
                cand = k
                tol = np.abs(L[k]-element)
        return cand

    # Fonction qui associe le Reynolds le plus proche du vrai Reynolds :

    def Reynolds(vitesse, L_c):
        mu = 1.56 * 10**-5
        all_reys = [50000,100000,500000,1000000]
        return all_reys[NN(L_c*vitesse/mu, all_reys)]

    # Calcul du Cx, Cz pour une géométrie donnée : 

    def C_global(thetas, chords):
        sz = 0
        sx = 0
        div = 0
        for i in range(N_elements-1):
            v_i = speed(rayons[i])
            S_i = area(chords[i], chords[i+1])
            reynolds = Reynolds(v_i, chords[i])
            C_x = func_Cx[str(reynolds)]
            sx+= v_i**2 * S_i * C_x(thetas[i]-beta(rayons[i]))
            C_z = func_Cz[str(reynolds)]
            sz+= v_i**2 * S_i * C_z(thetas[i]-beta(rayons[i]))

            div+= v_i**2 * S_i

        Cz = sz/div
        Cx = sx/div
        return Cz, Cx

    # Calcul du Cx induit pour une géométrie donnée :
    
    def C_induit(thetas, chords, Cz):
        e = 0.8
        S_tot = 0
        for i in range(N_elements-1):
            c_i = chords[i]
            c_i2 = chords[i+1]
            S_tot+=area(c_i,c_i2)

        allongement = (r_max - r_min)**2 / S_tot
        C_xi = Cz**2/(np.pi * allongement * e)
        return C_xi
    # Calcul de la puissance de trainee (contrainte) :

    def puissance_trainee(args):
        thetas = fonction_vrillage(args)
        chords = fonction_corde(args)
        P_tot = 0
        pho_air = 1.3
        S_tot = 0
        for i in range(N_elements-1):
            v_i = speed(rayons[i])
            S_i = area(chords[i],chords[i+1])
            reynolds = Reynolds(v_i, chords[i])
            f_C_x = func_Cx[str(reynolds)]
            Cx_loc = f_C_x(thetas[i]-beta(rayons[i]))
            P_tot += 2 * 1/2 * pho_air * v_i**3 * Cx_loc * S_i
            S_tot += S_i
        Cz, Cx = C_global(thetas, chords)
        C_xi = C_induit(thetas,chords, Cz)
        v_moy = np.mean(speed(rayons))
        P_tot += 2 * 1/2 * pho_air * v_moy**3 * C_xi * S_tot
        return P_tot
    
    def puissance_poussee(args):
        thetas = fonction_vrillage(args)
        chords = fonction_corde(args)
        P_tot = 0
        pho_air = 1.3
        for i in range(N_elements-1):
            v_i = speed(rayons[i])
            S_i = area(chords[i],chords[i+1])
            reynolds = Reynolds(v_i, chords[i])
            f_C_z = func_Cz[str(reynolds)]
            Cz_loc = f_C_z(thetas[i]-beta(rayons[i]))
            P_tot += 2 * 1/2 * pho_air * v_i**3 * Cz_loc * S_i
        return P_tot

    def puissance_totale(args):
        P = puissance_poussee(args) + puissance_trainee(args)
        return P

    def force_poussée(args):
        thetas = fonction_vrillage(args)
        chords = fonction_corde(args)
        F_tot = 0
        pho_air = 1.3
        Cz, Cx = C_global(thetas, chords)
        for i in range(N_elements-1):
            v_i = speed(rayons[i])
            S_i = area(chords[i], chords[i+1])
            reynolds = Reynolds(v_i, chords[i])
            f_C_z = func_Cz[str(reynolds)]
            Cz = f_C_z(thetas[i]-beta(rayons[i]))
            F_tot+= 1/2 * pho_air * v_i**2 * S_i * Cz
        return 2* F_tot

    def fonction_vrillage(args):
        #return args[2] * rayons**2 + args[1] * rayons + args[0]
        return args[0:N_elements]
    
    def fonction_corde(args):
        #return args[-1] * rayons**3 + args[-2] * rayons**2 + args[-3] * rayons + args[-4]
        return args[N_elements:]

    def func2optim(args):
        return -force_poussée(args)

    # Création de la contrainte :

    cons1 = NonlinearConstraint(puissance_totale, 0, 60)

    cons2 = NonlinearConstraint(fonction_corde,np.ones(N_elements)*0.008,np.ones(N_elements)*0.014)

    cons3 = NonlinearConstraint(fonction_vrillage,np.zeros(N_elements),np.ones(N_elements)*60)

    # Initialisation des coefficients : 

    #poly_vrillage_0 = [10,0,0]
    poly_corde_0 = [0.01 for k in range(N_elements)]
    #poly_corde_0 = [0.01, 0, 0,0]

    poly_vrillage_0 = [beta(rayons[k]) for k in range(N_elements)]
    #poly_corde_0 = [0.02 for k in range(N_elements)]


    x_init = np.array(poly_vrillage_0+poly_corde_0)

    # Exécution de l'optimisation :

    t_0 = time.time()

    res = minimize(func2optim, x0 = x_init, constraints = [cons1,cons2,cons3])

    print(f'Run in {int(time.time()-t_0)}s')
    print(res['success'], "in", res['nit'],"steps")

    fig, ax = plt.subplots(2,1, figsize=(10, 6))
    ax[0].plot(rayons,fonction_corde(res['x']))
    ax[0].set_title('Fonction de corde')
    ax[0].axis('equal')
    ax[0].fill_between(rayons, fonction_corde(res['x']), color='#539ecd')
    ax[1].set_title('Fonction de vrillage')
    ax[1].plot(rayons,fonction_vrillage(res['x']))

    val = res['fun']
    print(f"Force de portance {-val}N")
    args = res['x']
    print(f"Puissance de trainée = {puissance_trainee(args)}W")
    print(f"Puissance de portance = {puissance_poussee(args)}W")
    return rayons, res


rayons, res = simu(naca = 'clarky', N_elements = 1000)


