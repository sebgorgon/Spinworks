# adapted from 200604-RPavian-v.3
####################################################################################
# Preamble:
####################################################################################
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.linalg import block_diag as bd
import pandas as pd
import os

####################################################################################
# Plotting function:
####################################################################################
def plotres2(res, bvect, consts, k, fig=False, save=False):
    """Converts output of mesolve to dataarray of populations and generates figure"""
    Spop,T0pop,Tppop,Tnpop=[],[],[],[]
    for i in range(len(res.states)):
        Spop.append(np.real(res.states[i][8,8]))
        T0pop.append(np.real(res.states[i][9,9]))
        Tppop.append(np.real(res.states[i][10,10]))
        Tnpop.append(np.real(res.states[i][11,11]))
    Spop,T0pop,Tppop,Tnpop=np.array(Spop),np.array(T0pop),np.array(Tppop),np.array(Tnpop)
    Ttot=T0pop+Tppop+Tnpop
    Spopn,Ttotn,tn,T0popn = Spop[1:],Ttot[1:],t[1:],T0pop[1:]
    ratio=(Spopn-Ttotn)/(Spopn+Ttotn)*100
    T0frac=(T0popn)/(Ttotn)*100
    dataarray = pd.DataFrame(np.stack([Spop[1:],Ttot[1:],T0pop[1:],Tppop[1:],Tnpop[1:],ratio,T0frac], axis=1),index=t[1:],columns=['Spop','Ttot','T0pop','T+pop','T-pop','ratio','T0frac'])
    if fig==True:
        fig, ([ax1,ax3],[ax2,ax4]) = plt.subplots(2,2,figsize=(11,7))
        suptit=f"B field xyz = [{np.format_float_scientific(bvect[0], precision=2)},   {np.format_float_scientific(bvect[1], precision=2)},   {np.format_float_scientific(bvect[2], precision=2)}] T \n J_r = {consts[0]:.3e} ; J_pp = {consts[1]:.3e} ; D = {consts[2]:.3e} ; E = {consts[3]:.3e} mT \n k = {k} GHz"
        fig.suptitle(suptit)
        ax1.plot(t,Spop,'.-',label='S pop')
        ax1.plot(t,Ttot,'.-',label='T pop')
        ax1.plot(t,T0pop,'-',label='T0 pop')
        ax1.plot(t,Tppop,'-',label='T+ pop')
        ax1.plot(t,Tnpop,'-',label='T- pop')
        ax1.set_ylabel('level population')
        ax1.legend()
        ax2.plot(tn,ratio,label='% S excess')
        ax2.plot(tn,T0frac,label='% T0 in T')
        ax2.axhline(y=0,c='grey')
        ax2.legend()
        ax2.set_ylabel('%')
        ax2.set_xlabel('time / ns')
        ax3.semilogx(t,Spop,'.-',label=f"S pop fin = {Spop[-1]:.3f}")
        ax3.semilogx(t,Ttot,'.-',label=f"T pop fin = {Ttot[-1]:.3f}")
        ax3.semilogx(t,T0pop,'-',label=f"T0 pop fin = {T0pop[-1]:.3f}")
        ax3.semilogx(t,Tppop,'-',label=f"T+ pop fin = {Tppop[-1]:.3f}")
        ax3.semilogx(t,Tnpop,'-',label=f"T- pop fin = {Tnpop[-1]:.3f}")
        ax3.plot(0,0,lw=0,label=f"tot pp final = {(Spop[-1]+Ttot[-1]):.3f}")
        ax3.legend()
        ax4.semilogx(tn,ratio,label=f"% S excess fin = {ratio[-1]:.2f}")
        ax4.semilogx(tn,T0frac,label=f"% T0 in T fin = {T0frac[-1]:.2f}")
        ax4.axhline(y=0,c='grey')
        ax4.set_xlabel('time / ns')
        ax4.legend()
        if save==True:
            nam = f"J_r_{consts[0]:.3e} ; J_pp_{consts[1]:.3e} ; D_{consts[2]:.3e} ; Bz_{bvect[2]:.3e}"
            plt.savefig((nam+'.png'),dpi=300)
            #dataarray.to_csv((nam+'.csv'))
        plt.show()
    return dataarray

####################################################################################
# Conversion functions:
####################################################################################

def mTtoGHz(x):
    return 0.02802495*x

def Bfield(Ba0,p,t):
    return np.array([Ba0*np.cos(p)*np.sin(t), Ba0*np.sin(p)*np.sin(t), Ba0*np.cos(t)])

def dfsetindex(dflist, ixarray, ixtitle):
    newdfs = []
    for i in dflist:
        d = i.set_index(ixarray)
        d.index.name = ixtitle
        newdfs.append(d)
    return newdfs

def scale_ops(ops,k):
    """ Scales projection operators by recombination constant k """
    newops=[]
    for i in range(len(ops)):
        newops.append(k*ops[i])
    return newops

####################################################################################
# Pauli operators:
####################################################################################

Sx=sigmax()/2
Sy=sigmay()/2
Sz=sigmaz()/2
I=qeye(2)

####################################################################################
# Projectors:
####################################################################################

### NB: Operating in uncoupled representation (so s and t0 are explicitly linear combinations of up/down in projectors)

P1,P2,P3,P4,P5,P6,P7,P8=np.zeros((12,12)),np.zeros((12,12)),np.zeros((12,12)),np.zeros((12,12)),np.zeros((12,12)),np.zeros((12,12)),np.zeros((12,12)),np.zeros((12,12))

P1[8,1]=1; P1[8,2]=-1   #s  alpha   # should this matrix be normalised?
P2[9,1]=1; P2[9,2]=1    #t0 alpha   # should this matrix be normalised?
P3[10,0]=1              #t+ alpha
P4[11,3]=1              #t- alpha
P5[8,5]=1; P5[8,6]=-1   #s  beta    # should this matrix be normalised?
P6[9,5]=1; P6[9,6]=1    #t0 beta    # should this matrix be normalised?
P7[10,4]=1              #t+ beta
P8[11,7]=1              #t- beta

P1,P2,P3,P4,P5,P6,P7,P8=Qobj(P1),Qobj(P2),Qobj(P3),Qobj(P4),Qobj(P5),Qobj(P6),Qobj(P7),Qobj(P8)
c_ops = [P1,P2,P3,P4,P5,P6,P7,P8]
c_norm = [P1.unit(),P2.unit(),P3,P4,P5.unit(),P6.unit(),P7,P8]

####################################################################################
# Hamiltonian(s):
####################################################################################

def Hams2(J_r, J_pp, D, E, bvect, mu_B):

    """ Original Hamiltonian - now think form of zfs and order of states in tensors is wrong. """

    # 3 spin system: e1 and e2 PP + eR gxl - order: tensor(e1,e2,eR)

    # exchange
    H_ex_r = -2*J_r*(tensor(Sx,I,Sx)+tensor(Sy,I,Sy)+tensor(Sz,I,Sz))        # exchange e1 coupled to eR only
    H_ex_pp = -2*J_pp*(tensor(Sx,Sx,I)+tensor(Sy,Sy,I)+tensor(Sz,Sz,I))      # exchange e1 coupled to e2
    H_ex = H_ex_r + H_ex_pp

    # Zeeman
    g=2                                                                      # assume all spins same g value
    [Bx, By, Bz] = bvect
    H_Zee_pp = g*mu_B*(Bx*(tensor(I,Sx,I)+tensor(Sx,I,I))+By*(tensor(I,Sy,I)+tensor(Sy,I,I))+Bz*(tensor(I,Sz,I)+tensor(Sz,I,I)))
    H_Zee_r = g*mu_B*(Bx*tensor(I,I,Sx)+By*tensor(I,I,Sy)+Bz*tensor(I,I,Sz))
    H_Zee = H_Zee_pp + H_Zee_r

    # zfs
    H_zfs = D*(tensor(Sz,Sz)-2.0/3.0*(tensor(I,I)))+E*(tensor(Sx,Sx)-tensor(Sy,Sy))
    H_zfs2 = tensor(I,H_zfs)+tensor(H_zfs,I)

    # total
    H = H_ex + H_Zee + H_zfs2

    null4 = np.zeros((4,4))
    return Qobj(bd(H,null4))

def Hams2_eRe1e2_altzfs(J_r, J_pp, D, E, bvect, mu_B):

    """ Alternative Hamiltonian - tensor order changed and zfs acts only on PP """

    # swapped order from e1e2eR to eRe1e2
    # zfs acts only on pp

    # exchange
    H_ex_r = -2*J_r*(tensor(Sx,Sx,I)+tensor(Sy,Sy,I)+tensor(Sz,Sz,I))        # exchange e1 coupled to eR only
    H_ex_pp = -2*J_pp*(tensor(I,Sx,Sx)+tensor(I,Sy,Sy)+tensor(I,Sz,Sz))
    H_ex = H_ex_r + H_ex_pp

    # Zeeman
    g=2                                                                      # assume all spins same g value
    [Bx, By, Bz] = bvect
    H_Zee_pp = g*mu_B*(Bx*(tensor(I,Sx,I)+tensor(I,I,Sx))+By*(tensor(I,Sy,I)+tensor(I,I,Sy))+Bz*(tensor(I,Sz,I)+tensor(I,I,Sz)))
    H_Zee_r = g*mu_B*(Bx*tensor(Sx,I,I)+By*tensor(Sy,I,I)+Bz*tensor(Sz,I,I))
    H_Zee = H_Zee_pp + H_Zee_r

    # zfs?
    H_zfs = D*(tensor(Sz,Sz)-2.0/3.0*(tensor(I,I)))+E*(tensor(Sx,Sx)-tensor(Sy,Sy))
    H_zfs2 = tensor(I,H_zfs)

    # total
    H = H_ex + H_Zee + H_zfs2

    null4 = np.zeros((4,4))
    return Qobj(bd(H,null4))

def Hams2_eRe1e2_altzfs_doubleJr(J_r, J_pp, D, E, bvect, mu_B):

    """ Alternative Hamiltonian with gxl exchange to both PP spins """

    # swapped order from e1e2eR to eRe1e2
    # zfs acts only on pp
    # exchange between radical and both spins within pair

    # exchange
    H_ex_r = -2*J_r*(tensor(Sx,Sx,I)+tensor(Sy,Sy,I)+tensor(Sz,Sz,I)+tensor(Sx,I,Sx)+tensor(Sy,I,Sy)+tensor(Sz,I,Sz))
    H_ex_pp = -2*J_pp*(tensor(I,Sx,Sx)+tensor(I,Sy,Sy)+tensor(I,Sz,Sz))
    H_ex = H_ex_r + H_ex_pp

    # Zeeman
    g=2                                                                      # assume all spins same g value
    [Bx, By, Bz] = bvect
    H_Zee_pp = g*mu_B*(Bx*(tensor(I,Sx,I)+tensor(I,I,Sx))+By*(tensor(I,Sy,I)+tensor(I,I,Sy))+Bz*(tensor(I,Sz,I)+tensor(I,I,Sz)))
    H_Zee_r = g*mu_B*(Bx*tensor(Sx,I,I)+By*tensor(Sy,I,I)+Bz*tensor(Sz,I,I))
    H_Zee = H_Zee_pp + H_Zee_r

    # zfs
    H_zfs = D*(tensor(Sz,Sz)-2.0/3.0*(tensor(I,I)))+E*(tensor(Sx,Sx)-tensor(Sy,Sy))
    H_zfs2 = tensor(I,H_zfs)

    # total
    H = H_ex + H_Zee + H_zfs2

    null4 = np.zeros((4,4))
    return Qobj(bd(H,null4))

####################################################################################
# Initial states:
####################################################################################

# singlet initial state
psi0a=((fock(12,1)-fock(12,2)).unit()+(fock(12,5)-fock(12,6)).unit()).unit()

# mixed initial state
psi0b=((fock(12,1)-fock(12,2)).unit()+(fock(12,5)-fock(12,6)).unit()+      # 2 s    states
       (fock(12,1)+fock(12,2)).unit()+(fock(12,5)+fock(12,6)).unit()+      # 2 t0   states
        fock(12,0)+fock(12,3)+fock(12,4)+fock(12,7)).unit()                # 4 t+/- states

####################################################################################
# Calculation function:
####################################################################################

def runcalc2(Ham, consts, bvect, mu_B, t, k, psi0, c_ops, fig=False, save=False):

    [J_r, J_pp, D, E] = mTtoGHz(np.array(consts))            # unit conversion

    if Ham == 'original':                                    # choose and compute Hamiltonian
        H2=Hams2(J_r, J_pp, D, E, bvect, mu_B)
    elif Ham == 'eRe1e2_altzfs':
        H2=Hams2_eRe1e2_altzfs(J_r, J_pp, D, E, bvect, mu_B)
    elif Ham == 'eRe1e2_altzfs_doubleJr':
        H2=Hams2_eRe1e2_altzfs_doubleJr(J_r, J_pp, D, E, bvect, mu_B)

    c_new = scale_ops(c_ops, k)                              # compute recombination projectors
    res = mesolve(H2, psi0, t, c_new)                        # compute master equation
    dataarray = plotres2(res, bvect, consts, k, fig, save)   # compute populations
    return dataarray

####################################################################################
####################################################################################
####################################################################################

####################################################################################
# Parameters:
####################################################################################

mu_B = 13.99624493           # GHz/T
bz = 0                       # T
k = 0.1                      # GHz
t=np.logspace(-1,3,2000)     # ns

D = 10                       # mT  (converted to GHz in runcalc2)
E = D/10.0                   # mT  (converted to GHz in runcalc2)
J_r = 1                      # mT  (converted to GHz in runcalc2)
J_pp= 1                      # mT  (converted to GHz in runcalc2)

consts = [J_r, J_pp, D, E]
bvect  = [0, 0, bz]

####################################################################################
# Example calculation:
####################################################################################

r = runcalc2('eRe1e2_altzfs', consts, bvect, mu_B, t, k, psi0a, c_norm, fig=1, save=0)

####################################################################################
####################################################################################
####################################################################################

####################################################################################
# Interation function:
####################################################################################

def runiter(outer,inner,opar,ipar,pat,Ham,consts,bvect,mu_B,t,k,psi0,c_ops,fig=0,save=0):

    for outi in outer:
        finals = pd.DataFrame(columns=['Spop','Ttot','T0pop','T+pop','T-pop','ratio','T0frac'])
        Sevol, Tevol = pd.DataFrame(), pd.DataFrame()
        if os.path.isdir(pat)==False:
            os.mkdir(pat)
        os.chdir(pat)
        nam=namseg+str(outi)

        for ini in inner:
            consts,bvect=setpars(consts,bvect,outi,ini,opar,ipar)
            r = runcalc2(Ham, consts, bvect, mu_B, t, k, psi0, c_ops, fig=0, save=0)
            finals = finals.append(r.iloc[-1],ignore_index=True)
            Sevol, Tevol  = Sevol.append(r['Spop']), Tevol.append(r['Ttot'])

        [finals, Sevol, Tevol ] = dfsetindex([finals, Sevol, Tevol], inner, 'J_pp / mT')
        finals.to_csv((nam+'-finals.csv')), Sevol.to_csv((nam+'-Sevol.csv')), Tevol.to_csv((nam+'-Tevol.csv'))

# Iterate over 122x121 grid of J_r and J_pp values at zero field. Singlet inital state, normalised projectors, updated Hamiltonian

outer=np.concatenate(([0],np.logspace(-2,3,121)))
inner=np.logspace(-2,3,121)
opar,ipar = 'J_r','J_pp'
namseg='Jr_'

# Uncomment to run:
# runiter(outer,inner,opar,ipar,r"E:\3_MPT_PhD\200614-1a" ,'eRe1e2_altzfs',consts,[0,0,0],mu_B,t,k,psi0o,c_norm)
