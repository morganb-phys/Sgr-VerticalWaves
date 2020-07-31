import sys
sys.path.append('/home/bennett/VWCode/')

import misc
import astropy.units as apu
import wendy
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from units import *
from scipy.interpolate import interp1d

from galpy.potential import evaluatezforces, turn_physical_off, IsothermalDiskPotential, evaluatelinearForces, toVerticalPotential, evaluatelinearPotentials, verticalfreq
from galpy.util.bovy_conversion import force_in_2piGmsolpc2, force_in_pcMyr2,dens_in_msolpc3
import galpy 

import importlib
importlib.reload(misc)
import pickle

from multiprocessing import Pool

# Variables that need to be changed
def run_sim(o_sgr,apostr):
    sgr_m, sgr_a = [14.,13.]

    nt=2000
    n= int(1e6)
    nstr= str(int(n/10**int(np.floor(np.log10(n)))))+'e'+str(int(np.floor(np.log10(n))))

    zmax= 2.
    vzmax= 120.

    nbins= 201
    alpha=0.2
    astr= str(alpha).replace('.','')

    FilePath= '/epsen_data/scr/bennett/WaveModel/'
    times= o_sgr.t

    p_mid= 0.1/dens_in_msolpc3(MVU,MLU)
    mwd= IsothermalDiskPotential(p_mid,20.5/MVU)

    sgr_stlr = galpy.potential.HernquistPotential( (6.4*10**8)*apu.M_sun, 0.85*apu.kpc )
    sgr_halo = galpy.potential.HernquistPotential(amp=sgr_m*(10**10)*apu.M_sun, a=sgr_a*apu.kpc)
    sgr_pot = [sgr_halo,sgr_stlr]

    turn_physical_off(sgr_pot)

    def SgrForce(x,t):
        Force= evaluatezforces(sgr_pot,R=np.sqrt((o_sgr.x(t)-1.)**2+o_sgr.y(t)**2),z=(x*SLU/MLU-o_sgr.z(t)),
                               use_physical=False)*force_in_2piGmsolpc2(MVU,MLU)/1./SMU*(1000*SLU)**2   
        return Force

    def isoPot(pot,x):
        force= evaluatelinearForces(pot,x*SLU/MLU)*force_in_2piGmsolpc2(MVU,MLU)/1./SMU*(1000*SLU)**2
        return force

    try:
        z= np.loadtxt(FilePath+'py/Data/'+nstr+'/z.txt')
        v= np.loadtxt(FilePath+'py/Data/'+nstr+'/v.txt')
        f0= np.loadtxt(FilePath+'py/Data/'+nstr+'/f0.txt')

    except:    
        z,v,m= misc.SampleIsoDisc(n,zmax/SLU,vzmax/SVU,20.5*np.sqrt(2.3),0.1,20.5*np.sqrt(2.3))

        np.savetxt(FilePath+'py/Data/'+nstr+'/z.txt',z)
        np.savetxt(FilePath+'py/Data/'+nstr+'/v.txt',v)

        from galpy.actionAngle.actionAngleVertical import actionAngleVertical
        def calc_J(N,x,vx,pot):
            J= np.empty(N)
            for i in tqdm.trange(n):
                # Input: R,vR,vT [all 3 ignored], z,vz, pot=1D potential
                aAV= actionAngleVertical(0.1,0.1,0.1,x[i],vx[i],pot=pot) 
                J[i]= aAV.Jz()

            return J


        Jz= calc_J(n,z*SLU/MLU,v*SVU/MVU,mwd)
        sigma= 20.5/MVU
        try:
            freq= verticalfreq(mwd,1.)
        except:
            npts= 10001
            zz= np.linspace(-0.1,0.1,npts)
            z2deriv= np.gradient(np.gradient(evaluatelinearPotentials(mwd,zz),zz),zz)[int((npts-1)/2)]
            freq= np.sqrt(z2deriv)
        f0= 1./np.sqrt(2.*np.pi)/sigma*np.exp(-np.copy(Jz)*freq/sigma**2)
        f0= f0/np.sum(f0)

        np.savetxt(FilePath+'py/Data/'+nstr+'/f0.txt',f0)

        Jz=None
        Oz=None 

    dt= times[0]-times[1]
    
    p= np.zeros([nbins,nt+1])
    perr= np.zeros([nbins,nt+1])
    padj= np.zeros([nbins,nt+1])

    vv= np.zeros([nbins,nt+1])
    verr= np.zeros([nbins,nt+1])
    vadj= np.zeros([nbins,nt+1])
        
        
    # NEED TO MAKE THIS WRITE FIRST AND THEN APPEND
    p[:,0], edge= np.histogram(z*SLU,bins=nbins,range=[-zmax,zmax],
                          weights=f0*SSDU*(nbins/(2*zmax*1000)))
    vv[:,0]=np.histogram((z)*SLU,bins=nbins,range=[-zmax,zmax],
                        weights=f0*SSDU*(nbins/(2*zmax*1000))*v)[0]
    mid=np.diff(edge)/2.+edge[:-1]
    fz= interp1d(np.cumsum(p[:,0])/np.sum(p[:,0]),mid)
    fv= interp1d(np.cumsum(vv[:,0])/np.sum(vv[:,0]),mid)

    padj[:,0]= np.histogram((z-fz(0.5))*SLU,bins=nbins,range=[-zmax,zmax],
                            weights=f0*SSDU*(nbins/(2*zmax*1000)))[0]
    perr[:,0]= np.histogram((z)*SLU,bins=nbins,range=[-zmax,zmax],
                            weights=(f0*SSDU*(nbins/(2*zmax*1000)))**2)[0]
    vadj[:,0]= np.histogram((z)*SLU,bins=nbins,range=[-zmax,zmax],
                            weights=f0*SSDU*(nbins/(2*zmax*1000))*(v-fv(0.5)))[0]
    verr[:,0]= np.histogram((z)*SLU,bins=nbins,range=[-zmax,zmax],
                            weights=(f0*SSDU*(nbins/(2*zmax*1000))*v)**2)[0]

    gg= wendy.nbody(x=z,v=v,m=f0*alpha,dt=dt*MTU/STU,t0=0.,nleap=10,twopiG=1.,approx=True,sort='quick',
                      ext_force=lambda x,t:SgrForce(x,times[-1]+t*STU/MTU)+(1.-alpha)*isoPot(mwd,x))

        
    #ADJUST IT PROPERLY HERE!!!!!!!
    for i in tqdm.trange(nt):
        xt,vt= next(gg)

        p[:,i+1], edge= np.histogram(xt*SLU,bins=nbins,range=[-zmax,zmax],
                          weights=f0*SSDU*(nbins/(2*zmax*1000)))
        vv[:,i+1]=np.histogram((xt)*SLU,bins=nbins,range=[-zmax,zmax],
                        weights=f0*SSDU*(nbins/(2*zmax*1000))*vt)[0]
        mid=np.diff(edge)/2.+edge[:-1]
        fz= interp1d(np.cumsum(p[:,0])/np.sum(p[:,0]),mid)
        fv= interp1d(np.cumsum(vv[:,0])/np.sum(vv[:,0]),mid)

        padj[:,i+1]= np.histogram((xt-fz(0.5))*SLU,bins=nbins,range=[-zmax,zmax],
                                weights=f0*SSDU*(nbins/(2*zmax*1000)))[0]
        perr[:,i+1]= np.histogram((xt)*SLU,bins=nbins,range=[-zmax,zmax],
                                weights=(f0*SSDU*(nbins/(2*zmax*1000)))**2)[0]
        vadj[:,i+1]= np.histogram((xt)*SLU,bins=nbins,range=[-zmax,zmax],
                                weights=f0*SSDU*(nbins/(2*zmax*1000))*(vt-fv(0.5)))[0]
        verr[:,i+1]= np.histogram((xt)*SLU,bins=nbins,range=[-zmax,zmax],
                                weights=(f0*SSDU*(nbins/(2*zmax*1000))*vt)**2)[0]


    with open(FilePath+'py/Data/'+nstr+'/rho_'+str(nbins)+'_'+astr+'_'+apostr+'apo.dat','wb') as f1,\
         open(FilePath+'py/Data/'+nstr+'/rho_adj_'+str(nbins)+'_'+astr+'_'+apostr+'apo.dat','wb') as f2,\
         open(FilePath+'py/Data/'+nstr+'/rho_err_'+str(nbins)+'_'+astr+'_'+apostr+'apo.dat','wb') as f3,\
         open(FilePath+'py/Data/'+nstr+'/v_'+str(nbins)+'_'+astr+'_'+apostr+'apo.dat','wb') as f4, \
         open(FilePath+'py/Data/'+nstr+'/v_adj_'+str(nbins)+'_'+astr+'_'+apostr+'apo.dat','wb') as f5,\
         open(FilePath+'py/Data/'+nstr+'/v_err_'+str(nbins)+'_'+astr+'_'+apostr+'apo.dat','wb') as f6:

        np.savetxt(f1,p)
        np.savetxt(f2,padj)
        np.savetxt(f3,perr)
        np.savetxt(f4,vv)
        np.savetxt(f5,vadj)
        np.savetxt(f6,verr)
    
    mid=np.diff(edge)/2.+edge[:-1]
    with open(FilePath+'py/Data/'+nstr+'/mid_'+str(nbins)+'_'+astr+'_'+apostr+'apo.dat','wb') as f:
        np.savetxt(f,mid)
    f.close()
    
    return 0

if __name__ == '__main__':

    with open('/epsen_data/scr/bennett/WaveModel/py/Orbits/ModelOrbits/MW1_SGR1_1apo',"rb") as f:
        o1 = pickle.load(f)
    with open('/epsen_data/scr/bennett/WaveModel/py/Orbits/ModelOrbits/MW1_SGR1_2apo',"rb") as f:
        o2 = pickle.load(f)
    
    run_sim(o1,str(1))
    run_sim(o2,str(2))