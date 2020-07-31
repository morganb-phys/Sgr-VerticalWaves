import sys
sys.path.append('/home/bennett/VWCode/')

import misc
from units import SLU,SVU,STU,SMU,SDU,SSDU
import astropy.units as apu
import wendy
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from galpy.potential import evaluatezforces, turn_physical_off, IsothermalDiskPotential, evaluatelinearForces, toVerticalPotential, evaluatelinearPotentials, verticalfreq
from galpy.util.bovy_conversion import force_in_2piGmsolpc2, force_in_pcMyr2,dens_in_msolpc3
import galpy 

import importlib
importlib.reload(misc)
import pickle

from multiprocessing import Pool

# Variables that need to be changed
def run_sim(alpha):
    
    nt=2000
    n= int(1e6)
    nstr= str(int(n/10**int(np.floor(np.log10(n)))))+'e'+str(int(np.floor(np.log10(n))))

    zmax= 2.
    vzmax= 120.

    nbins= 201
    astr= str(alpha).replace('.','')

    FilePath= '/epsen_data/scr/bennett/WaveModel/'
    
    with open(FilePath+'py/Orbits/ModelOrbits/MW1_SGR1_1apo',"rb") as f:
        o_sgr = pickle.load(f)
    o_sgr.turn_physical_off()
    MLU,MVU,MTU,MMU,MDU,MSDU= misc.get_units(o_sgr)
    times= o_sgr.t

    p_mid= 0.1/dens_in_msolpc3(MVU,MLU)
    mwd= IsothermalDiskPotential(p_mid,20.5/MVU)

    StlrI= 1

    with open(FilePath+'py/Data/halo_masses.dat','rb') as f:
        halo_m= np.loadtxt(f)[StlrI]
    with open(FilePath+'py/Data/halo_radius.dat','rb') as f:
        halo_a= np.loadtxt(f)[StlrI]
    with open(FilePath+'py/Data/stlr_masses.dat','rb') as f:
        stlr_m= np.loadtxt(f)[StlrI]
    with open(FilePath+'py/Data/stlr_radius.dat','rb') as f:
        stlr_a= np.loadtxt(f)[StlrI]

    halo_pot= galpy.potential.HernquistPotential(amp=2.*halo_m*apu.M_sun,a=halo_a*apu.kpc,ro=MLU,vo=MVU) 
    stlr_pot= galpy.potential.HernquistPotential(amp=2.*stlr_m*apu.M_sun,a=halo_a*apu.kpc,ro=MLU,vo=MVU) 

    sgr_pot= halo_pot+stlr_pot
    turn_physical_off(sgr_pot)
    
    rotfreq= 1.

    self.Force= F
    return F
    def SgrForce(x,t):
        dx,dy,dz= rotfreq*np.cos(t)-o_sgr.x(t), rotfreq*np.sin(t)-o_sgr.y(t), x*SLU/MLU-o_sgr.z(t)
        dR= np.sqrt(dx**2.+dy**2.)

        F= evaluatezforces(sgr_pot,R=dR,z=dz,use_physical=False)*force_in_2piGmsolpc2(MVU,MLU)/1./SMU*(1000*SLU)**2 
        return F

    def isoPot(pot,x):
        force= evaluatelinearForces(pot,x*SLU/MLU)*force_in_2piGmsolpc2(MVU,MLU)/1./SMU*(1000*SLU)**2
        return force
    
    def nloglikelihood(params,data):
        N,z= data
        model= n_model(params,z)
        if (params[2]<0. or params[2]>5.):
            return np.inf
        if (params[1]<-2. or params[1]>2.):
            return np.inf
        loglike= -model+N*np.log(model)
        return -np.sum(loglike)

    def n_model(params,zdata):
        ln_n0,zsun,H1 = params
        n0= 10.**(ln_n0)
        return n0*(1./np.cosh((zdata+zsun)/(2.*H1))**2)

    def calc_sechfit(data,guess):
        fit= minimize(lambda x: nloglikelihood(x,[data[0],data[1]]),guess)
        return fit.x


    try:
        z= np.loadtxt(FilePath+'py/Data/'+nstr+'/z.txt')
        v= np.loadtxt(FilePath+'py/Data/'+nstr+'/v.txt')
        f0= np.loadtxt(FilePath+'py/Data/'+nstr+'/f0.txt')

    except:    
        z,v,m= misc.SampleIsoDisc(n,zmax/SLU,vzmax/SVU,20.5,0.1,20.5)

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
    padjerr= np.zeros([nbins,nt+1])
    
    vv= np.zeros([nbins,nt+1])
    verr= np.zeros([nbins,nt+1])
    vadj= np.zeros([nbins,nt+1])
    
    zsun= np.zeros([nt+1])
        
    # NEED TO MAKE THIS WRITE FIRST AND THEN APPEND
    p[:,0], edge= np.histogram(z*SLU,bins=nbins,range=[-zmax,zmax],
                          weights=f0*SSDU*(nbins/(2*zmax*1000)))
    vv[:,0]=np.histogram((z)*SLU,bins=nbins,range=[-zmax,zmax],
                        weights=f0*SSDU*(nbins/(2*zmax*1000))*v)[0]
    mid=np.diff(edge)/2.+edge[:-1]
    
    zsun[0]= calc_sechfit([p[:,0],mid],[np.log10(max(p[:,0])),0.0001,0.3])[1]

    padj[:,0]= np.histogram((z*SLU+zsun[0]),bins=nbins,range=[-zmax,zmax],
                            weights=f0*SSDU*(nbins/(2*zmax*1000)))[0]
    perr[:,0]= np.histogram((z)*SLU,bins=nbins,range=[-zmax,zmax],
                            weights=(f0*SSDU*(nbins/(2*zmax*1000)))**2)[0]
    padjerr[:,0]= np.histogram((z*SLU+zsun[0]),bins=nbins,range=[-zmax,zmax],
                               weights=(f0*SSDU*(nbins/(2*zmax*1000)))**2)[0]
    verr[:,0]= np.histogram((z)*SLU,bins=nbins,range=[-zmax,zmax],
                            weights=(f0*SSDU*(nbins/(2*zmax*1000))*v)**2)[0]

    gg= wendy.nbody(x=z,v=v,m=f0*alpha,dt=dt*MTU/STU,t0=0.,nleap=10,twopiG=1.,approx=True,sort='quick',
                      ext_force=lambda x,t:SgrForce(x,times[-1]+t*STU/MTU)+(1.-alpha)*isoPot(mwd,x))

         
    #ADJUST IT PROPERLY HERE!!!!!!!
    for i in tqdm.trange(nt):
        
        
        xt,vt= next(gg)

        fullp,edge= np.histogram(xt*SLU,bins=nbins,range=[-zmax+np.median(xt*SLU),zmax+np.median(xt*SLU)],
                          weights=f0*SSDU*(nbins/(2*zmax*1000)))
        mid=np.diff(edge)/2.+edge[:-1]
        zsun[i+1]= calc_sechfit([fullp,mid],[np.log10(max(fullp)),-np.median(xt*SLU),0.3])[1]
        p[:,i+1],edge= np.histogram(xt*SLU,bins=nbins,range=[-zmax,zmax],
                                    weights=f0*SSDU*(nbins/(2*zmax*1000)))
        mid=np.diff(edge)/2.+edge[:-1]
        vv[:,i+1]= np.histogram((xt)*SLU,bins=nbins,range=[-zmax,zmax],
                        weights=f0*SSDU*(nbins/(2*zmax*1000))*vt)[0]
        padj[:,i+1]= np.histogram((xt*SLU+zsun[i+1]),bins=nbins,range=[-zmax,zmax],
                                weights=f0*SSDU*(nbins/(2*zmax*1000)))[0]
        mid=np.diff(edge)/2.+edge[:-1]
        padjerr[:,i+1]= np.histogram((xt*SLU+zsun[i+1]),bins=nbins,range=[-zmax,zmax],
                                weights=(f0*SSDU*(nbins/(2*zmax*1000)))**2)[0]
        perr[:,i+1]= np.histogram((xt)*SLU,bins=nbins,range=[-zmax,zmax],
                                weights=(f0*SSDU*(nbins/(2*zmax*1000)))**2)[0]
        verr[:,i+1]= np.histogram((xt)*SLU,bins=nbins,range=[-zmax,zmax],
                                weights=(f0*SSDU*(nbins/(2*zmax*1000))*vt)**2)[0]


    with open(FilePath+'py/Data/'+nstr+'/rho_'+str(nbins)+'_'+astr+'.dat','wb') as f1,\
         open(FilePath+'py/Data/'+nstr+'/rho_adj_'+str(nbins)+'_'+astr+'.dat','wb') as f2,\
         open(FilePath+'py/Data/'+nstr+'/rho_err_'+str(nbins)+'_'+astr+'.dat','wb') as f3,\
         open(FilePath+'py/Data/'+nstr+'/rho_adj_err_'+str(nbins)+'_'+astr+'.dat','wb') as f4,\
         open(FilePath+'py/Data/'+nstr+'/v_'+str(nbins)+'_'+astr+'.dat','wb') as f5, \
         open(FilePath+'py/Data/'+nstr+'/v_adj_'+str(nbins)+'_'+astr+'.dat','wb') as f6,\
         open(FilePath+'py/Data/'+nstr+'/v_err_'+str(nbins)+'_'+astr+'.dat','wb') as f7,\
         open(FilePath+'py/Data/'+nstr+'/zsun_'+str(nbins)+'_'+astr+'.dat','wb') as f8:

        np.savetxt(f1,p)
        np.savetxt(f2,padj)
        np.savetxt(f3,perr)
        np.savetxt(f4,padjerr)
        np.savetxt(f5,vv)
        np.savetxt(f6,vadj)
        np.savetxt(f7,verr)
        np.savetxt(f8,zsun)
    
    mid=np.diff(edge)/2.+edge[:-1]
    with open(FilePath+'py/Data/'+nstr+'/mid_'+str(nbins)+'_'+astr+'.dat','wb') as f:
        np.savetxt(f,mid)
    f.close()
        
    return 0

if __name__ == '__main__':

    run_sim(0.01)