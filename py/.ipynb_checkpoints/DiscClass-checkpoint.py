from galpy.potential import MWPotential2014, toVerticalPotential
from galpy.util.bovy_conversion import get_physical
from galpy.actionAngle.actionAngleVertical import actionAngleVertical
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from galpy.orbit import Orbit
from scipy.integrate import simps, cumtrapz
from galpy.util import bovy_plot
import matplotlib.animation as animation
from IPython.display import HTML
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from galpy.potential import evaluatezforces,evaluateRforces, turn_physical_off, verticalfreq,evaluatelinearPotentials
from galpy.util.bovy_conversion import dens_in_msolpc3
import time
import tqdm
from scipy.interpolate import interp1d

class StellarDisc():
    
    def __init__(self,discpot,zlim,vlim,times=np.linspace(0,1,1001)*u.Gyr,zpt=101,vpt=101,zarray=False,varray=False):
    
        """
        NAME:
           __init__
        PURPOSE:
           Initialize a vertical disc in equilibrium
        INPUT:
           pot - should be a 1-dimensional potential
           zlim - maximum height of the galactic disc in kpc
           vlim - maximum velocity of the galactic disc in km/s
        OPTIONAL INPUTS:
           times - time array over which to integrate the disc orbits. 
                    Otherwise, it defaults to 1001 timesteps over 5 Gyr.
        OUTPUT:
           instance
        """

        turn_physical_off(discpot) #1-dimensional potential object from Galpy
        self.pot= discpot
        self.InternalUnits= get_physical(self.pot)
        if zarray:
            self.zmax=np.max(np.abs(zlim))
            self.znpt= len(zlim)
            self.z= zlim
        else:
            self.zmax= zlim/self.InternalUnits['ro']
            self.znpt= zpt
            self.z= np.linspace(-self.zmax,self.zmax,self.znpt) # in galpy internal units            
        if varray:
            self.vmax=np.max(np.abs(vlim))
            self.vnpt= len(vlim)
            self.v= vlim
        else:
            self.vmax= vlim/self.InternalUnits['vo']
            self.vnpt= vpt
            self.v= np.linspace(-self.vmax,self.vmax,self.vnpt) # in galpy internal units
        
        self.tt= times
            
        self.Jz,self.Oz= self.calc_JO() 
        self.integOrbits(self.tt)
    
    def calc_JO(self):
        J= np.empty([self.znpt,self.vnpt])
        O= np.empty([self.znpt,self.vnpt])
    
        for i in range(self.znpt):
            for j in range(self.vnpt):
                # Input: R,vR,vT [all 3 ignored], z,vz, pot=1D potential
                aAV= actionAngleVertical(0.1,0.1,0.1,self.z[i],self.v[j],pot=self.pot) 
                J[i,j]= aAV.Jz()
                O[i,j]= 2.*np.pi/aAV.Tz()
                if (abs(self.z[i])<1e-16) and (abs(self.v[j])<1e-16):
                    print("""One of your points was at (0,0) so we had to perturb it to: 
                    z= {0:.1e} and v= {1:.1e}
                    """.format(self.z[i+1]/100.,
                                self.v[j+1]/100.))
                    aAV= actionAngleVertical(0.1,0.1,0.1,self.z[i+1]/100.,self.v[j+1]/100.,pot=self.pot)
                    O[i,j]= 2.*np.pi/aAV.Tz()
                    
        return J,O
    
    def integOrbits(self,tsteps):
        vxvv= np.array(np.meshgrid(self.z,self.v)).T
        o= Orbit(vxvv,ro=self.InternalUnits['ro'],vo=self.InternalUnits['vo']) 
        o.turn_physical_off()
        o.integrate(tsteps,self.pot)
        self.discOrb= o
        
        return o
    
    def statSatForce(self,satpot,sat,tdep=False,method='slow',tstep=0):
            
        '''
        INPUTS
        
        satpot- 3D potential describing the shape of the potential
        sat- Orbit instance integrated over t
        t- numpy array with the timesteps over which the satellite was integrated
        '''
        if tdep:
            if method=='fast':
                nt= len(self.t)
                
                F= np.zeros([nt,nt,self.vnpt,self.znpt])
                
                satx,saty,satz= np.tile(sat.x(self.t),(nt,1)), np.tile(sat.y(self.t),(nt,1)), np.tile(sat.z(self.t),(nt,1))
                satx= np.array([np.roll(s,nt-int(i)) for  i,s in enumerate(satx)])[:,:,None]
                saty= np.array([np.roll(s,nt-int(i)) for  i,s in enumerate(saty)])[:,:,None]
                satz= np.array([np.roll(s,nt-int(i)) for  i,s in enumerate(satz)])[:,:,None]
                                
                for i in range(self.znpt):
                    F[:,:,:,i]= evaluatezforces(satpot,R=np.sqrt((satx-1.)**2+saty**2),
                                                z=(self.discOrb.x(self.t)[i,:,:,None].T-satz))
                    
                self.Force= F.T
                return F.T
      
            elif method=='slow':
                F= [None]*len(self.t)
                nt= len(self.t)
                for i in range(len(self.t)):
                    F[i]= evaluatezforces(satpot,
                                           R=np.sqrt((sat.x(self.t[(nt-i-1):])-1.)**2+sat.y(self.t[(nt-i-1):])**2),
                                           z=(self.discOrb.x(self.t[:i+1]))-sat.z(self.t[(nt-i-1):])) 
                    if i==0:
                        F[i]= F[i][:,:,None]
                self.Force= F
                return F
            
            elif method=='slowest':
                F= [None]*len(self.t)
                nt= len(self.t)
                for i in range(len(self.t)):
                    F[i]= evaluatezforces(satpot,
                                           R=np.sqrt((sat.x(self.t[(nt-i-1):])-1.)**2+sat.y(self.t[(nt-i-1):])**2),
                                           z=(self.discOrb[tstep].x(self.t[:i+1]))-sat.z(self.t[(nt-i-1):])) 
                    if i==0:
                        F[i]= F[i][:,None]
                return F
        else:
            F= evaluatezforces(satpot,R=np.sqrt((sat.x(self.t)-1.)**2+sat.y(self.t)**2),
                               z=(self.discOrb.x(self.t))-sat.z(self.t))
            
            self.Force= F
            return F
    
    def rotSatForce(self,satpot,sat,freq,tdep,method,tstep=0):
        
        if tdep:
            if method=='fast':
                 
                nt= len(self.t)
            
                discz= np.tile(np.array(self.discOrb.x(self.t)).T,(nt,1,1,1))
                satx,saty,satz= np.tile(sat.x(self.t),(nt,1)), np.tile(sat.y(self.t),(nt,1)), np.tile(sat.z(self.t),(nt,1))
                satx= np.array([np.roll(s,nt-int(i)) for  i,s in enumerate(satx)])
                saty= np.array([np.roll(s,nt-int(i)) for  i,s in enumerate(saty)])
                satz= np.array([np.roll(s,nt-int(i)) for  i,s in enumerate(satz)])[:,:,None,None]

                dx= np.tile(freq*np.cos(self.t),(nt,1))-satx
                dy= np.tile(freq*np.sin(self.t),(nt,1))-saty
                F= evaluatezforces(satpot,R=np.sqrt(dx**2+dy**2)[:,:,None,None],z=(discz-satz))
                discz,satx,saty,satz= np.array([[],[],[],[]])
                
                self.Force= F.T
                return F.T
            
            elif method=='slow':

                F= [None]*len(self.t)
                nt= len(self.t)
                for i in range(nt):
                    dx= freq*np.cos(self.t[(nt-i-1):])-sat.x(self.t[(nt-i-1):])
                    dy= freq*np.sin(self.t[(nt-i-1):])-sat.y(self.t[(nt-i-1):])
                    F[i]= evaluatezforces(satpot,R=np.sqrt(dx**2+dy**2),
                                           z=(self.discOrb.x(self.t[:i+1]))-sat.z(self.t[(nt-i-1):]))
                    if i==0:
                        F[i]= F[i][:,:,None]
                    
                self.Force= F
                return F

            elif method=='slowest':

                F= [None]*len(self.t)
                nt= len(self.t)
                for i in range(nt):
                    dx= freq*np.cos(self.t[(nt-i-1):])-sat.x(self.t[(nt-i-1):])
                    dy= freq*np.sin(self.t[(nt-i-1):])-sat.y(self.t[(nt-i-1):])
                    F[i]= evaluatezforces(satpot,R=np.sqrt(dx**2+dy**2),
                                           z=(self.discOrb[tstep].x(self.t[:i+1]))-sat.z(self.t[(nt-i-1):]))
                    if i==0:
                        F[i]= F[i][:,None]
                
                self.Force= F
                return F
            
        else:
            dx,dy,dz= freq*np.cos(self.t)-sat.x(self.t), freq*np.sin(self.t)-sat.y(self.t), self.discOrb.x(self.t)-sat.z(self.t)
            dR= np.sqrt(dx**2.+dy**2.)
            
            F= evaluatezforces(satpot,R=dR,z=dz)
            
            self.Force= F
            return F

    def calc_dJ(self,force,integ,tdep,method,tstep=0):
        
        vel= self.discOrb.vx(self.t)
        nt= len(self.t)
        if tdep:
            #Maybe put something that checks you aren't truing to integrate AND do time dependence
            if method=='fast':
                vt= np.reshape(np.tile(vel,nt),[self.znpt,self.vnpt,nt,nt])
                vt= np.tril(vt[:,:,:,::-1])[:,:,:,::-1]

                dJ= (np.sum(vt*np.transpose(force,(0,1,3,2)),axis=3)*(self.t[1]-self.t[0])/self.Oz[:,:,None])
                return dJ
            
            elif method=='slow':
                dJ= np.zeros([self.znpt,self.vnpt,len(self.t)])
                for i,f in enumerate(force):
                    dJ[:,:,i]= (np.sum(vel[:,:,:(i+1)]*f,axis=2)*(self.t[1]-self.t[0])/self.Oz)
                return dJ
            
            elif method=='slowest':
                dJ= np.zeros([self.vnpt,len(self.t)])
                for i,f in enumerate(force):
                    dJ[:,i]= (np.sum(vel[tstep,:,:(i+1)]*f,axis=1)*(self.t[1]-self.t[0])/self.Oz[tstep,:])
                return dJ
            
            else:
                print('Please choose between fast, slow, and slowest for the method of calculating force.')
                return None
        
        else:
            if integ:
                dJ= simps(vel*force,self.t,axis=2)/self.Oz[:,:,None]
            else:
                dJ= (np.sum(vel*force,axis=2)*(self.t[1]-self.t[0])/self.Oz)[:,:,None]
            return dJ
    
    ## Adding perturbations starts
    
    def add_satellite(self,satpot,sat,df_prop,integ=False,ftype='rotate',
                      rotFreq=1.,zsun='mean',tdep=True,method='slow'):
        
        turn_physical_off(satpot)
        sat.turn_physical_off()
        self.t= sat.time()
        
        #Check if the satellite has the same time array as the disc orbits
        if (len(self.t) != len(self.tt)) or (np.array(self.t)!=np.array(self.tt)).any():
            self.discOrb= self.integOrbits(self.t)
        else:
            pass
        
        if method!='slowest':

            if ftype=="static":
                force= self.statSatForce(satpot,sat,tdep,method=method)
            else:
                try: 
                    if (rotFreq.unit==u.km/u.s/u.kpc):
                        rotFreq= rotFreq.value/27.5
                    else:
                        rotFreq= rotFreq.to(u.km/u.s/u.kpc)
                        rotFreq= rotFreq.value/27.5
                except:
                    pass     
                force= self.rotSatForce(satpot,sat,rotFreq,tdep,method=method)
            
            self.add_force(force,self.t,df_prop,integ=integ,rotFreq=rotFreq,zsun=zsun,
                           tdep=tdep,method=method)
   
            return None
        elif method=='slowest':
            
            dJ= np.zeros([self.znpt,self.vnpt,len(self.t)])
            for i in tqdm.trange(self.znpt):
                #Calculate the force
                if ftype=="static":
                    force= self.statSatForce(satpot,sat,tdep,method=method,tstep=i)
                else:
                    try: 
                        if (rotFreq.unit==u.km/u.s/u.kpc):
                            rotFreq= rotFreq.value/27.5
                        else:
                            rotFreq= rotFreq.to(u.km/u.s/u.kpc)
                            rotFreq= rotFreq.value/27.5
                    except:
                        pass     
                    force= self.rotSatForce(satpot,sat,rotFreq,tdep,method=method)
            
                #Calculate deltaJ
                dJ[i]= self.calc_dJ(force,integ,tdep,method,tstep=i)
            
            self.deltaJ= dJ
            
            #calculate the perturbation (skip the add_force step)
            self.calc_pert(self.deltaJ,self.t,df_prop,integ=integ,rotFreq=rotFreq,zsun=zsun,
                           tdep=tdep,method=method)
            return None
    
    def VertFreq(self):
        return np.sqrt(self.z2deriv(0.))

    def z2deriv(self,z):
        npts= 10001
        zz= np.linspace(z-0.1,z+0.1,npts)
        return np.gradient(np.gradient(evaluatelinearPotentials(self.pot,zz),zz),zz)[int((npts-1)/2)]
    
    def add_force(self,force,t,df_stats,integ=False,rotFreq=1.,zsun='mean',
                  tdep=False,method='slow'):

        self.t= t
        if (len(self.t) != len(self.discOrb.time())) or (np.array(self.t)!=np.array(self.discOrb.time())).any():
            self.discOrb= self.integOrbits(self.t)
        else:
            pass
        self.Force= force
        self.deltaJ= self.calc_dJ(force,integ,tdep,method)
        self.calc_pert(self.deltaJ,t,df_stats,integ=integ,rotFreq=rotFreq,zsun=zsun,
                  tdep=tdep,method=method)
        
        return None
        
    def calc_pert(self,dJ,t,df_stats,integ=False,rotFreq=1.,zsun='mean',
                  tdep=False,method='slow'):    
        
        self.rho0= df_stats[0]
        self.sig= df_stats[1]
        
        self.df= IsoDF(self.rho0,self.sig)
        try:
            freq= verticalfreq(self.pot,1.)
        except:
            freq= self.VertFreq()
        self.f0= self.df.calc_df(self.Jz[:,:,None]+self.deltaJ,freq)
        
        if integ:
            self.rho= simps(self.f0,self.v,axis=1)
            self.meanV= simps(self.f0*self.v[None,:,None],axis=1)/simps(self.f0,axis=1)
        else:
            self.rho= np.sum(self.f0,axis=1)*(self.v[1]-self.v[0])
            self.meanV= np.sum(np.atleast_2d(self.v[None,:,None])*self.f0,axis=1)/self.f0.sum(axis=1)

        if zsun=='mean':
            #self.z0= -simps(self.rho.T*self.z,self.z)
            f= interp1d(np.cumsum(np.reshape(self.rho,self.znpt))/np.sum(self.rho),self.z)
            self.z0= f(0.5)
        elif zsun=='fit':
            
            if tdep:
                self.fit= np.array([self.calc_sechfit([self.rho[:,i],self.z],[np.log(np.max(self.rho[:,i])),0.,0.02]) for i in range(len(self.t))])
                self.z0= -self.fit[:,1]
            else:
                self.fit= self.calc_sechfit([self.rho[:,0],self.z],[np.log(np.max(self.rho[:,0])),0.,0.02])
                self.z0= -self.fit[1]
                                                                     
        elif zsun==None:
            self.z0=None
        else:
            print("""No method for finding zsun was specified, so it was assumed to be zero.\
            Please specify either 'mean' or 'fit' to adjust for movement of the disc""")
            self.z0=0.
        
        self.rawA= (self.rho-self.rho[::-1])/(self.rho+self.rho[::-1])
        
        #Calculate the asymmetry
        if tdep:
            try:
                if self.z0==None:
                    nt= int(len(self.t))
                    zA= np.tile(self.z,(nt,1)).T
                    self.zA= zA[int(self.znpt/2):]
                    self.A= self.rawA[int(self.znpt/2):]
            except:
                if self.z0.dtype=='float64':
                    nt= len(self.t)
                    
                    self.A= np.zeros([nt,self.znpt])
                    zf= np.zeros([nt,self.znpt])
                    self.zA= np.zeros([nt,self.znpt])
                    
                    for i,z0 in enumerate(self.z0):
                        zs= 2.*z0-self.z
                        zf= np.sort(np.append(self.z,zs))

                        funcp= interp1d(self.z,np.log(self.rho[:,i]), fill_value='extrapolate',kind='cubic')
                        p= np.exp(funcp(zf))
                        A= (p-p[::-1])/(p+p[::-1])

                        zA= zf-z0
                        self.zA[i]= zA[zA>=0.]
                        self.A[i]= A[zA>=0.]
 
                else:
                    print('Please choose fit, mean, or None as a method of finding zsun.')
                
        else:        

            if self.z0==None:
                zA= self.z
                self.zA= zA[zA>=0.]
                self.A= self.rawA[:,-1][zA>=0]
                    
            elif self.z0.dtype=='float64':
                zs= 2.*self.z0-self.z
                zf= np.sort(np.append(self.z,zs))

                funcp= interp1d(self.z,np.log(self.rho[:,0]), fill_value='extrapolate',kind='cubic')
                p= np.exp(funcp(zf))
                A= (p-p[::-1])/(p+p[::-1])

                zA= zf-self.z0
                self.zA= zA[zA>=0.]
                self.A= A[zA>=0.]
        return None
    
    def change_Iso(self,df_stats,integ=False,tdep=False):
        
        rho0= df_stats[0]
        sig= df_stats[1]
        
        df= IsoDF(rho0,sig)
        
        try:
            freq= verticalfreq(self.pot,1.)
        except:
            freq= self.VertFreq()
        f0= df.calc_df(self.Jz[:,:,None]+self.deltaJ,freq)
        
        if integ:
            rho= simps(f0,self.v,axis=1)
            meanV= simps(f0*self.v[None,:,None],axis=1)/simps(f0,axis=1)
        else:
            rho= np.sum(f0,axis=1)*(self.v[1]-self.v[0])
            meanV= np.sum(np.atleast_2d(self.v[None,:,None])*f0,axis=1)/f0.sum(axis=1)
                
        rawA= (rho-rho[::-1])/(rho+rho[::-1])
        
        if tdep:
            try:
                if self.z0==None:
                    nt= int(len(self.t))
                    zAA= np.tile(self.z,(nt,1))
                    zA= zAA[zAA>=0.]
                    A= rawA[zAA>=0.]
            except:
                if (zsun=='fit') or (zsun=='mean'):
                    nt= len(self.t)
                    A= np.zeros([nt,self.znpt*2])
                    zf= np.zeros([nt,self.znpt*2])
                
                    for i,z0 in enumerate(self.z0):
                        zs= 2.*z0-self.z
                        zf[i]= np.sort(np.append(self.z,zs))
                        
                        funcp= interp1d(self.z,np.log(rho[:,i]), fill_value='extrapolate',kind='cubic')
                        p= np.exp(funcp(zf[i]))
                        A[i]= (p-p[::-1])/(p+p[::-1])

                    A= A
                    zA= zf-self.z0[:,None]
                    zA= zA[zA>=0]
                else:
                    print('Please choose fit, mean, or None as a method of finding zsun.')
                
        else:        

            if self.z0==None:
                zAA= self.z
                zA= zAA[zAA>=0.]
                A= self.rawA[:,-1][zAA>=0]
                    
            if self.z0.dtype=='float64':
                zs= 2.*self.z0-self.z
                zf= np.sort(np.append(self.z,zs))

                funcp= interp1d(self.z,np.log(rho[:,0]), fill_value='extrapolate',kind='cubic')
                p= np.exp(funcp(zf))
                A= (p-p[::-1])/(p+p[::-1])

                zAA= zf-self.z0
                zA= zAA[zAA>=0.]
                A= A[zAA>=0.]
        
        return f0, rho, meanV, zA, A

    def nloglikelihood(self,params,data):
        N,z= data
        model= self.n_model(params,z)
        if (params[2]<0. or params[2]>5.):
            return np.inf
        if (params[1]<-2. or params[1]>2.):
            return np.inf
        loglike= -model+N*np.log(model)
        return -np.sum(loglike)

    def n_model(self,params,zdata):
        ln_n0,zsun,H1 = params
        n0= 10.**(ln_n0)
        return n0*(1./np.cosh((zdata+zsun)/(2.*H1))**2)

    def calc_sechfit(self,data,guess):
        fit= minimize(lambda x: self.nloglikelihood(x,[data[0],data[1]]),guess)
        return fit.x


    def plot_J(self):
        bovy_plot.bovy_print(axes_labelsize=17.,text_fontsize=12.,xtick_labelsize=15.,ytick_labelsize=15.)

        plt.figure(figsize=(10,8))
        plt.xlabel(r'$z\,\,(kpc)$')
        plt.ylabel(r'$v_z\,\,(km/s)$')
        plt.imshow(self.Jz[:,:,indx].T,aspect='auto',extent=[-self.zmax,self.zmax,-self.vmax,self.vmax])
        plt.colorbar()
    
    
    #Plotting functions
    
    def plot_orbit(self,mint=0,maxt=None):
        if maxt==None:
            maxt=len(self.tt)
        bovy_plot.bovy_print(axes_labelsize=17.,text_fontsize=12.,xtick_labelsize=15.,ytick_labelsize=15.)

        plt.figure(figsize=(14,7))
        plt.xlabel(r'$time\,\,(Gyr)$')
        plt.ylabel(r'$z\,\,(kpc)$')
        plt.plot(self.tt[mint:maxt],np.array(self.discOrb.x(self.tt[mint:maxt])).T)

    def plot_dJ(self,indx=-1):
        bovy_plot.bovy_print(axes_labelsize=17.,text_fontsize=12.,xtick_labelsize=15.,ytick_labelsize=15.)

        plt.figure(figsize=(7,6))
        plt.xlabel(r'$z\,\,(kpc)$')
        plt.ylabel(r'$v_z\,\,(km/s)$')
        plt.imshow(self.deltaJ[:,:,indx].T,aspect='auto',extent=[-self.zmax,self.zmax,-self.vmax,self.vmax])
        plt.colorbar()
        
    def plot_A(self,indx=-1):
        bovy_plot.bovy_print(axes_labelsize=17.,text_fontsize=12.,xtick_labelsize=15.,ytick_labelsize=15.)
        
        plt.plot(self.z*self.InternalUnits['ro'],self.A[:,indx])
        plt.xlabel(r'$z\,\,(kpc)$')
        plt.ylabel(r'$A$')
        
    def show_animate_A(self,skip=5,save=False,filename='A.gif'):

        self.skip= skip
        fig, ax= plt.subplots()
        ii= 0
        a= ax.plot(self.z*self.InternalUnits['ro'],self.A[:,0])
        ax.set_ylim(-0.5,0.5)
        ax.set_xlabel(r'$z\,\,(kpc)$')
        ax.set_ylabel(r'$A$')
        plt.tight_layout()
        
        anim = animation.FuncAnimation(fig,self.animate_A,fargs=[ax],
                                       frames=len(self.t)//skip,interval=40,blit=True,repeat=True)
        if save:
            anim.save(filename,writer='imagemagick',dpi=80)
        
        # The following is necessary to just get the movie, and not an additional initial frame
        plt.close()
        out= HTML(anim.to_html5_video())
        
        return out
        
    def animate_A(self,ii,axis):
        axis.clear()
        a= axis.plot(self.z*self.InternalUnits['ro']
                     ,self.A[:,ii*self.skip])
        axis.set_ylim(-np.max(self.A)*1.1,np.max(self.A)*1.1)
        axis.set_xlabel(r'$z\,\,(kpc)$')
        axis.set_ylabel(r'$A$')
        plt.tight_layout()
        return a  
    
    
class IsoDF():
    
    def __init__(self,Rho,Sigma):
        
        try:
            self.sig= [s.to(u.km/u.s).value/self.InternalUnits['vo'] for s in Sigma]
        except:
            #print('''You did not specify units for the velocity dispersion, 
            #so they have been assumed to be in galpy internal units.''')
            self.sig= Sigma
        
        try:
            self.rho0= [r.to(u.Msun/(u.pc**3)).value/dens_in_msolpc3(self.InternalUnits['vo'], self.InternalUnits['ro']) for r in Rho]
        except:
            #print('''You did not specify units for the mid-plane density, 
            #so they have been assumed to be in galpy internal units.''')
            self.rho0= Rho
            
    def calc_df(self,Jz,nu):
        f0= np.sum(np.array([self.rho0[i]/self.sig[i]*np.exp(-Jz*nu/self.sig[i]**2) for i in range(len(self.rho0))]),axis=0)/np.sqrt(2.*np.pi)
        return f0