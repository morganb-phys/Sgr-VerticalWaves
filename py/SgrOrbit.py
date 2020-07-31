from galpy.orbit import Orbit
import astropy.units as apu
import pickle
import time
import galpy
from galpy.potential.mwpotentials import McMillan17, Irrgang13I, MWPotential2014
from misc import Sgr_Force
import numpy as np

np.random.seed(1)

FilePath= '/epsen_data/scr/bennett/Sgr-VerticalWaves/'
df_File='data/Orbits/dynfric/'
nf_File='data/Orbits/nofric/'

# Number of times to sample orbit
n_samp = 10010

# Define parameters of the orbits
n_snaps = 1000
t_orbit = -5.0 # Gyr
times = np.linspace(0, t_orbit, n_snaps) * apu.Gyr

### Orbit definitions

# Define the kinematics of Sgr. Errors are averages of +/- interval. (Helmi+2018)
# Use LH Bovy system.
ra,dec,d,ur,ud,vlos = [283.8313,-30.5453,26.,-2.692,-1.359,140.]
sd,sur,sud,svlos = [2.,0.001,0.001,2.]

### Sample orbit properties

# Sample the properties of Sag orbit
d_samp = np.random.normal(loc=d, scale=sd, size=n_samp)
ur_samp = np.random.normal(loc=ur, scale=sur, size=n_samp)
ud_samp = np.random.normal(loc=ud, scale=sud, size=n_samp)
vlos_samp = np.random.normal(loc=vlos, scale=svlos, size=n_samp)

mw_halo_props = np.array([1.,1.5,2.])
with open(FilePath+'data/SgrModel/sgr_rhm.dat','rb') as f:
    rhm= np.loadtxt(f)
with open(FilePath+'data/SgrModel/halo_masses.dat','rb') as f:
    halo_masses= np.loadtxt(f)
with open(FilePath+'data/SgrModel/stlr_masses.dat','rb') as f:
    stlr_masses= np.loadtxt(f)
sgr_masses= halo_masses+stlr_masses

bp= galpy.potential.PowerSphericalPotentialwCutoff(alpha=1.8,rc=1.9/8.,normalize=0.05,ro=8.178*apu.kpc)
dp= galpy.potential.MiyamotoNagaiPotential(a=3./8.,b=0.28/8.,normalize=.6,ro=8.178*apu.kpc)
hp= galpy.potential.NFWPotential(a=16/8.,normalize=.35,ro=8.178*apu.kpc)
    
for j in range(3):
    print('Orbits with no dynamical friction. Milky Way Model',j)

    pot= [bp,dp,hp*mw_halo_props[j]]
    galpy.potential.turn_physical_on(pot)

    # Initialize the orbits of Sgr
    init= [[ra*apu.deg,dec*apu.deg,d_samp[i]*apu.kpc,ur_samp[i]*apu.mas/apu.yr,
            ud_samp[i]*apu.mas/apu.yr,vlos_samp[i]*apu.km/apu.s] for i in range(n_samp)]
    o_sgr = galpy.orbit.Orbit(vxvv=init,radec=True,ro=8.178*apu.kpc)
    o_sgr.turn_physical_on()

    # Now integrate
    o_sgr.integrate(times, pot)

    fname= FilePath+nf_File+"osgr_halo"+str(j)+".pickle"
    print("Writing to",fname)
    pickle_out= open(fname,"wb")
    pickle.dump(o_sgr, pickle_out)
    pickle_out.close()
        
        
for j in range(3):
    print('MW model: ',j)
    for k in range(5):
        print('\t Sgr model: ',k)
    
        pot = [bp,dp,hp*mw_halo_props[j]]
        galpy.potential.turn_physical_on(pot)
        
        # Dynamical friction force
        chand_dynfric = galpy.potential.ChandrasekharDynamicalFrictionForce(amp=1.,
                                                                            GMs=sgr_masses[k]*apu.M_sun, 
                                                                            rhm=rhm[k]*apu.kpc,
                                                                            dens=pot,ro=8.178*apu.kpc)
        
        pot_df = pot+chand_dynfric
        galpy.potential.turn_physical_on(pot_df)

        # Initialize the orbits of Sgr and the Ophiuchus stream.

        init= [[ra*apu.deg,dec*apu.deg,d_samp[i]*apu.kpc,ur_samp[i]*apu.mas/apu.yr,
                ud_samp[i]*apu.mas/apu.yr,vlos_samp[i]*apu.km/apu.s] for i in range(n_samp)]
        o_sgr = galpy.orbit.Orbit(vxvv=init,radec=True,ro=8.178*apu.kpc)
        o_sgr.turn_physical_on()

        # Now integrate
        o_sgr.integrate(times, pot_df)
        o_sgr.turn_physical_off()
        
        fname= FilePath+df_File+"osgr_halo"+str(j)+"_sgr"+str(k)+".pickle"
        print('\t \t Saving sgr properties',sgr_masses[k]/10**9,'x 10^9 Msun',rhm[k],'kpc to ',fname)
        pickle_out= open(fname,"wb")
        pickle.dump(o_sgr, pickle_out)
        pickle_out.close()
        

realpot= [McMillan17,Irrgang13I]
for j in range(2):
    print('MW model: ',j+3)
    for k in range(5):
        print('\t Sgr model: ',k) 

        pot= realpot[j]
        MLU= galpy.util.bovy_conversion.get_physical(pot)['ro']
        MVU= galpy.util.bovy_conversion.get_physical(pot)['vo']
        galpy.potential.turn_physical_on(pot)
        
        chand_dynfric = galpy.potential.ChandrasekharDynamicalFrictionForce(amp=1.,
                                                                            GMs=sgr_masses[k]*apu.M_sun, 
                                                                            rhm=rhm[k]*apu.kpc,
                                                                            dens=pot,ro=MLU, vo=MVU)
        galpy.potential.turn_physical_on(chand_dynfric)
        pot_df = pot+chand_dynfric
        galpy.potential.turn_physical_on(pot_df)

        init= [[ra*apu.deg,dec*apu.deg,d_samp[i]*apu.kpc,ur_samp[i]*apu.mas/apu.yr,
                ud_samp[i]*apu.mas/apu.yr,vlos_samp[i]*apu.km/apu.s] for i in range(n_samp)]
        o_sgr = galpy.orbit.Orbit(vxvv=init,radec=True,ro=MLU,vo=MVU)
        o_sgr.turn_physical_on()
        
        # Now integrate
        o_sgr.integrate(times, pot_df)
        o_sgr.turn_physical_off()
        
        delattr(o_sgr,'_pot')
        
        fname= FilePath+df_File+"osgr_halo"+str(j+3)+"_sgr"+str(k)+".pickle"
        print('\t \t Saving sgr properties',sgr_masses[k],'x 10^9 Msun',rhm[k],'kpc to ',fname)
        with open(fname,"wb") as f:
            pickle.dump(o_sgr,f)