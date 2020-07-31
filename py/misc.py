# Define units
import numpy
import galpy
import galpy.potential
import galpy.orbit
from galpy.potential import turn_physical_off
import astropy.units as apu
import matplotlib.pyplot as plt

def SampleIsoDisc(n,zmax,vmax,sig,p0,pot_sig):
    z= numpy.random.uniform(low=-zmax, high=zmax,size=n)
    v= numpy.random.uniform(low=-vmax, high=vmax,size=n)
    
    m= _iso_f(z,v,sig,p0,pot_sig)
    m= m/sum(m)
    
    v_shift= numpy.sum(v*m)/numpy.sum(m)
    v-= v_shift
    
    return z,v,m

def _iso_pot(z):
    p= 2.*numpy.log(numpy.cosh(z/(2.)))
    return p

def _iso_f(x,v,sig,p0,pot_sig):
    E=(0.5*v**2+_iso_pot(x))
    df= p0/numpy.sqrt(2*numpy.pi)*numpy.e**-E
    return df
'''    
def _iso_pot(z,p0,pot_sig):
    H= numpy.sqrt((pot_sig)**2/(4*p0*TwoPiG))/(SLU*1000)
    p= 2*(pot_sig/SVU)**2*numpy.log(numpy.cosh(z/(2*H)))
    return p

def _iso_f(x,v,sig,p0,pot_sig):
    E=(0.5*v**2+_iso_pot(x,p0,pot_sig))/(sig/SVU)**2
    df= p0/SDU/numpy.sqrt(2*numpy.pi)/(sig/SVU)*numpy.e**-E
    return df
'''
def findpeak(z_pos, mass):
    sindx= numpy.argsort(z_pos)
    cdf= numpy.cumsum(mass[sindx])
    cdf= cdf/cdf[-1] #at some point I should check that these are all summing to 1.
    return z_pos[sindx][numpy.argmin(abs(cdf-0.5))]

def Sgr_Force(tback,nt,sgr_pos=[-25.2,2.5,-6.4],sgr_vel=[-221.3,-266.5,197.4],
              sgr_pot=None,mw_pot=None,dynfric=False,corrections=True):
    
    MLU,MVU,MTU,MMU,MDU,MSDU= get_units(sgr_pot)

    # Define parameters of the orbits
    times = numpy.linspace(0, tback, nt)/MTU

    # Define the kinematics of Sag. Errors are averages of +/- interval. (Helmi+2018)
    # Use LH Bovy system.
    gcx, gcy, gcz = sgr_pos
    gcvx, gcvy, gcvz = sgr_vel
    
    if corrections:
        # Define the LSR kinematics
        # Peculiar velocites from astropy documentation (suggested). Solar peculiar
        # velocities from Schonrich+ 2012, Rotational velocity from Bovy+ 2012
        # Reverse helio_vx to account for Bovy LH system.
        helio_vx = -11.1
        helio_vy = 12.24 + 218
        helio_vz = 7.25

        # Get the galactocentric positions for Sag
        gcx += 8.
        gcy += 0.
        gcz += 0.0208

        # Get the galactocentric velocities for Sag
        gcvx += helio_vx
        gcvy += helio_vy
        gcvz += helio_vz

    # Convert to cylindrical coordinates
    gcr = numpy.sqrt( numpy.square(gcx) + numpy.square(gcy) )
    gcphi = numpy.arctan2( gcy, gcx )
    gcvr = gcvx * numpy.cos(gcphi) + gcvy * numpy.sin(gcphi)
    gcvphi = -gcvx * numpy.sin(gcphi) + gcvy * numpy.cos(gcphi)

    if mw_pot== None:
        # MW Potential
        mw_pot = galpy.potential.MWPotential2014
        
    if dynfric:
        if sgr_pot==None:
            # Non-changing parts of Sgr
            sgr_m, sgr_a = [8,16]

            sgr_stlr = galpy.potential.HernquistPotential( (6.4*10**8)*apu.M_sun, 0.85*apu.kpc )
            sgr_halo = galpy.potential.HernquistPotential(amp=sgr_m*(10**10)*apu.M_sun, a=sgr_a*apu.kpc)
            sgr_pot = [sgr_halo,sgr_stlr]

        turn_physical_off(sgr_pot)
        
        chand_dynfric = galpy.potential.ChandrasekharDynamicalFrictionForce(amp=1.,
                                                                    GMs=sgr_m*(10**10)*apu.M_sun, 
                                                                    rhm=(1+numpy.sqrt(2))*sgr_a*apu.kpc,
                                                                    dens=pot)
        mw_pot.append(chand_dynfric)

    turn_physical_off(mw_pot)
        
    # Initialize the orbits of Sgr and the Ophiuchus stream.
    o_sgr = galpy.orbit.Orbit(vxvv=[gcr/MLU, gcvr/MVU,
                                    gcvphi/MVU, gcz/MLU,
                                    gcvz/MVU, gcphi])

    # Now integrate
    o_sgr.integrate(times, mw_pot)
    
    return o_sgr, times


def calc_rho_A(z,m,nbins,nt,zmax):

    p= numpy.zeros([nbins,nt+1])
    p_err= numpy.zeros([nbins,nt+1])
    for i in range(nt+1):
        p[:,i],edge= numpy.histogram(z[:,i],bins=nbins,range=[-zmax,zmax],
                                     weights=m*(nbins/(2*zmax*1000)))
        p_err[:,i]= numpy.histogram(z[:,i],bins=nbins,range=[-zmax,zmax],
                                    weights=(m*(nbins/(2*zmax*1000)))**2)[0]
    
    mid=numpy.diff(edge)/2.+edge[:-1]
    A= (p-p[::-1])/(p+p[::-1])

    A_err= numpy.sqrt(4*p[::-1]**2/(p+p[::-1])**4*p_err**2+4*p**2/(p+p[::-1])**4*p_err[::-1]**2)

    return mid,p,p_err,A,A_err

def get_units(obj):
    
    from galpy.util.bovy_conversion import time_in_Gyr, mass_in_msol
    
    MLU= obj._ro
    MVU= obj._vo
    MTU= time_in_Gyr(MVU,MLU)*1000.
    MMU= mass_in_msol(MVU,MLU)
    MDU= MMU/MLU**3
    MSDU= MMU/MLU**2
    
    return MLU,MVU,MTU,MMU,MDU,MSDU