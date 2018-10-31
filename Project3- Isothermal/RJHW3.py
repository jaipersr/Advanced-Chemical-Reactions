# -*- coding: utf-8 -*-
"""
Ryan Jaipersaud
Advanced Chemical Reactions
10/25/2018

The following code solves is for the production of ethylene. The code solves 
for the composition of hydrocarbons and hydrogen as a function of reactor length.
The code simulataneous solves 9 fluxes ODEs and 1 pressure ODE while updating 
the  velocity using RK4. 
"""
import numpy
import math
import matplotlib.pyplot as plt



# The purpose of the function below is to calculate rate constants for each
# reaction based on the temperature of the system
def k_values(T):
    R = 8.314/4184 # R in kcal/mol/K
    k1_f = (4.625*(10**13))*numpy.exp(-65.20/(R*T))
    k1_r = (8.75*(10**5))*numpy.exp(-32.70/(R*T))
    k2   = (3.850*(10**11))*numpy.exp(-65.25/(R*T))
    k5_f = (9.814*(10**8))*numpy.exp(-36.92/(R*T))
    k5_r = (5.87*(10**1))*numpy.exp(-7.043/(R*T))
    k6   = (1.026*(10**9))*numpy.exp(-41.26/(R*T))
    k8   = (7.083*(10**10))*numpy.exp(-60.43/(R*T))
    
    k_vector = numpy.array([k1_f,k1_r,k2,k5_f,k5_r,k6,k8])
    k_vector = numpy.array([k_vector]).T
  
    return k_vector # the output is a vector of rate constant values

# The purpose of the below function is to calculate the viscosity as a function of
# temperature and the fluxes at a specific point along the length of the reactor
def Mu(T,Fluxes):
    #global J
    viscosity = numpy.zeros((9,1))
    
    Mu_matrix = numpy.array([[2.5906E-7, 0.67988, 98.902, 0],   # Ethane
          [2.0789E-6, 0.4163, 352.7, 0 ],                       # Ethylene
          [1.797E-7, 0.685, -0.59, 140],                        # Hydrogen
          [4.905E-8, 0.90125, 0, 0],                            # Propane
          [5.2546E-7, 0.59006, 105.67, 0],                      # Methane
          [7.3919E-7, 0.5432, 263.73, 0],                       # Propylene
          [1.2025E-6, 0.4952, 291.4, 0],                        # Acetylene
          [2.696E-7, 0.6715, 134.7, 0],                         # 1,3-butadiene
          [1.7096E-8, 1.1146, 0, 0] ]  )                        # Water
    
    for i in range(9):
        viscosity[i,0] = Mu_matrix[i,0]*(T**Mu_matrix[i,1]) / (1 + (Mu_matrix[i,2]/T) + (Mu_matrix[i,3]/(T**2)) ) # Calculate the viscosity of each component
        
    Total_Fluxes = numpy.sum(Fluxes[0:9])
   
    mole_fractions = numpy.divide(Fluxes[0:9],Total_Fluxes) # calculate the mole fraction of each species
    #J = mole_fractions
    mole_fractions = numpy.array([mole_fractions])
    
    mu = numpy.multiply(viscosity,mole_fractions) # multiply each component viscosity by its mole fraction
    mu = numpy.sum(mu) # sum the contributions to the viscosity
    
    return mu # the output is a scalar value of mu

# The function below calculates the changes in flux for each species and pressure
# based on the fluxes at a point along the length of the reactor and the velocity at 
# that point and the temperature
def reactions(T,Fluxes,u):
    [NE,NY,NH,NP,NM,NL,NA,NB,NW,P] = Fluxes
        
    mu = Mu(T,Fluxes) # Calculate mu. This will change Pressure.
   
    k = k_values(T) # Calculate rate constants. This will change reaction rates
    
    # Rate and Pressure ODEs
    RE = -k[0]*(NE/u) + k[1]*( NY*NH/(u**2) ) - 2*k[2]*((NE/u)**1) - k[6]*(NE*NY/(u**2))
    RY =  k[0]*(NE/u) - k[1]*( NY*NH/(u**2) ) - k[5]*( NA*NY/(u**2) ) - k[6]*( NE*NY/(u**2) )
    RH =  k[0]*(NE/u) - k[1]*( NY*NH/(u**2) )
    RP =  k[2]*((NE/u)**1)
    RM =  k[2]*((NE/u)**1) + k[3]*(NL/u) - k[4]*( NA*NM/(u**2) ) + k[6]*( NE*NY/(u**2) )
    RL = -k[3]*(NL/u) + k[4]*( NA*NM/(u**2) ) + k[6]*( NE*NY/(u**2) )
    RA =  k[3]*(NL/u) - k[4]*( NA*NM/(u**2) ) - k[5]*( NA*NY/(u**2) )
    RB =  k[5]*( NA*NY/(u**2) )
    RW =  0
    P  =  -312.62*u*(mu**0.25)
    
    R_vector = numpy.hstack((RE,RY,RH,RP,RM,RL,RA,RB,RW,P))
    R_vector = numpy.array([R_vector]).T
     
    return R_vector # the output is a vector of flux evaluations
    

# Main Function

# Inlet flow rates in moles per second for ethane, ethylene and stm
nEo = 99           
nYo = 1            
nWo = 66             

# Geometry of tubular reactor
d = 0.1              # Reactor diameter, meters
Ac =math.pi*d**2/4   # Reactor cross-sectional area, meters squared

# Define the range for the independent variable z
L = 95         # Reactor length, meters
z0 = 0         # Initial length (inlet to reactor)
zmax = L       # Final length (end of reactor)
h = 0.1        # Step size for RK4

z = numpy.linspace(z0,zmax,zmax/h+1) # Evenly divide the space interval

# Initial conditions for all species, Pressure and Temperature
NEo = nEo/Ac
NYo = nYo/Ac
NHo = 0
NPo = 0
NMo = 0
NLo = 0
NAo = 0
NBo = 0
NWo = nWo/Ac
Po = 11*101325 # Inlet pressure in Pascals
T = 1093       # in Kelvin
R = 8.314      # Gas constant in J/mol/K


# This will hold fluxes for each species and pressure as the species move through the reactor
Fluxes = numpy.zeros([10,len(z)])
Fluxes = numpy.array(Fluxes)
v = numpy.zeros(len(z))


# First guess is the initial conditions
Fluxes[0,0] = NEo
Fluxes[1,0] = NYo
Fluxes[2,0] = NHo
Fluxes[3,0] = NPo
Fluxes[4,0] = NMo
Fluxes[5,0] = NLo
Fluxes[6,0] = NAo
Fluxes[7,0] = NBo
Fluxes[8,0] = NWo
Fluxes[9,0] = Po
v[0] = R*T*numpy.sum(Fluxes[0:9,0])/Po


for i in range(len(z)-1):   
    u = v[i]
    # All ks are 10 by 1 to where rows 0:8 specify a species and row 9 specifies pressure
    Fluxes_i = numpy.array([Fluxes[:,i]]).T # This assigns the ith column of the Fluxes matrix to a temporary variable 
    k1 = h*reactions(T,Fluxes_i,u)
    k2 = h*reactions(T, numpy.add(Fluxes_i, (0.5*k1)) ,u)
    k3 = h*reactions(T, numpy.add(Fluxes_i, (0.5*k2)) ,u)
    k4 = h*reactions(T, numpy.add(Fluxes_i,      k3)  ,u)
    
    k = (k1 + 2*k2 + 2*k3 + k4)/6  
    Fluxes[:,i+1] = numpy.transpose(Fluxes_i + k) # iteration step for the next Fluxes column
    v[i+1] = R*T*(numpy.sum(Fluxes[0:9,i+1]))/Fluxes[9,i+1] # iteration step for the next velocity

# The steps below calculates the total flux at each point along the reactor
# and then find the mole fractions at each point along the reactor
Total_Fluxes = numpy.sum(Fluxes[0:9,:],axis = 0)
Ethane_x = numpy.divide(Fluxes[0,:],Total_Fluxes)
Ethylene_x = numpy.divide(Fluxes[1,:],Total_Fluxes)

plt.figure(1)
plt.title('Mole fractions Along Length of Reactor')
plt.plot(z,Ethane_x,'b',label = 'Ethane fraction') 
plt.plot(z,Ethylene_x,'k',label = 'Ethylene fraction')
plt.xlabel('Distance (meters)')
plt.ylabel('mole fraction')
plt.ylim((0, 1))
plt.xlim((0,95))
plt.legend()
#fname='mole_fraction.pdf'
#plt.savefig(fname)

# ethylene flux entering reactor minus ethylene flux leavving reactor
# divided by total flux minus the outlet flux of STM and ethane and inlet ethylene
Selectivity = (Fluxes[1,-1] - Fluxes[1,0]) / (Total_Fluxes[-1] - Fluxes[8,-1] - Fluxes[1,0] - Fluxes[0,-1]) 
print('Selectivity: ',Selectivity )

conversion_ethane = ( (Fluxes[0,0]) - (Fluxes[0,-1]) )/ (Fluxes[0,0])
print('conversion_ethane:',conversion_ethane)

#print('Fluxes')
#print(numpy.sum(numpy.array([Fluxes[0:9,-1]]).T))


#mu = Mu(1093,Fluxes[:,0])
#k_vector = k_values(1093)
#print(mu)
#print('---')

#print(k_vector)
