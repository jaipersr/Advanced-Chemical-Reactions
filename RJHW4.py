# -*- coding: utf-8 -*-
"""
Ryan Jaipersaud
Advanced Chemical Reactions
11/1/2018

The following code solves for the concentration profile of hydrocarbons as 
along the z direction of a non isothermal tubular reactor. The reactor is 95 
meters long with a diameter of 0.1 meters. The product of interest is ethylene.
The code simulataneous solves 9 fluxes ODEs, 1 pressure ODE and 1 temperature ODE
and then updates the  velocity using RK4. Functions were created to calculate 
specific heats, heats of reactions, thermal conductivities, overall heat transfer
and viscosity. Parameter such as the wall temperature, length of reactor, and feed
can be varied to increase the selectivity of ethylene and decrease the conversion
of ethane. If the conversion of ethane is above 0.6 the reactor will coke up.
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
    viscosity = numpy.zeros((9,1)) 
    Mu_matrix = numpy.array([[2.5906E-7, 0.67988, 98.902, 0],                       # Ethane
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
    mole_fractions = numpy.array([mole_fractions])
    mu = numpy.multiply(viscosity,mole_fractions) # multiply each component viscosity by its mole fraction
    mu = numpy.sum(mu) # sum the contributions to the viscosity
    
    return mu # the output is a scalar value of mu Pa*s

# The purpose of this function is to calculate the average thermal conductivities
# of a misture of species 
def thermal_conductivites(T,Fluxes):
#https://app.knovel.com/web/view/itable/show.v/rcid:kpDIPPRPF7/cid:kt00CZDV42/viewerType:eptble//root_slug:thermal-conductivity-vapor-phase/url_slug:dippr-proj-thermal-conductivity?filter=graph&b-toc-cid=kpDIPPRPF7&b-toc-root-slug=&b-toc-url-slug=dippr-proj-thermal-conductivity&b-toc-title=DIPPR%20Project%20801%20-%20Full%20Version&start=0&columns=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24&q=water
    parameters = numpy.array([[7.3869e-005,1.1689,500.73,0],      # ethane
                             [8.6806e-006,1.4559,299.72,-29403], # ethylene
                             [0.002653,0.7452,12,0],             # hydrogen
                             [-1.12,0.10972,-9834.6,-7.5358e+006], # propane
                             [8.3983e-006,1.4268,-49.654,0],       # methane
                             [4.49e-005,1.2018,421,0],              # propylene
                             [7.5782e-005,1.0327,-36.227,31432],    # acetylene
                             [-20890,0.9593,-9.382e+010,0],         # 1,3-butadiene
                             [6.2041e-006,1.3973,0,0] ] )           # water
    k_thermal = numpy.zeros((9,1))
     
    for i in range(9):
        #Y = = (A*(T**B)) / (1+(C/T) +(D/(T**2))) General form of equation taken for DIPPR
        k_thermal[i,0] =  (parameters[i,0]*(T**parameters[i,1])) / (1+(parameters[i,2]/T) +(parameters[i,3]/(T**2)))
        
    Total_Fluxes = numpy.sum(Fluxes[0:9])
    mole_fractions = numpy.divide(Fluxes[0:9],Total_Fluxes) # calculate the mole fraction of each species
    mole_fractions = numpy.array([mole_fractions])
    
    k = numpy.multiply(k_thermal,mole_fractions) # multiply each component viscosity by its mole fraction
    k_avr= numpy.sum(k) # sum the contributions to the viscosity
    
    return k_avr # W/K/m

# The purpose of this function is to return the specific heats of each
# individual species to calculate the heat of formation for the heat of reaction
# later on. This function also returns the average mass specific heat
def specific_heat(T,Fluxes):
# The link below shows where the parameters to calculate specifc heat were taken from
#https://app.knovel.com/web/view/itable/show.v/rcid:kpDIPPRPF7/cid:kt00CZDUX1/viewerType:eptble//root_slug:heat-capacity-ideal-gas/url_slug:heat-capacity-ideal-gas?filter=graph&b-toc-cid=kpDIPPRPF7&b-toc-root-slug=&b-toc-url-slug=heat-capacity-ideal-gas&b-toc-title=DIPPR%20Project%20801%20-%20Full%20Version&start=0&columns=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,25,26,20,21,22,23,24&q=ethylene
    # paramters give C_p ion units of J/kmol/K
    parameters = numpy.array([ [44256,84737,872.24,67130,2430.4], # ethane
                             [33380,94790,1596,55100,740.8],    # ethylene
                             [27617,9560,2466,3760,567.6],      # hydrogen
                             [59474,126610,844.31,86165,2482.7],# propane
                             [33298,79933,2086.9,41602,991.96],  # methane
                             [43852,150600,1398.8,74754,616.46], # propylene
                             [36921,31793,678.05,33430,3036.6], # acetylene
                             [50950,170500,1532.4,133700,685.6], # 1,3-butadiene
                             [33363,26790,2610.5,8896,1169] ] )  # water
    C_p = numpy.zeros((9,1))
    
    for i in range(9):
        #Y = = A + B*((C/(T*numpy.sinh(C/T)))**2) + D*((E/(T*numpy.cosh(E/T)))**2)  General form of equation taken for DIPPR
        C_p[i,0] =  (parameters[i,0] + parameters[i,1]*((parameters[i,2]/(T*numpy.sinh(parameters[i,2]/T)))**2) + parameters[i,3]*((parameters[i,4]/(T*numpy.cosh(parameters[i,4]/T)))**2))/1000  #J/mol/K

    Total_Fluxes = numpy.sum(Fluxes[0:9])
    mole_fractions = numpy.divide(Fluxes[0:9],Total_Fluxes) # calculate the mole fraction of each species
    mole_fractions = numpy.array([mole_fractions])
    
    C_p_avr = numpy.multiply(C_p,mole_fractions) # multiply each component viscosity by its mole fraction
    C_p_avr = numpy.sum(C_p_avr) # sum the contributions to the specific heat J/mol/K
    
    Mw = numpy.array([[30],[28],[2],[44],[16],[42],[26],[54],[18]])/1000 # kg/mol
    Mw_avr = numpy.sum(numpy.multiply(Mw,mole_fractions))
    C_p_avr = C_p_avr/Mw_avr # J/kg/K
    
    return C_p , C_p_avr # C_p_pure is a vector, C_p_avr is scalar, J/mol/K

# This function calculates the overall heat transfer coefficent which has been 
# approxiamted to be the convective heat transfer of the fluid
def overall_HT(T,Fluxes): 
    mu_bulk = Mu(T,Fluxes)    # calculate bulk mu based on temperature of fluid Pa*s
    Tw = 1173                 # Wall Temperature K
    mu_wall = Mu(Tw,Fluxes)   # calculate mu based on wall temperature of fluid Pa*s
    Re = 53.6/mu_bulk         # calculate Reynolds number
    
    C_p, C_p_avr = specific_heat(T,Fluxes)  # calculate average specific heat  J/Kg/K
    k_avr = thermal_conductivites(T,Fluxes) # calculate thermal conductivity W/m/K
    Pr = C_p_avr*mu_bulk/k_avr              # calculate Prandlt number
    
    Nu = 0.023*(Re**0.8)*(Pr**(1/3))*((mu_bulk/mu_wall)**0.14) # calculate Nusselt number based on correlation
    
    D = 0.1 # diameter  in meters
    h = k_avr*Nu/D # calculate convective heat transfer 
    
    rin = 0.05
    rout = 0.053
    U = h*(rin/rout) # calculate overall heat transfer
    return U # W/m^2/K

# This function determines the enthalpy of each reaction based on the heats of formation
# and the specific heat of each species.
def enthalpy_of_reaction(T,Fluxes):
    
    global M_v 
    C_p,C_p_avr = specific_heat(T,Fluxes) # get the specific heat of each species J/mol/K
   
    # https://app.knovel.com/web/view/itable/show.v/rcid:kpDIPPRPF7/cid:kt00CZDUR2/viewerType:itble//root_slug:thermodynamic-properties/url_slug:thermodynamic-properties?&filter=table&b-toc-cid=kpDIPPRPF7&b-toc-root-slug=&b-toc-url-slug=thermodynamic-properties&b-toc-title=DIPPR%20Project%20801%20-%20Full%20Version&start=0&columns=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23&q=ethane
    # enthalpy of formations in J/mol at 298.15 K
    H_f = numpy.array([[-8.382*(10**4)+ C_p[0,0]*(T - 298.15)], # ethane
                       [5.251*(10**4) + C_p[1,0]*(T - 298.15)], # ethylene
                       [0 + C_p[2,0]*(T - 298.15)],             #hydrogen
                       [-1.0468*(10**5) + C_p[3,0]*(T - 298.15)],# propane
                       [-7.452*(10**4) + C_p[4,0]*(T - 298.15)], # methane
                       [2.023*(10**4)+ C_p[5,0]*(T - 298.15)],   # propylene
                       [2.282*(10**5)+ C_p[6,0]*(T - 298.15)],   # acetylene
                       [1.0924*(10**5)+ C_p[7,0]*(T - 298.15)],  # 1,3-butadiene
                       [-2.8583*(10**5)+ C_p[8,0]*(T - 298.15)]]) # water
    
    H_rxn = numpy.matmul(M_v.T,H_f) # J/mol
    return H_rxn # J/mol
    
# The function below calculates the changes in flux for each species and pressure
# based on the fluxes at a point along the length of the reactor and the velocity at 
# that point and the temperature
def reactions(Fluxes,u):
    global M_v
    [NE,NY,NH,NP,NM,NL,NA,NB,NW,P,T] = Fluxes
    
    T = numpy.reshape(T,()) # makes T a scalar
    k = k_values(T) # Calculate rate constants
    R = numpy.array([[ -k[1]*(NY*NH/(u**2)) + k[0]*(NE/u)],
                    [k[2]*((NE/u)**1)],
                    [ - k[4]*( NA*NM/(u**2) ) + k[3]*(NL/u)],
                    [k[5]*( NA*NY/(u**2) )],
                    [k[6]*( NE*NY/(u**2) )]] )
    R = numpy.reshape(R,(5,1)) # R lives in 5 by 1
    
    # mu is needed for the Pressure ODE
    mu = Mu(T,Fluxes) # Pa * s
    
    # variables below are needed for the temperature ODE
    H_rxn = enthalpy_of_reaction(T,Fluxes) # Calculate heat of reaction for each reaction J/mol
    total_rxn_heat = numpy.matmul(H_rxn.T,R) # Calculate total heat of reaction J/m^3/s
    C_p_vector, C_p_avr = specific_heat(T,Fluxes) # calculate the average specifc heat J/kg/K
    U = overall_HT(T,Fluxes) # Calculate the overall heat transfer coefficent W/m^2/K

    # Rate and Pressure  and Tempertature ODEs
    product = numpy.matmul(M_v,R) # calculate the fluxes for each species
    dPdz  =  -312.62*u*(mu**0.25) # calculate the pressure change
    dTdz = -(total_rxn_heat/(536*C_p_avr)) + (2*U*(1173-T)/(26.8*C_p_avr)) -(u/(536*C_p_avr))* dPdz # calculate the temperature change
    N_vector = numpy.vstack((product,dPdz,dTdz))# vertically stack the fluxes, dPdz, and dTdz
    
    return N_vector # the output is a vector of flux evaluations lives in 11 by 1
    

# Main Function

# Inlet flow rates in moles per second for ethane, ethylene and stm
nEo = 99           
nYo = 1            
nWo = 66             

# Geometry of tubular reactor
d = 0.1              # Reactor diameter, meters
Ac =(math.pi*d**2/4)#*0.9   # Reactor cross-sectional area, meters squared

# Define the range for the independent variable z
L = 95        # Reactor length, meters
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
Po = 11*101325  # Inlet pressure in Pascals
To = 1093       # in Kelvin
R = 8.314       # Gas constant in J/mol/K


# This will hold fluxes for each species and pressure as the species move through the reactor
Fluxes = numpy.zeros([11,len(z)])
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
Fluxes[10,0] = To
v[0] = R*To*numpy.sum(Fluxes[0:9,0])/Po

# Define stoichiometric coefficents
M_v = numpy.array([[-1,-2,0,0,-1],     # ethane
                     [1,0,0,-1,-1],    # ethylene
                     [1,0,0,0,0],      # hydrogen
                     [0,1,0,0,0],      # propane
                     [0,1,1,0,1],      # methane
                     [0,0,-1,0,1],     # propylene
                     [0,0,1,-1,0],     # acetylene
                     [0,0,0,1,0],      # 1,3-butadiene
                     [0,0,0,0,0]])     # water

for i in range(len(z)-1):   
    u = v[i]
    # All ks are 11 by 1 where rows 0:8 specify a species, row 9 specifies pressure, row 10 specifies temperature
    Fluxes_i = numpy.array([Fluxes[:,i]]).T # This assigns the ith column of the Fluxes matrix to a temporary variable 
    k1 = h*reactions(Fluxes_i,u)
    k2 = h*reactions(numpy.add(Fluxes_i, (0.5*k1)) ,u)
    k3 = h*reactions(numpy.add(Fluxes_i, (0.5*k2)) ,u)
    k4 = h*reactions(numpy.add(Fluxes_i,      k3)  ,u)
    
    k = (k1 + 2*k2 + 2*k3 + k4)/6  
    Fluxes[:,i+1] = numpy.transpose(Fluxes_i + k) # iteration step for the next Fluxes column
    v[i+1] = R*Fluxes[10,i+1]*(numpy.sum(Fluxes[0:9,i+1]))/Fluxes[9,i+1] # iteration step for the next velocity
    
# The steps below calculates the total flux at each point along the reactor
# and then find the mole fractions at each point along the reactor
Total_Fluxes = numpy.sum(Fluxes[0:9,:],axis = 0)
Ethane_x = numpy.divide(Fluxes[0,:],Total_Fluxes)
Ethylene_x = numpy.divide(Fluxes[1,:],Total_Fluxes)

cash_flow_second = 358*Fluxes[1,-1]*Ac*28/1000000 # $/second
cash_flow_hr = cash_flow_second *3600
cash_flow_yr = cash_flow_hr * 8760

plt.figure(1)
plt.title('Mole fractions Along Length of Reactor')
plt.plot(z,Ethane_x,'b',label = 'Ethane fraction') 
plt.plot(z,Ethylene_x,'k',label = 'Ethylene fraction')
plt.xlabel('Distance (meters)')
plt.ylabel('mole fraction')
plt.ylim((0, 1))
plt.xlim((0,L))
plt.legend()
plt.grid()

# ethylene flux entering reactor minus ethylene flux leavving reactor
# divided by total flux minus the outlet flux of STM and ethane and inlet ethylene
Selectivity = (Fluxes[1,-1] - Fluxes[1,0]) / (Total_Fluxes[-1] - Fluxes[8,-1] - Fluxes[1,0] - Fluxes[0,-1]) 
print('Selectivity: ',Selectivity )

conversion_ethane = ( (Fluxes[0,0]) - (Fluxes[0,-1]) )/ (Fluxes[0,0])
print('conversion_ethane: ',conversion_ethane)
print('cash_flow per hour: ',cash_flow_hr)
print('cash_flow per year: ',cash_flow_yr)


#Plot T/Tin, P/Pin, v/vin
T_Tin = Fluxes[10,:]/To
P_Pin = Fluxes[9,:]/Po
v_vin = v/v[0]
plt.figure(2)
plt.title('T,P,v Along Length of Reactor')
plt.plot(z,T_Tin,'r',label = 'T/Tin') 
plt.plot(z,P_Pin,'b',label = 'P/Pin')
plt.plot(z,v_vin,'k',label = 'v/vin')
plt.xlabel('Distance (meters)')
plt.ylabel('Dimensionless')
plt.xlim((0,L))
plt.legend()
plt.grid()
