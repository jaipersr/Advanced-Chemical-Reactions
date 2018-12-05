# -*- coding: utf-8 -*-
"""
Ryan Jaipersaud
ChE 421 Advance Chemical Reactions
12/5/18

The following code models a non isothermal packed bed reactor that is producing
pthalic anhydride from ortho xylene. The reactions is very exothermic and produces
-307 kcal/mol of ortho xylene reacted. The code simultaneously solves for the 
concentration of ortho xylene and the temperature at all points in the reactor.
Plots of the maximum centerline temperature are generated as functions of the 
wall temperature at various inlet feed temperatures. The maximum allowable center
line temperature for this system is 690 K. Wall temperatures that allow maximum 
conversion of orthoxylene without having the centerline temperature go above the 
limit are printed in a data frame.
Problem parameters taken from below
Nauman,E.; Chemical Reactor Design, Optimization and Scaleup, 2nd ed.; John Wiley & Sons: New Jersey,2008;pp 324 - 329.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# The function below implements method of lines to reduce a system of  partial 
# differential equations to a system of ordinary differential equations and then 
# solve the ODE using Euler's method. The output consists of two matrixes detailing 
# the concentration and temperature at all point in the reactor, the maximum center
# line temperature and the pressure at each point along the length of the reactor.
def Method_of_Lines(T_wall,T_inlet):
    # Parameters
    d = 0.05             # diameter in m
    L = 5                # length of reactor in m
    r = d/2              # radius in m
    D = 3*10**-4         # effective radial diffusivity m^2/s
    alpha = 1.6*10**-3   # effective radial thermal conductivity  m^2/s
    hr = 150             # heat transfer coefficent cal/s/K
    k = 0.5              # thermal conductivity cal/m/s/K
    I = 5                # number of radial increments
    dr = r/I             # radial step size
    dz = dr**2/(4*alpha) # axial step size
    M = int(L/dz) + 1    # number of axial increments
    
    C_Matrix = np.zeros((I+1,M+1))  # Concentration of ortho xylene matrix
    T_Matrix = np.zeros((I+1,M+1))  # Temperature Matrix
    P_Array = np.zeros((1,M+1))     # Pressure Array 
    
    # The following declarations are for intializing the boundary conditions
    # at the wall. 
    C_Matrix[:,0] = 0.415    # inlet concentration in mol/m^3
    T_Matrix[:,0] = T_inlet  # inlet temperature in K
    P_Array[0] = 222915      # inlet pressure in Pa
    
    for j in range(1,C_Matrix.shape[1]): # Marches along axial direction
        for i in range(0,C_Matrix.shape[0]): # Marches in the radial direction
            
            if i == 0 : # coefficents for the centerline boundary condition
                # a,b,c are used to represent coefficents in concentration PDE
                a = (2*D)/(dr**2) # C+
                b = (-4*D)/(dr**2) # Co
                c = a # C-
                
                # d,e,f are used to represent coefficents in temperature PDE
                d = (2*alpha)/(dr**2) # E+
                e = (-4*alpha)/(dr**2) # Eo
                f = d # E-
            else: # coefficents for anywhere else in the reactor
                a = (D*(2*i+1))/(2*i*dr**2)
                b = (-2*D)/(dr**2)
                c = (D*(2*i-1))/(2*i*dr**2)
                d = (alpha*(2*i+1))/(2*i*dr**2)
                e = (-2*alpha)/(dr**2)
                f = (alpha*(2*i-1))/(2*i*dr**2)
            
            # Depending on what point the for loop is at in the reactor the iterative step
            # for determining the concentration and temperature will change
            if i == I: # wall boundary conditions for concentration and temperature
                C_Matrix[i,j] = (18*C_Matrix[i-1,j] - 9*C_Matrix[i-2,j] + 2*C_Matrix[i-3,j])/11 
                T_Matrix[i,j] = (6*dr*hr*T_wall + 18*k*T_Matrix[i-1,j] - 9*k*T_Matrix[i-2,j] + 2*k*T_Matrix[i-3,j])/(6*dr*hr + 11*k)
            elif i == 0: # centerline boundary conditions for concentration and temperature
                C_Matrix[i,j] = C_Matrix[i,j-1] + dz*(a*C_Matrix[1,j-1] + b*C_Matrix[i,j-1] + c*C_Matrix[1,j-1] - (4.12*10**8)*np.exp(-13636/T_Matrix[i,j-1])*C_Matrix[i,j-1]) # C is minimum at r = 0, dc/dr = 0
                T_Matrix[i,j] = T_Matrix[i,j-1] + dz*(d*T_Matrix[1,j-1] + e*T_Matrix[i,j-1] + f*T_Matrix[1,j-1] + (4.11*10**11)*np.exp(-13636/T_Matrix[i,j-1])*C_Matrix[i,j])
            else: # any other point in the reactor
                C_Matrix[i,j] = C_Matrix[i,j-1] + dz*(a*C_Matrix[i+1,j-1] + b*C_Matrix[i,j-1] + c*C_Matrix[i-1,j-1] -(4.12*10**8)*np.exp(-13636/T_Matrix[i,j-1])*C_Matrix[i,j-1])
                T_Matrix[i,j] = T_Matrix[i,j-1] + dz*(d*T_Matrix[i+1,j-1] + e*T_Matrix[i,j-1] + f*T_Matrix[i-1,j-1] + (4.11*10**11)*np.exp(-13636/T_Matrix[i,j-1])*C_Matrix[i,j])
        P_Array[0,j] = P_Array[0,j-1] - 1763*dz # keeps track of the pressure drop in the reactor
    max_center_T = max(T_Matrix[0,:]) # determines the maximum temperature along the centerline
    return max_center_T, C_Matrix, T_Matrix,P_Array


max_center_T, C_Matrix, T_Matrix,P_Array = Method_of_Lines(T_wall = 640,T_inlet = 600) # function call to MOL
average_C = np.mean(C_Matrix,axis =0) # Calculates the average concentration at each point in the axial direction
average_T = np.mean(T_Matrix,axis =0) # Calculates the average temperature at each point in the axial direction
conversion = (average_C[0]-average_C[-1])/average_C[0] # Computes conversion
delta_P = (P_Array[0,0] - P_Array[0,-1])/101325 # Calculates the Pressure drop across the reactor
print('At a wall temperature of 640 K and inlet temperature of 600 K')
print('Maximum center line temperature:',round(max_center_T,3))
print('Average Conversion:',round(conversion,3))
print('delta P in atm:',delta_P )



df = pd.DataFrame(columns=['T_inlet', 'T_Center','T_Wall', 'Conversion'])
i = 0 # index to add entries to dataframe
wall_T_array = np.arange(630,701) # array of wall Temperatures
print('-----')
for T_inlet in [550,600,650]: # moves through a range of inlet temperatures
    switch = 0                        # switch variable for determining the first instance of being below 690 K
    max_center_T_array = np.array([]) # array containing the maximum center line temperatures
    for T_wall in wall_T_array: # moves through a range of wall temperatures
        # uses the MOL function to recompute the max center line temperature at the wall and inlet temperatures
        max_center_T, C_Matrix, T_Matrix,P_Array = Method_of_Lines(T_wall = T_wall,T_inlet = T_inlet) 
        max_center_T_array = np.append(max_center_T_array,max_center_T) # appends to maximum center line temperature array
        if max_center_T >= 690 and switch == 0: # checks for the temperature right below the max allowable
            average_C = np.mean(C_Matrix,axis =0) # computes average concentration
            conversion = (average_C[0]-average_C[-1])/average_C[0] # computes conversion
            switch = 1 # turns the switch on
            df.loc[i] = [T_inlet, max_center_T_array[-2],T_wall,conversion] # adds the list to the data frame
            i = i + 1 # increments i
    plt.plot(wall_T_array,max_center_T_array, label = 'Inlet Temp = '+ str(T_inlet)) # plot the maximum center line temperature as a function of the wall_T_array
    
plt.xlabel('Wall Temperature K')
plt.ylabel('Maximum Center Line Temperature K')
plt.title('Center Line Temperature versus Wall Temperature')
plt.legend()
plt.plot(wall_T_array,690*np.ones((wall_T_array.shape)),label = 'cutoff') # cutoff for maximum allowable centerline temperature
print(df)
        










