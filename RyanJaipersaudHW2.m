% Ryan Jaipersaud
% Partner Tushar Nichwakawade
% Advanced chemical reactions
% 10/3/2018
% The following code solves a system of ODEs using RK4.
% The ODEs consist of the fluxes of species formed during the
% reaction of ethane to produce ethylene

clear all;
clc;
format long;

a = 0; 
b = 85;
N = 100; % iterations
h = (b-a)/N; % step size
flux = zeros(N,6); % matrix of fluxes
x = zeros(N,1); % matrix of steps along reactor

% initial conditions for ethane, inerts, pressure and velocity
flux(1,:) = [12605 0 254 0 0 8403]; % mol/s/m^2
P(1) =[1.2*10^6];  % Pa
v(1) = [163]; % m/s


% function handle for ODEs
f = @(NE,v) [(-3.6*NE/v) + -28*(NE^0.5)/(v^0.5) ; % Ethane
              2.4*NE/v; % Methane
              28*(NE/v)^0.5; % Ethylene
              28*(NE/v)^0.5; % Hydrogen
              1.2*NE/v; % Butane
              0;]; % Inerts
% function handle for Pressure ODE
Pfunct = @(Ntot,P) -203524*(Ntot)/P;
       
 for i = 1:N
 % RK4 method to obtain fluxes
 K(1,:)= h*f(flux(i,1),v(i));
 K(2,:) = h*f(flux(i,1) + 0.5*K(1,1),v(i));
 K(3,:) = h*f(flux(i,1) + 0.5*K(2,1),v(i));
 K(4,:) = h*f(flux(i,1) + K(3,1), v(i));
 
 % iteration step to solve for fluxes
 flux(i+1,:) = flux(i,:) + (1/6)*(  K(1,:) + 2*K(2,:) +2*K(3,:) + K(4,:) ); 
 
 % after you obtain the fluxes solve for P using RK4 again
 PK(1) = h*Pfunct(sum(flux(i,:)),P(i));
 PK(2) = h*Pfunct(sum(flux(i,:)),P(i)+0.5*PK(1));
 PK(3) = h*Pfunct(sum(flux(i,:)),P(i)+0.5*PK(2));
 PK(4)  = h*Pfunct(sum(flux(i,:)),P(i)+PK(3));
 % iteration step to solve for Pressure
 P(i+1) = P(i) + (1/6)* ( PK(1) + 2*PK(2) + 2*PK(3) + PK(4));
 
 % Compute velocity for the next step since fluxes depend on them
 v(i+1) = sum(flux(i,:))*9336/P(i+1);
 x(i+1) = x(i) + h;
 end


% Convert fluxes back to concentrations
C_matrix = flux./transpose(v);
C_matrix(101,:);

selectivity = (flux(101,3)-flux(1,3))/sum(flux(101,2:5)) % determine selectivity of species
plot(x(:,1),C_matrix(:, 1),'g-','LineWidth',2)
hold on
plot(x(:,1),C_matrix(:,2),'r-','LineWidth',2)
hold on
plot(x(:,1),C_matrix(:,3),'k-','LineWidth',2)
hold on
plot(x(:,1),C_matrix(:,5),'b-','LineWidth',2)
title('Cocentration of species as a function of length using RK4');
ylabel('C [mol/m^3]');
xlabel('length [m]');
legend('Ethane','Methane', 'Etylene','Butane','Location', 'Northeast');

figure(2)
plot(x(:,1),P(:),'b-','LineWidth',2)
title('Pressure versus length');
ylabel('Pressure [Pa]');
xlabel('length [m]');