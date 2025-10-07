## To be ran in Gilbreth

#Import packages:
import numpy as np
import CoolProp.CoolProp as CP
from fipy import CellVariable, FaceVariable, Grid1D, TransientTerm, DiffusionTerm, ConvectionTerm
import os, csv

# Constants:
R = 287 #J/kg(air)*K
R_mol = 8.314 # J/mol·K
v = 0.8 # m/s Average feed velocity
pH = 101325 # Pa, Inital pressure (1 atm)
pL = 1000  # Pa, Vacuum pressure (Stampi-Bombelli)
Twf = 373.15 # K, Desorption temperature
CO2_conc = 430 # ppm
pCO2_air = pH/1000 * (CO2_conc/1000000) # kPa, partial pressure CO2
pCO2_des = 1 # kPa, pressure of CO2 in chamber during desorption
rho_d = CP.PropsSI('D','P',pL,'T',Twf,'air') #kg/m3 constant density
mu_d = CP.PropsSI('V','P',pL,'T',Twf,'air') #Pa*s constant viscosity

# Packed Bed properites:
dp = 0.001 # m, pellet size
rp = dp/2 # m, pellet radius
rho_p = 1044 # kg/m3, Particle density TRI-al (pellets are porous)
L = 0.40 # m, 40cm
D = 0.04 # m
r = D/2 # m, radius
V = np.pi * r**2 * L # m3, volume(cylinder)
epsilon = 0.39 + 1.74/((D/dp + 1.14)**2) # bed porosity
psi = 1 # 1 for 40 cm single bed, 0.4 for 5cm beds in series.
u = v / epsilon # m/s interstitial velocity
rho_bed = rho_p*(1-epsilon) # kg/m^3, bed density

# Toth parameters:
T0 = 298 # K, reference temperature for Toth parameters
ns0 = 1.067 # mol/kg Toth sorbent maximum capacity at reference temperature
b0 = 93.89 # 1/kPa Toth affinity coefficient at reference temperature
t0 = 0.461 # Toth exponent (heterogeneity parameter) at reference temperature
deltaH0 = 70000 # J/mol isosteric heat of adsorption
chi = 6.49E-13 # Temperature adjustment factor for ns(T)
alpha = 0.708 # Temperature adjustment factor for t(T)

# Toth function:
def find_q_star(p_CO2, T):
    ns_val = ns0 * np.exp(chi * (1 - T / T0))
    b_val = b0 * np.exp((deltaH0 / (R_mol * T0)) * ((T0 / T) - 1))
    t_val = t0 + alpha * (1 - T0 / T)
    numerator = ns_val * b_val * p_CO2
    denominator = (1 + (b_val * p_CO2)**t_val)**(1 / t_val)
    return numerator / denominator # mol capacity CO2 / kg sorbent

# Transport constant variables:
Dm = 1.5e-5  # m2/s molecular diffusivity
D_TRI = 6e-7 # m2/s TRI-al diffusivity
rc = 3.5e-10 # m crystalline radius Wefers & Misra 1987
ks = 15 * D_TRI / (rc**2)  # 1/s
De_p = D_TRI  # m2/s Effective pore diffusivity in TRI-al
epsilon_p = 0.71
epsilon_star = epsilon + epsilon_p * (1-epsilon)
kp = 15 * epsilon_p * De_p / (rp**2)  # 1/s
Re_d = rho_d * v * dp / mu_d # Reynolds number
Sc_d = mu_d / (rho_d * Dm) # Schmidt number
Sh_d = 2 + 1.1 * Re_d**0.6 * Sc_d**0.33 # Sherwood number
kf_prime_d = Dm / dp * Sh_d # m/s
kf_d = 3 / rp * kf_prime_d  # 1/s
q_star_des = find_q_star(1,Twf)
qv_star_d = q_star_des * rho_bed * psi # mol/m3 (mol/kg * kg/m3)
c_CO2_d = (pCO2_des * 1000) / (R_mol * Twf) # mol/m3 
k_inv_d = (1 / kf_d) * (qv_star_d / c_CO2_d) + (1 / kp) * (qv_star_d / c_CO2_d) + (1 / ks)
k_d = 1/k_inv_d


# Make .csv
out_file = "capture_1hr.csv"
write_header = not os.path.exists(out_file)

csv_fh = open(out_file, "a", newline="")
writer = csv.writer(csv_fh)
if write_header:
    writer.writerow(["Temp_K", "kg_CO2"])

# Temperature range: 270 - 320K
for Temp in np.arange(270, 322, 2):
    
    # Temperature dependent variables:
    mu = CP.PropsSI('V','P',pH,'T',Temp,'air') #Pa*s gaseous dynamic viscosity
    rho = CP.PropsSI('D','P',pH,'T',Temp,'air') #kg/m3 density of dry air
    c_CO2 = (pCO2_air * 1000) / (R_mol * Temp) # mol/m3, Ideal gas law
    Re = rho * v * dp / mu # Reynolds number
    Sc = mu / (rho * Dm) # Schmidt number
    Sh = 2 + 1.1 * Re**0.6 * Sc**0.33 # Sherwood number
    kf_prime = Dm / dp * Sh # m/s
    kf = 3 / rp * kf_prime  # 1/s
    DL = u * dp * ((0.45 + 0.55 * epsilon) / (Re * Sc) + 0.5)  # m2/s
    q_star_ads = find_q_star(pCO2_air, Temp)
    qv_star = q_star_ads * rho_bed * psi # mol/m3 (mol/kg * kg/m3)
    k_inv = (1 / kf) * (qv_star / c_CO2) + (1 / kp) * (qv_star / c_CO2) + (1 / ks)
    k = 1/k_inv
    
    # q* function:
    def q_star_func(c_val):
        p_CO2 = c_val * R_mol * Temp / 1000  # mol/m3 to kPa
        q_mass = find_q_star(p_CO2, Temp)  # mol/kg
        return q_mass * rho_bed * psi # mol/m3
    
    # Adsorption:
    nx = 20 # number of spatial points
    dx = L / nx
    mesh = Grid1D(dx=dx, nx=nx)
    
    # Variables
    c = CellVariable(name="Gas Concentration", mesh=mesh, value=0.0) # mol/m3 Gas phase concentration: initially 0 everywhere
    q = CellVariable(name="Solid Loading", mesh=mesh, value=0.0) # mol/m3 Solid phase concentration: initially 0 everywhere
    
    # Inlet CO2 concentration (mol/m3)
    c_in = (pCO2_air * 1000) / (R_mol * Temp) # mol/m3
    
    # Boundary conditions: Serna-Guerro paper
    c.constrain(c_in, mesh.facesLeft) # CO2 inlet value fixed at ambient
    c.faceGrad.constrain(0.0, mesh.facesRight) # zero concentration gradient at outlet
    
    v_vector = FaceVariable(mesh=mesh, rank=1, value=v)
    
    # PDE
    mass_balance = (TransientTerm(coeff=epsilon_star, var=c)
                    == DiffusionTerm(coeff=epsilon * DL, var=c)
                    - ConvectionTerm(coeff=v_vector, var=c)
                    - (1 - epsilon_star) * TransientTerm(var=q))
    
    # Time loop
    dt = 0.025
    steps = 144000 # 1 hour
    x = mesh.cellCenters.value[0]
    
    for step in range(steps):
        mass_balance.solve(var=c, dt=dt)
        q_star_val = q_star_func(c.value)
        q.setValue(q.value + k * (q_star_val - q.value) * dt)
     
    mol_ads = np.trapz(q.value, x) * np.pi * r**2 # mol CO2 in whole column
    
    
    # Desorption:
    c_d = CellVariable(name="Gas Concentration", mesh=mesh, value=0.0)
    q_d = CellVariable(name="Solid Loading", mesh=mesh, value=q.value.copy())
    
    c_in_des = (pCO2_des * 1000) / (R_mol * Twf) # mol/m3 
    
    c_d.constrain(c_in_des, mesh.facesRight)
    c_d.faceGrad.constrain(0.0, mesh.facesLeft)
    
    v_vector_d = FaceVariable(mesh=mesh, rank=1, value=-v)
    
    mass_balance_d = (TransientTerm(coeff=epsilon_star, var=c_d)
                    == DiffusionTerm(coeff=epsilon * DL, var=c_d)
                    - ConvectionTerm(coeff=v_vector_d, var=c_d)
                    - (1 - epsilon_star) * TransientTerm(var=q_d))
    
    dt_d = 0.025
    steps_d = 144000 # 1 hour desorption
    x_d = mesh.cellCenters.value[0]
    
    for step in range(steps_d):
        mass_balance_d.solve(var=c_d, dt=dt_d)
        q_d.setValue(q_d.value + k_d * (q_star_des - q_d.value) * dt_d)
    
    mol_des = np.trapz(q_d.value, x) * np.pi * r**2
    kg_CO2 = (mol_ads - mol_des)*44.01/1000

    writer.writerow([Temp, round(kg_CO2, 6)])
    print(f"T = {Temp} K, {kg_CO2:.4f} kg captured")

csv_fh.close()

        
