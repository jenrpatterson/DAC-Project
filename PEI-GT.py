## To be ran in Gilbreth
# Global Thermostat single "CELOR" stick

#Import packages:
import numpy as np
import CoolProp.CoolProp as CP
from fipy import CellVariable, FaceVariable, Grid1D, TransientTerm, DiffusionTerm, ConvectionTerm
import os, csv

# Constants:
R = 287 #J/kg(air)*K
R_mol = 8.314 # J/mol·K
v = 2.5 # m/s Average feed velocity
Twf = 376.15 # K, Desorption temperature
pL = 10  # mbar, Vacuum pressure 

# Monolith / Sorbent properites:
dp = 9.96e-10 # m, pore diameter (0.996nm)
rho_s = 1590.0 # kg/m3, sorbent density (not including voids)
rho_p = 500.0 # kg/m3, particle density (with proper voids)
rho_b = 377.1 # kg/m3, bulk density
cp_sorbent = 892.5 # J/kg/K
rho_cord = 2050 # kg/m3, cordierite wall density

CPSI = 230 # cells per square inch
N = 8280 # number of cells (36 sq in face)
L = 0.15 # m
W = 0.15 # m
H = 0.15 # m
area = W * H # m2
w_wall = 0.265e-3 # m, wall thickness
w2 = 1.676e-3 # m, 1.676mm cell width
w1 = w2 - w_wall # m, actual channel width
V = L * W * H # m3
V_chan = w1**2 * L # m3, channel volume
V_wall = V - N*V_chan # m3, wall volume
omega = 0.5 # loading, range from 10-100%, kg sorbent / kg total

rho_wall = (omega / rho_p + (1 - omega) / rho_cord) ** -1 # total wall desnity (including voids)
rho_bed = rho_p * V_wall / V # bed density
epsilon = 1 - rho_bed / rho_p # bed porosity (calculated to be ~0.73)
u = v / epsilon # m/s interstitial velocity

# Toth parameters:
T0 = 270 # K, reference temperature for Toth parameters
chem_b0 = 996 # 1/mbar Toth affinity coefficient 
chem_dH = 68.3e3 # J/mol
chem_t0 = 0.243
chem_alpha = 1.802
chem_ns0 = 3.450 # mol/kg
chem_chi = 4.504

phys_b0 = 9.32e-3 # 1/mbar 93.2 1/MPa to mbar
phys_dH = 40.1e3 # J/mol
phys_t0 = 0.163
phys_alpha = 2.287
phys_ns0 = 6.205 # mol/kg
phys_chi = 0.579

# Toth parameters model:
def toth_coeff(T):
    ns_c = chem_ns0 * np.exp(chem_chi * (1.0 - T / T0))
    b_c  = chem_b0 * np.exp((chem_dH / (R_mol * T0)) * ((T0 / T) - 1.0))
    t_c  = chem_t0 + chem_alpha * (1.0 - T0 / T)
    
    ns_p = phys_ns0 * np.exp(phys_chi * (1.0 - T / T0))
    b_p  = phys_b0 * np.exp((phys_dH / (R_mol * T0)) * ((T0 / T) - 1.0))
    t_p  = phys_t0 + phys_alpha * (1.0 - T0 / T)

    return ns_c, b_c, t_c, ns_p, b_p, t_p

# Toth Equation:
def q_CO2_Toth(p_CO2, T): # p_CO2 in mbar and T in Kelvin
    ns_c, b_c, t_c, ns_p, b_p, t_p = toth_coeff(T)
    # chemisorption
    q_chem = ns_c * b_c * p_CO2 / (1 + (b_c * p_CO2)**t_c)**(1/t_c)
    # physisorption
    q_phys = ns_p * b_p * p_CO2 / (1 + (b_p * p_CO2)**t_p)**(1/t_p)

    return q_chem + q_phys # mol capacity CO2 / kg sorbent

# q* function:
def q_star_func(c_val,T):
    p_CO2 = c_val * R_mol * T / 100  # mol/m3 to mbar
    q_mass = q_CO2_Toth(p_CO2, T)  # mol/kg
    return q_mass * rho_bed * omega # mol/m3


# Make .csv
out_file = "capturePEI-GT.csv"
write_header = not os.path.exists(out_file)

csv_fh = open(out_file, "a", newline="")
writer = csv.writer(csv_fh)
if write_header:
    writer.writerow(["Temp_K","Pressure_mbar","kg_CO2"])


# Temperature range: 102 - 306K (Equivalent Temp considering RH)
for Temp in np.arange(102, 306, 2):
    for Press in np.arange(990, 1036, 2):
        
        # Transport variables:
        # adsorption 'k'
        Dm = 1.5e-5  # m2/s molecular diffusivity
        DK = (dp/3.0) * np.sqrt(8*R_mol*Temp/(np.pi*0.044009)) # Knudsen Diffusivity, with Molar mass CO2 (kg/mol)
        D_MIL = DK / 3 # where 3 is the tortiousity (Sabatino & Stampi-Bombelli)
        De_m = omega * D_MIL + (1 - omega) * Dm # m2/s
        mu = CP.PropsSI('V','P',Press * 100,'T',Temp,'air') #Pa*s gaseous dynamic viscosity
        rho = CP.PropsSI('D','P',Press * 100,'T',Temp,'air') #kg/m3 density of dry air
        CO2_conc = 430 # ppm
        pCO2_air = Press * (CO2_conc/1000000) # mbar, partial pressure CO2
        c_CO2 = (pCO2_air * 100) / (R_mol * Temp) # mol/m3, Ideal gas law
        Re = rho * u * w1 / mu # Reynolds number 
        Sc = mu / (rho * Dm) # Schmidt number
        Sh = 2.696 * (1 + 0.139 * (w1 / L) * Re * Sc) ** 0.81 # Sherwood number
        kf_prime = Dm / w1 * Sh # m/s
        kf = 4 * w1 / (w2**2 - w1**2) * kf_prime  # 1/s
        
        r1 = 2 * w1 / np.pi
        r2 = np.sqrt((4 * w_wall * w2 / np.pi) + r1**2)
        kp_inv = (24 * De_m * r1) / ((r2 - r1)**2 * (5 * r2 + 3 * r1)) # s
        kp = 1 / kp_inv  # 1/s
        
        ks = 10**4 # 1/s (assumed negligible from Stampi-Bombelli)
        
        DL = Dm + (u**2 * w1**2) / (192 * Dm)  # m2/s
        
        q_star_ads = q_CO2_Toth(pCO2_air, Temp)
        qv_star = q_star_ads * rho_bed * omega # mol/m3 (mol/kg * kg/m3)
        k_inv = (1 / kf) * (qv_star / c_CO2) + (1 / kp) * (qv_star / c_CO2) + (1 / ks)
        k = 1/k_inv
        
        # desorption 'k'
        Dm_d = Dm * (Twf / 300)**1.5 * (1000 / pL) #Dm changes with P and T
        DK_d = (dp/3.0) * np.sqrt(8*R_mol*Twf/(np.pi*0.044009)) # Knudsen Diffusivity, with Molar mass CO2 (kg/mol)
        D_MIL_d = DK_d / 3 # where 3 is the tortiousity (Sabatino & Stampi-Bombelli)
        De_m_d = omega * D_MIL_d + (1 - omega) * Dm_d # m2/s
        mu_d = CP.PropsSI('V','P',pL * 100,'T',Twf,'air') #Pa*s constant viscosity
        rho_d = CP.PropsSI('D','P',pL * 100,'T',Twf,'air') #kg/m3 constant density
        pCO2_des = 10 # mbar, pressure of CO2 in chamber during desorption
        Re_d = rho_d * u * w1 / mu_d # Reynolds number
        Sc_d = mu_d / (rho_d * Dm_d) # Schmidt number
        Sh_d = 2.696 * (1 + 0.139 * (w1 / L) * Re_d * Sc_d) ** 0.81 # Sherwood number
        kf_prime_d = Dm_d / w1 * Sh_d # m/s
        kf_d = 4 * w1 / (w2**2 - w1**2) * kf_prime_d  # 1/s
        
        kp_inv = (24 * De_m_d * r1) / ((r2 - r1)**2 * (5 * r2 + 3 * r1)) # s
        kp = 1 / kp_inv  # 1/s
        
        q_star_des = q_CO2_Toth(10,Twf)
        qv_star_d = q_star_des * rho_bed * omega # mol/m3 (mol/kg * kg/m3)
        c_CO2_d = (pCO2_des * 100) / (R_mol * Twf) # mol/m3 
        k_inv_d = (1 / kf_d) * (qv_star_d / c_CO2_d) + (1 / kp) * (qv_star_d / c_CO2_d) + (1 / ks)
        k_d = 1/k_inv_d

        epsilon_p = 1.0 - rho_p / rho_s
        epsilon_star = epsilon + epsilon_p * (1-epsilon)
        
        # Adsorption:
        nx = 20 # number of spatial points
        dx = L / nx
        mesh = Grid1D(dx=dx, nx=nx)
        
        # Variables
        c = CellVariable(name="Gas Concentration", mesh=mesh, value=0.0) # mol/m3 Gas phase concentration: initially 0 everywhere
        q = CellVariable(name="Solid Loading", mesh=mesh, value=0.0) # mol/m3 Solid phase concentration: initially 0 everywhere
        
        # Inlet CO2 concentration (mol/m3)
        c_in = (pCO2_air * 100) / (R_mol * Temp) # mol/m3
        
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
        steps = 72000 # 30 minutes
        x = mesh.cellCenters.value[0]
        
        for step in range(steps):
            mass_balance.solve(var=c, dt=dt)
            q_star_val = q_star_func(c.value,Temp)
            q.setValue(q.value + k * (q_star_val - q.value) * dt)
         
        mol_ads = np.trapz(q.value, x) * area # mol CO2 in whole column
        
        
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
        steps_d = 24000 # 10 min desorption
        x_d = mesh.cellCenters.value[0]
        
        for step in range(steps_d):
            mass_balance_d.solve(var=c_d, dt=dt_d)
            q_star_desin = q_star_func(c_d.value, Twf)
            q_d.setValue(q_d.value + k_d * (q_star_desin - q_d.value) * dt_d)
        
        mol_des = np.trapz(q_d.value, x) * area
        kg_CO2 = (mol_ads - mol_des)*44.01/1000
    
        writer.writerow([Temp, Press, round(kg_CO2, 6)])
        print(f"T = {Temp} K, {kg_CO2:.4f} kg captured")
    
csv_fh.close()

        
