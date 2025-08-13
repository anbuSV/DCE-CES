#!/usr/bin/env python
# coding: utf-8

"""
@author: anbu (anbu@ksu.edu), huynhlam (huynhlam@ksu.edu)
JRM Lab, Dept. of Physics, Kansas State University, Manhattan, KS. 
"""

import time
import numpy as np
import pandas as pd
idx = pd.IndexSlice
from scipy.integrate import solve_ivp

#%% Set Directories and some parameters

" params = [offset in pos, energy in eV]"
params = [0.25, 0.5]

# Number of geometries to simulate
checkpoint_interval = 500
no_of_sims = 5000

atoms_xyz_orig = {'C1': [0.66699, 0.96658, -0.0],
                 'C2': [-0.667, 0.96658, 0.0],
                 'H1': [1.22333, 1.90229, 0.0],
                 'H2': [-1.22333, 1.90229, 0.0],
                 'Cl1': [1.6677, -0.45304, 0.0],
                 'Cl2': [-1.6677, -0.45305, 0.0]}
xyz_species_label =  ['C1', 'C2', 'H1', 'H2', 'Cl1', 'Cl2']

molname = 'cis_DCE'

align_dict = {'x_ref':{'method':'Subtract', 'ID':['Cl1', 'Cl2'], 'Inverse':False},
              'xy_ref':{'method':'Add', 'ID':['Cl1', 'Cl2'], 'Inverse':False}}

mass_species = [12.0,12.0,1.007825032,1.007825032,34.968852682,34.968852682]

#Singly charged
frag_charge = np.ones_like(xyz_species_label, dtype=int)
atom_to_adjust = 'Cl1'

# # Extract q_initial as a NumPy array based on the order in frag_list
q_buildup = False # charge buildup
# q_buildup = True # charge buildup
q_final = np.array([1,1,1,1,1,1]) * 1.602188e-19  # Final charges in SI units

# Define multiple max time and corresponding time steps
max_times = [500, 5000, 10000]  # List of maximum times for each interval
step_sizes  = [5, 50, 250]       # List of step sizes for each interval

#Parameterize r_sigma and initial KE
r_sigma_abs_pos, initial_energy = params

q_initial_full = np.array([0,0,0,0,0,0])
q_t0_full = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) * 1e-15

# %%

def generate_time_steps(max_times, step_sizes):

    # Initialize an empty list to store the combined time steps
    t_steps = []
    current_start = 0  # Start time for the first interval
    
    # Iterate over max_times and corresponding step_sizes
    for i, (max_time, step) in enumerate(zip(max_times, step_sizes)):
        
        # Generate time steps for the current interval using the specified step size
        steps = np.arange(current_start, max_time + step, step)
        
        # Skip the first element in steps if this is not the first interval to avoid duplicate boundary values
        if i > 0:
            steps = steps[1:]
        
        # Extend t_steps with the current interval's steps
        t_steps.extend(steps)
        
        # Update current_start to the end of this interval for the next iteration
        current_start = max_time

    return np.array(t_steps)

# %%

# Generate the continous time steps with the function call
t_steps = generate_time_steps (max_times, step_sizes)* 1e-15  # Convert to seconds


#%% Define Newtons equations of motion to solve

def newtons_equations(t, y, masses, q_initial, q_final):
       
    # Number of atoms, inferred from the structure of y (positions and velocities for each atom)
    n_atoms = len(y) // 6
    
    # Initialize dydt array to store derivatives of position and velocity
    dydt = np.zeros_like(y)

    # Split y into separate position and velocity arrays
    # Positions are the first half of y (reshaped to (n_atoms, 3) for 3D coordinates)
    positions = y[:3 * n_atoms].reshape(n_atoms, 3)
    
    # Velocities are the second half of y (reshaped to (n_atoms, 3))
    velocities = y[3 * n_atoms:].reshape(n_atoms, 3)
    
    charges = q_final

    # Step 2: Compute forces on each atom in a vectorized manner
    # This uses the pre-defined compute_forces_vectorized function, which applies Coulomb's law
    forces = compute_forces_vectorized(positions, charges)

    # Step 3: Fill dydt for positions with velocities (dy/dt for position is velocity)
    dydt[:3 * n_atoms] = velocities.flatten()

    # Step 4: Calculate accelerations for each atom (F = ma) and fill in dydt for velocities
    # Each atom's acceleration is given by dividing its force by its mass
    # The resulting acceleration array is reshaped to match the shape of dydt
    accelerations = forces / masses[:, np.newaxis]
    dydt[3 * n_atoms:] = accelerations.flatten()

    return dydt

#%% Calculate forces

def compute_forces_vectorized(positions, charges):
 
    n_atoms = len(positions)  # Number of atoms in the system
    forces = np.zeros((n_atoms, 3))  # Initialize the forces array for each atom

    # Compute pairwise differences between all positions in a vectorized manner
    # This results in a (n_atoms, n_atoms, 3) array where each element [i, j] is the 3D distance vector from atom j to atom i
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]

    # Calculate the Euclidean distance between each pair of atoms
    # This yields an (n_atoms, n_atoms) matrix where each element [i, j] is the distance from atom j to atom i
    dist = np.linalg.norm(diff, axis=2)

    # Set diagonal to 1 to prevent division by zero when calculating self-interactions (forces on self are zero)
    np.fill_diagonal(dist, 1)

    # Calculate Coulomb force prefactor for each atom pair based on charge product
    # This results in a (n_atoms, n_atoms) matrix with the scalar Coulomb force magnitude prefactor for each atom pair
    coulomb_prefactor = 8.98755e9 * charges[:, np.newaxis] * charges[np.newaxis, :]

    # Calculate the force magnitude for each atom pair, which is inversely proportional to the square of the distance
    # We divide by dist^3 to account for Coulomb's law and force direction normalization
    force_magnitude = coulomb_prefactor / (dist ** 3)

    # Set diagonal to 0, ensuring no self-interaction forces are computed
    np.fill_diagonal(force_magnitude, 0)

    # Sum up all the forces by multiplying the magnitude by direction (diff) and summing over all other atoms
    # This produces the net 3D force vector for each atom due to all other atoms
    forces = np.sum(force_magnitude[..., np.newaxis] * diff, axis=1)

    return forces

#%% Calculate center of mass from position and mass
def center_of_mass(r_list, m_list):
    
    n_atoms = len(m_list)
    
    # If r_list is 1D (single geometry), reshape it to 2D
    if r_list.ndim == 1:
        r_list = r_list[np.newaxis, :]  # Shape it to (1, n_atoms * 3)

    # Reshape the 2D array (n_geometries, n_atoms * 3) to (n_geometries, n_atoms, 3)
    r_reshaped = r_list.reshape(-1, n_atoms, 3)  # Shape: (n_geometries, n_atoms, 3)
    
    # Multiply each position by the corresponding mass and sum for each geometry
    mass_product = np.einsum('ijk,j->ik', r_reshaped, m_list)  # Shape: (n_geometries, 3)
    
    # Sum the masses to get the total mass (scalar)
    mass_sum = np.sum(m_list)

    # Calculate the center of mass for each geometry
    com = mass_product / mass_sum  # Shape: (n_geometries, 3)

    # If input was a single geometry, return a 1D array (3,)
    if com.shape[0] == 1:
        return com[0]
    
    return com
#%% # Function to adjust one atom's position to match the original CoM
def adjust_com_for_atom(df, atom_to_adjust, ref_CoM, mass_species):

    xyz_col = ['x', 'y', 'z']
    # Make a copy to avoid modifying the original DataFrame
    df_adjusted = df.copy()

    # Extract positions of all atoms for all geometries
    positions = df.loc[:, idx[:, xyz_species_label, xyz_col]].values

    # Calculate the current CoM for each geometry
    current_com = np.round(center_of_mass(positions, mass_species),2)

    # Calculate the necessary shift to align the CoM to the original
    com_shift = ref_CoM - current_com  # Shape: (no_of_sims, 3)

    # Get the mass of the atom to be adjusted
    atom_index = xyz_species_label.index(atom_to_adjust)
    atom_mass = mass_species[atom_index]

    # Adjust the position of the chosen atom to shift the CoM
    positions[:, atom_index*3:atom_index*3+3] += com_shift * (np.sum(mass_species) / atom_mass)

    # Update the DataFrame with the adjusted positions
    df_adjusted.loc[idx[:], idx[:, atom_to_adjust, xyz_col]] = positions[:, atom_index*3:atom_index*3+3]
    
    return df_adjusted

#%% Function to generate random geometries

def generate_random_geometries(
    no_of_sims, atoms_xyz_orig, mass_species, align_dict, atom_to_adjust, xyz_species_label,
    r_sigma_abs_pos=0.05, initial_energy=30
):

    tKE = initial_energy * 1.602188e-19

    # Adjust the standard deviation so that the expected 3D displacement magnitude equals r_sigma_abs_pos
    # Since the displacement is applied in 3D (x, y, z) with independent normal distributions,
    # the expected squared norm is 3 * sigma^2 → divide by sqrt(3) to maintain correct overall spread
    r_sigma_abs_pos *= 1/np.sqrt(3)

    # Convert original geometry dictionary to array
    orig_geom = np.array(list(atoms_xyz_orig.values()))

    # Calculate CoM and remove the offset
    orig_geom -= np.round(center_of_mass(orig_geom, mass_species), 2)
    ref_CoM = np.round(center_of_mass(orig_geom, mass_species), 2)
    # print(f"Ref CoM, after offset correction: {ref_CoM}")

    n_atoms = len(mass_species) #orig_geom.shape[0]

    # Generate randomized positions
    # Offset in angstroms
    rand_array_1 = np.random.normal(loc=0, scale=r_sigma_abs_pos, size=(no_of_sims,) + orig_geom.shape)
    
    rand_new_geom_sets = orig_geom+ rand_array_1

    #Generate randomized velocities
    m_list = np.array(mass_species) * 1.66054e-27  # Mass in kg for each atom
    
    # Generate velocities from a normal distribution
    vel_rand_array = np.random.normal(loc=0.0, scale=1.0, size=(no_of_sims, n_atoms, 3))
    
    # Rescale velocities to match the desired total kinetic energy
    mass_matrix = m_list[np.newaxis, :, np.newaxis]  # Shape (1, n_atoms, 1)
    vel_squared = np.sum(vel_rand_array**2, axis=-1)  # Sum over x, y, z -> shape (no_of_sims, n_atoms)
    total_kinetic_energy = np.sum(mass_matrix[:,:,0] * vel_squared, axis=1) # Shape (no_of_sims,)
    
    # Compute the scaling factor
    scaling_factor = np.sqrt(2 * tKE / total_kinetic_energy)[:, np.newaxis, np.newaxis]  # Shape (no_of_sims, 1, 1)
    vel_rand_array *= scaling_factor  # Apply scaling factor
    
    # Adjust velocities to ensure zero center-of-mass velocity
    vel_rand_array -= np.mean(vel_rand_array, axis=1, keepdims=True)

    # Combine positions and velocities into a single array
    rand_geom_data = np.concatenate((rand_new_geom_sets, vel_rand_array), axis=2)  # Shape: (no_of_sims, n_atoms, 6)

    # Reshape for DataFrame
    reshaped_data = rand_geom_data.reshape(no_of_sims, n_atoms * 6)

    # MultiIndex for DataFrame columns
    properties = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    multi_index = pd.MultiIndex.from_product(
        [['atoms'], xyz_species_label, properties], names=['Species', 'Fragment', 'Pos_Vel']
    )

    # Create the final DataFrame
    df_dl_pos_vel = pd.DataFrame(reshaped_data, columns=multi_index)

    # Compare CoM and adjust offset for atom to match original CoM
    df_dl_pos_vel_ref = adjust_com_for_atom(df_dl_pos_vel, atom_to_adjust, ref_CoM, mass_species)
    
    return df_dl_pos_vel_ref

#%% Function to simulate Coulomb explosion

def simulate_coulomb_explosion(
    df_dl_pos_vel_ref, masses_SI, q_initial, q_final, 
    t_steps, newtons_equations, frag_list,  
    checkpoint_interval=None, no_of_sims=500
):
    
    output_list = []  # To store final output of the simulation
    start_time = time.time()  # Start timer

    # Determine columns to use based on alignment status
    position_columns = ['x', 'y', 'z']
    velocity_columns = ['vx', 'vy', 'vz']  # Velocity columns (constant)
    
    num_sims = min(no_of_sims, len(df_dl_pos_vel_ref))
    n_atoms = len(masses_SI)
    
    # Simulation loop
    for geom_index in range(num_sims):
        # Step Extract position and velocity data for the current geometry
        atom_data = df_dl_pos_vel_ref.iloc[geom_index]
        
        # Extract positions and velocities together for efficiency
        r_list = atom_data.loc[idx[:, frag_list, position_columns]].values
        vel_list = atom_data.loc[idx[:, frag_list, velocity_columns]].values
        
        # Convert units to SI for the simulation
        r_list_SI = r_list * 1e-10 
        
        # Step Set up initial conditions array for ODE solver
        y0 = np.zeros(6 * n_atoms)
        y0[:3 * n_atoms] = r_list_SI  # Initial positions in SI units
        y0[3 * n_atoms:] = vel_list   # Initial velocities in SI units
        
        # Propagate using solve_ivp
        sol = solve_ivp(newtons_equations, [0, np.max(t_steps)], y0, t_eval=t_steps, 
                        args=(masses_SI, q_initial, q_final))

        # Save final velocities, charges, and masses for each atom
        output_array = np.zeros((n_atoms, 5))
        for i in range(n_atoms):
            final_v = sol.y[3 * n_atoms + 3 * i: 3 * n_atoms + 3 * i + 3, -1]
            output_array[i, :3] = final_v
            output_array[i, 3] = q_final[i]
            output_array[i, 4] = masses_SI[i]

        output_list.append(output_array)
        
        if checkpoint_interval:
            # Step 5: Progress reporting
            if (geom_index + 1) % checkpoint_interval == 0:
                elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                # round(time.time() - start_time, 2)
                print(f"Simulated {geom_index + 1} geometries in {elapsed_time} seconds")
    if checkpoint_interval:                
        # Report total simulation time
        total_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        print(f"Total simulation time: {total_time} seconds")
        
    # Convert output_list to a numpy array for easy processing
    output_array = np.array(output_list)
    # Calculate momenta for each atom (p = mv)
    p_list = output_array[:, :, :3] * output_array[:, :, 4][:, :, None]
    # Calculate squared magnitudes of the momenta
    p_sqr_mag = np.sum(np.square(p_list), axis=2)
    # Calculate kinetic energy for each atom (KE = p^2 / (2 * m)) and convert to eV
    ke_list = p_sqr_mag / (2 * output_array[:, :, 4]) / 1.602188e-19
    # Convert to atomic units (au)
    p_list *= 5.018e+23

    # Get all initial positions and velocities from the DataFrame for all geometries
    r_list_all = df_dl_pos_vel_ref.loc[:, idx[:, frag_list, position_columns]].values  # All geometries, all positions
    vel_list_all = df_dl_pos_vel_ref.loc[:, idx[:, frag_list, velocity_columns]].values  # All geometries, all velocities
    # Reshape each component to (n_geometries, n_atoms, 3) for positions, velocities, and momenta
    r_list_all_reshaped = r_list_all.reshape(-1, 3)
    vel_list_all_reshaped = vel_list_all.reshape(-1, 3)
    p_list_reshaped = p_list.reshape(-1, 3)  # Momentums in (n_geometries, n_atoms, 3)
    # KE needs to be (n_geometries, n_atoms, 1) to stack correctly along with x, y, z components
    ke_list_reshaped = ke_list.reshape(-1, 1)
    # Stack along the last axis to form (n_geometries, n_atoms, 10) with x, y, z, vx, vy, vz, Px, Py, Pz, KE
    ion_data_3D = np.hstack([r_list_all_reshaped[:num_sims * n_atoms],
                             vel_list_all_reshaped[:num_sims * n_atoms],
                             p_list_reshaped, ke_list_reshaped])
    # Reshape to (n_geometries, n_atoms * 10) to match DataFrame's expected structure
    ion_data = ion_data_3D.reshape(-1, n_atoms * 10)
    # Set up column names with MultiIndex for DataFrame
    column_ions_xyt_mom_ke = pd.MultiIndex.from_product(
        [['ions'], frag_list, ['x', 'y', 'z', 'vx', 'vy', 'vz', 'Px', 'Py', 'Pz', 'KE']],
        names=['Species', 'Hits', 'xyt'])
    # Create DataFrame from the compiled data
    df_ions_xyt_mom_ke = pd.DataFrame(data=ion_data, columns=column_ions_xyt_mom_ke)

    return df_ions_xyt_mom_ke

#%% Generate random geoms and simulates CE

t1 = time.time()

# Call the ramdom geometry generation function
df_rand_geoms = generate_random_geometries(
    no_of_sims=no_of_sims,
    atoms_xyz_orig=atoms_xyz_orig,
    mass_species=mass_species,
    align_dict=align_dict,
    atom_to_adjust = atom_to_adjust,
    xyz_species_label=xyz_species_label,
    r_sigma_abs_pos = r_sigma_abs_pos, 
    initial_energy = initial_energy, 
)

masses_SI = np.array(mass_species)*1.66054e-27

df_ions_xyt_mom_ke = simulate_coulomb_explosion(
    df_rand_geoms, masses_SI, q_initial_full, q_final, 
    t_steps, newtons_equations, xyz_species_label,
    checkpoint_interval, no_of_sims
)


print(f'Simulated CE of random geometries, in {time.time()-t1}')

