# DCE-CES
CES for DCE 6-body

1. System requirements
- Operating system: Windows OS (tested on Windows 10, standard desktop configuration)
- Python version: 3.12.3 (installed via Anaconda distribution, standard options)
- Dependencies:
  - fast-histogram
  - Standard Python scientific libraries: numpy, pandas, matplotlib, etc.
- Hardware: Standard desktop; no special hardware required.

2. Installation guide
1. Install Anaconda (https://www.anaconda.com/download) and create/activate a Python 3.12.3 environment.
2. Install the required packages:
   pip install fast-histogram
3. Store the simulation script in your working directory.

3. Demo
- Instructions:
  1. Open the provided script Static_Simulations_RandomGeoms_DCE_ML_Paper.py.
  2. Adjust simulation parameters in the script if desired (see Section 4).
  3. Run the script in your Python environment (Spyder, terminal, etc.).

- Expected output:
  - A pandas DataFrame containing event-by-event simulation results.
  - The DataFrame includes geometries, final velocities, momenta (Cartesian), and KE for each simulated event.

- Expected runtime: Typically a few minutes on a standard desktop.

4. Instructions for use
- The cis-1,2-DCE Cartesian coordinates are embedded directly in the script as the starting geometry.
- Randomized geometries are generated based on parameters in the script.
- Key user-adjustable parameters inside the script:
  - params: Controls random variations in geometry and the initial kinetic energy of the molecules.
  - Number of geometries to simulate.
  - Fragment charges.
- To use custom settings:
  1. Edit the relevant variables in the script.
  2. Run the script to generate the DataFrame.
  3. Save the DataFrame to file (DataFrame.to_csv() or to_hdf()).

5. Reproduction instructions
- Run the script with default parameters to reproduce the results described in the manuscript.
- The generated DataFrame will contain:
  - XYZ coordinates of all atoms for each event.
  - Final velocities and momenta (Cartesian).
  - Kinetic energies.

License: MIT
