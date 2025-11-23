import numpy as np 
import os 
from mpi4py import MPI
import sys 
import yaml 
import os

def _validate_config(config):
    """
    Validates the injection field configuration including 
    conditional checks for UTM_ZONE and Coupling Methods.
    """
    errors = []
    warnings = []

    # --- 1. Helper for Path Resolution ---
    def resolve_path(path_str):
        if not path_str: return None
        return os.path.abspath(os.path.expanduser(path_str))

    # --- 2. Retrieve Key Settings ---
    coupling = config.get('coupling_method', '').lower()
    system = config.get('SPECFEM_SYSTEM', '').lower()

    # --- 3. Path Validation (Conditional) ---
    # Always required
    mandatory_paths = ['AXISEM_DIR', 'SPECFEM_DB']
    for key in mandatory_paths:
        abs_path = resolve_path(config.get(key))
        if not abs_path or not os.path.exists(abs_path):
            errors.append(f"Missing required directory for {key}: {config.get(key)}")
        else:
            config[key] = abs_path # Update config with safe absolute path

    # Conditional: SPECFEM_DATA (Only for Equivalent Forces)
    data_raw = config.get('SPECFEM_DATA')
    data_path = resolve_path(data_raw)
    
    if coupling == 'ef':
        if not data_path or not os.path.exists(data_path):
            errors.append(f"[Critical] Coupling is 'ef', but SPECFEM_DATA not found: {data_raw}")
    elif data_path and not os.path.exists(data_path):
        warnings.append(f"SPECFEM_DATA path invalid ({data_raw}), but allowed for coupling '{coupling}'")

    # --- 4. System & UTM Zone Validation (NEW) ---
    valid_systems = ['cart','cube2sph']
    if system not in valid_systems:
        errors.append(f"Invalid SPECFEM_SYSTEM: '{system}'. Must be {valid_systems}")

    # CONDITIONAL CHECK: UTM_ZONE
    # Only needed if system is Cartesian
    if system == 'cart':
        utm_raw = config.get('UTM_ZONE')
        
        if utm_raw is None:
            errors.append(f"[Critical] SPECFEM_SYSTEM is '{system}', but UTM_ZONE is missing.")
        else:
            try:
                utm_val = int(utm_raw)
                # strict geographic validation: Zones are 1 to 60
                if not (1 <= utm_val <= 60):
                    errors.append(f"UTM_ZONE must be between 1 and 60. Got: {utm_val}")
            except ValueError:
                errors.append(f"UTM_ZONE must be an integer. Got: {utm_raw}")

    # --- 5. Other Parameters ---
    if config.get('intp_method') not in ['linear', 'savgol']:
        errors.append(f"Invalid intp_method: {config.get('intp_method')}")
    
    if config.get('coupling_method') not in ['wd', 'ef']:
        errors.append(f"Invalid coupling_method: {config.get('coupling_method')}")

    # check if only_eq_force is exist 
    if 'only_eq_force' not in config:
        config['only_eq_force'] = False

    # Time & Types
    try:
        dt = float(config.get('dt', 0.0))
        nt = int(config.get('nt', 0))
        if dt <= 0 or nt <= 0:
            errors.append("Time parameters (dt, nt) must be positive.")
    except ValueError:
        errors.append("Time parameters must be numeric.")

    if not isinstance(config.get('DOWN_SAMPLING'), bool):
        errors.append("DOWN_SAMPLING must be a boolean.")

    # --- 6. Report ---
    if warnings:
        print("\n[!] Warnings:")
        for w in warnings: print(f" - {w}")

    if errors:
        print("\n[X] Configuration Failed:")
        for e in errors: print(f" - {e}")
        raise ValueError("Invalid Configuration")
    

def main():
    if len(sys.argv) != 2:
        print("Usage: python coupling.py config.yaml")
        return -1
    
    # MPI init
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # read config file
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        param = yaml.safe_load(f)

    # sanity check
    if rank ==0 :
        _validate_config(param)

    # create output dir
    if rank ==0 :
        outdir = param['OUTPUT_DIR']
        os.makedirs(outdir,exist_ok=True)
    comm.Barrier()

    # create functions for each coupling method
    coupling_method = param['coupling_method'].lower()
    system = param['SPECFEM_SYSTEM'].lower()
    if coupling_method == 'wd' and system == 'cart':
        from driver import coupling_cart_stacey
        coupling_cart_stacey(param)
    elif coupling_method == 'wd' and system == 'cube2sph':
        from driver import coupling_cube2sph
        coupling_cube2sph(param)
    elif coupling_method == 'ef' and system == 'cube2sph':
        from driver import coupling_cart_equivforce
        coupling_cart_equivforce(param)
    else:
        if rank ==0 :
            print(f"[X] Coupling method '{coupling_method}' with system '{system}' not implemented.")
        return -1 


    # MPI finalize
    MPI.Finalize()

if __name__ == "__main__":
    main()