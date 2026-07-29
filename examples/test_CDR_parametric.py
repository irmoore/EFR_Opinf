import numpy as np
from pathlib import Path

from efr_opinf._paths import PROJECT_ROOT, MESH_DIR
import opinf
import itertools
import time

from efr_opinf.interfaces.cdr import fenicsx_class
import matplotlib.pyplot as plt

from tqdm import tqdm
import typing
import os

import json

class RK_solvers:
    def __init__(self, solver_dict):
        self.filter_bool = solver_dict["filtering"]
        self.dt = solver_dict["dt"]

        if self.filter_bool == True:
            self.chi = solver_dict["chi"]
            self.filter_time = solver_dict["filter_time"]
            self.filter_method = solver_dict["filter_method"]

            if self.filter_method == "Projection":
                self.r_restriction = solver_dict["r_restriction"]


            elif self.filter_method == "Differential":
                self.delta = solver_dict["delta"]
                self.fenicsx_interface = solver_dict["fenicsx_interface"]
                self.include_mean = solver_dict["include_mean"]
                self.mean_func = solver_dict["mean_func"]
                self.basis = solver_dict["basis"]
                self._setup_differential_filter(self.delta)

            elif self.filter_method == "Differential_Partial":
                self.delta = solver_dict["delta"]
                self.fenicsx_interface = solver_dict["fenicsx_interface"]
                self.include_mean = solver_dict["include_mean"]
                self.mean_func = solver_dict["mean_func"]
                self.basis = solver_dict["basis"]
                self.r_restriction = solver_dict["r_restriction"]
                self._setup_differential_filter(self.delta)


            else:
                raise ValueError("Filter method can only be one of `Projection`, `Differential`, `Differential_Partial'")
            
    def _setup_differential_filter(self, delta):
        fenicsx_interface = self.fenicsx_interface
        basis = self.basis
        Filter_LHS = fenicsx_interface.setup_DF_filter_LHS(delta)
        b_entries = basis.entries

        ROM_DF_LHS = b_entries.transpose() @ Filter_LHS @ b_entries
        self.ROM_DF_LHS = ROM_DF_LHS

    def update_filter_parameters(self, filter_dict):
        assert self.filter_bool == True, "This solver was not initialized for filtering"
        assert filter_dict["filter_method"] == self.filter_method, "Solver filtering type does not match input"
        if filter_dict["filter_method"] == "Projection":
            self.chi = filter_dict["chi"]
            self.filter_time = filter_dict["filter_time"]
            self.r_restriction = filter_dict["r_restriction"]

        if filter_dict["filter_method"] == "Differential":
            self.delta = filter_dict["delta"]
            self.chi = filter_dict["chi"]
            self.filter_time = filter_dict["filter_time"]
            self._setup_differential_filter(self.delta)

        if filter_dict["filter_method"] == "Differential_Partial":
            self.delta = filter_dict["delta"]
            self.chi = filter_dict["chi"]
            self.filter_time = filter_dict["filter_time"]
            self.r_restriction = filter_dict["filter_time"]
            self._setup_differential_filter(self.delta)

    def update_chi_only(self, chi: np.float64):
        self.chi = chi

    def _compute_filter_rhs(self, u_ROM: np.ndarray) -> np.ndarray:
        basis = self.basis
        basis_entries = basis.entries
        mean_func = self.mean_func
        reconstructed_u = basis.decompress(u_ROM)
        fenicsx_interface = self.fenicsx_interface

        u_func = fenicsx_interface.vector_to_func(reconstructed_u)
        rhs_arr = fenicsx_interface.assemble_filter_rhs(u_func, mean_func, self.include_mean)
        rhs_ROM = basis_entries.transpose() @ rhs_arr
        return rhs_ROM



    def check_filtering_at_timestep(self, t):
        if self.filter_bool:
            if t >= self.filter_time:
                return True
            
        return False

    def solve(self, model, IC, timespan):
        dt = self.dt 
        t = timespan[0]
        solution = np.zeros((IC.shape[0], timespan.shape[0]), dtype = np.float64)
        u = IC.copy()
        solution[:,0] = IC
        for i in range(timespan.shape[0]-1):
            unew = self.RK4_step(model,t, u, dt) # Evolve
            if self.check_filtering_at_timestep(t):
                ufilter = self.filter_solution(unew) # Filter
                u_EFR = self.relax_solution(unew, ufilter) # Relax
                solution[:,i+1] = u_EFR
                u = u_EFR
            else:
                solution[:,i+1] = unew
                u = unew
            t += dt
        return solution
    
    def filter_solution(self, unew):
        if self.filter_method == "Projection":
            return self.projection_filter(unew)
        elif self.filter_method == "Differential":
            return self.differential_filter(unew)
        elif self.filter_method == "Differential_Partial":
            return self.differential_partial_filter(unew)
        else: 
            raise ValueError("Available filtering options are Projection and Differential")
        
    def differential_partial_filter(self, unew):
        ROM_LHS = self.ROM_DF_LHS
        ROM_rhs = self._compute_filter_rhs(unew)

        fil_sol = np.linalg.solve(ROM_LHS, ROM_rhs)

        r_restriction = self.r_restriction

        sol = unew.copy()
        sol[r_restriction:] = fil_sol[r_restriction:]
        return sol
        
    def differential_filter(self, unew):
        ROM_LHS = self.ROM_DF_LHS
        ROM_rhs = self._compute_filter_rhs(unew)

        sol = np.linalg.solve(ROM_LHS, ROM_rhs)
        return sol

        
    def projection_filter(self,unew):
        r_restriction = self.r_restriction
        u_filter = np.zeros_like(unew, dtype = np.float64)
        u_filter[:r_restriction] = unew[:r_restriction]
        return u_filter

    def relax_solution(self, unew, ufilter):
        chi = self.chi
        u_EFR = chi*ufilter + (1-chi)*unew
        return u_EFR


    def RK4_step(self, model: opinf.models.ContinuousModel, t, u, dt):
        k1 = model.rhs(t, u)
        k2 = model.rhs(t, u + dt*(1/2)*k1)
        k3 = model.rhs(t, u + dt*(1/2)*k2)
        k4 = model.rhs(t, u + dt*1*k3)
        unew = u + dt*(1/6)*(k1+ 2*k2 + 2*k3 + k4)
        return unew
    
def setup_save_folder_CDR(cwd, r, filter_dict) -> typing.Tuple[Path, str]:
        result_folder = cwd / "Results"
        if not result_folder.exists():
            result_folder.mkdir()

        

        standard_opinf_folder = result_folder / "CDR" /f"r_{r}"
        if not standard_opinf_folder.exists():
            standard_opinf_folder.mkdir(parents= True)

        savefolder = standard_opinf_folder

        identification = "_standard_"
        if filter_dict["filtering"]: #bool
            ft = filter_dict["filter_time"]
            identification = f"_filter_time_{ft}_"

            if filter_dict["filter_method"] == "Projection":
                filter_folder = standard_opinf_folder / "Projection_Filtering"
                identification += f"Proj_r_restriction_{filter_dict['r_restriction']}_"
                identification += f"chi_{filter_dict['chi']}"
                if not filter_folder.exists():
                    filter_folder.mkdir()

            elif filter_dict["filter_method"] == "Differential":
                filter_folder = standard_opinf_folder / "Differential_Filtering"
                identification += f"delta_{filter_dict['delta']}_"
                identification += f"chi_{filter_dict['chi']}"
                if not filter_folder.exists():
                    filter_folder.mkdir()

            elif filter_dict["filter_method"] == "Differential_Partial":
                filter_folder = standard_opinf_folder / "Differential_Partial"
                identification += f"delta_{filter_dict['delta']}_"
                identification += f"chi_{filter_dict['chi']}"
                identification += f"_r_restriction_{filter_dict['r_restriction']}_"
                
                if not filter_folder.exists():
                    filter_folder.mkdir()


            savefolder = filter_folder

        return savefolder, identification
    
def save_basis_funcs(
    indices: list[int],
    directory: Path,
    filename: str,
    basis,
    transformer,
    fenicsx_interface,
    timestep: float = 0.0,
):
    """
    Save selected POD/basis functions (and optionally the centering trajectory)
    as FE functions using a provided fenicsx_interface.

    Parameters
    ----------
    indices
        Iterable of integer indices of basis columns to save (0-based).
    directory
        Path to directory where files will be saved. The directory will be
        created if it does not exist.
    filename
        Filename (or filename prefix) passed to fenicsx_interface.save_functions_at_timestep.
    basis
        Object providing `entries` (numpy array, shape (n_dofs, n_basis)).
    transformer
        Object providing:
          - `.centering` (bool)
          - `.mean_` (array-like) if centering is True
    fenicsx_interface
        Object providing:
          - `lumped_data_to_FE_function_list(data_arr)` that maps a (n_dofs x n_funcs)
            numpy array to a list of dolfinx.fem.Function objects
          - `save_functions_at_timestep(timestep, func_list, directory, filename)`
            to write the functions to disk.
    timestep
        Time value to pass to save_functions_at_timestep (default 0.0).

    Returns
    -------
    func_list
        The list of FE functions created and saved.
    """
    directory = Path(directory)
    assert directory.is_dir(), f"directory {directory} does not exist"

    # Ensure indices is a list/array of ints
    idx = np.asarray(list(indices), dtype=int)
    if idx.size == 0:
        raise ValueError("`indices` must be a non-empty iterable of integers.")

    entries = np.asarray(basis.entries)
    n_dofs = entries.shape[0]
    data_len = idx.size

    # Build data array: columns are the requested basis functions; optionally append mean
    if getattr(transformer, "centering", False):
        data_arr = np.zeros((n_dofs, data_len + 1), dtype=np.float64)
    else:
        data_arr = np.zeros((n_dofs, data_len), dtype=np.float64)

    # Fill with selected basis columns (safely handle index range)
    try:
        data_arr[:, 0:data_len] = entries[:, idx]
    except IndexError as e:
        raise IndexError("One or more indices are out of range for basis.entries") from e

    if getattr(transformer, "centering", False):
        mean_vec = np.asarray(transformer.mean_)
        if mean_vec.shape[0] != n_dofs:
            raise ValueError("transformer.mean_ has incompatible length with basis.entries")
        data_arr[:, -1] = mean_vec

    # Convert lumped numpy data to FE function list
    func_list = fenicsx_interface.lumped_data_to_FE_function_list(data_arr)

    # Assign human-readable names
    if getattr(transformer, "centering", False):
        # last function is the mean
        for i, f in enumerate(func_list[:-1]):
            f.name = f"Basis_Func_{int(idx[i]) + 1}"
        func_list[-1].name = "Mean Function"
    else:
        for i, f in enumerate(func_list):
            f.name = f"Basis_Func_{int(idx[i]) + 1}"

    # Save using the provided interface
    fenicsx_interface.save_functions_at_timestep(timestep, func_list, directory, filename)

cwd = PROJECT_ROOT
meshpath = MESH_DIR / "CDR_mesh.msh"
dt = 0.01


#Peclet_list = [4000,4500,5500,6000]
Peclet_list = [5000,5500,6500,7000]
solution_end_time = 5.0
element_order = 2

CDR_FOM = fenicsx_class(meshpath, element_order)

FOM_sols = []
FOM_nus = []
FOM_sol_times = []

for i, peclet in enumerate(Peclet_list):

    solution_arr, solution_times = CDR_FOM.solve_CDR(peclet, dt, solution_end_time)
    
    FOM_sols.append(solution_arr)
    FOM_nus.append(1/peclet)
    FOM_sol_times.append(solution_times)
    if i > 0:
        assert np.allclose(solution_times, FOM_sol_times[i-1])

train_end = 2.0
train_end_idx = np.argmin(np.abs(solution_times- train_end))

train_sols = []
train_times = []

for i in range(len(Peclet_list)):
    train_sols.append(FOM_sols[i][:,:train_end_idx +1])
    train_times.append(FOM_sol_times[i][:train_end_idx+1])



transformer = opinf.pre.ShiftScaleTransformer(centering = True)
r = 10
basis = opinf.basis.PODBasis(num_vectors= r)
ddt_estimator = opinf.ddt.UniformFiniteDifferencer(solution_times)


operators = [opinf.operators.ConstantOperator(),
             opinf.operators.AffineLinearOperator(1)]
solver = opinf.lstsq.L2Solver(regularizer= 1e-5)
model = opinf.models.ParametricContinuousModel(operators = operators,solver = solver)

opinf_ROM = opinf.roms.ParametricROM( model = model,transformer=transformer, basis = basis,  ddt_estimator = ddt_estimator)

regularization_candidates = np.logspace(-10,3,100)
test_end = 20.0

opinf_ROM.fit_regselect_continuous(regularization_candidates, train_times, FOM_nus, train_sols,  
                                   fit_transformer=True, fit_basis=True,
                                    test_time_length=test_end, stability_margin= 1.2, verbose = True)
print(f"Regularizer used is: {opinf_ROM.model.solver.regularizer}")

test_pec = 6000

strong_reg = opinf_ROM.model.solver.regularizer

test_sol, test_times = CDR_FOM.solve_CDR(test_pec, dt, test_end)

predict_times = test_times

ROM_model = opinf_ROM.model
fixed_model = ROM_model.evaluate(1./test_pec)

u0 = test_sol[:,0]

ROM_u0 = opinf_ROM.encode(u0)


FOM_sol_list = CDR_FOM.lumped_data_to_FE_function_list(test_sol)

FOM_norm = CDR_FOM.compute_solution_norm(FOM_sol_list, predict_times)


strong_reg_dict = {"filtering": False,
                "dt": dt,
                "include_mean": False,
                "mean_func": None,
                "fenicsx_interface": CDR_FOM,
                "filter_time": train_end}

strong_reg_solver = RK_solvers(strong_reg_dict)

strong_reg_savefolder, strong_reg_identification = setup_save_folder_CDR(cwd, r, strong_reg_dict)

opinf_sol = strong_reg_solver.solve(fixed_model, ROM_u0,predict_times)
recon_opinf_sol = opinf_ROM.decode(opinf_sol)

strong_reg_ROM_sol_list = CDR_FOM.lumped_data_to_FE_function_list(recon_opinf_sol)

strong_reg_ROM_norm = CDR_FOM.compute_solution_norm(strong_reg_ROM_sol_list, predict_times)

strong_abs_err = CDR_FOM.compute_pointwise_errors(FOM_sol_list, predict_times,strong_reg_ROM_sol_list, predict_times)
strong_rel_e,_ = strong_abs_err / FOM_norm

ROM_model = opinf_ROM.model
ROM_model.solver.regularizer = 0.0
ROM_model.refit()
fixed_model = ROM_model.evaluate(1./test_pec)



standard_dict = {"filtering": False,
                "dt": dt,
                "include_mean": False,
                "mean_func": None,
                "fenicsx_interface": CDR_FOM,
                "filter_time": train_end}

proj_dict = {"filtering": True,
                "dt": dt,
                "filter_method": "Projection",
                "r_restriction": 1,
                "delta": 0.02,
                "chi": 0.01,
                "basis": opinf_ROM.basis,
                "include_mean": False,
                "mean_func": None,
                "fenicsx_interface": CDR_FOM,
                "filter_time": train_end}


diff_dict = {"filtering": True,
                "dt": dt,
                "filter_method": "Differential",
                "r_restriction": 5,
                "delta": 0.02,
                "chi": 0.01,
                "basis": opinf_ROM.basis,
                "include_mean": False,
                "mean_func": None,
                "fenicsx_interface": CDR_FOM,
                "filter_time": train_end}

diff_partial_dict = {"filtering": True,
                "dt": dt,
                "filter_method": "Differential_Partial",
                "r_restriction": 1,
                "delta": 0.02,
                "chi": 0.01,
                "basis": opinf_ROM.basis,
                "include_mean": False,
                "mean_func": None,
                "fenicsx_interface": CDR_FOM,
                "filter_time": train_end}




    
diff_solver = RK_solvers(diff_dict)

diff_savefolder, diff_identification = setup_save_folder_CDR(cwd, r, diff_dict)

opinf_sol = diff_solver.solve(fixed_model, ROM_u0,predict_times)
recon_opinf_sol = opinf_ROM.decode(opinf_sol)

diff_ROM_sol_list = CDR_FOM.lumped_data_to_FE_function_list(recon_opinf_sol)

diff_ROM_norm = CDR_FOM.compute_solution_norm(diff_ROM_sol_list, predict_times)

diff_abs_err = CDR_FOM.compute_pointwise_errors(FOM_sol_list, predict_times,diff_ROM_sol_list, predict_times)
diff_rel_e,_ = diff_abs_err / FOM_norm

# Differential partial filtering -- mirror the diff_dict workflow exactly
diff_partial_solver = RK_solvers(diff_partial_dict)

diff_partial_savefolder, diff_partial_identification = setup_save_folder_CDR(cwd, r,diff_partial_dict)

diff_partial_opinf_sol = diff_partial_solver.solve(fixed_model, ROM_u0, predict_times)
diff_partial_recon_opinf_sol = opinf_ROM.decode(diff_partial_opinf_sol)

diff_partial_ROM_sol_list = CDR_FOM.lumped_data_to_FE_function_list(diff_partial_recon_opinf_sol)

diff_partial_ROM_norm = CDR_FOM.compute_solution_norm(diff_partial_ROM_sol_list, predict_times)

diff_partial_abs_err = CDR_FOM.compute_pointwise_errors(FOM_sol_list, predict_times,
                                                       diff_partial_ROM_sol_list, predict_times)
diff_partial_rel_e, _ = diff_partial_abs_err / FOM_norm

# Standard (no filtering) -- mirror the diff_dict workflow exactly
standard_solver = RK_solvers(standard_dict)

standard_savefolder, standard_identification = setup_save_folder_CDR(cwd, r,standard_dict)

standard_opinf_sol = standard_solver.solve(fixed_model, ROM_u0, predict_times)
standard_recon_opinf_sol = opinf_ROM.decode(standard_opinf_sol)

standard_ROM_sol_list = CDR_FOM.lumped_data_to_FE_function_list(standard_recon_opinf_sol)

standard_ROM_norm = CDR_FOM.compute_solution_norm(standard_ROM_sol_list, predict_times)

standard_abs_err = CDR_FOM.compute_pointwise_errors(FOM_sol_list, predict_times,
                                                   standard_ROM_sol_list, predict_times)
standard_rel_e, _ = standard_abs_err / FOM_norm


# Projection filtering -- mirror the diff_dict workflow exactly
proj_solver = RK_solvers(proj_dict)

proj_savefolder, proj_identification = setup_save_folder_CDR(cwd,r, proj_dict)

proj_opinf_sol = proj_solver.solve(fixed_model, ROM_u0, predict_times)
proj_recon_opinf_sol = opinf_ROM.decode(proj_opinf_sol)

proj_ROM_sol_list = CDR_FOM.lumped_data_to_FE_function_list(proj_recon_opinf_sol)

proj_ROM_norm = CDR_FOM.compute_solution_norm(proj_ROM_sol_list, predict_times)

proj_abs_err = CDR_FOM.compute_pointwise_errors(FOM_sol_list, predict_times,
                                               proj_ROM_sol_list, predict_times)
proj_rel_e, _ = proj_abs_err / FOM_norm

# basis_path = standard_savefolder
# basis_idcs = [i for i in range(r)]
# basis_filename = "basis_pics.bp"
# save_basis_funcs(    basis_idcs,
#     basis_path,
#     basis_filename,
#     opinf_ROM.basis,
#     opinf_ROM.transformer,
#     CDR_FOM,
#     timestep = 0.0)


# Save VTK/BP files for each model

# fom_filename = "FOM_solutions.bp"
# CDR_FOM.save_function_list_VTK(FOM_sol_list, standard_savefolder, fom_filename, predict_times)
# print(f"Saved FOM solutions to: {standard_savefolder / fom_filename}")

# strong_reg_opinf_filename = "ROM_solutions" + f'strong_reg_{strong_reg:.2e}' + ".bp"
# CDR_FOM.save_function_list_VTK(strong_reg_ROM_sol_list, standard_savefolder, strong_reg_opinf_filename, predict_times)
# print(f"Saved Strong Reg OpInf solutions to: {standard_savefolder / strong_reg_opinf_filename}")

# standard_filename = standard_identification + "_solutions.bp"
# CDR_FOM.save_function_list_VTK(standard_ROM_sol_list, standard_savefolder, standard_filename, predict_times)
# print(f"Saved standard solutions to: {standard_savefolder / standard_filename}")

# proj_filename = proj_identification + "_solutions.bp"
# CDR_FOM.save_function_list_VTK(proj_ROM_sol_list, proj_savefolder, proj_filename, predict_times)
# print(f"Saved projection-filtered solutions to: {proj_savefolder / proj_filename}")

# diff_filename = diff_identification + "_solutions.bp"
# CDR_FOM.save_function_list_VTK(diff_ROM_sol_list, diff_savefolder, diff_filename, predict_times)
# print(f"Saved differential-filtered solutions to: {diff_savefolder / diff_filename}")

# diff_partial_filename = diff_partial_identification + "_solutions.bp"
# CDR_FOM.save_function_list_VTK(diff_partial_ROM_sol_list, diff_partial_savefolder, diff_partial_filename, predict_times)
# print(f"Saved differential-partial-filtered solutions to: {diff_partial_savefolder / diff_partial_filename}")

import matplotlib.pyplot as plt

EFR_color = 'red'
weak_opinf_color = 'sienna'
strong_opinf_color = 'teal'
projection_color = 'cornflowerblue'
differential_color = 'mediumorchid'
diffrential_part_color = 'coral'

L2_compare_name = "L2_Errors" + standard_identification +  f'strong_reg_{strong_reg:.2e}' +  ".png"
L2savepath = standard_savefolder / L2_compare_name

fig, ax = plt.subplots()
ax.semilogy(predict_times, standard_rel_e, label = "Unregularized OpInf", color = weak_opinf_color)
ax.semilogy(predict_times, strong_rel_e, label = "Strongly Reg. OpInf", color = strong_opinf_color)
ax.semilogy(predict_times, proj_rel_e, label = "EFR-OpInf-Proj", color = projection_color)
ax.semilogy(predict_times, diff_rel_e, label = "EFR-OpInf-DF", color = differential_color)
ax.semilogy(predict_times, diff_partial_rel_e, label = "EFR-OpInf-HDF", color = diffrential_part_color)
ax.axvline(train_end, color = 'blue', linestyle = '--', label= 'End of training region')
ax.axvline(test_end, color = 'purple', linestyle = '--', label= 'End of validation region')
ax.set_xlabel("Time")
ax.set_ylabel(r"Relative $L^2$ solution error")
ax.set_title(r"Relative $L^2$ errors")
ax.legend()
#plt.show()
fig.savefig(L2savepath)


# Save a time-series plot of solution norms (analogous to the L2 error figure)
norm_compare_name = "Solution_Norms" + standard_identification +  f'strong_reg_{strong_reg:.2e}' + ".png"
normsavepath = standard_savefolder / norm_compare_name

fig, ax = plt.subplots()
ax.plot(predict_times, FOM_norm, label="FOM norm", color='black')
ax.plot(predict_times, standard_ROM_norm, label="Unregularized OpInf", color=weak_opinf_color)
ax.plot(predict_times, strong_reg_ROM_norm, label="Strongly Reg. OpInf", color=strong_opinf_color)
ax.plot(predict_times, proj_ROM_norm, label="EFR-OpInf-Proj", color=projection_color)
ax.plot(predict_times, diff_ROM_norm, label="EFR-OpInf-DF", color=differential_color)
ax.plot(predict_times, diff_partial_ROM_norm, label="EFR-OpInf-HDF", color=diffrential_part_color)
ax.axvline(train_end, color='blue', linestyle='--', label='End of training data')
ax.axvline(test_end, color = 'purple', linestyle = '--', label= 'End of validation region')
# ax.set_ylim((10**(-2),10**2))
ax.set_xlabel("Time")
ax.set_ylabel(r"$L^2$ solution norm")
ax.set_title(r"FOM vs ROM $L^2$ solution norms")
ax.legend()
#plt.show()
fig.savefig(normsavepath)
print(f"Saved solution-norm comparison to: {normsavepath}")