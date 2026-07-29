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


class OpInf_ROM:
    def __init__(self, opinf_dict):
        """  opinf_dict = {
        "time_start": 0.0,
        "time_end": time_end,
        "predict_time": 5.0,
        "r": 10,
        "centering": True,
        "time_integrator": RK4_solver,
        "fenicsx_interface": CDR_FOM,
        "FOM_sol": solution_arr,
        "FOM_times": solution_times
         }
        """
        for key, value in opinf_dict.items():
            setattr(self, key, value)

        self._process_data(self.fenicsx_interface)
        self._setup_opinf_ROM()

    def _process_data(self, fenicsx_interface: fenicsx_class):

        coords = fenicsx_interface.coords
        FOM_times = self.FOM_times
        data_arr = self.FOM_sol

        self.coords = coords

        data = fenicsx_interface.lumped_data_to_FE_function_list(data_arr)

        assert len(FOM_times) == len(data), "Loaded times and data do not match!"

        time_start = self.time_start
        time_end = self.time_end
        predict_time = self.predict_time

        time_start_idx = np.argmin(np.abs(FOM_times - time_start))

        if not np.isclose(FOM_times[time_start_idx],time_start):
            print(f'warning: closest timestep to requested train start of {time_start} is {FOM_times[time_start_idx]}')
        time_end_idx = np.argmin(np.abs(FOM_times - time_end)) + 1

        time_predict_idx = np.argmin(np.abs(FOM_times - predict_time))+1

        extract_times = FOM_times[time_start_idx: time_predict_idx]

        fom_sol_list = data[time_start_idx:time_predict_idx]

        train_and_predict_data = data_arr[:, time_start_idx:time_predict_idx]

        time_start_idx = np.argmin(np.abs(extract_times - time_start))
        time_end_idx = np.argmin(np.abs(extract_times - time_end)) + 1

        time_predict_idx = np.argmin(np.abs(extract_times - predict_time))+1

        train_times = extract_times[time_start_idx:time_end_idx]
        train_data = train_and_predict_data[:, time_start_idx:time_end_idx]
        predict_times = extract_times[time_start_idx:time_predict_idx]

        self.fom_sol_list = fom_sol_list
        self.train_times = train_times
        self.train_data = train_data
        self.predict_times = predict_times
        self.train_and_predict_data = train_and_predict_data

        self.time_start_idx = time_start_idx
        self.time_end_idx = time_end_idx
        self.time_predict_idx = time_predict_idx

    def _setup_opinf_ROM(self):
        centering = self.centering
        train_data = self.train_data
        r = self.r
        train_times = self.train_times

        transformer = opinf.pre.ShiftScaleTransformer(centering = centering)
        trans_data = transformer.fit_transform(train_data)

        basis = opinf.basis.PODBasis(num_vectors=r)
        comp_data = basis.fit_compress(trans_data)

        operators = [opinf.operators.LinearOperator(),
                     opinf.operators.QuadraticOperator(),
                     opinf.operators.ConstantOperator()]
        
        solver = opinf.lstsq.TikhonovSolver()

        linear_reg = 1e-5

        ddt_estimator = opinf.ddt.UniformFiniteDifferencer(train_times, scheme="bwd2")

        states, ddts = ddt_estimator.estimate(comp_data)

        solver = opinf.lstsq.L2Solver(linear_reg)

        my_model = opinf.models.ContinuousModel(operators=operators, solver=solver)

        my_model.fit(states,ddts)

        self.comp_data = comp_data
        self.u0 = comp_data[:,0]
        self.opinf_model = my_model
        self.transformer = transformer
        self.basis = basis

    def update_model_regularization(self, beta1):
        self.opinf_model.solver.regularizer = beta1
        self.opinf_model.refit()
    
    def optimize_model_training(self, linear_dict):
        """Optimize model while only looking inside training data"""
        linear_min = linear_dict["min"]
        linear_max = linear_dict["max"]
        linear_num = linear_dict["num"]

        opinf_model = self.opinf_model
        solver = opinf_model.solver
        train_times = self.train_times
        comp_data = self.comp_data
        transformer = self.transformer
        operators = opinf_model.operators
        time_integrator = self.time_integrator
        r = self.r
        predict_times = self.predict_times

        B1 = np.logspace(linear_min, linear_max, num=linear_num)

        # Set the threshold for the maximum growth of the inferred reduced
        # coefficients, used for selecting the optimal regularization parameter pair.
        max_growth = 1.2


        # Loop over all regularization pairs.
        best_train_err = 1e20
        best_beta1 = None

        mean_Qhat_train = np.mean(comp_data, axis=1)
        max_diff_Qhat_train = np.max(np.abs(comp_data - mean_Qhat_train[:,np.newaxis]), axis=1)
        for beta1 in tqdm(B1, desc = "Optimizing Regularization Within Training Data"):
            self.update_model_regularization(beta1)
            try: 
                opinf_model.refit()
            except:
                continue        
            u0 = comp_data[:,0]
            Q_ROM = time_integrator.solve(opinf_model, u0, train_times)

            if np.any(np.isnan(Q_ROM)):
                continue  # go to next iteration of the for loop.

            if Q_ROM.shape[1] != comp_data.shape[1]:
                continue # if integration failed, go to next iteration

            # Q_ROM_predict = time_integrator.solve(opinf_model, u0, predict_times)

            # max_diff_Qhat_trial = np.max(
            #     np.abs(Q_ROM_predict - mean_Qhat_train[:,np.newaxis]), axis=1
            # )
            # max_growth_trial = np.max(max_diff_Qhat_trial) / np.max(
            #     max_diff_Qhat_train
            # )
            # if max_growth_trial > max_growth:
            #     continue 

            abs_e, rel_e = opinf.post.lp_error(comp_data,Q_ROM, p = 2)
            train_err = np.mean(rel_e)
            if train_err < best_train_err:
                best_beta1 = beta1
                best_train_err = train_err
                best_rel_e = rel_e
                Q_ROM_opt = Q_ROM
        self.best_beta1 = best_beta1
        self.Q_ROM_opt = Q_ROM_opt
        print(
            f"Optimal regularization parameters: Linear Regularizer = {best_beta1} with training error = {np.mean(best_train_err):.2e}" 
        )

    def optimize_model(self, linear_dict):
        """Optimize model while looking beyond training data for instability only"""
        linear_min = linear_dict["min"]
        linear_max = linear_dict["max"]
        linear_num = linear_dict["num"]

        opinf_model = self.opinf_model
        solver = opinf_model.solver
        train_times = self.train_times
        comp_data = self.comp_data
        transformer = self.transformer
        operators = opinf_model.operators
        time_integrator = self.time_integrator
        r = self.r
        predict_times = self.predict_times

        B1 = np.logspace(linear_min, linear_max, num=linear_num)

        # Set the threshold for the maximum growth of the inferred reduced
        # coefficients, used for selecting the optimal regularization parameter pair.
        max_growth = 1.2


        # Loop over all regularization pairs.
        best_train_err = 1e20
        best_beta1 = None

        mean_Qhat_train = np.mean(comp_data, axis=1)
        max_diff_Qhat_train = np.max(np.abs(comp_data - mean_Qhat_train[:,np.newaxis]), axis=1)
        for beta1 in tqdm(B1, desc = "Optimizing Regularization in Training Data while maintaining stability"):
            self.update_model_regularization(beta1)
            try: 
                opinf_model.refit()
            except:
                continue        
            u0 = comp_data[:,0]
            Q_ROM = time_integrator.solve(opinf_model, u0, train_times)

            if np.any(np.isnan(Q_ROM)):
                continue  # go to next iteration of the for loop.

            if Q_ROM.shape[1] != comp_data.shape[1]:
                continue # if integration failed, go to next iteration

            Q_ROM_predict = time_integrator.solve(opinf_model, u0, predict_times)

            max_diff_Qhat_trial = np.max(
                np.abs(Q_ROM_predict - mean_Qhat_train[:,np.newaxis]), axis=1
            )
            max_growth_trial = np.max(max_diff_Qhat_trial) / np.max(
                max_diff_Qhat_train
            )
            if max_growth_trial > max_growth:
                continue 

            abs_e, rel_e = opinf.post.lp_error(comp_data,Q_ROM, p = 2)
            train_err = np.mean(rel_e)
            if train_err < best_train_err:
                best_beta1 = beta1
                best_train_err = train_err
                best_rel_e = rel_e
                Q_ROM_opt = Q_ROM
        self.best_beta1 = best_beta1
        self.Q_ROM_opt = Q_ROM_opt
        print(
            f"Optimal regularization parameters: Linear Regularizer = {best_beta1} with training error = {np.mean(best_train_err):.2e}" 
        )

    def compare_filter(self, RK_filter_solver: RK_solvers, timestep: float, directory: Path, filename: str, filter_type = "Projection"):
        data_source = self.train_and_predict_data
        data_times = self.predict_times

        time_idx = np.argmin(np.abs(data_times - timestep))

        chosen_func = data_source[:,time_idx]

        assert RK_filter_solver.filter_method == filter_type

        transformer = self.transformer
        basis = self.basis

        ROM_sol = basis.compress(transformer.transform(chosen_func))

        u_filter = RK_filter_solver.filter_solution(ROM_sol)

        fenicsx_interface = self.fenicsx_interface

        reconstructed_ROM_sol = basis.decompress(ROM_sol)
        reconstructed_ROM_sol_list = self.convert_lumped_data_to_FE_function_list(reconstructed_ROM_sol[:, np.newaxis])

        reconstructed_u_filter = basis.decompress(u_filter)
        reconstructed_u_filter_list = self.convert_lumped_data_to_FE_function_list(reconstructed_u_filter[:, np.newaxis])

        mean_centered_data = transformer.transform(chosen_func[:, np.newaxis])
        original_function_list = self.convert_lumped_data_to_FE_function_list(mean_centered_data)

        ROM_proj_func = reconstructed_ROM_sol_list[0]
        ROM_proj_func.name = "u_ROM"

        ROM_filter_func = reconstructed_u_filter_list[0]
        ROM_filter_func.name = "u_filter"

        FOM_func = original_function_list[0]
        FOM_func.name = "u_FOM"

        func_list = [ROM_proj_func, ROM_filter_func, FOM_func]

        fenicsx_interface.save_functions_at_timestep(timestep, func_list, directory, filename)

    def compute_FOM_KE(self):
        time_predict_idx = self.time_predict_idx
        fom_sol_list = self.fom_sol_list
        predict_times = self.predict_times
        fenicsx_interface = self.fenicsx_interface


        fenicsx_sol_list = fom_sol_list[0:time_predict_idx]
        fenicsx_KE_array = fenicsx_interface.compute_KE_arr(fenicsx_sol_list)

        return predict_times, fenicsx_KE_array
    
    def convert_lumped_data_to_FE_function_list(self, recon_data):
        fenicsx_interface = self.fenicsx_interface
        rom_sol_list = fenicsx_interface.lumped_data_to_FE_function_list(recon_data)
        return rom_sol_list
    
    def compute_ROM_KE(self, recon_data_function_list, verbose = False):
        
        predict_times = self.predict_times
        fenicsx_interface = self.fenicsx_interface

        rom_KE_array = fenicsx_interface.compute_KE_arr(recon_data_function_list, verbose)

        return predict_times, rom_KE_array

    def reconstruct_ROM_sol(self, QROM):
        transformer = self.transformer
        basis = self.basis

        reconstructed_ROM = transformer.inverse_transform(basis.decompress(QROM))
        return reconstructed_ROM

    def check_EFR_chi(self, trained_opinf_model: opinf.models.ContinuousModel, u0: np.ndarray, chi_list: np.ndarray, solver: RK_solvers, run_times: np.ndarray, max_growth =  None):
        comp_data = self.comp_data
        mean_Qhat_train = np.mean(comp_data, axis=1)
        #predict_times = self.predict_times

        best_growth = 10.0
        max_diff_Qhat_train = np.max(np.abs(comp_data - mean_Qhat_train[:,np.newaxis]), axis=1)

        if max_growth is None:
            max_growth = 1.2 
            print("Defaulting to max growth factor of 1.2")
        acceptable_chi = 0.5

        for chi in chi_list:
            solver.update_chi_only(chi)
            Q_ROM = solver.solve(trained_opinf_model, u0, run_times)

            if np.any(np.isnan(Q_ROM)):
                continue  # go to next iteration of the for loop.

            if Q_ROM.shape[1] != comp_data.shape[1]:
                continue # if integration failed, go to next iteration

            max_diff_Qhat_trial = np.max(
                np.abs(Q_ROM - mean_Qhat_train[:,np.newaxis]), axis=1
            )
            max_growth_trial = np.max(max_diff_Qhat_trial) / np.max(
                max_diff_Qhat_train)
            
            if max_growth_trial < max_growth:
                acceptable_chi = chi
                break

        if best_growth >= 10.0:
            print("No acceptable chi values found!")
        return acceptable_chi

    def save_basis_funcs(self, indices: list[int], directory: Path, filename):
        basis = self.basis
        transformer = self.transformer

        data_len = len(indices)

        if self.transformer.centering:
            data_arr = np.zeros((basis.entries.shape[0], data_len+1), dtype = np.float64)
        else: 
            data_arr = np.zeros((basis.entries.shape[0], data_len), dtype = np.float64)

        data_arr[:, 0:data_len] = self.basis.entries[:,indices]

        if self.transformer.centering:
            data_arr[:,-1] = transformer.mean_

        func_list = self.fenicsx_interface.lumped_data_to_FE_function_list(data_arr)

        if self.transformer.centering:
            for i in range(len(func_list)-1):
                func_list[i].name = f"Basis_Func_{indices[i]+1}"
            func_list[-1].name = f"Mean Function"
        else:
            for i in range(len(func_list)):
                func_list[i].name = f"Basis_Func_{indices[i]+1}"

        self.fenicsx_interface.save_functions_at_timestep(0.0, func_list, directory, filename)

    def plot_KE(self, FOM_KE_times: np.ndarray, ROM_KE_times: np.ndarray, FOM_KE: np.ndarray, ROM_KE: np.ndarray, 
                train_times: np.ndarray, savefolder: Path, identification: str):
        fig, ax = plt.subplots()
        ax.plot(FOM_KE_times, FOM_KE, label='FOM KE', color='black')
        ax.plot(ROM_KE_times, ROM_KE, label='OpInf KE', color='red')
        ax.set_ylim(0.5, 0.7)
        if not np.isclose(ROM_KE_times[-1], train_times[-1]): # means we are in predictive regime
            last_train_time = train_times[-1]
            ax.axvline(x=last_train_time, color='blue', linestyle='--', label='End of training data')
        if (FOM_KE_times[-1] + (FOM_KE_times[1] - FOM_KE_times[0])) < ROM_KE_times[-1]: # IF final FOM time + dt < final ROM time
            ax.axvline(x=FOM_KE_times[-1], color='purple', linestyle='--', label='End of FOM data')
            identification += f"_extended_time_{ROM_KE_times[-1]}_"
        ax.set_xlabel('Time')
        ax.set_ylabel('Kinetic Energy')
        ax.set_title('FOM vs OpInf Kinetic Energy')
        ax.legend()
        fig.savefig(savefolder / ("Opinf_KE" + identification + ".png") )

    def setup_save_folder(self,cwd, r, filter_dict) -> typing.Tuple[Path, str]:
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
    
if __name__ == "__main__":   
    cwd = PROJECT_ROOT
    meshpath = MESH_DIR / "CDR_mesh.msh"
    dt = 0.01
    Peclet = 5000
    solution_end_time = 5.0
    element_order = 2

    CDR_FOM = fenicsx_class(meshpath, element_order)

    solution_arr, solution_times = CDR_FOM.solve_CDR(Peclet, dt, solution_end_time)

    FOM_function_list = CDR_FOM.lumped_data_to_FE_function_list(solution_arr)

    CDR_FOM.plot_function(FOM_function_list[-1])

    RK4_dict = {"filtering": False,
                "dt": dt,
                }
    
    RK4_solver = RK_solvers(RK4_dict)

    time_end = 0.75
    
    opinf_dict = {
        "time_start": 0.0,
        "time_end": time_end,
        "predict_time": 5.0,
        "r": 10,
        "centering": True,
        "time_integrator": RK4_solver,
        "fenicsx_interface": CDR_FOM,
        "FOM_sol": solution_arr,
        "FOM_times": solution_times
         }

    opinf_ROM = OpInf_ROM(opinf_dict)

    linear_reg = {"min": -10,
                    "max": -1,
                    "num":40}
    
    opinf_ROM.optimize_model_training(linear_reg)

    best_beta1 = opinf_ROM.best_beta1

    opinf_ROM.update_model_regularization(best_beta1)
    print(f"Using L2 regularization {best_beta1}")


    opinf_model = opinf_ROM.opinf_model

    r = opinf_ROM.r

    savefolder, identification = opinf_ROM.setup_save_folder(cwd, r, RK4_dict)

    print(savefolder)
    print(identification)

    predict_times = opinf_ROM.predict_times

    filter_dict = {"filtering": False,
                    "dt": dt,
                    "filter_method": "Projection",
                    "r_restriction": 5,
                    "delta": 0.02,
                    "chi": 0.1,
                    "basis": opinf_ROM.basis,
                    "include_mean": False,
                    "mean_func": None,
                    "fenicsx_interface": CDR_FOM,
                    "filter_time": time_end}
    
    filter_solver = RK_solvers(filter_dict)

    u0 = opinf_ROM.u0
    Q_ROM = filter_solver.solve(opinf_model, u0, predict_times)
    recon_data = opinf_ROM.reconstruct_ROM_sol(Q_ROM)
    ROM_sol_list = opinf_ROM.convert_lumped_data_to_FE_function_list(recon_data)

    CDR_FOM.plot_function(ROM_sol_list[-1])

    FOM_data_arr = opinf_ROM.train_and_predict_data

    times = opinf_ROM.predict_times

    FOM_sol_list = opinf_ROM.fom_sol_list

    fom_norm = CDR_FOM.compute_solution_norm(FOM_sol_list, times)
    abs_L2_e, _ = CDR_FOM.compute_pointwise_errors(FOM_sol_list, times, ROM_sol_list, times)
    rom_norm = CDR_FOM.compute_solution_norm(ROM_sol_list, times)

    rel_L2_e = abs_L2_e / fom_norm

    import matplotlib.pyplot as plt

    plt.semilogy(times, rel_L2_e)
    plt.title('L2 Error')
    plt.show()

    plt.plot(times, rom_norm, label = r'ROM $L^2$ Norm')
    plt.plot(times, fom_norm, label = r'FOM $L^2$ Norm')
    plt.title(r'$L^2$ Norms')
    plt.axvline(x = time_end, label = "End of training data")
    plt.legend()
    plt.show()


    # basis_function_filename = "basis_funcs.bp"
    # basis_idcs = [i for i in range(10)]
    # opinf_ROM.save_basis_funcs(basis_idcs,savefolder, basis_function_filename)


    





    print("hold up")