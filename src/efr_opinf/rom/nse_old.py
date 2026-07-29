import numpy as np
from pathlib import Path

from efr_opinf._paths import MESH_DIR, DATA_DIR, RESULTS_DIR
import opinf
import itertools
import time

from efr_opinf.interfaces.nse_old import fenicsx_class
import matplotlib.pyplot as plt

class RK_solvers:
    def __init__(self, solver_dict):
        self.filter_bool = solver_dict["filtering"]
        self.dt = 0.05 # Hardcoded value for this sim     ]
        if self.filter_bool == True:
            self.chi = solver_dict["chi"]
            self.filter_time = solver_dict["filter_time"]
            self.filter_method = solver_dict["filter_method"]

            if self.filter_method == "Projection":
                self.r_restriction = solver_dict["r_restriction"]


            elif self.filter_method == "Differential":
                self.delta = solver_dict["delta"]
                print("Warning: Differential filtering not fully implemented")

            else:
                raise ValueError("Filter method can only be one of `Projection`, `Differential`")

    def update_filter_parameters(self, filter_dict):
        assert self.filter_bool == True, "This solver was not initialized for filtering"
        assert filter_dict["filter_method"] == self.filter_method, "Solver filtering type does not match input"
        if filter_dict["filter_method"] == "Projection":
            self.chi = filter_dict["chi"]
            self.filter_time = filter_dict["filter_time"]
            self.r_restriction = filter_dict["r_restriction"]

        if filter_dict["filter_method"] == "Projection":
            raise ValueError("Differential Filtering NYI")
            


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
            raise ValueError("Differential Filtering is NYI")
        else: 
            raise ValueError("Available filtering options are Projection and Differential")
        
    def projection_filter(self,unew):
        r_restriction = self.r_restriction
        u_filter = np.zeros_like(unew, dtype = np.float64)
        u_filter[:r_restriction] = unew[:r_restriction]
        return u_filter

    def relax_solution(self, unew, ufilter):
        chi = self.chi
        u_EFR = chi*ufilter + (1-chi)*unew
        return u_EFR


    def RK4_step(self, model, t, u, dt):
        k1 = model.rhs(t, u)
        k2 = model.rhs(t, u + dt*(1/2)*k1)
        k3 = model.rhs(t, u + dt*(1/2)*k2)
        k4 = model.rhs(t, u + dt*1*k3)
        unew = u + dt*(1/6)*(k1+ 2*k2 + 2*k3 + k4)
        return unew

class OpInf_ROM:
    def __init__(self, opinf_dict):
        """  opinf_dict = {
        "time_start": 70.0,
        "time_end": 105.0,
        "predict_time": 150.0,
        "r": 30,
        "centering": True,
        "datapath": data_dir,
        "time_integrator": RK4_solver,
        "fenicsx_interface": fenicsx_interface
         }
        """
        for key, value in opinf_dict.items():
            setattr(self, key, value)

        self._load_data(self.datapath)
        self._setup_opinf_ROM()

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

        reconstructed_ROM_sol = transformer.inverse_transform(basis.decompress(ROM_sol))
        reconstructed_ROM_sol_list = self.convert_lumped_data_to_FE_function_list(reconstructed_ROM_sol[:, np.newaxis])

        reconstructed_u_filter = transformer.inverse_transform(basis.decompress(u_filter))
        reconstructed_u_filter_list = self.convert_lumped_data_to_FE_function_list(reconstructed_u_filter[:, np.newaxis])

        original_function_list = self.convert_lumped_data_to_FE_function_list(chosen_func[:, np.newaxis])

        ROM_proj_func = reconstructed_ROM_sol_list[0]
        ROM_proj_func.name = "u_ROM"

        ROM_filter_func = reconstructed_u_filter_list[0]
        ROM_filter_func.name = "u_filter"

        FOM_func = original_function_list[0]
        FOM_func.name = "u_FOM"

        func_list = [ROM_proj_func, ROM_filter_func, FOM_func]

        fenicsx_interface.save_function_list_VTK(func_list, directory, filename, time = timestep)


        








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
    
    def compute_ROM_KE(self, recon_data_function_list):
        
        predict_times = self.predict_times
        fenicsx_interface = self.fenicsx_interface

        rom_KE_array = fenicsx_interface.compute_KE_arr(recon_data_function_list)

        return predict_times, rom_KE_array

    def reconstruct_ROM_sol(self, QROM):
        transformer = self.transformer
        basis = self.basis

        reconstructed_ROM = transformer.inverse_transform(basis.decompress(QROM))
        return reconstructed_ROM

    def _setup_opinf_ROM(self):
        centering = self.centering
        train_data = self.train_data
        r = self.r
        train_times = self.train_times

        transformer = opinf.pre.ShiftScaleTransformer(centering = centering)
        trans_data = transformer.fit_transform(train_data)

        basis = opinf.basis.PODBasis(num_vectors=r)
        comp_data = basis.fit_compress(trans_data)

        if transformer.centering:
            operators = [
                opinf.operators.ConstantOperator(),
                opinf.operators.LinearOperator(),
                opinf.operators.QuadraticOperator()]
        else:
            operators = [
                opinf.operators.LinearOperator(),
                opinf.operators.QuadraticOperator()]
        
        solver = opinf.lstsq.TikhonovSolver()

        linear_reg = 1e-5
        quadratic_reg = 1e-1
        if transformer.centering:
            tikreg = solver.get_operator_regularizer(operators, [ linear_reg, linear_reg, quadratic_reg], r)
        else:
            tikreg = solver.get_operator_regularizer(operators, [ linear_reg, quadratic_reg], r)

        solver.regularizer = tikreg

        my_model = opinf.models.ContinuousModel(operators=operators, solver=solver)

        ddt_estimator = opinf.ddt.UniformFiniteDifferencer(train_times, scheme="bwd2")

        states, ddts = ddt_estimator.estimate(comp_data)

        my_model.fit(states,ddts)

        self.comp_data = comp_data
        self.u0 = comp_data[:,0]
        self.opinf_model = my_model
        self.transformer = transformer
        self.basis = basis


    def _load_data(self, datapath):
        data = np.load(datapath / "compressed_componentwise_data.npz")
        ux = data['ux']
        uy = data['uy']
        NSE_times = data['times']
        dofs = data['dofs']

        time_start = self.time_start
        time_end = self.time_end
        predict_time = self.predict_time

        fenicsx_interface = self.fenicsx_interface

        time_start_idx = np.argmin(np.abs(NSE_times - time_start))
        time_end_idx = np.argmin(np.abs(NSE_times - time_end)) + 1

        time_predict_idx = np.argmin(np.abs(NSE_times - predict_time))+1

        extract_times = NSE_times[time_start_idx: time_predict_idx]

        extract_ux = ux[:,time_start_idx:time_predict_idx]
        extract_uy = uy[:,time_start_idx: time_predict_idx]

        # This ensures the the ordering of DoFs is correct wrt the FE environment
        fom_sol_list = fenicsx_interface.split_data_to_FE_function_list(dofs, extract_ux, extract_uy)
        train_and_predict_data = fenicsx_interface.FE_function_list_to_arr(fom_sol_list)

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

    def update_model_regularization(self, beta1, beta2):
        operators = self.opinf_model.operators
        r = self.r
        if self.transformer.centering:
                tikreg = self.opinf_model.solver.get_operator_regularizer(operators, [beta1, beta1, beta2], r)
        else:
            tikreg = self.opinf_model.solver.get_operator_regularizer(operators, [beta1, beta2], r)
        self.opinf_model.solver.regularizer = tikreg
        self.opinf_model.refit()

    def optimize_model(self, linear_dict, quadratic_dict):
        linear_min = linear_dict["min"]
        linear_max = linear_dict["max"]
        linear_num = linear_dict["num"]

        quadratic_min = quadratic_dict["min"]
        quadratic_max = quadratic_dict["max"]
        quadratic_num = quadratic_dict["num"]

        opinf_model = self.opinf_model
        solver = opinf_model.solver
        train_times = self.train_times
        comp_data = self.comp_data
        transformer = self.transformer
        operators = opinf_model.operators
        time_integrator = self.time_integrator
        r = self.r

        B1 = np.logspace(linear_min, linear_max, num=linear_num)
        B2 = np.logspace(quadratic_min, quadratic_max, num=quadratic_num)
        # Get the Cartesian product of all regularization pairs (beta1, beta2).
        reg_pairs_global = list(itertools.product(B1, B2))
        n_reg_global = len(reg_pairs_global)

        # Set the threshold for the maximum growth of the inferred reduced
        # coefficients, used for selecting the optimal regularization parameter pair.
        max_growth = 1.2


        # Loop over all regularization pairs.
        best_train_err = 1e20
        best_beta1, best_beta2 = None, None

        mean_Qhat_train = np.mean(comp_data, axis=1)
        max_diff_Qhat_train = np.max(np.abs(comp_data - mean_Qhat_train[:,np.newaxis]), axis=1)
        for beta1, beta2 in reg_pairs_global:
            if transformer.centering:
                tikreg = solver.get_operator_regularizer(operators, [beta1, beta1, beta2], r)
            else:
                tikreg = solver.get_operator_regularizer(operators, [beta1, beta2], r)
            solver.regularizer = tikreg
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

            max_diff_Qhat_trial = np.max(
                np.abs(Q_ROM - mean_Qhat_train[:,np.newaxis]), axis=1
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
                best_beta2 = beta2
                best_train_err = train_err
                best_rel_e = rel_e
                Q_ROM_opt = Q_ROM
        self.best_beta1 = best_beta1
        self.best_beta2 = best_beta2
        self.Q_ROM_opt = Q_ROM_opt
        print(
            f"Optimal regularization parameters: Linear Regularizer = {best_beta1}, Quadratic Regularizer = {best_beta2} with training error = {np.mean(best_train_err):.2e}" 
        )

if __name__ == "__main__":
    data_dir = DATA_DIR

    RK4_dict = {"filtering": False}

    RK4_solver = RK_solvers(RK4_dict)

    Mesh_fld = MESH_DIR
    Mesh_fld.mkdir(exist_ok = True)

    mesh_file = str(Mesh_fld) + "/BFS_Mesh"
    fenicsx_interface = fenicsx_class()
    fenicsx_interface.load_mesh(mesh_file)
    fenicsx_interface.create_function_space()

    opinf_dict = {
        "time_start": 70.0,
        "time_end": 105.0,
        "predict_time": 150.0,
        "r": 30,
        "centering": True,
        "datapath": data_dir,
        "time_integrator": RK4_solver,
        "fenicsx_interface": fenicsx_interface
    }

    

    opinf_ROM = OpInf_ROM(opinf_dict)

    linear_reg = {"min": -12,
                  "max": -5,
                  "num": 5}
    quad_reg = {"min": -2,
                  "max": 2,
                  "num": 4}
    # opinf_ROM.optimize_model(linear_reg, quad_reg)
    # beta1 = opinf_ROM.best_beta1
    # beta2 =  opinf_ROM.best_beta2
    opinf_ROM.update_model_regularization(1e-12, 100.0)

    filter_dict = {"filtering": True,
                   "filter_method": "Projection",
                   "r_restriction": 10,
                   "delta": 0.03,
                   "chi": 0.005,
                   "filter_time": 120.0}
    
    RK4_filter_solver = RK_solvers(filter_dict)

    opinf_model = opinf_ROM.opinf_model
    predict_times = opinf_ROM.predict_times
    train_times = opinf_ROM.train_times

    if train_times[-1] != predict_times[-1]:
        is_prediction = True

    u0 = opinf_ROM.u0

    Q_ROM = RK4_filter_solver.solve(opinf_model, u0, predict_times)
    recon_data = opinf_ROM.reconstruct_ROM_sol(Q_ROM)
    FOM_KE_times, FOM_KE = opinf_ROM.compute_FOM_KE()
    ROM_sol_list = opinf_ROM.convert_lumped_data_to_FE_function_list(recon_data)
    ROM_KE_times, ROM_KE = opinf_ROM.compute_ROM_KE(ROM_sol_list)

    result_folder = RESULTS_DIR
    if not result_folder.exists():
        result_folder.mkdir()

    standard_opinf_folder = result_folder / "Continuous_OpInf_Results"
    if not standard_opinf_folder.exists():
        standard_opinf_folder.mkdir()

    if filter_dict["filter_method"] == "Projection":
        filter_folder = standard_opinf_folder / "Projection_Filtering"
        identification = f"Projection_r_restriction_{filter_dict["r_restriction"]}"
        if not filter_folder.exists():
            filter_folder.mkdir()
    
    elif filter_dict["filter_method"] == "Differential":
        filter_folder = standard_opinf_folder / "Differential_Filtering"
        if not filter_folder.exists():
            filter_folder.mkdir()

    filter_comparison_filename = "Filter_comparison_" + identification + ".pvd" 
    

    # opinf_ROM.compare_filter(RK4_filter_solver, 150.0, filter_folder, filter_comparison_filename, filter_type = "Projection")

    fig, ax = plt.subplots()
    ax.plot(FOM_KE_times, FOM_KE, label='FOM KE', color='black')
    ax.plot(ROM_KE_times, ROM_KE, label='OpInf KE', color='red')
    if is_prediction:
        last_train_time = train_times[-1]
        ax.axvline(x=last_train_time, color='blue', linestyle='--', label='End of training data')
    ax.set_xlabel('Time')
    ax.set_ylabel('Kinetic Energy')
    ax.set_title('FOM vs OpInf Kinetic Energy')
    ax.legend()
    fig.savefig((standard_opinf_folder / "standard_opinf_KE.png"))


    




