import dolfinx.fem
import dolfinx.fem.petsc
import numpy as np
import ufl
import dolfinx.io
import mpi4py.MPI 
import dolfinx.geometry
from pathlib import Path

from efr_opinf._paths import DATA_DIR

import petsc4py.PETSc
import scipy

from ufl import(
    TrialFunction,
    TestFunction,
    inner,
    dx,
    grad
)

from dolfinx.io import VTXWriter
from mpi4py import MPI

from basix.ufl import element

import scipy.interpolate
from scipy.spatial import Delaunay

import adios4dolfinx

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

class fenicsx_class():
    def __init__(self, data_file: Path, time_start: float | None = None):
        self.data_file = data_file 
        assert data_file.suffix == ".bp", "Data needs to be in .bp format"
        self._load_mesh(data_file)
        self._create_function_space()
        self._load_functions(data_file, time_start)

    def setup_pressure(self):
        datafile = self.data_file
        self._setup_pressure_space()
        self._load_pressure(datafile)

    def pressure_poisson(self, velocity_funcs: list[dolfinx.fem.Function]) -> list[dolfinx.fem.Function]:
        inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2,3,4,5
        Q = self._Q
        V = self._V
        mesh = self.mesh

        q = ufl.TestFunction(Q)
        dp = ufl.TrialFunction(Q)
        u_prev = dolfinx.fem.Function(V)
        u = dolfinx.fem.Function(V)
        p_prev = dolfinx.fem.Function(Q)


        dx = ufl.dx
        inner = ufl.inner
        grad = ufl.grad
        dot = ufl.dot
        nabla_grad = ufl.nabla_grad

        mu = dolfinx.fem.Constant(mesh, petsc4py.PETSc.ScalarType(0.001)) 

        n = ufl.FacetNormal(mesh)

        A_ufl = inner(grad(dp), grad(q))*dx

        b_ufl = inner(dot(u, nabla_grad(u)), grad(q))*dx -inner(dot(u_prev, nabla_grad(u_prev)), grad(q))*dx - inner(grad(p_prev), grad(q))*dx

        ft= self.boundaries
        fdim = ft.dim

        bcp_outlet = dolfinx.fem.dirichletbc(petsc4py.PETSc.ScalarType(0), dolfinx.fem.locate_dofs_topological(Q, fdim, ft.find(outlet_marker)), Q)
        
        bcp = [bcp_outlet]

        A_form = dolfinx.fem.form(A_ufl)
        b_form = dolfinx.fem.form(b_ufl)

        Amat = dolfinx.fem.petsc.assemble_matrix(A_form, bcs = bcp)
        Amat.assemble()
        bvec = dolfinx.fem.petsc.create_vector(dolfinx.fem.extract_function_spaces(b_form))

        

        solver = petsc4py.PETSc.KSP().create(mesh.comm)
        solver.setOperators(Amat)
        solver.setType(petsc4py.PETSc.KSP.Type.CG)
        pc = solver.getPC()
        pc.setType(petsc4py.PETSc.PC.Type.SOR)

        p = dolfinx.fem.Function(Q)

        func_list = []
        func_list.append(self._loaded_pressure[0])

        for i in range(len(velocity_funcs)-1):
            p_app = dolfinx.fem.Function(Q)
            p_prev.x.array[:] = func_list[i]
            p_prev.x.scatter_forward()
            u_prev.x.array[:] = velocity_funcs[i].x.array[:]
            u_prev.x.scatter_forward()
            u.x.array[:] = velocity_funcs[i+1].x.array[:]
            u.x.scatter_forward()
            with bvec.localForm() as loc:
                loc.set(0)
                dolfinx.fem.petsc.assemble_vector(bvec, b_form)
                dolfinx.fem.petsc.apply_lifting(bvec, [A_form], [bcp])
                bvec.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD_VALUES, mode=petsc4py.PETSc.ScatterMode.REVERSE)
                dolfinx.fem.petsc.set_bc(bvec, bcp)
                solver.solve(bvec, p.x.petsc_vec)
                p.x.scatter_forward()
            p_app.x.array[:] = p.x.array[:]
            func_list.append(p_app)
        return func_list

    def pressure_poisson_old(self, velocity_funcs: list[dolfinx.fem.Function]) -> list[dolfinx.fem.Function]:
        inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2,3,4,5
        Q = self._Q
        V = self._V
        mesh = self.mesh

        q = ufl.TestFunction(Q)
        dp = ufl.TrialFunction(Q)
        u = dolfinx.fem.Function(V)

        dx = ufl.dx
        inner = ufl.inner
        grad = ufl.grad
        dot = ufl.dot
        nabla_grad = ufl.nabla_grad

        mu = dolfinx.fem.Constant(mesh, petsc4py.PETSc.ScalarType(0.001)) 

        n = ufl.FacetNormal(mesh)

        A_ufl = inner(grad(dp), grad(q))*dx

        b_ufl = -inner(dot(u, nabla_grad(u)), grad(q))*dx

        ft= self.boundaries
        fdim = ft.dim

        bcp_outlet = dolfinx.fem.dirichletbc(petsc4py.PETSc.ScalarType(0), dolfinx.fem.locate_dofs_topological(Q, fdim, ft.find(outlet_marker)), Q)
        ds = ufl.Measure("ds", domain = mesh, subdomain_data=ft)
        walls_bc = dot(dot(u, nabla_grad(u)), n)

        b_ufl += walls_bc*q*ds(inlet_marker)
        b_ufl += walls_bc*q*ds(wall_marker)
        b_ufl += -walls_bc*q*ds(obstacle_marker)

        dudt = dolfinx.fem.Function(V)

        p_bc = dot(n, mu*ufl.div(grad(u)) - dot(u, nabla_grad(u)) - dudt)

        b_ufl += -p_bc*q*ds(inlet_marker)
        b_ufl += -p_bc*q*ds(wall_marker)
        b_ufl += p_bc*q*ds(obstacle_marker)

        
        bcp = [bcp_outlet]

        A_form = dolfinx.fem.form(A_ufl)
        b_form = dolfinx.fem.form(b_ufl)

        Amat = dolfinx.fem.petsc.assemble_matrix(A_form, bcs = bcp)
        Amat.assemble()
        bvec = dolfinx.fem.petsc.create_vector(dolfinx.fem.extract_function_spaces(b_form))

        

        solver = petsc4py.PETSc.KSP().create(mesh.comm)
        solver.setOperators(Amat)
        solver.setType(petsc4py.PETSc.KSP.Type.CG)
        pc = solver.getPC()
        pc.setType(petsc4py.PETSc.PC.Type.SOR)

        p = dolfinx.fem.Function(Q)

        func_list = []

        for i in range(len(velocity_funcs)-1):
            p_app = dolfinx.fem.Function(Q)
            dudt.x.array[:] = (velocity_funcs[i+1].x.array[:] - velocity_funcs[i].x.array[:])/self._dt
            u.x.array[:] = velocity_funcs[i].x.array[:]
            u.x.scatter_forward()
            with bvec.localForm() as loc:
                loc.set(0)
                dolfinx.fem.petsc.assemble_vector(bvec, b_form)
                dolfinx.fem.petsc.apply_lifting(bvec, [A_form], [bcp])
                bvec.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD_VALUES, mode=petsc4py.PETSc.ScatterMode.REVERSE)
                dolfinx.fem.petsc.set_bc(bvec, bcp)
                solver.solve(bvec, p.x.petsc_vec)
                p.x.scatter_forward()
            p_app.x.array[:] = p.x.array[:]
            func_list.append(p_app)
        return func_list



    def _load_pressure(self, data_file):
        mesh = self._mesh
        time_start = self._time_start
        sample_times = adios4dolfinx.read_timestamps(data_file, mesh.comm, "p")
        dt = sample_times[0]
        timecheck = sample_times[1::] - sample_times[:-1:]
        assert np.allclose(timecheck, dt), "Sample time in loaded data are not even"

        if time_start is not None:
            sample_idx = np.argmin(np.abs(sample_times - time_start))
            sample_times = sample_times[sample_idx:]

        self._dt = dt
        Q = self._Q

        p = dolfinx.fem.Function(Q)
        function_list = []

        iterator = tqdm(range(len(sample_times)), desc = "Loading in FE Pressure functions")

        for i in iterator:
            app_p = dolfinx.fem.Function(Q)
            timestep = sample_times[i]
            adios4dolfinx.read_function(data_file, p,name= "p", time = timestep)
            app_p.x.array[:] = p.x.array
            function_list.append(app_p)
        
        self._loaded_pressure = function_list


    def _setup_pressure_space(self):
        mesh = self._mesh
        s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
        Q = dolfinx.fem.functionspace(mesh, s_cg1)
        self._Q = Q

    def compute_liftdrag(self, velocity_list: list[dolfinx.fem.Function], velocity_times: np.ndarray):
        inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2,3,4,5
        Q = self._Q
        V = self._V
        mesh = self.mesh
        ft = self.boundaries

        assert len(velocity_times) == len(velocity_list)

        mu = dolfinx.fem.Constant(mesh, petsc4py.PETSc.ScalarType(0.001))  # Dynamic viscosity
        rho = dolfinx.fem.Constant(mesh, petsc4py.PETSc.ScalarType(1)) 

        u_ = dolfinx.fem.Function(V)
        p_ = dolfinx.fem.Function(Q)

        n = -ufl.FacetNormal(mesh)  # Normal pointing out of obstacle
        dObs = ufl.Measure("ds", domain=mesh, subdomain_data=ft, subdomain_id=obstacle_marker)
        u_t = inner(ufl.as_vector((n[1], -n[0])), u_)
        drag = dolfinx.fem.form(2 / 0.1 * (mu / rho * inner(grad(u_t), n) * n[1] - p_ * n[0]) * dObs)
        lift = dolfinx.fem.form(-2 / 0.1 * (mu / rho * inner(grad(u_t), n) * n[0] + p_ * n[1]) * dObs)

        pressure_list = self._loaded_pressure
        assert np.allclose(self._data_times, velocity_times)
        assert len(pressure_list) == len(velocity_list)

        drag_arr = np.zeros_like(velocity_times, dtype = np.float64)
        lift_arr = np.zeros_like(velocity_times, dtype = np.float64)


        for i in range(len(velocity_times)):
            u_.x.array[:] = velocity_list[i].x.array
            u_.x.scatter_forward()
            p_.x.array[:] = pressure_list[i].x.array
            p_.x.scatter_forward()

            drag_arr[i] = dolfinx.fem.assemble_scalar(drag)
            lift_arr[i] = dolfinx.fem.assemble_scalar(lift)
        return drag_arr, lift_arr, velocity_times



    

    def _load_mesh(self, mesh_file) -> None:
        """Load a mesh from a given file."""
        mesh = adios4dolfinx.read_mesh(mesh_file, MPI.COMM_WORLD)
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        facet_tags = adios4dolfinx.read_meshtags(mesh_file, mesh, "facet_tags")

        self._mesh = mesh
        self._facet_tags = facet_tags

    def _create_function_space(self)-> None:
        """Create a function space on the loaded mesh."""
        mesh = self._mesh
        v_cg2 = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
        V = dolfinx.fem.functionspace(mesh, v_cg2)
        self._V = V

    def _load_extra_functions(self, data_file, time_startstop: tuple[float, float] | None = None):
        mesh = self._mesh
        sample_times = adios4dolfinx.read_timestamps(data_file, mesh.comm, "u")
        dt = sample_times[0]
        timecheck = sample_times[1::] - sample_times[:-1:]
        assert np.allclose(timecheck, dt), "Sample time in loaded data are not even"

        if time_startstop is not None:
            time_start = time_startstop[0]
            time_end = time_startstop[1]

            start_idx = np.argmin(np.abs(sample_times - time_start))
            end_idx = np.argmin(np.abs(sample_times - time_end))
            sample_times = sample_times[start_idx:end_idx + 1]

        self._dt = dt
        V = self._V

        u = dolfinx.fem.Function(V)
        function_list = []

        iterator = tqdm(range(len(sample_times)), desc = "Loading in FE functions")

        for i in iterator:
            app_u = dolfinx.fem.Function(V)
            timestep = sample_times[i]
            adios4dolfinx.read_function(data_file, u,name= "u", time = timestep)
            app_u.x.array[:] = u.x.array
            function_list.append(app_u)

        self.loaded_extended_data = function_list
        self.loaded_extended_times = sample_times

    def _load_functions(self, data_file, time_start: float | None = None):
        mesh = self._mesh
        sample_times = adios4dolfinx.read_timestamps(data_file, mesh.comm, "u")
        dt = sample_times[0]
        timecheck = sample_times[1::] - sample_times[:-1:]
        assert np.allclose(timecheck, dt), "Sample time in loaded data are not even"

        if time_start is not None:
            sample_idx = np.argmin(np.abs(sample_times - time_start))
            sample_times = sample_times[sample_idx:]

        self._dt = dt
        V = self._V

        u = dolfinx.fem.Function(V)
        function_list = []

        iterator = tqdm(range(len(sample_times)), desc = "Loading in FE functions")

        for i in iterator:
            app_u = dolfinx.fem.Function(V)
            timestep = sample_times[i]
            adios4dolfinx.read_function(data_file, u,name= "u", time = timestep)
            app_u.x.array[:] = u.x.array
            function_list.append(app_u)

        self._time_start = time_start

        self._loaded_data = function_list
        self._data_times = sample_times

    def filter_basis_functions(self, basis: list[dolfinx.fem.Function], delta: float):
        mesh = self.mesh
        V = self.functionspace

        ft= self.boundaries
        fdim = ft.dim
        u_nonslip = np.array((0,) * mesh.geometry.dim, dtype=petsc4py.PETSc.ScalarType)

        inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2,3,4,5

        du = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        bcu_inflow = dolfinx.fem.dirichletbc(u_nonslip, dolfinx.fem.locate_dofs_topological(V, fdim, ft.find(inlet_marker)), V)
        # Walls
        bcu_walls = dolfinx.fem.dirichletbc(u_nonslip, dolfinx.fem.locate_dofs_topological(V, fdim, ft.find(wall_marker)), V)
        # Obstacle
        bcu_obstacle = dolfinx.fem.dirichletbc(u_nonslip, dolfinx.fem.locate_dofs_topological(V, fdim, ft.find(obstacle_marker)), V)
        bcu = [bcu_inflow, bcu_obstacle, bcu_walls]

        v = ufl.TestFunction(V)
        du = ufl.TrialFunction(V)
        dx = ufl.dx
        inner = ufl.inner
        grad = ufl.grad

        delc = dolfinx.fem.Constant(mesh, delta**2)

        a = ufl.inner(du,v) * dx + delc*inner(grad(du), grad(v)) *dx

        centered_u = dolfinx.fem.Function(V)

        L = inner(centered_u,v) * dx 

        num_funcs = len(basis)
        function_dim = basis[0].x.array.shape[0]

        fil_basis_arr = np.zeros((function_dim, num_funcs))

        fil_func_list = []

        if num_funcs > 50: 
            print(f'Warning! Basis filtering is not optimized for large numbers of functions - you supplied {num_funcs}!')

        for i in tqdm(range(num_funcs)):
            centered_u.x.array[:] = basis[i].x.array
            centered_u.x.scatter_forward()
            problem = dolfinx.fem.petsc.LinearProblem(
                a,
                L,
                bcs=bcu,
                petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                petsc_options_prefix="filter_problem",
            )
            uh = problem.solve()
            fil_func_list.append(uh)
            fil_basis_arr[:,i] = uh.x.array

        return fil_func_list, fil_basis_arr


    def setup_DF_filter_LHS(self, delta: float):
        mesh = self.mesh
        V = self.functionspace

        ft= self.boundaries
        fdim = ft.dim
        u_nonslip = np.array((0,) * mesh.geometry.dim, dtype=petsc4py.PETSc.ScalarType)

        inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2,3,4,5

        du = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        bcu_inflow = dolfinx.fem.dirichletbc(u_nonslip, dolfinx.fem.locate_dofs_topological(V, fdim, ft.find(inlet_marker)), V)
        # Walls
        bcu_walls = dolfinx.fem.dirichletbc(u_nonslip, dolfinx.fem.locate_dofs_topological(V, fdim, ft.find(wall_marker)), V)
        # Obstacle
        bcu_obstacle = dolfinx.fem.dirichletbc(u_nonslip, dolfinx.fem.locate_dofs_topological(V, fdim, ft.find(obstacle_marker)), V)
        bcu = [bcu_inflow, bcu_obstacle, bcu_walls]

        v = ufl.TestFunction(V)
        du = ufl.TrialFunction(V)
        dx = ufl.dx
        inner = ufl.inner
        grad = ufl.grad

        delc = dolfinx.fem.Constant(mesh, delta**2)

        a = ufl.inner(du,v) * dx + delc*inner(grad(du), grad(v)) *dx

        self.delc = delc

        Af = dolfinx.fem.form(a)

        LHS_MAT = dolfinx.fem.assemble_matrix(Af, bcs = bcu)
        rhs = dolfinx.fem.create_vector(V)
        rhs.array[:] = 0

        dolfinx.fem.apply_lifting(rhs.array, [Af],bcs = [bcu], alpha = -1.0)
        [bc.set(rhs.array, alpha = -1.0) for bc in bcu]

        AS = LHS_MAT.to_scipy()

        self.LHS_Filter_form = Af
        
        self.zero_BCs = bcu
        self.LHS_filter_MAT = AS


        return AS
    
    def assemble_mass_0_BCs(self):
        mesh = self.mesh
        V = self.functionspace

        ft= self.boundaries
        fdim = ft.dim
        u_nonslip = np.array((0,) * mesh.geometry.dim, dtype=petsc4py.PETSc.ScalarType)

        inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2,3,4,5

        du = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        bcu_inflow = dolfinx.fem.dirichletbc(u_nonslip, dolfinx.fem.locate_dofs_topological(V, fdim, ft.find(inlet_marker)), V)
        # Walls
        bcu_walls = dolfinx.fem.dirichletbc(u_nonslip, dolfinx.fem.locate_dofs_topological(V, fdim, ft.find(wall_marker)), V)
        # Obstacle
        bcu_obstacle = dolfinx.fem.dirichletbc(u_nonslip, dolfinx.fem.locate_dofs_topological(V, fdim, ft.find(obstacle_marker)), V)
        bcu = [bcu_inflow, bcu_obstacle, bcu_walls]

        v = ufl.TestFunction(V)
        du = ufl.TrialFunction(V)
        dx = ufl.dx
        inner = ufl.inner
        grad = ufl.grad

        a = ufl.inner(du,v) * dx 

        Af = dolfinx.fem.form(a)

        LHS_MAT = dolfinx.fem.assemble_matrix(Af, bcs = bcu)

        AS = LHS_MAT.to_scipy()
        return AS


    def assemble_filter_rhs(self,centered_u: dolfinx.fem.Function, mean_func:dolfinx.fem.Function, include_mean: bool):
        V = self.functionspace
        v = ufl.TestFunction(V)
        delc = self.delc
        bcu = self.zero_BCs
        Af = self.LHS_Filter_form

        if include_mean:
            l = inner(centered_u,v) * dx - delc * inner(grad(v), grad(mean_func)) * dx
        else: l = inner(centered_u,v) * dx

        Lf = dolfinx.fem.form(l)

        rhs = dolfinx.fem.create_vector(V)
        rhs.array[:] = 0
        dolfinx.fem.assemble_vector(rhs.array,Lf)

        dolfinx.fem.apply_lifting(rhs.array, [Af],bcs = [bcu], alpha = -1.0)
        [bc.set(rhs.array, alpha = -1.0) for bc in bcu]

        return rhs.array
    
    def vector_to_func(self, data: np.ndarray) -> dolfinx.fem.Function:
        d = data.squeeze()
        assert d.ndim == 1

        V = self.functionspace
        u = dolfinx.fem.Function(V)

        u.x.array[:] = d
        return u




    def obtain_inlet_dofs(self):
        V = self._V
        dofs = V.tabulate_dof_coordinates()

        left_idc, = np.where(np.isclose(dofs[:,0],0))

        left = dofs[left_idc,:]

        return left 
    
    def save_functions_at_timestep(self, timestep, func_list: list[dolfinx.fem.Function],directory: Path, filename: str):

        assert directory.is_dir()
        mesh = self._mesh 
        savefile = directory / filename

        vtx_u = VTXWriter(mesh.comm, savefile, func_list, engine="BP4")
        vtx_u.write(timestep)

    def save_function_list_VTK(self, func_list: list[dolfinx.fem.Function], directory: Path, filename: str, time_arr: np.array) -> None:
        assert directory.is_dir()
        mesh = self._mesh 
        savefile = directory / filename

        u_ = dolfinx.fem.Function(self._V)
        vtx_u = VTXWriter(mesh.comm, savefile, [u_], engine="BP4")

        iterator = tqdm(range(len(func_list)), desc = "Saving functions for visualization")

        for i in iterator:
            t = time_arr[i]
            u_.x.array[:] = func_list[i].x.array
            vtx_u.write(t)

    def lumped_data_to_FE_function_list(self, data: np.ndarray) -> list[dolfinx.fem.Function]:
        """convert array of pre-organized data to FE function space"""
        func_list = []
        # if tqdm is not None:
        #     iterator = tqdm(range(data.shape[1]), desc="Converting lumped data to FE functions")
        # else:
        iterator = range(data.shape[1]) # This happens too quickly for tqdm to be useful

        for i in iterator:
            func = dolfinx.fem.Function(self.functionspace)  
            func.x.array[:] = data[:,i]
            func_list.append(func)
        return func_list
    
    def FE_function_list_to_arr(self, FE_list: list[dolfinx.fem.Function]) -> np.ndarray:
        first_func = FE_list[0]
        dim = first_func.x.array.shape[0]
        length = len(FE_list)

        np_arr = np.zeros((dim,length), dtype = np.float64, order = 'F')
        for i in range(length):
            np_arr[:,i] = FE_list[i].x.array[:]
        return np_arr
    
    def eval_func(self, f_h: dolfinx.fem.Function, x: np.ndarray) -> np.ndarray:
        '''Evaluate a dolfinx function on a set of 2D points.'''
        x = np.concatenate([x, np.zeros((len(x), 1))], axis=1)
        mesh = self.mesh
        bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim, padding=1e-10)
        potential_colliding_cells = dolfinx.geometry.compute_collisions_points(bb_tree, x)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, potential_colliding_cells, x)
        points_on_proc = []
        cells = []
        for i, point in enumerate(x):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])

        points_on_proc = np.array(points_on_proc, dtype=np.float64).reshape(-1, 3)
        cells = np.array(cells, dtype=np.int32)

        return f_h.eval(points_on_proc, cells)
    
    def compute_solution_norm(self, list_1: list[dolfinx.fem.Function], times_1: np.ndarray) -> np.ndarray:
        time_shape = times_1.shape[0]
        assert len(list_1) == time_shape
        V = self.functionspace
        
        FOM_error_arr = np.zeros((time_shape,), dtype = np.float64)
        FOM_func = dolfinx.fem.Function(V)
        FOM_norm_form = dolfinx.fem.form(ufl.inner(FOM_func, FOM_func) * ufl.dx)

        for i in range(time_shape):
            FOM_func.x.array[:] = list_1[i].x.array
            FOM_func.x.scatter_forward()
            FOM_error_arr[i] = dolfinx.fem.assemble_scalar(FOM_norm_form)
        FOM_error_arr = np.sqrt(FOM_error_arr)
        return FOM_error_arr
    
    def compute_pointwise_errors(self, list_1: list[dolfinx.fem.Function], times_1: np.ndarray, list_2: list[dolfinx.fem.Function], times_2: np.ndarray) -> np.ndarray:
        time_shape = times_1.shape[0]
        assert time_shape == times_2.shape[0], "Times do not match vetween provided function lists"
        assert len(list_1) == time_shape
        assert len(list_2) == time_shape
        assert np.allclose(times_1, times_2)

        L2_error_list = np.zeros((time_shape,))

        V = self.functionspace

        diff = dolfinx.fem.Function(V)


        error_form = dolfinx.fem.form( ufl.inner(diff, diff) * ufl.dx)

        for i in range(time_shape):
            diff.x.array[:] = list_1[i].x.array - list_2[i].x.array
            diff.x.scatter_forward()
            L2_error_list[i] = dolfinx.fem.assemble_scalar(error_form)
        return L2_error_list, times_1
    
    
    def compute_KE_arr(self, sol_list: list[dolfinx.fem.Function], verbose = False) -> np.ndarray:
        KE = np.zeros(len(sol_list))
        if verbose:
            iterator = tqdm(range(len(sol_list)), desc="Computing Kinetic Energy")
        else:
            iterator = range(len(sol_list))     
        for i in iterator:
            KE[i] = self.compute_KE_single(sol_list[i])
        return KE
    
    def compute_KE_single(self, current_sol: dolfinx.fem.Function) -> float:
        inner = ufl.inner(current_sol, current_sol)*ufl.dx
        inner_cpp = dolfinx.fem.form(inner)
        result = dolfinx.fem.assemble_scalar(inner_cpp)
        KE = 0.5*result
        return KE
    
    def plot_KE(self, times: np.ndarray, KE_array: np.ndarray, savefile=None):
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(times, KE_array, label='Kinetic Energy', color='black')
        plt.xlabel('Time')
        plt.ylabel('Kinetic Energy')
        plt.title('Kinetic Energy over Time')
        plt.legend()
        plt.show()
        if savefile is not None:
            plt.savefig(savefile)
            plt.close()
    
    def plot_function(self, func: dolfinx.fem.Function):
        try :
            import pyvista
        except ImportError:
            raise ImportError("pyvista is required for plotting functions. Please install pyvista to use this feature.")

        from dolfinx.plot import vtk_mesh

        mesh = self.mesh

        topology, cell_types, geometry = vtk_mesh(self.functionspace)
        values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
        values[:, : len(func)] = func.x.array.real.reshape((geometry.shape[0], len(func)))

        # Create a point cloud of glyphs
        function_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        function_grid["u"] = values
        glyphs = function_grid.glyph(orient="u", factor=0.1)

        # Create a pyvista-grid for the mesh
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim, tdim)
        grid = pyvista.UnstructuredGrid(*vtk_mesh(mesh, tdim))

        # Create plotter
        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, style="wireframe", color="k")
        plotter.add_mesh(glyphs)
        plotter.view_xy()

        plotter.show()

    def plot_pressure(self, func: dolfinx.fem.Function):
        """
        Plot a scalar pressure function. Assumes the correct scalar function space is self._Q.
        """
        try:
            import pyvista
        except ImportError:
            raise ImportError("pyvista is required for plotting functions. Please install pyvista to use this feature.")
        import numpy as np
        from dolfinx.plot import vtk_mesh

        # use the pressure function space provided by the class
        Q = self._Q

        # VTK topology/geometry for the pressure function space
        topology, cell_types, geometry = vtk_mesh(Q)
        npoints = geometry.shape[0]

        # Extract DOF data and ensure it matches the geometry
        data = np.asarray(func.x.array.real).ravel()
        if data.size % npoints != 0:
            raise ValueError(
                f"Cannot reshape DOF array of length {data.size} into ({npoints}, 1). "
                "Check that `func` lives in `self._Q` and that `self._Q` matches the vtk geometry."
            )
        ncomp = data.size // npoints
        if ncomp != 1:
            raise ValueError(f"Expected scalar-valued function (1 component per point), found {ncomp} components.")

        scalars = data.reshape((npoints,))

        # Create an UnstructuredGrid for the pressure data and attach scalars
        function_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        function_grid["p"] = scalars

        # Create a pyvista grid for the mesh (wireframe context)
        tdim = self.mesh.topology.dim
        self.mesh.topology.create_connectivity(tdim, tdim)
        grid = pyvista.UnstructuredGrid(*vtk_mesh(self.mesh, tdim))

        # Plot
        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, style="wireframe", color="k")
        plotter.add_mesh(function_grid,
                        scalars="p",
                        cmap="viridis",
                        render_points_as_spheres=True,
                        point_size=8,
                        show_scalar_bar=True)
        plotter.view_xy()
        plotter.show()

    @property
    def functionspace(self):
        return self._V
    @property
    def mesh(self):
        return self._mesh   
    @property
    def boundaries(self):
        """returns meshtags locating array boundaries"""
        return self._facet_tags
    
if __name__ == "__main__":      
    

    data_fld = DATA_DIR
    data_fld.mkdir(exist_ok = True)

    datafile = data_fld / "Cylinder" / "NSE_save_data.bp"

    BFS_FE = fenicsx_class(datafile)

    loaded_data = BFS_FE._loaded_data
    time = BFS_FE._data_times

    func = loaded_data[-1]

    BFS_FE.plot_function(func)
    sol_arr = BFS_FE.FE_function_list_to_arr(loaded_data)
    reconfigured_sol = BFS_FE.lumped_data_to_FE_function_list(sol_arr)

    KE_from_data = BFS_FE.compute_KE_arr(loaded_data)
    KE_reconfigured = BFS_FE.compute_KE_arr(reconfigured_sol)

    assert np.allclose(KE_from_data, KE_reconfigured)








