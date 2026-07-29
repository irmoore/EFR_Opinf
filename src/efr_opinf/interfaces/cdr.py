import gmsh
import os
import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
from petsc4py import PETSc

from basix.ufl import element, mixed_element

from dolfinx.cpp.mesh import to_type, cell_entity_type
from dolfinx.fem import (Constant, Function, Expression, functionspace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological, set_bc)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, create_matrix, set_bc)
from dolfinx.graph import adjacencylist
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.io import (VTXWriter, distribute_entity_data, gmsh as gmshio)
from dolfinx.mesh import create_mesh, meshtags_from_entities
from ufl import (FacetNormal, Identity, Measure, TestFunction, TrialFunction, split, derivative,
                 as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym, system, inner)

import dolfinx

from pathlib import Path

from efr_opinf._paths import MESH_DIR

from tqdm import tqdm


class fenicsx_class():
    def __init__(self, mesh_file: Path, element_order: int = 2):
        self._load_mesh(mesh_file)
        self._setup_function_space(element_order)

    def solve_CDR(self, Peclet, dt, time_end) -> tuple[np.ndarray, np.ndarray]:
        mesh = self.mesh
        V = self.functionspace
        self._Peclet = Peclet
        self._time_end = time_end
        

        boundary_conditions = self.BCs

        num_steps = int(time_end / dt)
        timespace = np.linspace(0,time_end,num_steps+1)

        u = TrialFunction(V)
        v = TestFunction(V)

        f = Constant(mesh, PETSc.ScalarType(1))

        b = as_vector([1.,0.])
        nu = 1/ Peclet
        eps = Constant(mesh,PETSc.ScalarType(nu))
        sigma =Constant(mesh,PETSc.ScalarType(1))

        a = u * v *dx + eps* dt * dot(grad(u), grad(v)) * dx + dt * dot(b,grad(u))* v *dx + sigma * dt *u * v *dx

        u_n = Function(V)
        solution = Function(V)
        L = (u_n + dt * f) * v * dx

        bilinear_form = form(a)
        linear_form = form(L)

        A = assemble_matrix(bilinear_form, bcs=boundary_conditions)
        A.assemble()
        b1 = create_vector(dolfinx.fem.extract_function_spaces( linear_form))

        solver = PETSc.KSP().create(mesh.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)

        solution_arr = np.nan*np.ones((solution.x.array.shape[0], num_steps + 1))
        solution_arr[:,0] = solution.x.array[:]

        for i in range(num_steps):
            with b1.localForm() as loc_b:
                loc_b.set(0)
                assemble_vector(b1, linear_form)

            # Apply Dirichlet boundary condition to the vector
            apply_lifting(b1, [bilinear_form], [boundary_conditions])
            b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b1, boundary_conditions)

            # Solve linear problem
            solver.solve(b1, solution.x.petsc_vec)
            solution.x.scatter_forward()

            # Update solution at previous time step (u_n)
            u_n.x.array[:] = solution.x.array
            solution_arr[:,i+1] = solution.x.array[:]

        self._last_sol = solution

        
        return solution_arr, timespace

    def setup_DF_filter_LHS(self, delta: float):
        mesh = self.mesh
        V = self.functionspace

        ft= self.facet_tags
        fdim = ft.dim


        du = TrialFunction(V)
        v = TestFunction(V)

        bcs_0 = self._boundaries


        delc = dolfinx.fem.Constant(mesh, delta**2)

        a = inner(du,v) * dx + delc*inner(grad(du), grad(v)) *dx

        self.delc = delc

        Af = dolfinx.fem.form(a)

        LHS_MAT = dolfinx.fem.assemble_matrix(Af, bcs = bcs_0)
        rhs = dolfinx.fem.create_vector(V)
        rhs.array[:] = 0

        dolfinx.fem.apply_lifting(rhs.array, [Af],bcs = [bcs_0], alpha = -1.0)
        [bc.set(rhs.array, alpha = -1.0) for bc in bcs_0]

        AS = LHS_MAT.to_scipy()

        self.LHS_Filter_form = Af
        

        self.LHS_filter_MAT = AS


        return AS
    
    def assemble_filter_rhs(self,centered_u: dolfinx.fem.Function, mean_func:dolfinx.fem.Function, include_mean: bool):
        V = self.functionspace
        v = TestFunction(V)
        delc = self.delc
        bcu = self._boundaries
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

    def compute_solution_norm(self, list_1: list[dolfinx.fem.Function], times_1: np.ndarray) -> np.ndarray:
        time_shape = times_1.shape[0]
        assert len(list_1) == time_shape
        V = self.functionspace
        
        FOM_error_arr = np.zeros((time_shape,), dtype = np.float64)
        FOM_func = dolfinx.fem.Function(V)
        FOM_norm_form = dolfinx.fem.form(inner(FOM_func, FOM_func) * dx)

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


        error_form = dolfinx.fem.form( inner(diff, diff) * dx)

        for i in range(time_shape):
            diff.x.array[:] = list_1[i].x.array - list_2[i].x.array
            diff.x.scatter_forward()
            L2_error_list[i] = dolfinx.fem.assemble_scalar(error_form)
        return L2_error_list, times_1
    

    def plot_function(self, u: Function):
        V = self.functionspace
        
        from dolfinx.plot import vtk_mesh
        import pyvista

        # Extract topology from mesh and create pyvista mesh
        topology, cell_types, x = vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(topology, cell_types, x)

        # Set deflection values and add it to plotter
        grid.point_data["u"] = u.x.array
        warped = grid.warp_by_scalar("u", factor=1)

        plotter = pyvista.Plotter(off_screen = False)
        plotter.add_mesh(warped, show_edges=False, show_scalar_bar=True, scalars="u")
        plotter.show()

    def lumped_data_to_FE_function_list(self, data: np.ndarray) -> list[Function]:
        """convert array of pre-organized data to FE function space"""
        func_list = []
        # if tqdm is not None:
        #     iterator = tqdm(range(data.shape[1]), desc="Converting lumped data to FE functions")
        # else:
        iterator = range(data.shape[1]) # This happens too quickly for tqdm to be useful

        for i in iterator:
            func = Function(self.functionspace)  
            func.x.array[:] = data[:,i]
            func_list.append(func)
        return func_list
    
    def FE_function_list_to_arr(self, FE_list: list[Function]) -> np.ndarray:
        first_func = FE_list[0]
        dim = first_func.x.array.shape[0]
        length = len(FE_list)

        np_arr = np.zeros((dim,length), dtype = np.float64, order = 'F')
        for i in range(length):
            np_arr[:,i] = FE_list[i].x.array[:]
        return np_arr
    
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

    def _setup_function_space(self, element_order: int):
        mesh = self.mesh
        if element_order <= 0:
            raise ValueError("Element order must be positive int")
        V = functionspace(mesh, ("Lagrange", element_order)) # eg element_order = 2 implies quadratic
        self._element_order = element_order
        self._V = V
        coords = V.tabulate_dof_coordinates()
        self._coords = coords

        ft = self.facet_tags
        wall_marker = 1
        fdim = mesh.topology.dim - 1

        u_0 = Function(V)
        bcu_walls = dirichletbc(u_0, locate_dofs_topological(V, fdim, ft.find(wall_marker)))
        boundary_conditions = [bcu_walls]

        self._boundaries = boundary_conditions

    def _load_mesh(self, meshfile) -> None:
        gdim = 2
        mesh_comm = MPI.COMM_WORLD
        mesh_data = gmshio.read_from_msh(meshfile, mesh_comm, gdim = 2)
        mesh = mesh_data.mesh
        assert mesh_data.facet_tags is not None
        ft = mesh_data.facet_tags
        ct = mesh_data.cell_tags
        ct.name = "Cell Markers"
        ft.name = "Facet markers"
        self._ft = ft

        wall_marker = 1
        overlap_marker = 3

        self.wall_marker = wall_marker
        self._mesh = mesh

    @property
    def BCs(self):
        return self._boundaries

    @property
    def facet_tags(self):
        return self._ft
    
    @property
    def coords(self):
        return self._coords

    @property
    def element_order(self):
        return self._element_order

    @property
    def mesh(self):
        return self._mesh

    @property
    def functionspace(self):
        return self._V

    

if __name__ == "__main__":   
    meshpath = MESH_DIR / "CDR_mesh.msh"
    dt = 0.01
    Peclet = 10000
    time_end = 5.0
    element_order = 2

    CDR_FOM = fenicsx_class( meshpath, dt, Peclet, time_end, element_order)

    sol = CDR_FOM._last_sol
    CDR_FOM.plot_function(sol)

    solution_arr = CDR_FOM.FOM_sol

    FOM_function_list = CDR_FOM.lumped_data_to_FE_function_list(solution_arr)

    assert np.allclose(sol.x.array, FOM_function_list[-1].x.array)
    CDR_FOM.plot_function(FOM_function_list[-1])

    print("hold up")


