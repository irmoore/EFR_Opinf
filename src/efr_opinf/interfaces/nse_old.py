import dolfinx.fem
import numpy as np
import ufl
import dolfinx.io
import mpi4py.MPI 
import dolfinx.geometry
from pathlib import Path

from efr_opinf._paths import MESH_DIR, DATA_DIR, RESULTS_DIR

import petsc4py.PETSc

from ufl import(
    TrialFunction,
    TestFunction,
    inner,
    dx,
    grad
)

from basix.ufl import element

import scipy.interpolate
from scipy.spatial import Delaunay

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

class fenicsx_class():
    def __init__(self):
        pass
    def load_mesh(self, mesh_file):
        """Load a mesh from a given file."""
        with dolfinx.io.XDMFFile(mpi4py.MPI.COMM_WORLD, mesh_file, "r") as file:
            mesh = file.read_mesh()
            mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
            boundaries = file.read_meshtags(mesh, 'Facet tags')
            subdomains = file.read_meshtags(mesh, 'Cell tags')
        self._mesh = mesh
        self._boundaries = boundaries
        self._subdomains = subdomains
        self._delta = None

    def create_function_space(self):
        """Create a function space on the loaded mesh."""
        mesh = self.mesh
        v_cg2 = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
        V = dolfinx.fem.functionspace(mesh, v_cg2)
        self._V = V

    def save_function_list_VTK(self, func_list: list[dolfinx.fem.Function], directory: Path, filename: str, time = 0.0) -> None:
        assert directory.is_dir()
        mesh = self._mesh 
        savefile = directory / filename

        with dolfinx.io.VTKFile(mesh.comm, savefile, 'w') as vtk_file:
            vtk_file.write_mesh(mesh)
            vtk_file.write_function(func_list, t = time)

    def assemble_filter_components(self, delta, mean_vec: np.ndarray):
        LHS_filter, FOM_mass = self._assemble_LHS_filter(delta)
        LHS_mat = LHS_filter.to_dense()
        Mass_mat = FOM_mass.to_dense()
        mean_term = self._assemble_filter_mean_term(delta, mean_vec)
        mean_vec = mean_term.array
        self._delta = delta
        return LHS_mat, Mass_mat, mean_vec


    def _assemble_filter_mean_term(self, delta, mean_vec: np.ndarray):
        V = self.functionspace
        mesh = self.mesh

        mean = dolfinx.fem.Function(V)
        mean.x.array[:] = mean_vec
        v = TestFunction(V)
        delc = dolfinx.fem.Constant(mesh, delta**2)

        g = -inner(mean, v) * dx - delc*inner(grad(mean), grad(v)) *dx
        gform = dolfinx.fem.form(g)

        gvec = dolfinx.fem.assemble_vector(gform)
        return gvec


    def _assemble_LHS_filter(self, delta):
        mesh = self.mesh
        V = self.functionspace
        boundaries = self.boundaries

        u = TrialFunction(V)
        v = TestFunction(V)
        delc = dolfinx.fem.Constant(mesh, delta**2)

        LHS = inner(u,v) * dx + delc*inner(grad(u), grad(v)) *dx

        dirichlet_idcs = np.concatenate((boundaries.find(1), boundaries.find(2)))

        
        bc_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim-1, dirichlet_idcs)

        u_noslip = np.array((0,) * mesh.geometry.dim, dtype=petsc4py.PETSc.ScalarType)

        bc_wall = dolfinx.fem.dirichletbc(u_noslip, bc_dofs, V)

        LHS_form = dolfinx.fem.form(LHS)
        LHS_filter = dolfinx.fem.assemble_matrix(LHS_form, bcs = [bc_wall])

        M = inner(u,v)*dx
        M_form = dolfinx.fem.form(M)
        Mass_assembled = dolfinx.fem.assemble_matrix(M_form, bcs = [bc_wall])
        return LHS_filter, Mass_assembled

    def lumped_data_to_FE_function_list(self, data) -> list[dolfinx.fem.Function]:
        """convert array of pre-organized data to FE function space"""
        func_list = []
        if tqdm is not None:
            iterator = tqdm(range(data.shape[1]), desc="Converting lumped data to FE functions")
        else:
            iterator = range(data.shape[1])

        for i in iterator:
            func = dolfinx.fem.Function(self.functionspace)  
            func.x.array[:] = data[:,i]
            func_list.append(func)
        return func_list


    def split_data_to_FE_function_list(self, points_arr, ux_data, uy_data) -> list[dolfinx.fem.Function]:
        '''convert array of dofs and corresponding split data to list of dolfinx vector functions'''
        func_list = []
        tesselation = Delaunay(points_arr[:,:-1])
        if tqdm is not None:
            iterator = tqdm(range(ux_data.shape[1]), desc="Converting split data to FE functions")
        else:
            iterator = range(ux_data.shape[1])
        for i in iterator:
            func = self.FE_interpolate(tesselation, ux_data[:,i], uy_data[:,i])
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

    def FE_interpolate(self, tesselation: scipy.spatial.Delaunay, x, y) -> dolfinx.fem.Function:
        uh = dolfinx.fem.Function(self.functionspace)  
        interpolator = self._interpolate_2D_function_from_data(tesselation, x, y)
        uh.interpolate(lambda x: interpolator(x[:2].T).T)
        return uh
    
    def _interpolate_2D_function_from_data(self, tesselation, x, y):
        interp = scipy.interpolate.LinearNDInterpolator(tesselation, np.hstack((x[:,np.newaxis],y[:,np.newaxis])))
        return interp
    
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
    
    def compute_KE_arr(self, sol_list: list[dolfinx.fem.Function]) -> np.ndarray:
        KE = np.zeros(len(sol_list))
        if tqdm is not None:
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
    
    def plot_function(self, func):
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
        glyphs = function_grid.glyph(orient="u", factor=1)

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

    @property
    def delta_val(self):
        return self._delta

    @property
    def functionspace(self):
        return self._V
    @property
    def mesh(self):
        return self._mesh   
    @property
    def boundaries(self):
        """returns meshtags locating array boundaries"""
        return self._boundaries
    @property
    def subdomains(self):
        """returns meshtags locating array subdomains (these are unused, monolithic domain.)"""
        return self._subdomains
    
if __name__ == "__main__":      
    BFS_FE = fenicsx_class()

    Mesh_fld = MESH_DIR
    Mesh_fld.mkdir(exist_ok = True)

    data_fld = DATA_DIR
    data_fld.mkdir(exist_ok = True)

    # res_fld = RESULTS_DIR
    # res_fld.mkdir(exist_ok = True)

    mesh_file = str(Mesh_fld) + "/BFS_Mesh"

    BFS_FE.load_mesh(mesh_file)

    BFS_FE.create_function_space()

    V = BFS_FE.functionspace

    LHS, M = BFS_FE._assemble_LHS_filter(0.1)

    S = LHS.to_dense()
    mean = np.random.rand(S.shape[0])
    g = BFS_FE._assemble_filter_mean_term(0.1, mean)
    FOM_data_file = data_fld / "compressed_componentwise_data.npz"
    data = np.load(FOM_data_file)
    ux = data['ux']
    uy = data['uy']
    times = data['times']
    dofs = data['dofs']

    # func_list = BFS_FE.data_to_FE_function_list(dofs, ux, uy)
    # KE_array = BFS_FE.compute_KE_arr(func_list)

    # BFS_FE.plot_KE(times, KE_array)


    tesselation = Delaunay(dofs[:,:-1])

    func = BFS_FE.FE_interpolate(tesselation, ux[:,-1], uy[:,-1])

    dofs_len = dofs.shape[0]

    eval_points = dofs[:,:-1]
    compare_x = ux[:, -1]
    compare_y = uy[:, -1]
    eval_values = BFS_FE.eval_func(func, eval_points)

    BFS_FE.plot_function(func)

    eval_x = eval_values[:,0]
    eval_y = eval_values[:,1]

    # dolfinx function and supplied data are at least close at FOM DoFs if this passes
    assert np.allclose(compare_x, eval_x)
    assert np.allclose(compare_y, eval_y)





