import gmsh
import os
import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
import petsc4py.PETSc as PETSc

from basix.ufl import element

from dolfinx.cpp.mesh import to_type, cell_entity_type
from dolfinx.fem import (Constant, Function, Expression, functionspace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological, set_bc, extract_function_spaces)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, create_matrix, set_bc)
from dolfinx.graph import adjacencylist
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.io import (VTXWriter, distribute_entity_data, gmsh as gmshio)
from dolfinx.mesh import create_mesh, meshtags_from_entities
from ufl import (FacetNormal, Identity, Measure, TestFunction, TrialFunction,
                 as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym, system)

from efr_opinf._paths import RESULTS_DIR
from efr_opinf.fom.inlet import InletVelocity
import shutil

import adios4dolfinx
import tqdm

r = 0.05

compare = False
plot = True

save_monolithic = True
save_liftdrag = True

gmsh.initialize()

L = 2.2
H = 0.41
c_x = c_y = 0.2
r = 0.05
gdim = 2
mesh_comm = MPI.COMM_WORLD
model_rank = 0
if mesh_comm.rank == model_rank:
    rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
    obstacle = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)

if mesh_comm.rank == model_rank:
    fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle)])
    gmsh.model.occ.synchronize()

fluid_marker = 1
if mesh_comm.rank == model_rank:
    volumes = gmsh.model.getEntities(dim=gdim)
    assert len(volumes) == 1
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
    gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")

inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2,3,4,5
inflow, outflow, walls, obstacle = [], [], [], []
if mesh_comm.rank == model_rank:
    boundaries = gmsh.model.getBoundary(volumes, oriented=False)
    for boundary in boundaries:
        center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
        if np.allclose(center_of_mass, [0, H / 2, 0]):
            inflow.append(boundary[1])
        elif np.allclose(center_of_mass, [L, H / 2, 0]):
            outflow.append(boundary[1])
        elif np.allclose(center_of_mass, [L / 2, H, 0]) or np.allclose(
            center_of_mass, [L / 2, 0, 0]
        ):
            walls.append(boundary[1])
        else:
            obstacle.append(boundary[1])
    gmsh.model.addPhysicalGroup(1, walls, wall_marker)
    gmsh.model.setPhysicalName(1, wall_marker, "Walls")
    gmsh.model.addPhysicalGroup(1, inflow, inlet_marker)
    gmsh.model.setPhysicalName(1, inlet_marker, "Inlet")
    gmsh.model.addPhysicalGroup(1, outflow, outlet_marker)
    gmsh.model.setPhysicalName(1, outlet_marker, "Outlet")
    gmsh.model.addPhysicalGroup(1, obstacle, obstacle_marker)
    gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")

res_min = r / 3
if mesh_comm.rank == model_rank:
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", obstacle)
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.25 * H)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", r)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * H)
    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

if mesh_comm.rank == model_rank:
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.setOrder(2)
    gmsh.model.mesh.optimize("Netgen")

mesh_data = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
mesh = mesh_data.mesh
assert mesh_data.facet_tags is not None
ft = mesh_data.facet_tags
ct = mesh_data.cell_tags
ct.name = "Cell Markers"
ft.name = "Facet markers"

# mesh, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
# ft = facet_tags
# ft.name = "Facet markers"
# ct = cell_tags
# ct.name = "cell tags"

# gdim = 2
# mesh_comm = MPI.COMM_WORLD
# mesh, ct, ft = gmshio.read_from_msh(str(meshfile), mesh_comm, gdim = 2)
# ft.name = "Facet markers"

# inlet_marker = 1
# outlet_marker = 2
# obstacle_marker = 3
# wall_marker = 4
# fluid_marker = 5


t = 0
T = 10                # Final time
dt = 1 / 2000              # Time step size
num_steps = int(T / dt)
timespace = np.linspace(dt,T,num_steps)
sample_gap = 10
mono_save_file = f"NSE_snaps_gap_{sample_gap}.bp"
liftdrag_save_file = f"NSE_liftdrag_gap_{sample_gap}.npz"

total_times = int(num_steps / sample_gap) + 1

k = Constant(mesh, PETSc.ScalarType(dt))
mu = Constant(mesh, PETSc.ScalarType(0.001))  # Dynamic viscosity
rho = Constant(mesh, PETSc.ScalarType(1))     # Density

v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim, ))
s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
V = functionspace(mesh, v_cg2)
Q = functionspace(mesh, s_cg1)

fdim = mesh.topology.dim - 1

# Define boundary conditions

# Inlet
u_inlet = Function(V)
inlet_velocity = InletVelocity(t)
u_inlet.interpolate(inlet_velocity)
bcu_inflow = dirichletbc(u_inlet, locate_dofs_topological(V, fdim, ft.find(inlet_marker)))
# Walls
u_nonslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
bcu_walls = dirichletbc(u_nonslip, locate_dofs_topological(V, fdim, ft.find(wall_marker)), V)
# Obstacle
bcu_obstacle = dirichletbc(u_nonslip, locate_dofs_topological(V, fdim, ft.find(obstacle_marker)), V)
bcu = [bcu_inflow, bcu_obstacle, bcu_walls]
# Outlet
bcp_outlet = dirichletbc(PETSc.ScalarType(0), locate_dofs_topological(Q, fdim, ft.find(outlet_marker)), Q)
bcp = [bcp_outlet]

u = TrialFunction(V)
v = TestFunction(V)
u_ = Function(V)
u_.name = "u"
u_s = Function(V)
u_n = Function(V)
u_n1 = Function(V)
p = TrialFunction(Q)
q = TestFunction(Q)
p_ = Function(Q)
p_.name = "p"
phi = Function(Q)
u_f = Function(V)
w_ = Function(V)

f = Constant(mesh, PETSc.ScalarType((0, 0)))
F1 = rho / k * dot(u - u_n, v) * dx
F1 += inner(dot(1.5 * u_n - 0.5 * u_n1, 0.5 * nabla_grad(u + u_n)), v) * dx
F1 += 0.5 * mu * inner(grad(u + u_n), grad(v)) * dx - dot(p_, div(v)) * dx
F1 += dot(f, v) * dx
a1 = form(lhs(F1))
L1 = form(rhs(F1))
A1 = create_matrix(a1)
b1 = create_vector(extract_function_spaces(L1))

a2 = form(dot(grad(p), grad(q)) * dx)
L2 = form(-rho / k * dot(div(u_s), q) * dx)
A2 = assemble_matrix(a2, bcs=bcp)
A2.assemble()
b2 = create_vector(extract_function_spaces(L2))

a3 = form(rho * dot(u, v) * dx)
L3 = form(rho * dot(u_s, v) * dx - k * dot(nabla_grad(phi), v) * dx)
A3 = assemble_matrix(a3)
A3.assemble()
b3 = create_vector(extract_function_spaces(L3))

solver1 = PETSc.KSP().create(mesh.comm)
solver1.setOperators(A1)
solver1.setType(PETSc.KSP.Type.BCGS)
pc1 = solver1.getPC()
pc1.setType(PETSc.PC.Type.JACOBI)

# Solver for step 2
solver2 = PETSc.KSP().create(mesh.comm)
solver2.setOperators(A2)
solver2.setType(PETSc.KSP.Type.MINRES)
pc2 = solver2.getPC()
pc2.setType(PETSc.PC.Type.HYPRE)
pc2.setHYPREType("boomeramg")

# Solver for step 3
solver3 = PETSc.KSP().create(mesh.comm)
solver3.setOperators(A3)
solver3.setType(PETSc.KSP.Type.CG)
pc3 = solver3.getPC()
pc3.setType(PETSc.PC.Type.SOR)

##### Filter step
delta = res_min
chi = dt
delc = Constant(mesh, delta**2)
a4 = form( inner(u,v)*dx + delc* inner(grad(u), grad(v))*dx)
A4 = assemble_matrix(a4, bcs = bcu)
A4.assemble()
L4 = form( inner(w_, v) *dx)
b4 = create_vector(extract_function_spaces(L4))

solver4 = PETSc.KSP().create(mesh.comm)
solver4.setOperators(A4)
solver4.setType(PETSc.KSP.Type.CG)
pc4 = solver4.getPC()
pc4.setType(PETSc.PC.Type.SOR)

### Evalute dudt

#rhs_ufl = mu*div(grad(u_)) - dot(u_, nabla_grad(u_)) - grad(p_)


n = -FacetNormal(mesh)  # Normal pointing out of obstacle
dObs = Measure("ds", domain=mesh, subdomain_data=ft, subdomain_id=obstacle_marker)
u_t = inner(as_vector((n[1], -n[0])), u_)
drag = form(2 / 0.1 * (mu / rho * inner(grad(u_t), n) * n[1] - p_ * n[0]) * dObs)
lift = form(-2 / 0.1 * (mu / rho * inner(grad(u_t), n) * n[0] + p_ * n[1]) * dObs)
if mesh.comm.rank == 0:
    C_D = np.zeros(num_steps, dtype=PETSc.ScalarType)
    C_L = np.zeros(num_steps, dtype=PETSc.ScalarType)
    t_u = np.zeros(num_steps, dtype=np.float64)
    t_p = np.zeros(num_steps, dtype=np.float64)


if compare == True:
    tree = bb_tree(mesh, mesh.geometry.dim)
    points = np.array([[0.15, 0.2, 0], [0.25, 0.2, 0]])
    cell_candidates = compute_collisions_points(tree, points)
    colliding_cells = compute_colliding_cells(mesh, cell_candidates, points)
    front_cells = colliding_cells.links(0)
    back_cells = colliding_cells.links(1)
if mesh.comm.rank == 0:
    p_diff = np.zeros(num_steps, dtype=PETSc.ScalarType)

result_folder = RESULTS_DIR / "Test_Dir"
result_folder.mkdir(exist_ok=True, parents=True)



folder = result_folder / "Cylinder"
folder.mkdir(exist_ok=True, parents=True)

if plot == True:
    target_u = folder / ("r_" + str(r) + "_u.bp")
    target_p = folder / ("r_" + str(r) + "_p.bp")

    if target_u.is_dir():
        shutil.rmtree(target_u, ignore_errors= True)
    if target_p.is_dir():
        shutil.rmtree(target_p, ignore_errors=True)

    vtx_u = VTXWriter(mesh.comm, str(target_u), [u_], engine="BP4")
    vtx_p = VTXWriter(mesh.comm, str(target_p), [p_], engine="BP4")

    vtx_u.write(t)
    vtx_p.write(t)

dof_coordinates = V.tabulate_dof_coordinates()
progress = tqdm.tqdm(desc="Solving Monolothic Unstructured Mesh NSE", total=num_steps)
gathered_coords = mesh_comm.gather(dof_coordinates, root=0)
if save_monolithic == True:
    savefile = folder / mono_save_file
    adios4dolfinx.write_mesh(savefile, mesh, engine="BP4")
    
    adios4dolfinx.write_meshtags(savefile, mesh, ft, meshtag_name=f"facet_tags")
    adios4dolfinx.write_meshtags(savefile, mesh, ct, meshtag_name=f"cell_tags")
    
    # if mesh_comm.rank==0:
    #     stacked_coords = np.vstack(gathered_coords)
    #     _, indices = np.unique(stacked_coords[:,0:2].round(10), axis = 0, return_index = True)
    #     np_coords_unique = stacked_coords[indices,0:2]

    #     u_x_np = np.zeros((np_coords_unique.shape[0], total_times))
    #     u_y_np = np.zeros((np_coords_unique.shape[0], total_times))
    #     sample_times = np.zeros((1, total_times))
    #     j = 1

for i in range(num_steps):
    progress.update(1)
    # print(f'step {i}/{num_steps}')
    # Update current time step
    t = timespace[i]
    # Update inlet velocity
    inlet_velocity.t = t
    u_inlet.interpolate(inlet_velocity)

    # Step 1: Tentative velocity step
    A1.zeroEntries()
    assemble_matrix(A1, a1, bcs=bcu)
    A1.assemble()
    with b1.localForm() as loc:
        loc.set(0)
    assemble_vector(b1, L1)
    apply_lifting(b1, [a1], [bcu])
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b1, bcu)
    solver1.solve(b1, u_s.x.petsc_vec)
    u_s.x.scatter_forward()

    # Step 2: Pressure corrrection step
    with b2.localForm() as loc:
        loc.set(0)
    assemble_vector(b2, L2)
    apply_lifting(b2, [a2], [bcp])
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b2, bcp)
    solver2.solve(b2, phi.x.petsc_vec)
    phi.x.scatter_forward()

    p_.x.petsc_vec.axpy(1, phi.x.petsc_vec)
    p_.x.scatter_forward()

    # Step 3: Velocity correction step
    with b3.localForm() as loc:
        loc.set(0)
    assemble_vector(b3, L3)
    b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    solver3.solve(b3, w_.x.petsc_vec)
    w_.x.scatter_forward()

    #### Step 4: EFR

    if t >= 10.0:

        with b4.localForm() as loc:
            loc.set(0)
        assemble_vector(b4, L4)
        apply_lifting(b4, [a4], [bcu])
        b4.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b4, bcu)
        solver4.solve(b4, u_f.x.petsc_vec)
        u_f.x.scatter_forward()

        u_.x.array[:] = chi*(u_f.x.array) + (1 - chi)*w_.x.array[:]
        

    else:
        u_.x.array[:] = w_.x.array
    
    u_.x.scatter_forward()





    # Write solutions to file
    if plot == True: 
        vtx_u.write(t)
        vtx_p.write(t)

    #store snapshots into NP array
    if (i % sample_gap) == sample_gap - 1:
        if save_monolithic:

            adios4dolfinx.write_function(savefile, u_, time = t, name = u_.name)
            adios4dolfinx.write_function(savefile, p_, time = t, name = p_.name)




        
            # u_x = u_.sub(0).collapse().x.array
            # u_y = u_.sub(1).collapse().x.array

            
            # gathered_u_x = mesh_comm.gather(u_x, root=0)
            # gathered_u_y = mesh_comm.gather(u_y, root=0)


            # if mesh_comm.rank == 0:

            #     if save_monolithic == True:
                
            #         stacked_u_x =np.concatenate(gathered_u_x)
            #         stacked_u_y = np.concatenate(gathered_u_y)
            #         u_x_np[:,j] = stacked_u_x[indices]
            #         u_y_np[:,j] = stacked_u_y[indices]
            #         sample_times[0,j] = t

            #     j += 1
        if save_liftdrag:
            drag_coeff = mesh.comm.gather(assemble_scalar(drag), root=0)
            lift_coeff = mesh.comm.gather(assemble_scalar(lift), root=0)
            if mesh.comm.rank == 0:
                t_u[i] = t
                C_D[i] = sum(drag_coeff)
                C_L[i] = sum(lift_coeff)




    # Update variable with solution form this time step
    with u_.x.petsc_vec.localForm() as loc_, u_n.x.petsc_vec.localForm() as loc_n, u_n1.x.petsc_vec.localForm() as loc_n1:
        loc_n.copy(loc_n1)
        loc_.copy(loc_n)

    # Compute physical quantities
    # For this to work in paralell, we gather contributions from all processors
    # to processor zero and sum the contributions.
    if compare == True:
        drag_coeff = mesh.comm.gather(assemble_scalar(drag), root=0)
        lift_coeff = mesh.comm.gather(assemble_scalar(lift), root=0)
        p_front = None
        if len(front_cells) > 0:
            p_front = p_.eval(points[0], front_cells[:1])
        p_front = mesh.comm.gather(p_front, root=0)
        p_back = None
        if len(back_cells) > 0:
            p_back = p_.eval(points[1], back_cells[:1])
        p_back = mesh.comm.gather(p_back, root=0)
        if mesh.comm.rank == 0:
            t_u[i] = t
            t_p[i] = t - dt / 2
            C_D[i] = sum(drag_coeff)
            C_L[i] = sum(lift_coeff)
            # Choose first pressure that is found from the different processors
            for pressure in p_front:
                if pressure is not None:
                    p_diff[i] = pressure[0]
                    break
            for pressure in p_back:
                if pressure is not None:
                    p_diff[i] -= pressure[0]
                    break
progress.close()
if plot == True:
    vtx_u.close()
    vtx_p.close()

# V0temp = V.sub(0)
# V0, _ = V0temp.collapse()
# divuu = Function(V0)
# u_expr = Expression(div(u_), V0.element.interpolation_points())
# divuu.interpolate(u_expr)
# divu=mesh.comm.gather(divuu.x.array, root = 0)


# incomp_ufl = inner(div(u_), q) * dx
# incomp_form = form(incomp_ufl)
# incomp = mesh_comm.gather(np.mean(assemble_vector(incomp_form)), root = 0)
# if mesh_comm.rank == 0:
#     print(f"Integral of divergence of u tested against pressure basis is: {np.sum(incomp)}")
    # print(f"Mean of computed divergence is {np.mean(np.hstack(divu))}")
    

if mesh_comm.rank == 0:
    if save_monolithic:

        # assert np_coords_unique.shape[0] ==  u_x_np.shape[0]
        # assert np_coords_unique.shape[0] ==  u_y_np.shape[0]
        # assert u_x_np.shape[1] == sample_times.shape[1]
        # savefile = folder / "np_sol_dump.npz"

        # np.savez(savefile, coords = stacked_coords[indices,0:2], u_x = u_x_np, u_y = u_y_np, times = sample_times)

        # meshfile = folder / "NSE_mesh"

        print("Results saved in " + str(savefile))
    if save_liftdrag: 
        np.savez(liftdrag_save_file, times = t_u, drag = C_D, lift = C_L)


if compare == True:
    data_folder = folder.parent / "Benchmark_Problem"
    fig_path = folder / "structured_figures"
    if mesh.comm.rank == 0:
        fig_path.mkdir(exist_ok=True, parents = True)
        save_fig = str(fig_path)
        num_velocity_dofs = V.dofmap.index_map_bs * V.dofmap.index_map.size_global
        num_pressure_dofs = Q.dofmap.index_map_bs * V.dofmap.index_map.size_global

        turek = np.loadtxt(data_folder / "Comparison_data" / "bdforces_lv4")
        turek_p = np.loadtxt(data_folder / "Comparison_data"/"pointvalues_lv4")
        fig = plt.figure(figsize=(25, 8))
        l1 = plt.plot(t_u, C_D, label=r"FEniCSx  ({0:d} dofs)".format(num_velocity_dofs + num_pressure_dofs), linewidth=2)
        l2 = plt.plot(turek[1:, 1], turek[1:, 3], marker="x", markevery=50,
                    linestyle="", markersize=4, label="FEATFLOW (42016 dofs)")
        plt.title("Drag coefficient")
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(save_fig, "structured_drag_comparison.png"))

        fig = plt.figure(figsize=(25, 8))
        l1 = plt.plot(t_u, C_L, label=r"FEniCSx  ({0:d} dofs)".format(
            num_velocity_dofs + num_pressure_dofs), linewidth=2)
        l2 = plt.plot(turek[1:, 1], turek[1:, 4], marker="x", markevery=50,
                    linestyle="", markersize=4, label="FEATFLOW (42016 dofs)")
        plt.title("Lift coefficient")
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(save_fig, "structured_lift_comparison.png"))

        fig = plt.figure(figsize=(25, 8))
        l1 = plt.plot(t_p, p_diff, label=r"FEniCSx ({0:d} dofs)".format(num_velocity_dofs + num_pressure_dofs), linewidth=2)
        l2 = plt.plot(turek[1:, 1], turek_p[1:, 6] - turek_p[1:, -1], marker="x", markevery=50,
                    linestyle="", markersize=4, label="FEATFLOW (42016 dofs)")
        plt.title("Pressure difference")
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(save_fig, "structured_pressure_comparison.png"))




