## This code is lightly edited from the dolfinx tutorial
## and can be found at https://github.com/jorgensd/dolfinx-tutorial

import argparse
import json
import shutil
from pathlib import Path

import gmsh
import numpy as np
import yaml

from mpi4py import MPI
import petsc4py.PETSc as PETSc

from basix.ufl import element

from dolfinx.fem import (Constant, Function, functionspace, assemble_scalar,
                         dirichletbc, form, locate_dofs_topological, set_bc,
                         extract_function_spaces)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, create_matrix, set_bc)
from dolfinx.io import VTXWriter, gmsh as gmshio
from ufl import (FacetNormal, Measure, TestFunction, TrialFunction,
                 as_vector, div, dot, dx, inner, lhs, grad, nabla_grad, rhs)

from efr_opinf._paths import DATA_DIR
from efr_opinf.fom.inlet import InletVelocity

import adios4dolfinx
import tqdm

L = 2.2
H = 0.41
c_x = c_y = 0.2
r = 0.05
gdim = 2
mu_val = 0.001
rho_val = 1.0
res_min = r / 3


def load_config(path: Path) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    if "T" not in config.get("time", {}):
        raise ValueError("YAML config must specify time.T")
    return config


def _build_mesh(comm: MPI.Comm, model_rank: int = 0):
    if not gmsh.isInitialized():
        gmsh.initialize()

    if comm.rank == model_rank:
        rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
        obstacle = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)
        gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle)])
        gmsh.model.occ.synchronize()

    fluid_marker = 1
    inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5

    if comm.rank == model_rank:
        volumes = gmsh.model.getEntities(dim=gdim)
        assert len(volumes) == 1
        gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
        gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")

        inflow, outflow, walls, obstacle = [], [], [], []
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

        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        gmsh.model.mesh.generate(gdim)
        gmsh.model.mesh.setOrder(2)
        gmsh.model.mesh.optimize("Netgen")

    mesh_data = gmshio.model_to_mesh(gmsh.model, comm, model_rank, gdim=gdim)
    mesh = mesh_data.mesh
    assert mesh_data.facet_tags is not None
    ft = mesh_data.facet_tags
    ct = mesh_data.cell_tags
    ct.name = "Cell Markers"
    ft.name = "Facet markers"

    markers = {
        "inlet": inlet_marker,
        "outlet": outlet_marker,
        "wall": wall_marker,
        "obstacle": obstacle_marker,
    }
    return mesh, ft, ct, markers


def run_cylinder_fom(config: dict) -> Path:
    """Run the FOM cylinder-flow solve and save training snapshots.

    `config` is a dict shaped like a parsed YAML config (see
    examples/fom_configs/ for examples): {"time": {"T": ..., "dt": ...,
    "sample_gap": ...}, "efr": {...} | absent, "output": {...}}.

    Returns the path to the saved `.bp` snapshot data (only meaningful
    when output.save_monolithic is True).
    """
    comm = MPI.COMM_WORLD
    model_rank = 0

    mesh, ft, ct, markers = _build_mesh(comm, model_rank)
    inlet_marker = markers["inlet"]
    outlet_marker = markers["outlet"]
    wall_marker = markers["wall"]
    obstacle_marker = markers["obstacle"]

    time_cfg = config.get("time", {})
    T = time_cfg["T"]
    dt = time_cfg.get("dt", 1 / 2000)
    sample_gap = time_cfg.get("sample_gap", 10)
    num_steps = int(T / dt)
    timespace = np.linspace(dt, T, num_steps)
    total_times = int(num_steps / sample_gap) + 1

    DO_EFR = config.get("efr")  # None if the block is omitted -- EFR off

    output_cfg = config.get("output", {})
    save_monolithic = output_cfg.get("save_monolithic", True)
    save_KE = output_cfg.get("save_KE", True)
    save_liftdrag = output_cfg.get("save_liftdrag", False)
    plot = output_cfg.get("plot", False)

    k = Constant(mesh, PETSc.ScalarType(dt))
    mu = Constant(mesh, PETSc.ScalarType(mu_val))
    rho = Constant(mesh, PETSc.ScalarType(rho_val))

    v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim,))
    s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
    V = functionspace(mesh, v_cg2)
    Q = functionspace(mesh, s_cg1)
    fdim = mesh.topology.dim - 1

    t = 0.0
    u_inlet = Function(V)
    inlet_velocity = InletVelocity(t)
    u_inlet.interpolate(inlet_velocity)
    bcu_inflow = dirichletbc(u_inlet, locate_dofs_topological(V, fdim, ft.find(inlet_marker)))
    u_nonslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
    bcu_walls = dirichletbc(u_nonslip, locate_dofs_topological(V, fdim, ft.find(wall_marker)), V)
    bcu_obstacle = dirichletbc(u_nonslip, locate_dofs_topological(V, fdim, ft.find(obstacle_marker)), V)
    bcu = [bcu_inflow, bcu_obstacle, bcu_walls]
    bcp_outlet = dirichletbc(PETSc.ScalarType(0), locate_dofs_topological(Q, fdim, ft.find(outlet_marker)), Q)
    bcp = [bcp_outlet]

    u = TrialFunction(V)
    v = TestFunction(V)
    u_ = Function(V)
    u_.name = "u"
    u_s = Function(V)
    w_ = Function(V)
    u_n = Function(V)
    u_n1 = Function(V)
    p = TrialFunction(Q)
    q = TestFunction(Q)
    p_ = Function(Q)
    p_.name = "p"
    phi = Function(Q)

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
    solver1.getPC().setType(PETSc.PC.Type.JACOBI)

    solver2 = PETSc.KSP().create(mesh.comm)
    solver2.setOperators(A2)
    solver2.setType(PETSc.KSP.Type.MINRES)
    pc2 = solver2.getPC()
    pc2.setType(PETSc.PC.Type.HYPRE)
    pc2.setHYPREType("boomeramg")

    solver3 = PETSc.KSP().create(mesh.comm)
    solver3.setOperators(A3)
    solver3.setType(PETSc.KSP.Type.CG)
    solver3.getPC().setType(PETSc.PC.Type.SOR)

    if DO_EFR is not None:
        filter_start = DO_EFR["filter_start"]
        chi = DO_EFR["chi"]
        delta = DO_EFR.get("delta", res_min)

        u_f = Function(V)
        delc = Constant(mesh, PETSc.ScalarType(delta ** 2))
        a4 = form(inner(u, v) * dx + delc * inner(grad(u), grad(v)) * dx)
        A4 = assemble_matrix(a4, bcs=bcu)
        A4.assemble()
        L4 = form(inner(w_, v) * dx)
        b4 = create_vector(extract_function_spaces(L4))

        solver4 = PETSc.KSP().create(mesh.comm)
        solver4.setOperators(A4)
        solver4.setType(PETSc.KSP.Type.CG)
        solver4.getPC().setType(PETSc.PC.Type.SOR)

    n = -FacetNormal(mesh)
    dObs = Measure("ds", domain=mesh, subdomain_data=ft, subdomain_id=obstacle_marker)
    u_t = inner(as_vector((n[1], -n[0])), u_)
    drag = form(2 / 0.1 * (mu / rho * inner(grad(u_t), n) * n[1] - p_ * n[0]) * dObs)
    lift = form(-2 / 0.1 * (mu / rho * inner(grad(u_t), n) * n[0] + p_ * n[1]) * dObs)
    KE_form = form(0.5 * inner(u_, u_) * dx)

    if mesh.comm.rank == 0:
        if save_liftdrag:
            C_D = np.zeros(total_times, dtype=PETSc.ScalarType)
            C_L = np.zeros(total_times, dtype=PETSc.ScalarType)
            t_liftdrag = np.zeros(total_times, dtype=np.float64)
        if save_KE:
            KE_arr = np.zeros(total_times, dtype=PETSc.ScalarType)
            KE_times = np.zeros(total_times, dtype=np.float64)

    flavor = "efr" if DO_EFR is not None else "standard"
    out_dir = DATA_DIR / "Cylinder" / flavor
    out_dir.mkdir(exist_ok=True, parents=True)
    stem = f"T_{T:g}"
    savefile = out_dir / f"{stem}.bp"

    if save_monolithic:
        adios4dolfinx.write_mesh(savefile, mesh, engine="BP4")
        adios4dolfinx.write_meshtags(savefile, mesh, ft, meshtag_name="facet_tags")
        adios4dolfinx.write_meshtags(savefile, mesh, ct, meshtag_name="cell_tags")

    if plot:
        viz_dir = out_dir / f"{stem}_viz"
        target_u = viz_dir / "u.bp"
        target_p = viz_dir / "p.bp"
        if mesh.comm.rank == 0:
            if target_u.is_dir():
                shutil.rmtree(target_u, ignore_errors=True)
            if target_p.is_dir():
                shutil.rmtree(target_p, ignore_errors=True)
        vtx_u = VTXWriter(mesh.comm, str(target_u), [u_], engine="BP4")
        vtx_p = VTXWriter(mesh.comm, str(target_p), [p_], engine="BP4")
        vtx_u.write(t)
        vtx_p.write(t)

    progress = tqdm.tqdm(desc="Solving cylinder FOM", total=num_steps, disable=(mesh.comm.rank != 0))

    j = 0
    for i in range(num_steps):
        progress.update(1)
        t = timespace[i]
        inlet_velocity.t = t
        u_inlet.interpolate(inlet_velocity)

        # Step 1: tentative velocity
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

        # Step 2: pressure correction
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

        # Step 3: velocity correction
        with b3.localForm() as loc:
            loc.set(0)
        assemble_vector(b3, L3)
        b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        solver3.solve(b3, w_.x.petsc_vec)
        w_.x.scatter_forward()

        # Step 4: optional EFR filter/relax, applied to the FOM velocity itself
        if DO_EFR is not None and t >= filter_start:
            with b4.localForm() as loc:
                loc.set(0)
            assemble_vector(b4, L4)
            apply_lifting(b4, [a4], [bcu])
            b4.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b4, bcu)
            solver4.solve(b4, u_f.x.petsc_vec)
            u_f.x.scatter_forward()
            u_.x.array[:] = chi * u_f.x.array + (1 - chi) * w_.x.array[:]
        else:
            u_.x.array[:] = w_.x.array
        u_.x.scatter_forward()

        if plot:
            vtx_u.write(t)
            vtx_p.write(t)

        if (i % sample_gap) == sample_gap - 1:
            if save_monolithic:
                adios4dolfinx.write_function(savefile, u_, time=t, name=u_.name)
                adios4dolfinx.write_function(savefile, p_, time=t, name=p_.name)
            if save_KE:
                KE_t = mesh.comm.gather(assemble_scalar(KE_form), root=0)
                if mesh.comm.rank == 0:
                    KE_times[j] = t
                    KE_arr[j] = sum(KE_t)
            if save_liftdrag:
                drag_coeff = mesh.comm.gather(assemble_scalar(drag), root=0)
                lift_coeff = mesh.comm.gather(assemble_scalar(lift), root=0)
                if mesh.comm.rank == 0:
                    t_liftdrag[j] = t
                    C_D[j] = sum(drag_coeff)
                    C_L[j] = sum(lift_coeff)
            if mesh.comm.rank == 0:
                j += 1

        with u_.x.petsc_vec.localForm() as loc_, u_n.x.petsc_vec.localForm() as loc_n, u_n1.x.petsc_vec.localForm() as loc_n1:
            loc_n.copy(loc_n1)
            loc_.copy(loc_n)

    progress.close()
    if plot:
        vtx_u.close()
        vtx_p.close()

    if mesh.comm.rank == 0:
        if save_KE:
            np.savez(out_dir / f"{stem}_KE.npz", times=KE_times, KE=KE_arr)
        if save_liftdrag:
            np.savez(out_dir / f"{stem}_liftdrag.npz", times=t_liftdrag, drag=C_D, lift=C_L)
        if save_monolithic:
            sidecar = {"T": T, "dt": dt, "sample_gap": sample_gap, "efr": DO_EFR if DO_EFR is not None else False}
            with open(out_dir / f"{stem}.json", "w") as sidecar_f:
                json.dump(sidecar, sidecar_f, indent=2)
            print(f"Saved FOM snapshots to {savefile}")

    return savefile


def _cli():
    parser = argparse.ArgumentParser(
        description="Run the FOM cylinder-flow solve from a YAML config."
    )
    parser.add_argument(
        "config", type=Path,
        help="Path to a YAML config file (see examples/fom_configs/ for examples).",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    run_cylinder_fom(config)


if __name__ == "__main__":
    _cli()
