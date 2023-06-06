import os
import mfem.ser as mfem
from glvis import glvis, to_stream


def run(order=1, meshfile=''):
    #  2. Read the mesh from the given mesh file, and refine once uniformly.
    mesh = mfem.Mesh(meshfile, 1, 1)
    mesh.UniformRefinement()

    # 3. Define a finite element space on the mesh. Here we use H1 continuous
    #    high-order Lagrange finite elements of the given order.
    fec = mfem.H1_FECollection(order, mesh.Dimension())
    fespace = mfem.FiniteElementSpace(mesh, fec)
    print('Number of finite element unknowns: ' + str(fespace.GetTrueVSize()))

    # 4. Extract the list of all the boundary DOFs. These will be marked as
    #    Dirichlet in order to enforce zero boundary conditions.
    boundary_dofs = mfem.intArray()
    fespace.GetBoundaryTrueDofs(boundary_dofs)

    # 5. Define the solution x as a finite element grid function in fespace. Set
    #    the initial guess to zero, which also sets the boundary conditions.
    x = mfem.GridFunction(fespace)
    x.Assign(0.0)

    # 6. Set up the linear form b(.) corresponding to the right-hand side.
    one = mfem.ConstantCoefficient(1.0)
    b = mfem.LinearForm(fespace)
    b.AddDomainIntegrator(mfem.DomainLFIntegrator(one))
    b.Assemble()

    # 7. Set up the bilinear form a(.,.) corresponding to the -Delta operator.
    a = mfem.BilinearForm(fespace)
    a.AddDomainIntegrator(mfem.DiffusionIntegrator(one))
    a.Assemble()

    # 8. Form the linear system A X = B. This includes eliminating boundary
    #    conditions, applying AMR constraints, and other transformations.
    A = mfem.SparseMatrix()
    B = mfem.Vector()
    X = mfem.Vector()
    a.FormLinearSystem(boundary_dofs, x, b, A, X, B)
    print("Size of linear system: " + str(A.Height()))

    # 9. Solve the system using PCG with symmetric Gauss-Seidel preconditioner.
    M = mfem.GSSmoother(A)
    mfem.PCG(A, M, B, X, 1, 200, 1e-12, 0.0)

    # 10. Recover the solution x as a grid function and save to file. The output
    #     can be viewed using GLVis as follows: "glvis -m mesh.mesh -g sol.gf"
    a.RecoverFEMSolution(X, b, x)
    x.Save('./logs/sol.gf')
    mesh.Save('./logs/mesh.mesh')


def main():
    from mfem.common.arg_parser import ArgParser
    from os.path import expanduser, join

    parser = ArgParser(description='Ex1 (Laplace Problem)')
    parser.add_argument('-m', '--mesh',
                        default='star.mesh',
                        action='store', type=str,
                        help='Mesh file to use.')
    parser.add_argument('-o', '--order',
                        action='store', default=1, type=int,
                        help="Finite element order (polynomial degree) or -1 for isoparametric space.")

    args = parser.parse_args()
    parser.print_options(args)

    order = 1
    mesh = "star.mesh"
    meshfile = expanduser(join(os.path.dirname(__file__), 'data', mesh))

    run(order=order, meshfile=meshfile)


def debug():
    mesh = mfem.Mesh(5, 5, "TRIANGLE")
    mesh.UniformRefinement()

    # Create H1 finite element function space
    fec = mfem.H1_FECollection(2, mesh.Dimension())  # order=2
    fespace = mfem.FiniteElementSpace(mesh, fec)

    # Determine essential degrees of freedom (the whole boundary here)
    ess_tdof_list = mfem.intArray()
    ess_bdr = mfem.intArray([1] * mesh.bdr_attributes.Size())
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list)

    # Define Bilinear and Linear forms for the Laplace problem -Δu=1
    one = mfem.ConstantCoefficient(1.0)
    a = mfem.BilinearForm(fespace)
    a.AddDomainIntegrator(mfem.DiffusionIntegrator(one))
    a.Assemble()
    b = mfem.LinearForm(fespace)
    b.AddDomainIntegrator(mfem.DomainLFIntegrator(one))
    b.Assemble()

    # Create a grid function for the solution and initialize with 0
    x = mfem.GridFunction(fespace)
    x.Assign(0.0)

    # Form the linear system, AX=B, for the FEM discretization
    A = mfem.OperatorPtr()
    B = mfem.Vector()
    X = mfem.Vector()
    a.FormLinearSystem(ess_tdof_list, x, b, A, X, B)
    print("Size of the linear system: " + str(A.Height()))

    # Solve the system using PCG solver and get the solution in x
    Asm = mfem.OperatorHandle2SparseMatrix(A)
    Msm = mfem.GSSmoother(Asm)
    mfem.PCG(Asm, Msm, B, X, 1, 200, 1e-12, 0.0)
    a.RecoverFEMSolution(X, b, x)

    glvis((mesh, x), 400, 400)
    glvis(mesh)

    g = glvis(to_stream(mesh, x) + 'keys ARjlmcbp*******')
    g.set_size(600, 400)
    g.plot(mesh)

    print("done")


if __name__ == "__main__":
    debug()
