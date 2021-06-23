#!/usr/bin/env python3
#
# Introduction
# ============
# This example considers the plastic deformation of a perforated strip. More
# information regarding this prototypical nonlinear solid mechanics problem
# can be found in, for example, *The Finite Element Method for Solid and
# Structural Mechanics* by Zienkiewicz and Taylor.
#
# Preliminaries
# =============
# *   This example builds on the Nutils **elasticity** example. Make sure to have a good
#     understanding of the **elasticity** example before proceeding with this example.
# *   In this example unicode variables are employed to improve readability. For
#     example, the stress tensor will be represented by the variable σ.
#
# We start by importing the necessary modules. Note that the `function` module is
# imported in order to construct a dedicated function for the plastic strain evolution.
from nutils import mesh, solver, types, cli, export, numeric, function, types
import numpy, treelog, pathlib, typing
from matplotlib import collections
_ = numpy.newaxis

unit = types.unit(m=1e3, s=1, g=1e-6, N='kg*m/s2', Pa='N/m2')

# This example can be run with the arguments listed below. The default arguments are based
# on the plane stress simulation with strain hardening as presented in Zienkiewicz and
# Taylor. The reference result for this parameter set is visualized in the output of
# this script.

def main(fname: str, degree: int, Δuload: unit['mm'], nsteps: int, E: unit['GPa'], nu: float, σyield: unit['MPa'], hardening: unit['GPa'], referencedata: typing.Optional[str], testing: bool):

  '''
  Plastic deformation of a perforated strip.

  .. arguments::

     fname [strip.msh]
       Mesh file with units in [mm]

     degree [1]
       Finite element interpolation order

     Δuload [5μm]
       Load boundary displacement steps

     nsteps [11]
       Number of load steps

     E [70GPa]
       Young's modulus

     nu [0.2]
       Poisson ratio

     σyield [243MPa]
       Yield strength

     hardening [2.25GPa]
       Hardening parameter

     referencedata [zienkiewicz.csv]
       Reference data file name

     testing [False]
       Use a 1 element mesh for testing

  .. presets::

     testing
       nsteps=30
       referencedata=
       testing=True
  '''

  # We commence with reading the mesh from the specified GMSH file, or, alternatively,
  # we continue with a single-element mesh for testing purposes.
  domain, geom = mesh.gmsh(pathlib.Path(__file__).parent/fname)
  Wdomain, Hdomain = domain.boundary['load'].integrate([function.J(geom),geom[1]*function.J(geom)], degree=1)
  Hdomain /= Wdomain

  if testing:
    domain, geom = mesh.rectilinear([numpy.linspace(0,Wdomain/2,2),numpy.linspace(0,Hdomain,2)])
    domain = domain.withboundary(hsymmetry='left', vsymmetry='bottom', load='top')

  # We next initiate the point set in which the constitutive bahvior will be evaluated.
  # Note that this is also the point set in which the history variable will be stored.
  gauss  = domain.sample('gauss', 2)

  # Elasto-plastic formulation
  # ==========================
  # The weak formulation is constructed using a `Namespace`, which is initialized and
  # populated with the necessary problem parameters and coordinate system. Note that the
  # coefficients `mu` and `λ` have been defined such that 'C_{ijkl}' is the fourth-order
  # **plane stress** elasticity tensor.

  ns = function.Namespace(fallback_length=domain.ndims)

  ns.x      = geom
  ns.Δuload = Δuload
  ns.mu     = E/(1-nu**2)*((1-nu)/2)
  ns.λ      = E/(1-nu**2)*nu
  ns.delta  = function.eye(domain.ndims)
  ns.C_ijkl = 'mu ( delta_ik delta_jl + delta_il delta_jk ) + λ delta_ij delta_kl'

  # We make use of a Lagrange finite element basis of arbitrary `degree`. Since we
  # approximate the displacement field, the basis functions are vector-valued. Both
  # the displacement field at the current load step `u` and displacement field of the
  # previous load step `u0` are approximated using this basis:

  ns.basis = domain.basis('std',degree=degree).vector(domain.ndims)

  ns.u0_i = 'basis_ni ?lhs0_n'
  ns.u_i  = 'basis_ni ?lhs_n'

  # This simulation is based on a standard elasto-plasticity model. In this
  # model the *total strain*, ε_kl = ½ (∂u_k/∂x_l + ∂u_l/∂x_k)', is comprised of an
  # elastic and a plastic part:
  #
  #   ε_kl = εe_kl + εp_kl
  #
  # The stress is related to the *elastic strain*, εe_kl, through Hooke's law:
  #
  #   σ_ij = C_ijkl εe_kl = C_ijkl (ε_kl - εp_kl)

  ns.ε_kl   = '(u_k,l + u_l,k) / 2'
  ns.ε0_kl  = '(u0_k,l + u0_l,k) / 2'
  ns.gbasis = gauss.basis()
  ns.εp0_ij = 'gbasis_n ?εp0_nij'
  ns.κ0     = 'gbasis_n ?κ0_n'

  ns.εp    = PlasticStrain(ns.ε, ns.ε0, ns.εp0, ns.κ0, E, nu, σyield, hardening)
  ns.εe_ij = 'ε_ij - εp_ij'
  ns.σ_ij  = 'C_ijkl εe_kl'

  # Note that the plasticity model is implemented through the user-defined function `PlasticStrain`,
  # which implements the actual yielding model including a standard return mapping algorithm. This
  # function is discussed in detail below.
  #
  # The components of the residual vector are then defined as:
  #
  #   r_n = ∫_Ω (∂N_ni/∂x_j) σ_ij dΩ

  res = domain.integral('basis_ni,j σ_ij d:x' @ ns, degree=2)

  # The problem formulation is completed by supplementing prescribed displacement boundary conditions
  # (for a load step), which are computed in the standard manner:

  sqr  = domain.boundary['hsymmetry,vsymmetry'].integral('(u_k n_k)^2 d:x' @ ns, degree=2)
  sqr += domain.boundary['load'].integral('(u_k n_k - Δuload)^2 d:x' @ ns, degree=2)
  cons = solver.optimize('lhs', sqr, droptol=1e-15)

  # Incremental-iterative solution procedure
  # ========================================
  # We initialize the solution vector for the first load step `lhs` and solution vector
  # of the previous load step `lhs0`. Note that, in order to construct a predictor step
  # for the first load step, we define the previous load step state as the solution
  # vector corresponding to a negative elastic loading step.

  lhs0 = -solver.solve_linear('lhs', domain.integral('basis_ni,j C_ijkl ε_kl d:x' @ ns, degree=2), constrain=cons)
  lhs  = numpy.zeros_like(lhs0)
  εp0  = numpy.zeros((gauss.npoints,)+ns.εp0.shape)
  κ0   = numpy.zeros((gauss.npoints,)+ns.κ0.shape)

  # To store the force-dispalcement data we initialize an empty data array with the
  # inital state solution substituted in the first row.
  fddata      = numpy.empty(shape=(nsteps+1,2))
  fddata[:]   = numpy.nan
  fddata[0,:] = 0

  # Load step incrementation
  # ------------------------
  with treelog.iter.fraction('step', range(nsteps)) as counter:
    for step in counter:

      # The solution of the previous load step is set to `lhs0`, and the Newton solution
      # procedure is initialized by extrapolation of the state vector:
      lhs_init = lhs + (lhs-lhs0)
      lhs0     = lhs

      # The non-linear system of equations is solved for `lhs` using Newton iterations,
      # where the `step` variable is used to scale the incremental constraints.
      lhs = solver.newton(target='lhs', residual=res, constrain=cons*step, lhs0=lhs_init, arguments={'lhs0':lhs0,'εp0':εp0,'κ0':κ0}).solve(tol=1e-6)

      # The computed solution is post-processed in the form of a loading curve - which
      # plots the normalized mean stress versus the maximum 'ε_{yy}' strain
      # component - and a contour plot showing the 'σ_{yy}' stress component on a
      # deformed mesh. Note that since the stresses are defined in the integration points
      # only, a post-processing step is involved that transfers the stress information to
      # the nodal points.
      εyymax = gauss.eval(ns.ε[1,1], arguments=dict(lhs=lhs)).max()

      basis = domain.basis('std', degree=1)
      bw, b = domain.integrate([basis * ns.σ[1,1] * function.J(geom), basis * function.J(geom)], degree=2, arguments=dict(lhs=lhs,lhs0=lhs0,εp0=εp0,κ0=κ0))
      σyy = basis.dot(bw / b)

      uyload, σyyload = domain.boundary['load'].integrate(['u_1 d:x'@ns,σyy * function.J(geom)], degree=2, arguments=dict(lhs=lhs,εp0=εp0,κ0=κ0))
      uyload /= Wdomain
      σyyload /= Wdomain
      fddata[step,0] = (E*εyymax)/σyield
      fddata[step,1] = (σyyload*2)/σyield

      with export.mplfigure('forcedisp.png') as fig:
        ax = fig.add_subplot(111, xlabel=r'${E \cdot {\rm max}(\varepsilon_{yy})}/{\sigma_{\rm yield}}$', ylabel=r'${\sigma_{\rm mean}}/{\sigma_{\rm yield}}$')
        if referencedata:
          data = numpy.genfromtxt(pathlib.Path(__file__).parent/referencedata, delimiter=',', skip_header=1)
          ax.plot(data[:,0], data[:,1], 'r:', label='Reference')
          ax.legend()
        ax.plot(fddata[:,0], fddata[:,1], 'o-', label='Nutils')
        ax.grid()

      bezier = domain.sample('bezier', 3)
      points, uvals, σyyvals = bezier.eval(['(x_i + 25 u_i)' @ ns, ns.u, σyy], arguments=dict(lhs=lhs))
      with export.mplfigure('stress.png') as fig:
        ax = fig.add_subplot(111, aspect='equal', xlabel=r'$x$ [mm]', ylabel=r'$y$ [mm]')
        im = ax.tripcolor(points[:,0]/unit('mm'), points[:,1]/unit('mm'), bezier.tri, σyyvals/unit('MPa'), shading='gouraud', cmap='jet')
        ax.add_collection(collections.LineCollection(points.take(bezier.hull, axis=0), colors='k', linewidths=.1))
        ax.autoscale(enable=True, axis='both', tight=True)
        cb = fig.colorbar(im)
        im.set_clim(0, 1.2*σyield)
        cb.set_label(r'$σ_{yy}$ [MPa]')

      # Load step convergence
      # ---------------------
      # At the end of the loading step, the plastic strain state and history parameter are updated,
      # where use if made of the strain hardening relation for the history variable:
      #
      #   Δκ = √(Δεp_ij Δεp_ij)

      Δεp = ns.εp-ns.εp0
      Δκ  = function.sqrt((Δεp*Δεp).sum((0,1)))
      κ0  = gauss.eval(ns.κ0+Δκ, arguments=dict(lhs0=lhs0,lhs=lhs,εp0=εp0,κ0=κ0))
      εp0 = gauss.eval(ns.εp, arguments=dict(lhs0=lhs0,lhs=lhs,εp0=εp0,κ0=κ0))

# The plastic strain function
# ===========================
# The core of the plasticity model is implemented in the following user-defined
# Nutils function. The essence of this function is that, based on the current
# *total strain* level, the yield function is used to determine the corresponding
# *plastic strain*.

class PlasticStrain(function.Custom):
  '''Von Mises plane stress with strain hardening

  Plastic strain tensor function for a plane stress Von Mises model with strain hardening.

  Parameters
  ----------
  ε : :class:`function.Array`
      Strain tensor function
  ε0 : :class:`function.Array`
      Strain tensor function at the previous load step
  εp0 : :class:`function.Array`
      Plastic strain tensor function at the previous load step
  κ0 : :class:`function.Array`
      History variable function at the previous load step
  E : :class:`unit['Pa']`
      Young's modulus
  nu : :class:`float`
      Poisson ratio
  σyield : :class:`unit['Pa']`
      Yield stress
  h : :class:`unit['Pa']`
      Hardening parameter
  rtol : :class:`float`, optional
      Return mapping tolerance
  maxiter : :class:`float`, optional
      Maximum number of return mapping iterations
  '''

  __slots__ = 'rtol', 'maxiter', 'δ', 'Isym', 'C', 'σyield', 'h', 'ε', 'ε0', 'εp0', 'κ0', 'P'

  def __init__(self, ε: function.Array, ε0: function.Array, εp0: function.Array, κ0: function.Array, E:unit['Pa'], nu:float, σyield:unit['Pa'], h:unit['Pa'], rtol:float = 1e-8, maxiter:int = 20):
    ε = function.asarray(ε)
    ε0 = function.asarray(ε0)
    εp0 = function.asarray(εp0)
    κ0 = function.asarray(κ0)

    assert ε.shape == (2, 2)

    # We set the numerical parameters for the return mapping algorithm:
    self.rtol    = rtol
    self.maxiter = maxiter

    # We store various useful variables:
    self.δ      = numpy.eye(ε.shape[1])
    self.Isym   = (self.δ[:,_,:,_]*self.δ[_,:,_,:]+self.δ[:,_,_,:]*self.δ[_,:,:,_])/2
    self.C      = E/(1-nu**2)*((1-nu)*self.Isym+nu*self.δ[:,:,_,_]*self.δ[_,_,:,:])
    self.σyield = σyield
    self.h      = h

    # We store various functions that are needed within this object:
    self.ε   = ε
    self.ε0  = ε0
    self.εp0 = εp0
    self.κ0  = κ0

    # We construct a projector to vectorize (higher-order) tensors:
    self.P = numpy.zeros(shape=(3,2,2))
    self.P[0,0,0] = 1.
    self.P[1,1,1] = 1.
    self.P[2,1,0] = numpy.sqrt(0.5)
    self.P[2,0,1] = numpy.sqrt(0.5)

    # The constructor of the base class is finally called to conclude the
    # constructor:
    super().__init__(args=[ε,ε0,εp0,κ0,E,nu,σyield,h,rtol,maxiter], shape=ε.shape, dtype=float)

  # Plastic strain evaluation
  # -------------------------
  # A prediction for the stress is computed by assuming the behavior of the
  # material to be elastic. If the yield criterion is violated the `_returnmap`
  # function is called to compute the required update of the plastic strain so
  # that the stress state is on the yield surface.
  def evalf(self, ε: numpy.ndarray, ε0: numpy.ndarray, εp0: numpy.ndarray, κ0: numpy.ndarray, E:unit['Pa'], nu:float, σyield:unit['Pa'], h:unit['Pa'], rtol:float, maxiter:int) -> numpy.ndarray:
    ε, ε0, εp0 = numpy.broadcast_arrays(ε, ε0, εp0)
    εp = εp0.copy()
    σ = numpy.einsum('ijkl,pkl->pij', self.C, ε-εp0)
    yielding = F(σ) > numpy.sqrt(2/3) * (self.σyield + self.h*κ0)
    for p in yielding.nonzero()[0]:
      εp[p] += self._returnmap(Δε=ε[p]-ε0[p], εe0=ε0[p]-εp0[p], κ0=κ0[p])
    return εp


  # A **return mapping procedure** is employed to map the stress back to the yield surface.
  # Given the elastic strain of the previous load step, εe0_kl (such that σ0_ij = C_ijkl εe0_kl),
  # and iterate for the total strain increment, 'Δε_kl', this procedure returns the corresponding
  # increment of plastic strain, Δεp_kl = Δλ n_kl, satisfying:
  #
  #   rσ_ij = Δσ_ij - C_ijkl (Δε_kl - Δλ n_kl) = 0
  #   r_F = F(σ_0 + Δσ) - √⅔ (σyield + h (κ0 + Δκ)) = 0
  #
  # The solution to this nonlinear system of equations is computed using a Newton-Raphson
  # procedure. Convergence of the procedure is checked based on the norm of the residual.
  # A `RuntimeError` is thrown if the procedure does not converge in `maxiter` iterations.
  def _returnmap(self, Δε, εe0, κ0):

    σ0 = numpy.einsum('ijkl,kl->ij', self.C, εe0)
    Δσ = numpy.einsum('ijkl,kl->ij', self.C, Δε)
    Δλ = 0

    with treelog.iter.fraction('rmap', range(self.maxiter)) as counter:
      for iiter in counter:

        n  = dF(σ0+Δσ)
        dn = d2F(σ0+Δσ)

        κ  = κ0 + Δλ*numpy.sqrt(numpy.einsum('ij,ij', n, n))

        rσ = Δσ - numpy.einsum('ijkl,kl->ij', self.C, Δε-Δλ*n)
        rF = F(σ0+Δσ) - numpy.sqrt(2/3) * (self.σyield + self.h*κ)

        b = numpy.empty(len(self.P)+1)
        b[:-1] = numpy.einsum('ijk,jk', self.P, rσ)
        b[-1] = rF

        error = numpy.linalg.norm(b) / self.σyield
        treelog.debug('rmap residual = {}'.format(error))
        if error < self.rtol:
          break

        drσdσ = self.Isym + Δλ * numpy.einsum('ijkl,klmn->ijmn', self.C, dn)
        drσdλ = numpy.einsum('ijkl,kl->ij', self.C, n)
        drFdσ = n
        drFdλ = - numpy.sqrt(2*numpy.einsum('ij,ij', n, n)/3) *self.h

        A = numpy.empty((len(self.P)+1,)*2)
        A[:-1,:-1] = numpy.einsum('ikl,klmn,jmn->ij', self.P, drσdσ, self.P)
        A[-1,:-1]  = numpy.einsum('mn,jmn->j', drFdσ, self.P)
        A[:-1,-1]  = numpy.einsum('ikl,kl->i', self.P, drσdλ)
        A[-1,-1]   = drFdλ

        x = -numpy.linalg.solve(A,b)

        Δσ += numpy.einsum('ijk,i->jk', self.P, x[:-1])
        Δλ += x[-1]

      else:
        raise RuntimeError('Return mapping solver did not converge')

    return Δλ * n

  # Consistent tangent
  # ------------------
  def partial_derivative(self, iarg, ε: function.Array, ε0: function.Array, εp0: function.Array, κ0: function.Array, E:unit['Pa'], nu:float, σyield:unit['Pa'], h:unit['Pa'], rtol:float, maxiter:int) -> function.Array:

    if iarg == 0:
      # Function objects are created for the stress, the second invariant
      # of its deviatoric part, and the derivatives thereof:
      σ    = ( self.C[:,:,:,:]*(ε-self)[_,_,:,:] ).sum((2,3)) # σ_Nij = C_ijkl (ε_Nkl - εp_Nkl)
      J2   = ((σ*σ).sum((0,1)) - (1./3.)*((σ*self.δ[:,:]).sum((0,1)))**2) # J2_N = ½ (σ_Nij σ_Nij - ⅓ σ_Nii^2)
      dJ2  = σ - (1./3.)*((σ*self.δ[:,:]).sum((0,1)))[_,_]*self.δ[:,:] # dJ2_Nkl = σ_Nkl - ⅓ σ_Nii δ_kl
      d2J2 = self.Isym[:,:,:,:] - (1./3.)*self.δ[:,:,_,_]*self.δ[_,_,:,:] # d2J2_Nklmn = Isym_klmn - ⅓ δ_kl δ_mn

      # Functions are then created for the flow direction and its derivative:
      nf  = (1./function.sqrt(2.*J2[_,_]))*dJ2 # nf_Nkl = '(1 / sqrt(2 J2_N)) dJ2_Nkl'
      dnf = (1./function.sqrt(2.*J2[_,_,_,_]))*(d2J2-(1./(2.*J2[_,_,_,_]))*dJ2[:,:,_,_]*dJ2[_,_,:,:]) # dnf_Nklmn = (1 / √(2 J2_N)) (d2J2_Nklmn - (1 / (2 J2_N)) dJ2_Nkl dJ2_Nmn)

      # The plastic multiplier can then be computed as:
      Δεp = self-εp0 # Δεp_Nkl = εp_Nkl - εp0_Nkl
      Δλ  = (Δεp*nf).sum((0,1))/((nf*nf).sum((0,1))) # Δλ_N = (Δεp_Nkl nf_Nkl) / (nf_Nmn nf_Nmn)

      # The consistent tangent can then be computed as:
      N     = nf-function.sqrt(2./(3.*(nf[:,:,_,_]*nf[:,:,_,_]).sum((0,1))))*self.h*Δλ[_,_]*((nf[:,:,_,_]*dnf).sum((0,1))) # N_Nkl = nf_Nkl - √(⅔ / nf_Nij nf_Nij) h Δλ_N nf_Nmn dnf_Nmnkl
      B     = self.Isym[:,:,:,:]+Δλ[_,_,_,_]*((self.C[:,:,:,:,_,_]*dnf[_,_,:,:,:,:]).sum((2,3))) # B_Nklmn = Isym_klmn + Δλ_N C_klij dnf_Nijmn
      Binv  = (self.P[:,_,:,:,_,_]*(function.inverse((self.P[:,_,:,:,_,_]*B[_,_,:,:,:,:]*self.P[_,:,_,_,:,:]).sum((2,3,4,5))))[:,:,_,_,_,_]*self.P[_,:,_,_,:,:]).sum((0,1)) # Binv_Nijkl = P_aij inv(P_aqr B_Nqrst P_bst) P_bkl
      Cstar = (N[:,:,_,_,_,_]*Binv[:,:,:,:,_,_]*self.C[_,_,:,:,:,:]*nf[_,_,_,_,:,:]).sum((0,1,2,3,4,5))+function.sqrt(2.*((nf*nf).sum((0,1)))/3.)*self.h # Cstar_N = N_Nkl Binv_Nklmn C_mnpq nf_Npq + √(⅔ nf_Nij nf_Nij) h
      dΔλdε = ((((N[:,:,_,_]*Binv).sum((0,1)))[:,:,_,_]*self.C[:,:,:,:]).sum((0,1)))/Cstar[_,_] # dΔλdε_Nkl = N_Nij Binv_Nijmn C_mnkl / Cstar_N
      dσdε  = (Binv[:,:,:,:,_,_]*self.C[_,_,:,:,:,:]).sum((2,3))-((Binv[:,:,:,:,_,_]*self.C[_,_,:,:,:,:]*nf[_,_,_,_,:,:]).sum((2,3,4,5)))[:,:,_,_]*dΔλdε[_,_,:,:] # dσdε_Nijkl = Binv_Nijpq C_pqkl - Binv_Nijpq C_pqrs nf_Nrs dΔλdε_Nkl
      dεpdε = nf[:,:,_,_]*dΔλdε[_,_,:,:]+Δλ[_,_,_,_]*((dnf[:,:,:,:,_,_]*dσdε[_,_,:,:,:,:]).sum((2,3))) # dεpdε_Nijkl = nf_Nij dΔλdε_Nkl + Δλ_N dnf_Nijrs dσdε_Nrskl

      # Note that, based on the value of the plastic multiplier increment, either the
      # elastic (zero) or the plastic tangent derivative is used:
      return function.greater(Δλ, numpy.spacing(1000)*function.ones(Δλ.shape))[_,_,_,_]*dεpdε
    else:
      raise NotImplementedError


# Supplementary function definitions
# ==================================
# The second invariant of the deviatoric stress tensor
# ----------------------------------------------------
def J2(σ):
  return numpy.einsum('...ij,...ij->...', σ, σ) / 2 - numpy.einsum('...ii', σ)**2 / 6

def dJ2(σ):
  retval = σ.copy()
  numeric.takediag(retval)[...] -= numpy.einsum('...ii', σ) / 3
  return retval

def d2J2(σ):
  δ = numpy.eye(σ.shape[-1])
  retval = δ[:,_,:,_] * δ[_,:,_,:] / 2 + δ[:,_,_,:] * δ[_,:,:,_] / 2 - δ[:,:,_,_] * δ[_,_,:,:] / 3
  return retval[(_,)*(σ.ndim-2)]

# The Von Mises yield function and its derivatives
# ------------------------------------------------
def F(σ):
  return numpy.sqrt(2*J2(σ))

def dF(σ):
  return dJ2(σ) / F(σ)

def d2F(σ):
  Fσ = F(σ)
  dJ2σ = dJ2(σ)
  return d2J2(σ) / Fσ - dJ2σ[:,:,_,_] * dJ2σ[_,_,:,:] / Fσ**3


if __name__ == '__main__':
  cli.run( main )
