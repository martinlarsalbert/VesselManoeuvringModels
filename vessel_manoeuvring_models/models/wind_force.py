import sympy as sp
from vessel_manoeuvring_models.symbols import *
from .wind_force_helpers import *
from vessel_manoeuvring_models.apparent_wind import eq_cog, eq_aws, eq_awa
from vessel_manoeuvring_models.models.subsystem import EquationSubSystem
from vessel_manoeuvring_models.models.modular_simulator import ModularVesselSimulator
from vessel_manoeuvring_models.angles import smallest_signed_angle

C_x, C_y, C_n = sp.symbols("C_x,C_y,C_n")

deg_x = 5
C_xs = eq_abs(deg=deg_x, symbol_prefix="C_x")
eq_C_x = sp.Eq(C_x, C_xs)
eq_X_W = sp.expand(
    sp.Eq(X_W, rho_A * A_XV * aws**2 * C_xs)
)  # 1/2 is stored in the C_x...

deg_y = 3
C_ys = eq_sign(
    deg=deg_y, symbol_prefix="C_y", const=False
)  # 1/2 is stored in the C_y...
eq_C_y = sp.Eq(C_y, C_ys)
eq_Y_W = sp.expand(sp.Eq(Y_W, rho_A * A_YV * aws**2 * C_ys))

deg_n = 3
C_ns = eq_sign(deg=deg_n, symbol_prefix="C_n", const=False)
eq_C_n = sp.Eq(C_n, C_ns)
eq_N_W = sp.expand(
    sp.Eq(N_W, rho_A * A_YV * aws**2 * C_ns * L)
)  # 1/2 is stored in the C_n...


class WindForceSystem(EquationSubSystem):
    def __init__(self, ship: ModularVesselSimulator, create_jacobians=True):

        equations = [eq_cog, eq_aws, eq_awa, eq_X_W, eq_Y_W, eq_N_W]

        super().__init__(
            ship=ship, equations=equations, create_jacobians=create_jacobians
        )

        # Need to force the awa on the [-pi,pi] intervall:
        lambda_ = self.lambdas["awa"]

        def awa_signed_angle(U, cog, psi, twa, tws):
            result = lambda_(U=U, cog=cog, psi=psi, twa=twa, tws=tws)
            return smallest_signed_angle(result)

        self.lambdas["awa"] = awa_signed_angle


class DummyWindForceSystem(EquationSubSystem):
    def __init__(self, ship: ModularVesselSimulator, create_jacobians=True):

        eq_X_W_dummy = sp.Eq(X_W,0)
        eq_Y_W_dummy = sp.Eq(Y_W,0)
        eq_N_W_dummy = sp.Eq(N_W,0)
                
        equations = [eq_X_W_dummy, eq_Y_W_dummy, eq_N_W_dummy]

        super().__init__(
            ship=ship, equations=equations, create_jacobians=create_jacobians
        )
       