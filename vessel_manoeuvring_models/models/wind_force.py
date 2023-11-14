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


class WindForceSystemSimple(EquationSubSystem):
    def __init__(self, ship: ModularVesselSimulator, create_jacobians=True):
        CX, CY, CN = sp.symbols("CX, CY, CN")
        X_WC, Y_WC, N_WC = sp.symbols("X_WC, Y_WC, N_WC")
        eq_X_W_simple = sp.Eq(X_W, CX * X_WC)
        eq_Y_W_simple = sp.Eq(Y_W, CY * Y_WC)
        eq_N_W_simple = sp.Eq(N_W, CN * N_WC)

        eq_X_W_subs = eq_X_W.subs(X_W, X_WC)
        eq_Y_W_subs = eq_Y_W.subs(Y_W, Y_WC)
        eq_N_W_subs = eq_N_W.subs(N_W, N_WC)

        equations = [
            eq_cog,
            eq_aws,
            eq_awa,
            eq_X_W_subs,
            eq_Y_W_subs,
            eq_N_W_subs,
            eq_X_W_simple,
            eq_Y_W_simple,
            eq_N_W_simple,
        ]

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
        eq_X_W_dummy = sp.Eq(X_W, 0)
        eq_Y_W_dummy = sp.Eq(Y_W, 0)
        eq_N_W_dummy = sp.Eq(N_W, 0)

        equations = [eq_X_W_dummy, eq_Y_W_dummy, eq_N_W_dummy]

        super().__init__(
            ship=ship, equations=equations, create_jacobians=create_jacobians
        )


F_aftfan, F_forefan = sp.symbols("F_aftfan, F_forefan")
alpha_aftfan, alpha_forefan = sp.symbols("alpha_aftfan, alpha_forefan")
x_aftfan, x_forefan = sp.symbols("x_aftfan, x_forefan")

X_F_aft, X_F_fore = sp.symbols("X_F_aft, X_F_fore")
Y_F_aft, Y_F_fore = sp.symbols("Y_F_aft, Y_F_fore")
N_F_aft, N_F_fore = sp.symbols("N_F_aft, N_F_fore")


class WindFanForceSystem(EquationSubSystem):
    def __init__(self, ship: ModularVesselSimulator, create_jacobians=True):
        eq_X_F_aft = sp.Eq(X_F_aft, F_aftfan * sp.cos(alpha_aftfan))
        eq_Y_F_aft = sp.Eq(Y_F_aft, F_aftfan * sp.sin(alpha_aftfan))
        eq_N_F_aft = sp.Eq(N_F_aft, x_aftfan * Y_F_aft)

        eq_X_F_fore = sp.Eq(X_F_fore, F_forefan * sp.cos(alpha_forefan))
        eq_Y_F_fore = sp.Eq(Y_F_fore, F_forefan * sp.sin(alpha_forefan))
        eq_N_F_fore = sp.Eq(N_F_fore, x_forefan * Y_F_fore)

        eq_X_W = sp.Eq(X_W, X_F_aft + X_F_fore)
        eq_Y_W = sp.Eq(Y_W, Y_F_aft + Y_F_fore)
        eq_N_W = sp.Eq(N_W, N_F_aft + N_F_fore)

        equations = [
            eq_X_F_aft,
            eq_Y_F_aft,
            eq_N_F_aft,
            eq_X_F_fore,
            eq_Y_F_fore,
            eq_N_F_fore,
            eq_X_W,
            eq_Y_W,
            eq_N_W,
        ]

        super().__init__(
            ship=ship, equations=equations, create_jacobians=create_jacobians
        )
