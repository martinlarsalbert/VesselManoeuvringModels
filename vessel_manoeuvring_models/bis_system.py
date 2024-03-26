import numpy as np
import quantities as pq


class BisSystem:
    """Class that handles conversion from and to bis system"""

    def __init__(self, lpp, volume, g=9.81, rho=1000, units={}):
        self.lpp = lpp
        self.volume = volume
        self.g = g
        self.rho = rho
        self.__update_denomintors()

        self.qd = {}
        self.qd["D"] = "length"
        self.qd["rho"] = "density"
        self.qd["lpp"] = "length"
        self.qd["ta"] = "length"
        self.qd["tf"] = "length"
        self.qd["g"] = "linear_acceleration"
        self.qd["V"] = "linear_velocity"
        self.qd["fx"] = "force"
        self.qd["fy"] = "force"
        self.qd["fz"] = "force"
        self.qd["YR"] = "force"
        self.qd["fx_rudders"] = "force"
        self.qd["fy_rudders"] = "force"
        self.qd["mz_rudders"] = "moment"
        self.qd["fx_rudder"] = "force"
        self.qd["fy_rudder"] = "force"
        self.qd["mz_rudder"] = "moment"
        self.qd["mx"] = "moment"
        self.qd["my"] = "moment"
        self.qd["mz"] = "moment"
        self.qd["p"] = "angular_velocity"
        self.qd["r"] = "angular_velocity"
        self.qd["phi"] = "angle"
        self.qd["beta"] = "angle"
        self.qd["delta"] = "angle"
        self.qd["thrust"] = "force"
        self.qd["thrust_propeller"] = "force"
        self.qd["thrust_dysa"] = "force"
        self.qd["torque"] = "moment"
        self.qd["rev"] = "hz"
        self.qd["u"] = "linear_velocity"
        self.qd["v"] = "linear_velocity"
        self.qd["V_round"] = "linear_velocity"
        self.qd["psi"] = "angle"
        self.qd["theta"] = "angle"
        self.qd["q"] = "angular_velocity"
        self.qd["w"] = "linear_velocity"
        self.qd["x0"] = "length"
        self.qd["y0"] = "length"
        self.qd["z0"] = "length"
        self.qd["S"] = "area"
        self.qd["Dp"] = "length"
        self.qd["vtm"] = "linear_velocity"
        self.qd["gtm"] = "angle"
        self.qd["volume"] = "volume"
        self.qd["eta0"] = "non_dimensional"
        self.qd.update(units)

        self.exclude = [
            "sub method",
            "test type",
            "bis",
            "bis_vct",
            "method",
            "method_vct",
            "rudder",
            "description",
            "ship",
            "type of test",
            "result_file_path",
        ]

    def __update_denomintors(self):
        rho = self.rho
        volume = self.volume
        Lpp = self.lpp
        g = self.g

        denominators = {}
        denominators["non_dimensional"] = 1
        denominators["mass"] = rho * volume
        denominators["length"] = Lpp
        denominators["area"] = Lpp**2
        denominators["volume"] = Lpp**3
        denominators["density"] = denominators["mass"] / denominators["volume"]
        denominators["time"] = np.sqrt(Lpp / g)
        denominators["hz"] = 1 / denominators["time"]
        denominators["linear_velocity"] = np.sqrt(Lpp * g)
        denominators["linear_acceleration"] = g
        denominators["angle"] = 1
        denominators["angular_velocity"] = np.sqrt(g / Lpp)
        denominators["angular_acceleration"] = g / Lpp
        denominators["force"] = rho * g * volume
        denominators["moment"] = rho * g * volume * Lpp
        self.denominators = denominators

        quantities = {}
        quantities["non_dimensional"] = 1
        quantities["mass"] = pq.kg
        quantities["length"] = pq.m
        quantities["area"] = pq.m**2
        quantities["volume"] = pq.m**3
        quantities["density"] = quantities["mass"] / quantities["volume"]
        quantities["time"] = pq.s
        quantities["hz"] = 1 / quantities["time"]
        quantities["linear_velocity"] = pq.m / pq.s
        quantities["linear_acceleration"] = pq.m / (pq.s**2)
        quantities["angle"] = pq.rad
        quantities["angular_velocity"] = pq.rad / pq.s
        quantities["angular_acceleration"] = pq.rad / (pq.s**2)
        quantities["force"] = pq.N
        quantities["moment"] = pq.N * pq.m
        self.quantities = quantities

    def get_pq(self, key):
        quantity = self.get_quantity(key=key)

        if quantity in self.quantities:
            return self.quantities[quantity]
        else:
            raise ValueError(
                'Cannot find a quantity in self.quantities matching:"%s"' % key
            )

    def get_quantity(self, key):
        if key in self.qd:
            return self.qd[key]
        else:
            raise ValueError('Cannot find a quantity in self.qd matching:"%s"' % key)

    def get_denominator(self, key):
        quantity = self.get_quantity(key=key)
        if quantity in self.denominators:
            return self.denominators[quantity]
        else:
            raise ValueError('Cannot find a denominator for quantity:"%s"' % quantity)

    def to_bis(self, key, value):
        if key in self.exclude:
            nondimensional_value = value
        else:
            denominator = self.get_denominator(key=key)
            nondimensional_value = value / denominator

        return nondimensional_value

    def from_bis(self, key, nondimensional_value):
        if key in self.exclude:
            value = nondimensional_value
        else:
            denominator = self.get_denominator(key=key)
            value = nondimensional_value * denominator

        return value

    @staticmethod
    def only_numeric(df):
        mask = df.dtypes != "object"
        numeric_columns = df.columns[mask]
        return df[numeric_columns]

    def df_to_bis(self, df):
        nondimensional_df = df.copy()

        for key, data in self.only_numeric(nondimensional_df).items():
            nondimensional_df[key] = self.to_bis(key=key, value=data)

        nondimensional_df["bis"] = True
        return nondimensional_df

    def df_from_bis(self, nondimensional_df):
        df = nondimensional_df.copy()
        for key, data in self.only_numeric(nondimensional_df).items():
            df[key] = self.from_bis(key=key, nondimensional_value=data)

        df["bis"] = False
        return df
