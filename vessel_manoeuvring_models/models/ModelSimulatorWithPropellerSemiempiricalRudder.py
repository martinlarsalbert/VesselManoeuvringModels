import numpy as np
from vessel_manoeuvring_models.models.vmm import ModelSimulatorWithPropeller, Simulator
from vessel_manoeuvring_models.prime_system import PrimeSystem
from vessel_manoeuvring_models.substitute_dynamic_symbols import run
import vessel_manoeuvring_models.models.semiempirical_rudder as semiempirical_rudder
from vessel_manoeuvring_models.symbols import *


class ModelSimulatorWithPropellerSemiempiricalRudder(ModelSimulatorWithPropeller):
    def __init__(
        self,
        simulator: Simulator,
        parameters: dict,
        ship_parameters: dict,
        prime_system: PrimeSystem,
        lambda_thrust,
        control_keys: list = ["delta", "rev"],
        name="simulation",
        primed_parameters=True,
        include_accelerations=True,
    ):
        """Generate a simulator that is specific to one ship with a specific set of parameters.
        This is done by making a copy of an existing simulator object and add freezed parameters.
        Parameters
        ----------
        simulator : Simulator
            Simulator object with predefined odes
        parameters : dict
            [description]
        ship_parameters : dict
            [description]
        control_keys : list
            [description]
        prime_system : PrimeSystem
            [description]
        lambda_thrust
            method that calculates the thrust, based on current state and parameters
        name : str, optional
            [description], by default 'simulation'
        primed_parameters : bool, optional
            [description], by default True
        """
        super().__init__(
            simulator=simulator,
            parameters=parameters,
            ship_parameters=ship_parameters,
            control_keys=control_keys,
            prime_system=prime_system,
            name=name,
            primed_parameters=primed_parameters,
            include_accelerations=include_accelerations,
            lambda_thrust=lambda_thrust,
        )

    def calculate_X_force(
        self,
        inputs: dict,
        parameters: dict,
        ship_parameters: dict,
        states_dict: dict,
        control_: dict,
        U: float,
    ) -> np.ndarray:
        """Method that calculates the quasi static forces in X direction
        This method can be changed in inherrited class to alter the force model.
        """
        X_qs = run(
            function=self.X_qs_lambda,
            **inputs,
            **parameters,
            **ship_parameters,
            **states_dict,
            **control_,
        )
        return X_qs

    def calculate_Y_force(
        self,
        inputs: dict,
        ship_parameters: dict,
        states_dict: dict,
        control_: dict,
        U: float,
    ) -> np.ndarray:
        """Method that calculates the quasi static forces in Y direction
        This method can be changed in inherrited class to alter the force model.
        """

        Y_qs = run(
            function=self.Y_qs_lambda,
            inputs=states_dict,
            **inputs,
            **self.parameters,
            **ship_parameters,
            **control_,
        )  # (prime)

        fy_rudders = self.calculate_rudder_Y_force(
            inputs=inputs,
            states_dict=states_dict,
            control=control_,
            U=U,
        )
        y_force = Y_qs + fy_rudders

        return y_force

    def calculate_rudder_Y_force(
        self,
        inputs: dict,
        states_dict: dict,
        control: dict,
        U: float,
    ) -> np.ndarray:
        ## Semi empirical rudder
        # Propeller induced speed:

        V_x = run(
            function=semiempirical_rudder.lambdas_propeller[semiempirical_rudder.V_x],
            inputs=states_dict,
            **self.ship_parameters,
            **control,
        )
        inputs_SI = self.prime_system.unprime(values=inputs, U=U)

        fy_rudders = run(
            function=semiempirical_rudder.lambdas_lift[Y_R],
            inputs=states_dict,
            V_x=V_x,
            C_L_tune=self.parameters["C_L_tune"],
            delta_lim=self.parameters["delta_lim"],
            kappa=self.parameters["kappa"],
            **inputs_SI,
            **self.ship_parameters,
            **control,
        )  # (SI)
        # fy_rudders = self.prime_system._unprime(value=fy_rudders_SI, unit="force", U=U)
        return fy_rudders

    def calculate_N_force(
        self,
        inputs: dict,
        parameters: dict,
        ship_parameters: dict,
        states_dict: dict,
        control_: dict,
        U: float,
    ) -> np.ndarray:
        """Method that calculates the quasi static forces in N direction
        This method can be changed in inherrited class to alter the force model.
        """
        N_qs = run(
            function=self.N_qs_lambda,
            **inputs,
            **parameters,
            **ship_parameters,
            **states_dict,
            **control_,
        )
        return N_qs
