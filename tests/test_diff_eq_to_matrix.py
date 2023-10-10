from vessel_manoeuvring_models.models.diff_eq_to_matrix import DiffEqToMatrix
import sympy as sp
import pytest
import pandas as pd

y, z, k, l, x_1, x_2, m = sp.symbols("y, z, k, l, x_1, x_2, m")


@pytest.fixture
def eq():
    yield sp.Eq(z, k * x_1)


@pytest.fixture
def eq2():
    yield sp.Eq(z, k * x_1 + m)


@pytest.fixture
def eq3():
    yield sp.Eq(y, k * x_1)


def test_create(eq):
    diff_eq_to_matrix = DiffEqToMatrix(eq, label=y, base_features=[x_1])


def test_create_y_label(eq3):
    with pytest.raises(AssertionError):
        diff_eq_to_matrix = DiffEqToMatrix(eq3, label=y, base_features=[x_1])


def test_X_matrix(eq):
    diff_eq_to_matrix = DiffEqToMatrix(eq, label=y, base_features=[x_1])
    assert diff_eq_to_matrix.X_matrix == sp.matrices.ImmutableDenseMatrix([[x_1]])


def test_coefficients(eq):
    diff_eq_to_matrix = DiffEqToMatrix(eq, label=y, base_features=[x_1])
    assert diff_eq_to_matrix.coefficients == [k, 1]


def test_calculate_features(eq):
    diff_eq_to_matrix = DiffEqToMatrix(eq, label=y, base_features=[x_1])
    data = pd.DataFrame({"x_1": [1]})
    X = diff_eq_to_matrix.calculate_features(data=data)
    assert X["k"].iloc[0] == 1


def test_calculate_features_and_label(eq2):
    diff_eq_to_matrix = DiffEqToMatrix(eq2, label=z, base_features=[x_1, x_2])
    data = pd.DataFrame({"x_1": [1], "z": [2]})
    X, y_ = diff_eq_to_matrix.calculate_features_and_label(data=data, y=data["z"])
    assert (X.values == data[["x_1"]].values).all()
    assert (y_ == data["z"]).all()


def test_exclude_parameters(eq2):
    exclude_parameters = {"m": 1}
    diff_eq_to_matrix = DiffEqToMatrix(
        eq2, label=z, base_features=[x_1], exclude_parameters=exclude_parameters
    )
    data = pd.DataFrame({"x_1": [1], "z": [2]})

    X, y_ = diff_eq_to_matrix.calculate_features_and_label(data=data, y=data["z"])
    assert list(X.columns) == ["k"]
    assert (y_ == data["z"] - exclude_parameters["m"]).all()
    a = 1
