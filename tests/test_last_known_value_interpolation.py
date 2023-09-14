from vessel_manoeuvring_models.extended_kalman_vmm import last_known_value_interpolation
from numpy.testing import assert_almost_equal
import numpy as np
import pandas as pd


def test1():
    signal = pd.Series(
        data=[
            1,
        ],
        index=[
            1,
        ],
        name="signal",
    )
    data = pd.DataFrame(index=[1, 2])
    last_known_value_interpolation(signal=signal, data=data)

    desired = [1, 1]
    assert_almost_equal(data["signal"].values, desired)


def test2():
    signal = pd.Series(
        data=[1, 2],
        index=[1, 3],
        name="signal",
    )
    data = pd.DataFrame(index=[1, 2])
    last_known_value_interpolation(signal=signal, data=data)

    desired = [1, 1]
    assert_almost_equal(data["signal"].values, desired)


def test3():
    signal = pd.Series(
        data=[1, 2],
        index=[1, 2],
        name="signal",
    )
    data = pd.DataFrame(index=[1, 2, 3])
    last_known_value_interpolation(signal=signal, data=data)

    desired = [1, 2, 2]
    assert_almost_equal(data["signal"].values, desired)


def test4():
    signal = pd.Series(
        data=[1, 2],
        index=[1, 2],
        name="signal",
    )
    data = pd.DataFrame(index=[0, 1, 2])
    last_known_value_interpolation(signal=signal, data=data)

    desired = [np.NaN, 1, 2]
    assert_almost_equal(data["signal"].values, desired)


def test5():
    signal = pd.Series(
        data=[1, 2],
        index=[1, 2],
        name="signal",
    )
    data = pd.DataFrame(index=[0, 0, 1, 2])
    last_known_value_interpolation(signal=signal, data=data)

    desired = [np.NaN, np.NaN, 1, 2]
    assert_almost_equal(data["signal"].values, desired)


def test6():
    signal = pd.Series(
        data=[1, 2],
        index=[10, 20],
        name="signal",
    )
    data = pd.DataFrame(index=[10, 20])
    last_known_value_interpolation(signal=signal, data=data)

    desired = [1, 2]
    assert_almost_equal(data["signal"].values, desired)


def test6():
    signal = pd.Series(
        data=[1, 2],
        index=[10, 20],
        name="signal",
    )
    data = pd.DataFrame(index=[10, 15, 20])
    last_known_value_interpolation(signal=signal, data=data)

    desired = [1, 1, 2]
    assert_almost_equal(data["signal"].values, desired)
