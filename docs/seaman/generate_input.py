import inspect
import pandas as pd

def list_parameters(lambda_function):
    sig = inspect.signature(lambda_function)
    parameters = list(sig.parameters.keys())
    return parameters

def figure_out_ship_dict_names(parameters):
    shipdict_names = {}

    for variable in parameters:
        ship_dict_name = variable.lower()
        ship_dict_name = ship_dict_name.replace('_', '')
        ship_dict_name = ship_dict_name.replace('delta', 'd')
        ship_dict_name = ship_dict_name.replace('T', 't')

        shipdict_names[ship_dict_name] = variable

    #shipdict_names['disp'] = 'volume'

    return shipdict_names

def load_ship_dict_data(shipdict):

    shipdict_data = dict()
    shipdict_data.update(shipdict.main_data)
    shipdict_data.update(shipdict.design_particulars)
    shipdict_data.update(shipdict.lin_hull_coeff_data)
    shipdict_data.update(shipdict.non_lin_coeff_data)
    shipdict_data.update(shipdict.rudder_coeff_data)
    shipdict_data.update(shipdict.resistance_data)
    shipdict_data.update(shipdict.rudder_particulars[0])

    shipdict_data = pd.Series(shipdict_data)

    shipdict_data['nrud'] = len(shipdict.rudder_particulars)

    return shipdict_data

def get_inputs(lambda_function,shipdict):

    parameters = list_parameters(lambda_function=lambda_function)
    shipdict_names = figure_out_ship_dict_names(parameters=parameters)

    shipdict_data = load_ship_dict_data(shipdict=shipdict)

    interesting = list(set(shipdict_names) & set(shipdict_data.keys()))
    inputs = shipdict_data[interesting]
    return inputs,shipdict_names


def add_shipdict_inputs(lambda_function,shipdict,df):

    df=df.copy()


    inputs,shipdict_names = get_inputs(lambda_function=lambda_function,shipdict=shipdict)

    data = [inputs.values]
    index = df.index
    columns = inputs.index

    shipdict_stuff = pd.DataFrame(data=data, columns=columns, index=index)
    assert shipdict_stuff.columns.is_unique

    shipdict_stuff_rename = shipdict_stuff.rename(columns=shipdict_names)
    assert shipdict_stuff_rename.columns.is_unique
    assert df.columns.is_unique

    df_input = shipdict_stuff_rename.combine_first(df)
    assert df_input.columns.is_unique

    parameters = list_parameters(lambda_function=lambda_function)

    df_input_cut = df_input[parameters]
    assert df_input_cut.columns.is_unique

    return df_input_cut



