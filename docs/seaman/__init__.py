import os.path
path = os.path.dirname(__file__)

'''Module containing helper functionality'''
import json
from collections import OrderedDict
import numpy as np
import copy


def fix_floats(data):
    """This function iterates though lists and dictionaries and tries to
    convert strings to floats."""
    if isinstance(data, list):
        iterator = enumerate(data)
    elif isinstance(data, dict):
        iterator = data.items()
    else:
        raise TypeError("can only traverse list or dict")

    for i,value in iterator:
        if isinstance(value, (list, dict)):
            fix_floats(value)
        elif isinstance(value, str):
            try:
                data[i] = float(value)
            except ValueError:
                pass
    return data


def read_shipdict(path):
    '''Method returning the shipdict for the provided
    shipfile'
    '''
    shipdict = None
    with open(path) as f:
        shipdict = json.load(f,object_pairs_hook=OrderedDict)
    shipdict = fix_floats(shipdict)
    return shipdict


class ShipDict(OrderedDict):

    @classmethod
    def load(cls,path):
        shipdict = read_shipdict(path = path)
        return cls(shipdict)

    def to_json(self):
        return json.dumps(self,indent=4,sort_keys=True)

    def save(self,path):
        with open(path,'w') as f:
            f.write(self.to_json())

    def copy(self):
        return copy.deepcopy(self)

    #def __getitem__(self, item):
    #   super().__getitem__(item)

    @property
    def lin_hull_coeff_data(self):
        return self['lin_hull_coeff_data']

    @property
    def non_lin_coeff_data(self):
        return self['non_lin_coeff_data']

    @property
    def resistance_data(self):
        return self['resistance_data']

    @property
    def res_data(self):
        return self['res_data']

    @property
    def main_data(self):
        return self['main_data']

    @property
    def design_particulars(self):
        return self['design_particulars']

    @property
    def rudder_coeff_data(self):
        return self['rudder_coeff_data']

    @property
    def rudder_particulars(self):
        return self['rudder_particulars']

    @property
    def wind_data(self):
        return self['wind_data']
        
    @property
    def n_prop(self):
        # Count propellers:
        n = 0
        if 'fix_prop_data' in self:
            n+=len(self['fix_prop_data'])
        
        if 'cp_prop_data' in self:
            n+=len(self['cp_prop_data'])
        
        return n


def set_inputs_from_dict(sys, dikt):
    """ Force try setting all pars/inputs in shipdict"""
    for key, value in dikt.items():
        if not isinstance(value, dict):
            # Try forcing this to float

            if isinstance(value,np.ndarray):
                value = list(value)

            try:
                if isinstance(value, list):
                    value = [float(x) for x in value]
                else:
                    value = float(value)
            except (ValueError, TypeError):

                if isinstance(value,str):
                    value = value
                else:
                    continue

            # Try setting this value as a par
            try:
                setattr(sys.pars, key, value)
            except AttributeError:
                pass

            #Try setting this value as an input
            try:
                setattr(sys.inputs, key, value)
            except AttributeError:
                pass
        else:
            set_inputs_from_dict(sys, value)

class ShipDictError(ValueError): pass
class ShipDictFixPropError(ValueError): pass
class ShipDictCPPropError(ValueError): pass
