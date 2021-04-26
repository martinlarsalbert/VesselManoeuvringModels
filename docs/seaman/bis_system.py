import sympy as sp
import seaman_symbol as ss

class BisSystem():
    """
    This class has the mathematical definition of the bis-system
    """

    quantities = {}
    quantities['non_dimensional'] = 1
    quantities['mass'] = 'kg'
    quantities['length'] = 'm'
    quantities['area'] = 'm2'
    quantities['volume'] = 'm3'
    quantities['density'] = 'kg/m3'
    quantities['time'] = 's'
    quantities['hz'] = '1/s'
    quantities['linear_velocity'] = 'm/s'
    quantities['linear_acceleration'] = 'm/s2'
    quantities['angle'] = 'rad'
    quantities['angular_velocity'] = 'rad/s'
    quantities['angular_acceleration'] = 'rad/s2'
    quantities['force'] = 'N'
    quantities['moment'] = 'Nm'


    rho_s = ss.Symbol('rho', 'Water density', unit='kg/m3')
    g_s = ss.Symbol('g', 'gravity', unit='m/s2')
    L_s = ss.Symbol('L', 'Perpendicular length', unit='m')
    m_s = ss.Symbol('m', 'Ship mass', unit='kg')
    volume_s = ss.Symbol('disp', 'Ship volume', unit='m3')

    non_dimensional = 1
    mass = rho_s * volume_s
    length = L_s
    area = L_s ** 2
    volume = L_s ** 3
    density = mass / volume
    time = sp.sqrt(L_s / g_s)
    hz = 1 / time
    linear_velocity = sp.sqrt(L_s * g_s)
    linear_acceleration = g_s
    angle = 1
    angular_velocity = sp.sqrt(g_s / L_s)
    angular_acceleration = g_s / L_s
    force = rho_s * volume_s * g_s
    moment = rho_s * volume_s * g_s * L_s

    @classmethod
    def html_table(cls):

        html = """
            <tr>
                <th>Quantity</th>
                <th>Denominator</th> 
                <th>SI Unit</th> 
            </tr>
            """
        skip = ('rho_s', 'g_s', 'L_s', 'm_s', 'volume_s')

        for name, denominator in cls.__dict__.items():

            if name in skip:
                continue

            if not isinstance(denominator,sp.Basic):
                continue

            unit = cls.quantities[name]

            html_row = """
            <tr>
                <td>%s</td>
                <td>$%s$</td> 
                <td>%s</td> 
            </tr>
            """ % (name, sp.latex(denominator), unit)

            html += html_row

        html_table = """
        <table>
        %s
        </table>
        """ % html

        return html_table

