from setuptools import find_packages, setup
import os
import subprocess
import sys
from setuptools.command.install import install as _install

class install(_install):
    def run(self):
        _install.run(self)

        ## Install nbstripout:
        print('installing nbstripout')
        if os.name == 'nt':
            print('Windows...')
            ## Windows:
            subprocess.run(['nbstripout', '--install', '--system'])
        else:
            ## Linux/OS X:
            print('Linux...')
            subprocess.run(['sudo', 'nbstripout', '--install', '--system'])

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def run(self):
            raise SystemExit('this is fine')
except ImportError:
    bdist_wheel = None

setup(
    name='vessel_manoeuvring_models',
    packages=find_packages(),
    version='0.1.0',
    description='open stuff for the wPCC project',
    author='Martin Alexandersson',
    license='MIT',
    install_requires=[

        'Sphinx',
        'coverage',
        'python-dotenv>=0.5.1',
        'pytest',
        'nbstripout',
        'numpy',
        'pandas',
        'matplotlib',
        'sklearn',
        'seaborn',
        'scipy',
        'jupyterlab',
        'sympy',
        'plotly',
        'ipywidgets>=7.5',
        'jupyterlab-mathjax3',
        'statsmodels',
        'dill',

    ],
    #cmdclass={'bdist_wheel': bdist_wheel, 'install': install},
    cmdclass={'bdist_wheel': bdist_wheel}
)
