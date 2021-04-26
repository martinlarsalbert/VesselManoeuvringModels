"""
This module contain methods to save the developed Sympy lambda function to python files and matlab filed.
"""

import inspect
import os.path

import git
from datetime import datetime

def get_git_hash():
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return sha

def get_source(lambda_function,function_name):
    lines = inspect.getsource(lambda_function)
    lines = lines.replace('_lambdifygenerated',function_name)
    return lines

def save_lambda_to_python_file(lambda_function,function_name,save_dir = ''):

    save_name = '%s.py' % function_name
    save_path = os.path.join(save_dir,save_name)

    lines = get_source(lambda_function=lambda_function,function_name=function_name)

    lines = 'from numpy import *\n%s' % lines

    with open(save_path,mode = 'w') as file:
        file.writelines(lines)

def python_function_source_to_matlab(source):

    sha = get_git_hash()
    version_string = """
%% This Matlab function was automatically generated from the Seaman mathematical model documentation using 
%% Python package Sympy.
%% date:%s
%% version GIT hash:%s
""" % (datetime.now(),sha)

    source = source.replace(':', '')
    source = source.replace('return', 'result =')
    source = source.replace('def', 'function result =')
    source = source.replace('**', '.^')
    source = source.replace('*', '.*')
    source = source.replace(r'/', r'./')
    source = source.replace('disp', 'displ')

    source = '%s\n%s;' % (version_string,source[0:-1])
    return source

def save_lambda_to_matlab_file(lambda_function,function_name,save_dir = ''):

    save_name = '%s.m' % function_name
    save_path = os.path.join(save_dir,save_name)

    source = get_source(lambda_function=lambda_function,function_name=function_name)
    matlab_source = python_function_source_to_matlab(source=source)

    with open(save_path,mode = 'w') as file:
        file.writelines(matlab_source)
