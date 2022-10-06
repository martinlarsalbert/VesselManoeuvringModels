import papermill as pm
import os.path
from multiprocessing import Pool
import vessel_manoeuvring_models.data.mdl
import data
import models

template_dir = os.path.join(os.path.dirname(__file__),'templates')

simulation_dir = os.path.join(os.path.dirname(__file__),'simulations')
if not os.path.exists(simulation_dir):
    os.mkdir(simulation_dir)

def setup_run_params():
    
    run_params_template = {
        'experiment' : 'VCT_linear',
        'model_test_id' : 22770,
        'model_test_dir_path' : os.path.join(os.path.dirname(data.__file__),'processed','kalman'),
        'model' : os.path.join(os.path.dirname(models.__file__),'model_VCT_linear.pkl'),
        'run_name' : '22770',
    }

    experiments = {
        'VCT_linear' : os.path.join(os.path.dirname(models.__file__),'model_VCT_linear.pkl'),
        'VCT_abkowitz' : os.path.join(os.path.dirname(models.__file__),'model_VCT_abkowitz.pkl'),
    }
    
    model_test_ids = list(vessel_manoeuvring_models.data.mdl.runs().index)
    #model_test_ids = [22770]

    run_params_all = []
    for experiment, model in experiments.items():
        for model_test_id in model_test_ids:

            run_params = run_params_template.copy()
            run_params['experiment'] = experiment
            run_params['model'] = model
            run_params['model_test_id'] = model_test_id
            run_params['run_name'] = str(model_test_id)
            run_params_all.append(run_params)

    return run_params_all

def run_papermill(run_params):

    try:
        _run_papermill(run_params)
    except:
        return

def _run_papermill(run_params):
       
    parameters = {
        'run_params' : run_params,
    }
    
    input_notebook_name = '16.03_model_simulate'
    input_path = os.path.join(template_dir, f'{input_notebook_name}.ipynb')
    output_notebook_name = f'{run_params["experiment"]}_{run_params["run_name"]}'
    output_path = os.path.join(simulation_dir, f'{output_notebook_name}.ipynb')
    print(output_notebook_name)

    if os.path.exists(output_path):
        print('Skipping...')
        return
    
    pm.execute_notebook(input_path=input_path, 
                        output_path=output_path,
                        parameters=parameters,
                        cwd=simulation_dir, execution_timeout=100);

if __name__=='__main__':

    run_params_all = setup_run_params()

    in_parallell = True
    if in_parallell:
        p = Pool(5)
        with p:
            p.map(run_papermill, run_params_all)
    else:
        for run_params in run_params_all:
            run_papermill(run_params=run_params)
