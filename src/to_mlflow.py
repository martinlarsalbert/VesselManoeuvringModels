import notebooks
import os.path
tracking_uri = r'file:///' + os.path.join(os.path.dirname(notebooks.__file__),'model_simulate','mlruns')
import mlflow
mlflow.set_tracking_uri(tracking_uri)