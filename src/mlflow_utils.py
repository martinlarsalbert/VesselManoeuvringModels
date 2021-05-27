from typing import Dict, Any, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import recall_score, precision_score, roc_auc_score, roc_curve, precision_recall_curve, f1_score
import os

from logger import logger


def track_plot(df:pd.DataFrame, df_sim:pd.DataFrame, run_name:str, plot_dir: str = None):

    fig,ax=plt.subplots()
    ax.axis('equal')
    df.plot(x='y0', y='x0', ax=ax, label='MDL')
    df_sim.plot(x='y0', y='x0', style='--', ax=ax, label='SIM')
    ax.set_xlabel('y [m]')
    ax.set_ylabel('x [m]')
    ax.legend()    

    # Save figure
    if plot_dir:
        output_path = '{}/track_plot_{}.png'.format(plot_dir, run_name)
        plt.savefig(output_path)
        logger.info('track plot saved to: {}'.format(output_path))
        return output_path 

def plot(key:str, df:pd.DataFrame, df_sim:pd.DataFrame, run_name:str, units:dict, plot_dir: str = None):

    fig,ax=plt.subplots()

    df.plot(y=key, ax=ax, label='MDL')
    df_sim.plot(y=key, style='--', ax=ax, label='SIM')
    ax.legend()    
    ax.grid(True)
    ax.set_xlabel('time')
    
    unit = units.get(key,'')
    ax.set_ylabel(f'{key} [{unit}]')

    ylims_ = ax.get_ylim()
    ylims = list(ylims_)
    if (ylims_[0] > 0) and (ylims_[1] > 0):
        ylims[0] = 0
    elif (ylims_[0] < 0) and (ylims_[1] < 0):   
        ylims[1] = 0

    ax.set_ylim(ylims)

    # Save figure
    if plot_dir:
        output_path = f'{plot_dir}/{key}_plot_{run_name}.png'
        plt.savefig(output_path)
        logger.info(f'{key} plot saved to: {output_path}')
        return output_path 
       

def plot_roc(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, plot_dir: str = None) -> str:
    """
    Plot the area under curve for the ROC curve.
    Args:
        y_true: Array of true y values
        y_pred: Array of predicted y values
        model_name: Name of model
        plot_dir: Directory to save plot in
    Returns:
        Output path of plot
    """
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred)
    plt.figure(figsize=(5, 5))
    plt.grid()
    plt.plot(fpr, tpr, color='b')
    plt.title('ROC curve: {}'.format(model_name))

    # Save figure
    if plot_dir:
        output_path = '{}/plots/roc_curve_{}.png'.format(plot_dir, model_name)
        plt.savefig(output_path)
        logger.info('ROC curve saved to: {}'.format(output_path))
        return output_path


def plot_precision_recall(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, plot_dir: str = None) -> str:
    """
    Plots the precision-recall curve.
    Args:
        y_true: Array of true y values
        y_pred: Array of predicted y values
        model_name: Name of model
        plot_dir: Directory to save plot in
    Returns:Ø
        Output path of plot
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    plt.figure(figsize=(15, 5))
    plt.grid()
    plt.plot(thresholds, precision[1:], color='r', label='Precision')
    plt.plot(thresholds, recall[1:], color='b', label='Recall')
    plt.gca().invert_xaxis()
    plt.legend()
    plt.title('Precision-recall curve: {}'.format(model_name))

    # Save figure
    if plot_dir:
        output_path = '{}/plots/precision_recall_{}.png'.format(plot_dir, model_name)
        plt.savefig(output_path)
        logger.info('Precision-recall curve saved to: {}'.format(output_path))
        return output_path


def save_model(model: Any, model_name: str, model_dir: str) -> str:
    """
    Saves model in pickle format
    Args:
        model: Model binary
        model_name: Name of model
        model_dir: Directory to save model in
    Returns:
        Output path of model
    """
    output_path = '{}/models/{}.pickle'.format(model_dir, model_name)
    logger.info('Model saved to: {}'.format(output_path))
    dump(model, output_path)

    return output_path


def evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> \
        Tuple[float, float, float, float]:
    """
    Returns binary evaluation metrics
    Args:
        y_true: Array of true y values
        y_pred: Array of predicted y values
        threshold: Threshold to convert probabilies to binary values (default=0.5)
    Returns:
        Metrics for AUC, recall, precision, and F1
    """
    y_pred_thresholded = np.where(y_pred > threshold, 1, 0)

    auc = roc_auc_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred_thresholded)
    precision = precision_score(y_true, y_pred_thresholded)
    f1 = f1_score(y_true, y_pred_thresholded)

    logger.info('AUC: {:.3f} | Recall: {:.3f} | Precision: {:.3f} | F1: {:.3f}'.format(auc, recall, precision, f1))
    return auc, recall, precision, f1

def evaluate(df:pd.DataFrame, df_sim:pd.DataFrame):

    metrics = {}

    columns_pos = ['x0','y0']
    distance = np.sqrt(((df_sim[columns_pos] - df[columns_pos])**2).sum(axis=1))
    distance_norm = distance/np.sqrt(((df[columns_pos])**2).sum(axis=1))
    metrics['distance_rmse'] = rmse = distance_norm.mean()

    psi = df['psi']
    psi_sim = df_sim['psi']
    dx = np.cos(psi_sim) - np.cos(psi) 
    dy = np.sin(psi_sim) - np.sin(psi)
    course_diff = np.arctan2(dy,dx) 
    metrics['psi_rmse'] = np.mean(np.sqrt((course_diff**2)))

    return metrics

def log_mlflow(run_params: Dict, df:pd.DataFrame, df_sim:pd.DataFrame, units:dict) -> None:
    """
    Logs result of model training and validation to mlflow
    Args:
        run_params: Dictionary containing parameters of run.
                    Expects keys for 'experiment', 'artifact_dir', 'iteration', and 'index.
        df : true model test
        df_sim : simulation
    Returns:
        None
    """
    mlflow.set_experiment(run_params['experiment'])

    #auc, recall, precision, f1 = evaluate_binary(y_true, y_pred)

    #roc_path = plot_roc(y_true, y_pred, '{} (auc = {:.2f})'.format(model_name, auc), run_params['artifact_dir'])
    #pr_path = plot_precision_recall(y_true, y_pred,
    #                                '{} (prec: {:.2f}, recall: {:.2f})'.format(model_name, precision, recall),
    #                                run_params['artifact_dir'])
    #model_path = save_model(model, model_name, run_params['artifact_dir'])

    plot_dir = os.path.join('figures','twin_sim')
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    plot_keys = ['psi',
                 'V',
                 'beta',
                 'delta']

    with mlflow.start_run(run_name=run_params['iteration']):
        
        for key,value in run_params.items():
            mlflow.log_param(key, value)

        mlflow.log_artifact(track_plot(df=df, df_sim=df_sim, run_name=run_params['id'], plot_dir=plot_dir))
                
        
        for key in plot_keys:
            mlflow.log_artifact(plot(key=key, df=df, df_sim=df_sim, run_name=run_params['id'], plot_dir=plot_dir, units=units))
        
        metrics = evaluate(df=df, df_sim=df_sim)
        for key,value in metrics.items():
            mlflow.log_metric(key, value)


        #mlflow.log_param('index', run_params['index'])
        #mlflow.log_param('model', model_name)
        #mlflow.log_metric('auc', auc)
        #mlflow.log_metric('recall', recall)
        #mlflow.log_metric('precision', precision)
        #mlflow.log_metric('f1', f1)
        #mlflow.log_artifact(model_path)
        #mlflow.log_artifact(roc_path)
        #mlflow.log_artifact(pr_path)