import nbformat

# 1. Import the exporter
from nbconvert import HTMLExporter
import nbconvert.preprocessors
from traitlets.config import Config
import os
import shutil

class ChangeIbynbLink(nbconvert.preprocessors.Preprocessor):

    def preprocess_cell(self,cell, resources, index):

        if hasattr(cell,'outputs'):
            for i,output in enumerate(cell.outputs):
                if 'text/plain' in output.data:
                    cell.outputs[i].data['text/plain'] = cell.outputs[i].data['text/plain'].replace('.ibynb','.html')

        return cell, resources


def convert_notebook_to_html(notebook_path,html_path):

    notebook_filename = notebook_path
    with open(notebook_filename,encoding="utf8") as f:
        nb = nbformat.read(f, as_version=4)

    
    c = Config()
    #c.HTMLExporter.preprocessors = [ChangeIbynbLink]

    # 2. Instantiate the exporter. We use the `basic` template for now; we'll get into more details
    # later about how to customize the exporter further.
    html_exporter = HTMLExporter(config = c)
    path = os.path.dirname(__file__)
    html_exporter.template_paths.append(path)
    html_exporter.template_file = 'hidecode.tplx'
    
    # 3. Process the notebook we loaded earlier
    (body, resources) = html_exporter.from_notebook_node(nb)

    with open(html_path,'w',encoding='utf8') as file:
        file.write(body)

def find_notebooks(path = ''):

    notebook_paths = []

    file_names = os.listdir(path = os.path.abspath(path))
    for file_name in file_names:
        if os.path.splitext(file_name)[-1] == '.ipynb':
            notebook_paths.append(os.path.join(path,file_name))

    return notebook_paths

def find_figures(path = ''):

    figure_paths = []

    file_names = os.listdir(path = os.path.abspath(path))
    for file_name in file_names:
        ext = os.path.splitext(file_name)[-1]
        if  ext == '.png' or ext == 'jpg':

            figure_paths.append(os.path.join(path,file_name))

    return figure_paths

def convert_notebooks(build_path = 'html',path = None):

    if path is None:
        path = os.path.dirname(__file__)

    notebook_paths = find_notebooks(path = path)
    figure_paths = find_figures(path = path)

    if not os.path.exists(build_path):
        os.mkdir(build_path)

    for notebook_path in notebook_paths:
        path,file_name = os.path.split(notebook_path)
        base_name = os.path.splitext(file_name)[0]
        html_file_name = '%s.html' % base_name
        html_file_path = os.path.join(build_path,html_file_name)

        convert_notebook_to_html(notebook_path=notebook_path,html_path=html_file_path)

    for figure_path in figure_paths:
        base_path,file_name = os.path.split(figure_path)
        new_path = os.path.join(build_path,file_name)
        shutil.copy(figure_path,new_path)

if __name__ == "__main__":

    convert_notebooks()



