"""Convert notebook to LaTeX
"""

import nbformat

# 1. Import the exporter
from nbconvert import LatexExporter
from nbconvert.writers import FilesWriter
import nbconvert.preprocessors.extractoutput
from nbconvert.preprocessors import TagRemovePreprocessor, Preprocessor
import vessel_manoeuvring_models.bibpreprocessor
from vessel_manoeuvring_models.itemize_preprocessor import ItemizePreprocessor
from vessel_manoeuvring_models.quad_preprocessor import QuadPreprocessor
from traitlets.config import Config
import os
import shutil
import vessel_manoeuvring_models
import re
from collections import OrderedDict



# Import the RST exproter
from nbconvert import RSTExporter

class FigureRenameError(Exception): pass

def convert_notebook_to_latex(notebook_path:str, build_directory:str, save_main=True, skip_figures=False):

    assert os.path.exists(notebook_path)

    _,notebook_file_name = os.path.split(notebook_path)
    notebook_name,_ = os.path.splitext(notebook_file_name)

    notebook_filename = notebook_path
    with open(notebook_filename,encoding="utf8") as f:
        nb = nbformat.read(f, as_version=4)

    #path = os.path.split(os.path.abspath(__file__))[0]
    c = Config()
    # Employ nbconvert.writers.FilesWriter to write the markdown file 
    
    if not os.path.exists(build_directory):
        os.mkdir(build_directory)

    figure_directory = os.path.join(build_directory,'figures')
    if not os.path.exists(figure_directory):
        os.mkdir(figure_directory)

    c.FilesWriter.build_directory = figure_directory  # Only figures!!

    # Configure our tag removal
    c.TagRemovePreprocessor.remove_cell_tags = ("remove_cell",)
    c.TagRemovePreprocessor.remove_all_outputs_tags = ('remove_output',)
    c.TagRemovePreprocessor.remove_input_tags = ('remove_input',)
    #c.LatexExporter.preprocessors = [TagRemovePreprocessor,'vessel_manoeuvring_models.bibpreprocessor.BibTexPreprocessor']
    #c.LatexExporter.preprocessors = [TagRemovePreprocessor, FigureName, vessel_manoeuvring_models.bibpreprocessor.BibTexPreprocessor, ItemizePreprocessor, QuadPreprocessor]
    c.LatexExporter.preprocessors = [TagRemovePreprocessor, FigureName, vessel_manoeuvring_models.bibpreprocessor.BibTexPreprocessor, ItemizePreprocessor]


    # 2. Instantiate the exporter. We use the `basic` template for now; we'll get into more details
    # later about how to customize the exporter further.
    latex_exporter = LatexExporter(config = c)
    latex_exporter.template_file = os.path.join(vessel_manoeuvring_models.path,'mytemplate.tplx')
    latex_exporter.exclude_input=True  # No input cells.
    latex_exporter.exclude_input_prompt=True  # No cell numbering
    latex_exporter.exclude_output_prompt=True  # No cell numbering
    
    # 3. Process the notebook we loaded earlier
    (body, resources) = latex_exporter.from_notebook_node(nb)

    FilesWriter(body=body,resources=resources)
    
    fw = FilesWriter(config=c, input=False)
    
    if not skip_figures:
        fw.write(body, resources, notebook_name=notebook_name)
    else:
        fw.write(body, {}, notebook_name=notebook_name)

    # Creata a tree structure instead:
    tree_writer(body=body, build_directory=build_directory, save_main=save_main)


class FigureName(Preprocessor):
    """Give names to figures"""

    def preprocess(self, nb, resources):
        #self.log.info("I'll keep only cells from %d to %d", self.start, self.end)
        #nb.cells = nb.cells[self.start:self.end]
        
        for cell_id, cell in enumerate(nb['cells']):

            meta_data = cell['metadata']
            if 'name' in meta_data:

                if 'outputs' in cell:
                    outputs = cell['outputs']
                    output_id = 0
                    for output_id, output in enumerate(outputs):                    
                        output = outputs[output_id]
                        output_meta_data = output.get('metadata',None)
                        if output_meta_data is None:
                            continue

                        filenames = output_meta_data['filenames']

                        output_name = 'output_%i_%i' % (cell_id, output_id)
                        for key, value in filenames.items():
                            if output_name in value:

                                # Rename to "figure":
                                new_figure_name = value.replace(output_name, meta_data['name'])

                                # Meta data:
                                nb['cells'][cell_id]['outputs'][output_id]['metadata']['filenames'][key] = new_figure_name       

                                # resources:
                                resources['outputs'][new_figure_name] = resources['outputs'].pop(value)


        
        return nb, resources


def tree_writer(body:str, build_directory:str, save_main=True):
    """Splitting the generated LaTex document into sub files:
    
    * main.tex
        * section 1
        * section 2
        *...    

    Parameters
    ----------
    body : str
        LaTeX text

    build_directory : str
        Where should the tree be placed

    save_main : bool
        generate a main.tex with all subsections.
    """

    # Last minute LaTeX changes.
    body = latex_cleaner(body)
    body = change_figure_paths(body=body, build_directory=build_directory)
    body = change_inline_equations(body=body)
    body = anchor_links(body=body)
    body = anchor_links_section(body=body)
    body = clean_figure_warning(body=body)
    body = ole_ole_ole(body=body)
    body = star_sub_sections(body=body)
    body = citep(body=body)
    body = remove_hypertarget(body=body)
    
    pre, document, end = split_parts(body=body)
    sections = splitter_section(document=document)

    main = '%s\n\\begin{document}\n\maketitle\n' % pre
    
    for section_name, section_ in sections.items():
        
        section = capital_section(body=section_)  

        section = remove_whitespaces(body=section)

        # Create the section file:
        section_file_name = '%s.tex' % section_name
        section_file_path = os.path.join(build_directory, section_file_name)
        with open(section_file_path, mode='w') as file:
            file.write(section)

        # Make a ref in the main file:
        ref = r'\input{%s}' % section_name
        ref+='\n'

        main+=ref
        
    main+='\n\\end{document}%s' % end

    main_file_name = 'main.tex'
    main_file_path = os.path.join(build_directory, main_file_name)
    if save_main:
        with open(main_file_path, mode='w') as file:
            file.write(main)

def latex_cleaner(body:str):

    ## Clean equation:
    body = re.sub(r'\$\\displaystyle\W*\\begin{equation}',r'\\begin{equation}', body)
    body = re.sub(r'\\end{equation}\W*\$',r'\\end{equation}', body)

    ## Clean links:
    body = clean_links(body=body)
    
    return body

def clean_links(body:str):
    
    """Cleaning something like:
    \href{../../notebooks/01.3_select_suitable_MDL_test_KLVCC2_speed.ipynb\#yawrate}{yawrate}

    Returns
    -------
    [type]
        [description]
    """
    
    return re.sub(r"\\href\{.*.ipynb[^}]*}{[^}]+}\n*",'',body)

def anchor_links(body:str, prefixes = ['eq','fig','tab'], ref_tag='ref'):
    """
    Replace:
    Section \ref{eq_linear}
    With:
    \ref{eq:linear}

    Parameters
    ----------
    body : str
        [description]
    """
    for prefix in prefixes:

        pattern1 = r'Section \\ref\{%s_([^}]+)' % prefix
        
        for result in re.finditer(pattern1, body):
            name = result.group(1)
            eq_label = '%s:%s' % (prefix,name)
            pattern2 = r'Section \\ref\{%s_%s\}' % (prefix,name)
            body = re.sub(pattern2,
                r'\\%s{%s}' % (ref_tag, eq_label),
                body
            )

    return body

def anchor_links_section(body:str, prefixes = ['se'], ref_tag='nameref'):
    """
    Replace:
    Section \ref{eq_linear}
    With:
    \ref{eq:linear}

    Parameters
    ----------
    body : str
        [description]
    """
    for prefix in prefixes:

        pattern1 = r'Section \\ref\{%s_([^}]+)' % prefix
        
        for result in re.finditer(pattern1, body):
            name = result.group(1)
            eq_label = '%s:%s' % (prefix,name)
            pattern2 = r'Section \\ref\{%s_%s\}' % (prefix,name)
            body = re.sub(pattern2,
                r'"\\%s{%s}"' % (ref_tag, eq_label),
                body
            )

    return body

def change_figure_paths(body:str, build_directory:str, figure_directory_name='figures'):
    """The figures are now in a subfolder, 
    so the paths need to be changed.

    Parameters
    ----------
    body : str
    """
    figure_directory = os.path.join(build_directory,figure_directory_name)
    for file in os.listdir(figure_directory):
        _,ext = os.path.splitext(file)
        if ext=='.pdf' or ext=='.png':
            new_path =r'%s/%s' % (figure_directory_name,file)
            body = body.replace(file, new_path)

    return body

def change_inline_equations(body:str):
    """Modifying inline latex equations, ex:
    \(B_{BK}\) 
    to
    $B_{BK}$ 

    Parameters
    ----------
    body : str
        """

    body = re.sub(r'\\\(','$', body)
    body = re.sub(r'\\\)','$', body)
    return body

def clean_figure_warning(body:str):
    
    s=r"""findfont: Font family ['"serif"'] not found. Falling back to DejaVu Sans."""
    
    body = body.replace(s,'')

    s2 = r"""\begin{Verbatim}[commandchars=\\\{\}]

    \end{Verbatim}
"""    
    body = body.replace(s2,'')

    return body

def ole_ole_ole(body:str):
    """replacing é with \'e

    """

    body = body.replace(r'é',r"\'e")

    return body

def capital_section(body:str):
    """
    Change:
    \section{Abstract}
    To:
    \section*{ABSTRACT}
    """
    pattern = r'\\section\{([^}]+)'
    for title in re.findall(pattern=pattern, string=body):
        TITLE = title.upper()
        pattern2 = r'\\section\{%s\}' % title
        repl = r'\\section*{%s}' % TITLE
        body = re.sub(pattern2, repl=repl, string=body)

    return body

def star_sub_sections(body:str):
    """
    Change
    \subsection
    \subsubsection
    To:
    \subsection*
    \subsubsection*
    """
    body = body.replace(r'\subsection',r'\subsection*')
    body = body.replace(r'\subsubsection',r'\subsubsection*')
    return body

def citep(body:str):
    """
    Change
    \cite
    to
    \citep
    """
    body = body.replace(r'\cite',r'\citep')
    return body

def remove_whitespaces(body:str):
    body = re.sub(r' +\n','\n', body)
    body = re.sub(r'\n[ \n]+','\n', body)
    
    return body

def split_parts(body:str):
    """Split into:
    * Pre
    * Document
    * End

    Parameters
    ----------
    body : [type]
        [description]
    """
    
    

    p = re.split(pattern= r'\\begin{document}', string=body)
    pre = p[0]
    rest = p[1]

    p2 = re.split(pattern= r'\\end{document}', string=rest)
    document = p2[0]
    end = p2[1]

    return pre, document, end

def remove_hypertarget(body:str):
    """
    Remove something like:
    \hypertarget{analysis}{%
    \section{Analysis}\label{analysis}}

    to:
    \section{Analysis}\label{analysis}

    """

    #results = re.findall(r'\\section{[^}]+}\\label{[^}]+}', body)
    #for result in results:
    #    repl = re.sub(r'\\section', r'\\\\section', result)
    #    repl = re.sub(r'\\label', r'\\\\label', repl)
    #            
    #    body = re.sub(r'\\hypertarget{[^}]+}{%\n*' + repl + r'\}', repl=repl, string=body)
    
    #for result in re.finditer(r'\\hypertarget{[^}]+}{%\n*([^}]+})}', body):
    #    body = body.replace(result.group(0), result.group(1))

    for result in re.finditer(r'\\hypertarget{[^}]+}{%\n*([^}]+})}', body):
        body = body.replace(result.group(0), result.group(1))
    
    for result in re.finditer(r'\\hypertarget{[^}]+}{%\n*([^}]+}\\label\{[^}]+})}', body):
        body = body.replace(result.group(0), result.group(1))

    return body



def splitter_section(document):

    
    parts = re.split(pattern= r'\\section', string=document)
    # parts = re.split(pattern= r'\\hypertarget{[^}]+}{%\n*\\section{[^}]+}\\label{[^}]+}}', string=document)

    sections = OrderedDict()

    for part in parts[1:]:
        result = re.match(pattern=r'^\{([^}]+)', string=part)
        if not result:
            raise ValueError('Could not find the section name in:%s' % part)
        
        text = result.group(1)
        ps = re.findall(r'[\w]+', text)
        section_name = ps[0]
        if len(ps)>1:
            for p_ in ps[1:]:
                section_name+=' %s' % p_

        section_name = section_name.replace(' ','_')
        section_name = section_name.lower()


        section = '\\section%s' % part
        sections[section_name] = section

    return sections




