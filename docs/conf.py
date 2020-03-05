import sys
import os
from setuptools_scm import get_version
from unittest.mock import MagicMock as Mock

# Mocking to make RTD autobuild the documentation.
autodoc_mock_imports = ['numpy']
#sys.modules.update([("numpy", Mock())])
sys.path.insert(0,os.path.abspath("../chspy"))
sys.path.insert(0,os.path.abspath("../examples"))

needs_sphinx = '1.6'

extensions = [
	'sphinx.ext.autodoc',
	'sphinx.ext.autosummary',
	'sphinx.ext.mathjax',
	'matplotlib.sphinxext.plot_directive',
	'numpydoc',
]

source_suffix = '.rst'

master_doc = 'index'

project = 'CHSPy'
copyright = '2019, Gerrit Ansmann'

release = version = get_version(root='..', relative_to=__file__)

default_role = "any"

add_function_parentheses = True

add_module_names = False

html_theme = 'nature'
pygments_style = 'colorful'
htmlhelp_basename = 'CHSPydoc'

plot_html_show_formats = False
plot_html_show_source_link = False

numpydoc_show_class_members = False
autodoc_member_order = 'bysource'

def on_missing_reference(app, env, node, contnode):
	if node['reftype'] == 'any':
		return contnode
	else:
		return None

def setup(app):
	app.connect('missing-reference', on_missing_reference)
