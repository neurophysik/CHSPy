#!/usr/bin/python3

from setuptools import setup
from io import open

setup(
		name = 'CHSPy',
		description = 'Cubic Hermite splines',
		long_description = open('README.md', encoding='utf8').read(),
		python_requires=">=3.6",
		packages = ['chspy'],
		install_requires = ['numpy'],
		setup_requires = ['setuptools_scm'],
		tests_require = ['symengine'],
		use_scm_version = {'write_to': 'chspy/version.py'},
		classifiers = [
				'License :: OSI Approved :: BSD License',
				'Operating System :: POSIX',
				'Operating System :: MacOS :: MacOS X',
				'Operating System :: Microsoft :: Windows',
				'Programming Language :: Python',
				'Topic :: Scientific/Engineering',
			],
	)

