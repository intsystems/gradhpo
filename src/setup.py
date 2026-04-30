"""Setup script for the gradhpo package.

Layout: this file lives in src/, the importable package is src/gradhpo.
Build with ``python -m pip install ./src`` from the repository root, or
``python -m pip install .`` from inside src/.
"""

import io
import os
import re

from setuptools import find_packages, setup


HERE = os.path.abspath(os.path.dirname(__file__))


def read(file_path):
    """Read a UTF-8 text file relative to this setup.py."""
    with io.open(os.path.join(HERE, file_path), 'r', encoding='utf-8') as f:
        return f.read()


def get_version():
    """Read __version__ from gradhpo/__init__.py without importing the package."""
    content = read('gradhpo/__init__.py')
    match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", content, re.MULTILINE)
    if match:
        return match.group(1)
    raise RuntimeError("Unable to find __version__ in gradhpo/__init__.py")


readme = read('README.rst')
# Strip local version specifiers from requirements file (per PEP 440)
requirements = '\n'.join(
    re.findall(r'^([^\s^+]+).*$',
               read('requirements.txt'),
               flags=re.MULTILINE))


setup(
    name='gradhpo',
    version=get_version(),
    license='MIT',
    author='Eynullayev A., Rubtsov D., Karpeev G.',
    author_email='karpeev.ga@phystech.edu',
    description=(
        'gradhpo: gradient-based hyperparameter optimization in JAX. '
        'Implements T1-T2/DARTS, Greedy and HyperDistill bilevel algorithms.'
    ),
    long_description=read('README.rst'),
    long_description_content_type='text/x-rst',
    url='https://github.com/intsystems/gradhpo',
    project_urls={
        'Source': 'https://github.com/intsystems/gradhpo',
        'Documentation': 'https://intsystems.github.io/gradhpo/',
        'Tracker': 'https://github.com/intsystems/gradhpo/issues',
    },
    packages=find_packages(exclude=('tests', 'tests.*')),
    install_requires=get_requirements(),
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords=(
        'hyperparameter-optimization bilevel-optimization '
        'jax meta-learning hypergradient'
    ),
)
