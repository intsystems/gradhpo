import io
import re
from setuptools import setup, find_packages


def read(file_path):
    with io.open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def get_version():
    """Read __version__ from mylib/__init__.py without importing the package."""
    content = read('mylib/__init__.py')
    match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", content, re.MULTILINE)
    if match:
        return match.group(1)
    raise RuntimeError("Unable to find __version__ in mylib/__init__.py")


readme = read('README.rst')
# вычищаем локальные версии из файла requirements (согласно PEP440)
requirements = '\n'.join(
    re.findall(r'^([^\s^+]+).*$',
               read('requirements.txt'),
               flags=re.MULTILINE))


setup(
    name='mylib',
    version=get_version(),
    license='MIT',
    author='Eynullayev A., Rubtsov D., Karpeev G.',
    author_email="grabovoy.av@phystech.edu",
    description='GradHpO: gradient-based hyperparameter optimization in JAX',
    long_description=readme,
    url='https://github.com/intsystems/GradHpO',
    packages=find_packages(),
    install_requires=requirements,
)
