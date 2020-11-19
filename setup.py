from setuptools import setup, find_packages

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='SimString-cuda',
    version='0.1.0',
    url='https://github.com/fginter/simstring-cuda.git',
    author='Filip Ginter',
    author_email='filip.ginter@gmail.com',
    description="A poor-man's version of simistring-like lookup. Can hold its ground if the DB is few million strings, a GPU is present, and queries are batched by about a hundred strings.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),    
    install_requires=['sklearn', 'torch'],
    scripts=['simscuda']
)
