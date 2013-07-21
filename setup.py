import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "Lerot",
    version = "1.0",
    author = "Katja Hofmann, Anne Schuth",
    author_email = "katja.hofmann@microsoft.com, anne.schuth@uva.nl",
    description = ("This project is designed to run experiments on online learning to rank methods 
for information retrieval.")
    keywords = "online learning to rank for information retrieval",
    url = "http://ilps.science.uva.nl",
    package_dir = {'': 'src/python'}
    packages=['analysis',
    'comparison',
    'environment',
    'evaluation',
    'experiment',
    'query',
    'ranker',
    'retrieval_system',
    'utils'],
    long_description=read('README'),
)
