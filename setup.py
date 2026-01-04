# setup.py is responsible for creating my ML application as a package. So, this ML app can also be used in other ML projects just like numpy or pandas libraries.
# I can also deploy in Python PyPI. So, anyone can install and use it.

# "find_packages" is responsible for fetching all the packages that is being used in this ML application.
# How "find_packages" will come to know about all the packages is, when it is executed, it checks how many folder in this ML app contains "__init__.py". So, that specific folder (For Eg: 'src' folder) will be built. So, once built, it can be imported anywhere we want like seaborn or other libraries and use it if it is deployed to PyPI.
# Therefore, the entire project development will be happening in src folder. Similarly, whenever new folder is created, there also we will create "__init__.py" so that it can also be used as package.

from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the list of libraries mentioned in the requirements.txt file
    '''
    requirements=[]

    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements
    

# 'setup' contains the metadata about the project.

setup(
    name='mlproject',
    version='0.0.1',
    author='Karthik',
    author_email='karthik.rbhat21@gmail.com',
    packages=find_packages(),
    # install_requires = ['pandas','numpy','seaborn'] # A project may have hundreds of libraries. So, it not feasible to mention all the libraries in the form of list. Better to create a function which can fetch requirements.txt which will have all the packages required by the project.
    install_requires = get_requirements('requirements.txt')
)