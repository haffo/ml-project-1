from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'  # This is the string that we want to remove from the requirements.txt file

def get_requirements(file_path: str) -> List[str]:
    '''
    This function will return the list of requirements
    '''
    with open(file_path, 'r') as file:
        requirements = file.read().splitlines()
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
 name='mlproject',
 version='0.0.1',
 author='Harold',
 author_email='harold.affo@gmail.com',
 packages=find_packages(),
 install_requires=get_requirements('requirements.txt'),
)

