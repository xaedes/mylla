
from setuptools import setup, find_packages
import glob, os

setup(
    name='mylla',
    version='0.0.1',
    author='xaedes',
    author_email='',
    license='MIT',
    packages=find_packages(exclude=['tests']),
    install_requires=[],
    entry_points={'console_scripts': ['mylla=mylla.main:main']},
)
