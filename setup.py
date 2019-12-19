from setuptools import setup

setup(name='ibmqxbackend',
	description='IBM Quantum Experience backend',
	author='Ruslan Shaydulin',
	author_email='rshaydu@g.clemson.edu',
	packages=['ibmqxbackend'],
    install_requires=['qiskit'],
	zip_safe=False)
