from setuptools import setup, find_packages

DESCRIPTION = 'Mi primer paquete de Python'
LONG_DESCRIPTION = 'Mi primer paquete de Python con una descripción ligeramente más larga'

install_requires = [
    "numpy>=1.19",  # 1.19 required by tensorflow 2.6
    "pandas>1.0.3",
]

setup(
    name='malib',  # Nombre del paquete
    version='0.1',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(where='src'),  # Buscar paquetes dentro del directorio 'src'
    package_dir={'': 'src'},  
    python_requires=">=3.9",
    install_requires=install_requires,
    # test_suite='tests',
)