from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np

__version__ = "0.0.1"

house_ext = Extension(
    name="linalg.transforms.cy_householder",
    sources=["linalg/transforms/cy_householder.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-fopenmp"],
    extra_link_args=["-fopenmp"]
)

givens_ext = Extension(
    name="linalg.transforms.cy_givens",
    sources=["linalg/transforms/cy_givens.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-fopenmp"],
    extra_link_args=["-fopenmp"]
)

setup(
    name="linalg",
    version=__version__,
    description="Numerical linear algebra algorithms",
    author="gittasche",
    python_requires=">=3.9",
    install_requires=["numpy>=1.22.3"],
    packages=find_packages(),
    ext_modules=cythonize([house_ext, givens_ext]),
    zip_safe=False
)