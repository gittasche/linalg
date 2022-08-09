from setuptools import setup, find_packages

__version__ = "0.0.1"

setup(
    name="linalg",
    version=__version__,
    description="Numerical linear algebra algorithms",
    author="gittasche",
    python_requires=">=3.9",
    install_requires=["numpy>=1.22.3"],
    packages=find_packages(),
    zip_safe=False
)