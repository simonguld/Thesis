from setuptools import Extension, setup

module = Extension(
    "lempel_ziv_complexity.LempelZivModule", 
    sources=["lempel_ziv_complexity/LempelZivModule.c"]
)


## build with python setup.py build_ext --inplace in the same directory as setup.py
setup(
    name="lempel_ziv_complexity",
    version="1.0",
    description="Python interface for the LZ77 C implementation",
    ext_modules=[module],
    packages=["lempel_ziv_complexity"],
)
