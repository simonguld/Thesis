from setuptools import Extension, setup

module = Extension(
    "ComputableInformationDensity.lempel_ziv_complexity.LempelZivModule", 
    sources=["ComputableInformationDensity/lempel_ziv_complexity/LempelZivModule.c"]
)

setup(
    name='ComputableInformationDensity',
    version='1.0',
    description='Package for computing the Computable Information Density (CID).',
    packages=['ComputableInformationDensity'],
    url='https://github.com/BHAN1992/computable-information-density.git',
    install_requires=['numpy'],
    ext_modules=[module]
)
