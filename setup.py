from setuptools import setup, find_packages

setup(
    name="pronunciation_transformer",
    version="0.1.0",
    author="Alfie Beard",
    author_email="alfiebeard96@gmail.com",
    packages=find_packages(),
    description="A python package for running a transformer model for IPA pronunciations.",
    install_requires=[
        'ipatok==0.4.1',
        'matplotlib==3.5.2',
        'numpy==1.22.4',
        'pandas==1.4.3',
        'pydub==0.25.1',
        'tensorflow-macos==2.9.2',
        'tensorflow-metal==0.5.0'
    ]
)
