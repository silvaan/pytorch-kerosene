from setuptools import setup, find_packages


setup(
    name='pytorch-kerosene',
    version='0.0.5',
    author='Silvan Ferreira',
    author_email='silvanfj@gmail.com',
    description='PyTorch Kerosene is a tool for abstracting the training process of PyTorch models.',
    packages=find_packages(),
    keywords=['python', 'pytorch'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
