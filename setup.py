import setuptools

with open("README.md", 'r', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='jax-moseq',
    version='0.0.1',
    author='Caleb Weinreb',
    author_email='calebsw@gmail.com',
    include_package_data=True,
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent'
    ],
    python_requires='>=3.8',
    install_requires=[
        'numba',
        'dynamax',
        'chex==0.1.6', # not used in code; required to avoid installation issues
        'tqdm',
        'optree', # TODO: elimninate when all platforms can use JAX >= 0.4.6
    ], 
    url='https://github.com/dattalab/jax-moseq/tree/0.0.0',
    download_url='https://github.com/dattalab/jax-moseq/archive/refs/tags/0.0.0.tar.gz'
)
