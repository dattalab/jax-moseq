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
        'jax',
        'numpy',
        'scikit-learn',
        'dynamax',
        'chex==0.1.6', 
        'tqdm',
    ], 
    url='https://github.com/dattalab/jax-moseq/tree/0.0.0',
    download_url='https://github.com/dattalab/jax-moseq/archive/refs/tags/0.0.0.tar.gz'
)
