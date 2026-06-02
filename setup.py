from setuptools import setup, find_packages

setup(
    name='fullwave2d',
    version='2.0.0',
    author='Francesco Orlacchio',
    author_email='francesco.orlacchio@lpp.polytechnique.fr',
    description='',
    packages=find_packages(),  # Automatically discover and include all packages
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'h5py',
        'mpi4py',
        'matlabtools @ git+https://github.com/Frakkio10/pythonmatlabtools.git@master',
    ],
    classifiers=[
        # 'Development Status :: 3 - Alpha',
        # 'Intended Audience :: Developers',
        # 'License :: OSI Approved :: MIT License',
        # 'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.6',
        # 'Programming Language :: Python :: 3.7',
        # 'Programming Language :: Python :: 3.8',
        # Add other classifiers as needed
    ],
    entry_points={
        'console_scripts': [
            # Define any command-line scripts your package provides
            'submit=fullwave2d.submitter.submit_job:main',
        ],
    },
)
