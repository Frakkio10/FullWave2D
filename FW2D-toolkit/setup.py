from setuptools import setup, find_packages

setup(
    name='FW2D-toolkit',
    version='1.0.0',
    author='FO',
    author_email='francesco.orlacchio@lpp.polytechnique.fr',
    description='',
    packages=find_packages(),  # Automatically discover and include all packages
    install_requires=[
        # List the dependencies your package needs
        'numpy',
        'scipy',
        'matplotlib',
        'ipympl',
        'pandas',
        'tables',
        'scikit-image',
        'pyqtplotlib @ git+https://github.com/SaschaRienaecker/pyqtplotlib.git@master',
        'matlabtools @ git+ssh://git@github.com/Frakkio10/pythonmatlabtools.git@master',
        'plotfactory @ git+https://github.com/SaschaRienaecker/SR-plotfactory.git@master',
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
            'FW2Dgui=FW2D.gui.FW2Dgui:main',
        ],
    },
)