# FullWave2D

## Installation Instructions

### Clone the Repository
The git repository is private. To get access, send an email to francesco.orlacchio@lpp.polytechnique.fr

```bash
git clone https://github.com/Frakkio10/FullWave2D.git
```
Note that git integration is made a lot easier using e.g. VSCode and linking your GitHub account to it. See below for more details.

### Installation on the IRFM cluster

1. Load the necessary environment variables. This will also load the Anaconda environment.
   ```bash
   module load tools_dc # A general purpose environment for accessing IRFM public tools
   ```

2. Navigate to the root of the package
    ```bash
    cd /path/to/FullWave2D
    ```
  
   Install the package into the current envinronment.
   ```bash
   pip install --user --editable .
   ```

   Compile the C code using the Makefile (needed once after installation, or every time the C code is changed)
   ```bash
   make
   ```

3. Run python and import the package to check if everything works
   ```python
   import fullwave2d as fw2d
   print(fw2d.__file__)
   ```
To unload the environment, simply type `module unload tools_dc` or `module purge`.


## Getting Started with a first simulation.

- You can find a first example in the `examples` folder, that you can run using:
   ```bash
   python fullwave2d/examples/airy_reflection.py
   ```
   This will create a folder under data/fw2d called `airy_reflection` (the name of the simulation) containing the input files and the output files of the simulation.

## Contributing

Instructions on how to contribute to this project:

1. Fork this repository
2. Create a new branch: `git checkout -b feature-branch`
3. Make your changes and commit them: `git commit -am 'Add some feature'`
4. Push your branch to the remote repository: `git push origin feature-branch`
