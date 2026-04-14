setup(
       # the name must match the folder name 'verysimplemodule'
        name="SDEMatching", 
        packages=["SDEMatching"],
        install_requires=[
"gpytorch>=1.14",
"matplotlib>=3.10.3",
"normflows>=1.7.3",
"numpy>=2.2.6",
"pandas>=2.2.3",
"seaborn>=0.13.2",
"setuptools>=75.6.0",
"torch>=2.5.1",
"torchdyn>=1.0.6",
"torchsde>=0.2.6"], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
)