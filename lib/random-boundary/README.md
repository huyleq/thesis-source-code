generator 2D and 3D random boundary

>> ./randomB.x modelIn=velIn.H seed=2017 pad=40 pctG=0.9 modelOut=velOut.H

where:
    velIn.H: is the 2D or 3D velocity model with padding applied
    seed: is used to generate the random number, for the purpose of reproducibility
    pad: thickness of random boundary area
    velOut.H: is the 2D or 3D velocity model with random boundary

20180306 update:
    
    1. fix the bug that velocity goes to zero for a few points;

    2. fix the bug that binary file cannot be executed on cees-mazama;
    
    3. added source code and header files
