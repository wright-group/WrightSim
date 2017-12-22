WrightSim
=========
A simulation package for multidimensional spectroscopy.


Installation
------------

.. code-block:: bash
    
    $ git clone https://github.com/wright-group/WrightSim
    $ cd WrightSim
    $ python setup.py develop

Note: This will install all required dependencies.
PyCUDA is not a required dependency in general, but is required if GPU simulations are desired.

.. code-block:: bash

    $ pip install pycuda

PyCUDA requires an Nvida graphics card and drivers, and the CUDA libraries installed.

Usage
-----

An example script is provided at ``./scripts/target.py``

This script can be modified to suit an individual simulation.

The basic steps are:

#. Select an experiment
#. Set up the axes of the scan
#. Set the time interfal and buffers
#. Create a Hamiltonian object
#. Run the scan
#. (optional) review the results

Level of parallelism is selected by the ``mp`` parameter of teh ``exp.run`` method.

- ``True`` or ``"cpu"`` enables CPU multiprocessing
- ``"gpu"`` enables CUDA 
- ``False`` or ``""`` runs in single threaded mode


The script is set up to read the dimensions from arguments.
it can be run like so:

.. code-block:: bash
    
    $ ./target.py 32 16


This will run a 3D simulation of 32x32x16 Freq-Freq-Delay.

It will print the time it took to compute.

An example of how to visualize results is provided at the end of the script, but is not active, as it only works for 2D scans.

To convert this script to a 2D scan and visualize, uncomment the lines which set ``exp.d2.points`` and ``exp.d2.active`` and remove the triple quotes around the plotting routine.
There it will ignore the second argument, producing a 32x32 2D frequency spectrum. (The second argument is still parsed, so it is required, just ignored.)




