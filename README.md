# CVXMG

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://www.facebook.com/oviedojuancarlos)

CVXMG is a python based package that allows the users to compute the sizing of Isolated/Islanded MicroGrids (IMGs). Additionally to the sizing of the IMGs, CVXMG returns the optimal dispatch of the energy sources and the optimal tariffs for the energy. CVXMG allows implementing Demand Side Management (DSM) in the sizing of the IMGs. The package uses a plug and play approach, allowing its users to choose energy sources, DSM strategies, and types of solvers (deterministic, stochastic, multiyear, and stochastic multiyear). Here a brief description of the capabilities of the package:   

  - Obtain the sizing of the energy sources.
  - Perform optimal dispatch of the energy sources.
  - Compute the tariffs for the energy. 
  - Build different business models for the microgrid.
  - And more.

CVXMG has its own documentation that can be found at: link. 

# First release!

Just launch the first release! More to come!

### Requirements

CVXMG uses a number of open source projects to work properly:

* [Numpy](https://numpy.org/) 
* [Pandas](https://pandas.pydata.org/)
* [Scipy](https://www.scipy.org/)
* [CVXPY](https://www.cvxpy.org/)
* [Matplotlib](https://matplotlib.org/)

And of course cvxpymg itself is open source with a [public repository](https://github.com/juancaoviedo/cvxmg/tree/master)
 on GitHub.

### Installation

You can use "pip install cvxpymg" from your command window. Or clone the GitHub repository.


### Development

Want to contribute? Great!



### Todos

 - Write more examples.
 - Create Electrolizer-Tank-Fuel cell system source.
 - Create biomass generator source.
 - And more!

License
----

MIT


**Free Software, Hell Yeah!**