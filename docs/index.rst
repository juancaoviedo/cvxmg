.. cvxmg documentation master file, created by
   sphinx-quickstart on Thu Jun 25 08:35:34 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CVXMG 2.0.0
=================================

CVXMG is a python based package that allows the users to compute the sizing of Isolated/Islanded MicroGrids (IMGs). Additionally to the sizing of the IMGs, CVXMG returns the optimal dispatch of the energy sources and the optimal tariffs for the energy. 
CVXMG allows implementing Demand Side Management (DSM) in the sizing of the IMGs. The user of CVXMG can choose seven different DSM strategies based on dynamic pricing of the energy and one DSM based on direct control of the loads:

+ Time of Use of two price levels
+ Time of Use with an incentive for solar generation
+ Time of use of three levels of price
+ Critical Peak Pricing
+ Day-Ahead Dynamic Pricing
+ Fixed Shape Pricing
+ Incentive Based Pricing 
+ Directly Curtailing the Electrical Demand

CVXMG allows for creating different business models for IMG projects. CVXMG allows defining the percentage of public or private funding for the project. CVXMG defines the energy's tariffs for the customers using the business model information and the share of public and private financing. These capabilities make CVXMG a worth looking tool for different analyses for IMG planners and policymakers.

CVXMG uses CVXPY at its core. CVXPY is a Python-embedded modeling language for convex optimization problems. CVXPY allows CVXMG to build and solve deterministic and stochastic convex formulations to perform the analysis of the IMGs. 
Due to the speed of solution of convex formulations, CVXMG can perform a multiyear analysis in seconds! Moreover, CVXMG can execute multiyear stochastic analysis in a regular machine.  

Installing CVXMG
---------------------

CVXMG can be easily installed using `PyPi <https://pypi.org/project/cvxmg/>`_ or downloading the `GitHub <https://github.com/juancaoviedo/cvxmg>`_ repository. 
To install CVXMG just execute the following in your line of commands:

::

   pip install cvxmg


Using CVXMG
--------------

CVXMG uses two simple dictionaries "prob_info" and "sources_info" to create different architectures of IMGs. These two dictionaries are attributes for the constructor classes. 
CVXMG offers three different constructor classes: One for deterministic analysis, one for multiyear analysis, and one for multiyear stochastic analysis. 
Each of the constructors uses the information of "prob_info" to know the architecture of the IMG and the information of "sources_info" to know the characteristics of the energy sources.
CVXMG creates the energy sources of the IMG as objects using the Objected Oriented Programming capabilities offered by Phyton. 
The use of objects for the energy sources allows CVXMG to build the optimization formulation of the problem using a Plug and Play approach.  

Once the user defines "prob_info" and "sources_info" needs to execute a constructor. For more info on how to set "prob_info" and "sources_info," please refer to the example section. 
Suppose the user wants to compute the sizing of an IMG using a deterministic analysis of one year. In that case, the user must execute the following command:  

:: 

   import cvxmg as cm
   MicroGrid = cm.DeterministicDSMS(prob_info, sources_info)
    
The above line of commands will create the structure of the IMG in the object MicroGrid. Additionally, it will guarantee that all the optimization formulation follows the Disciplined Convex Programming rules already established in CVXPY.
However, at this moment, CVXMG did not solve the formulation yet. To solve the formulation, the user needs to execute the solve method: 

::

   MicroGrid.solveMG()

The above commands will solve the formulation and will store the results in the MicroGrid object. To extract the results, the user must execute: 

::

   summary, dispatch_results = MicroGrid.resultsMG()

The above line of commands will create a pandas structure in the variable summary with the sizing results' essential variables. Additionally, the method will create another pandas structure to store the dispatch results of energy sources.

Finally, if the user wants to create some predetermined plots of the results can call the method plotMG:

::

   MicroGrid.plotMG()


The following are the contents of this guide: 

.. toctree::
   :maxdepth: 2

   configini
   constructores
   example
   methodology
   dsm
   multiyear
   stochastic
   sources
   modules
   references



