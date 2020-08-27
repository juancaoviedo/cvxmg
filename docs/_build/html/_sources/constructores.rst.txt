
Types of constructors for the IMG 
=========================================

CVXMG have three different types of constructors that perform different analysis for IMGs. All the constructors hace the following constraints: 
* One constraint to perform the energy balance of the IMG.
* One constraint to guarantee that the total delivered energy remains cosntant after the aplication of the DSM. 
* One constraint to guarantee that the lack of energy do not exceed the desired reliability.
* One constraint to guarantee that the excess of energy do not exceed the desired reliability.
* One constraint to guarantee that the private investors recover their investments and the desired return of investment. 

The only difference between them is the horizon of optimization (one year, multiyear) and the type of analysis (deterministic, stochastic).

A brief description of each of the constructors proceeds. 

DeterministicDSMS
----------------------

The deterministic constructor performs the sizing of the microgrid considering one year of operation. Additionlly, as the name suggests, the deterministic constructor use a deterministic analysis. 

To use this constructor the user must execute: 
::

    import  cvxmg  as  cm
    MicroGrid = cm.DeterministicDSMS(prob_info, sources_info)

MultiyearDSMS
----------------------

The multiyear constructor performs the sizing of the microgrid considering one or several years of operation. The number of years are specified by the user in "prob_info" dictionary. The multiyear constructor use a deterministic analysis. 
The multy year constructor implements the adaptative method described in [pece2019]_. 
However, the multy year analysis here does not consider three day each month as the authors propose in the article. 
The multy-year analysis here considers the full year analysis (8760 hours).
    
To use this constructor the user must execute: 
::

    import  cvxmg  as  cm
    MicroGrid = cm.MultiyearDSMS(prob_info, sources_info)


StochasticDSMS
----------------------

The stochastic constructor performs the sizing of the microgrid considering one or several years of operation. However, the stochastic constructor performs a stochastic analysis. By using the functions resources_norm, resources_all and resources_noise CVXMG creates the data to perform the multiyear stochastic analysis. 
This constructor requires that the user specify the number of years for the multiyear analysis and the number of escenarios for the stochastic analysis. 

To use this constructor the user must execute: 
::

    import  cvxmg  as  cm
    MicroGrid = cm.StochasticDSMS(prob_info, sources_info)



















