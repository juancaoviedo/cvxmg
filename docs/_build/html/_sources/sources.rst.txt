Energy sources models
===========================

Photovoltaic system                                       
------------------------

References \cite{Li2017, Zhang2016, lasnier1990} describe the output power :math:`E_{PV,t}` of a :math:`N_{PV}` number of photovoltaic panels as:

.. math::
    \begin{equation}
        E_{PV,t} = N_{PV}\rho_{PV}P_{STC}\frac{G_{A,t}}{G_{STC}}(1+C_T(T_{C,t}-T_{STC}))
    \end{equation}
    :label: equ:23

where :math:`\rho_{PV}`, :math:`P_{STC}`, :math:`G_{A,t}`, :math:`G_{STC}`, and :math:`C_T` are the derating factor (unitless), output power  of the PV module (:math:`kW`), GHI (:math:`kW/m^2`), GHI at standard conditions (:math:`kW/m^2`), and temperature coefficient of the PV module (:math:`\%/^{\circ}C`), respectively. :math:`T_{C,t}`  is the working temperature of the PV cell at hour :math:`t` (:math:`^{\circ}C`), and :math:`T_{STC}` is the temperature at standard conditions (:math:`^{\circ}C`). Reference \cite{Skoplaki2009} describes :math:`T_{C,t}` as a function of the ambient temperature and incident solar radiation over the PV module. 

.. math::
    \begin{equation}
        T_{C,t} = T_{A,t} + \frac{G_{A,t}}{G_{NOCT}}(T_{NOCT}-T_{a,t,NOCT})
    \end{equation}
    :label: equ:23a

where :math:`G_{NOCT}`, :math:`T_{NOCT}` and :math:`T_{a,t,NOCT}` are the solar radiation (:math:`kW/m^2`), working temperature (:math:`^{\circ}C`) and ambient temperature (:math:`^{\circ}C`) at Nominal Operational Cell Temperature (NOCT) conditions \cite{A.Duffie2013, librosolar}. 


Battery energy storage system     
---------------------------------

A battery is an element strongly coupled in time \cite{Xiaoping2010}. The lack or excess of energy in one hour can be demanded or stored in the battery. To guarantee that the battery is not charged and discharged simultaneously, the BESS model can integrate binary variables. However, as discussed before, the proposed methodology tries to avoid using binary variables. The methodology proposes to model the BESS as an accumulator to avoid using binary variables. The battery is a deposit to store something temporarily. The deposit can *charge* if there is still space available, and *discharge* when required.  Operations Research modeled this problem long before, and it is well known as the inventory problem \cite{Silver2008}.

The model of the BESS does not use separate optimization variables for charging and discharging of the BESS. Instead uses one single variable for the dispatch that controls the residual energy of the battery \cite{Zhang2018ab}. Equation :eq:`equ:residual_energy` presents a simple way of defining the residual energy in a BESS.

.. math::
    \begin{equation}
        RE_{B,t} = SOC_{t}C_{B}
    \end{equation}
    :label: equ:residual_energy

If the following state of the residual energy is superior to the previous, the battery was charged :math:`E_{B,t}` units during time :math:`t`. If the following state of the residual energy is inferior to the previous, the battery was discharged :math:`E_{B,t}` units during time :math:`t`. Equations :eq:`equ:charge` and :eq:`equ:discharge` show this.

.. math::
    \begin{equation}
        RE_{B,t+1} = RE_{B,t} + E_{B,t}
    \end{equation}
    :label: equ:charge

.. math::
    \begin{equation}
        RE_{B,t+1} = RE_{B,t} - E_{B,t}
    \end{equation}
    :label: equ:discharge


Equation :eq:`equ:24` describes the initial residual energy of the BESS. The simulations assume that the battery starts half charged (50% of its nominal capacity). Additionally, the simulation assumes that the minimum level of discharge of the battery is 50% and that the maximum level of charge is 100% of its nominal capacity. Equation :eq:`equ:25` describes those limits. Moreover, the simulations consider the maximum rate of charge and discharge of the battery. The simulation assumes that the maximum charge and discharge rate in each time slot is 30% of its nominal capacity. For all the simulations, the slot of time is one hour. Equation :eq:`equ:27` and :eq:`equ:28` describes the limits of charge and discharge of the battery for each time slot, respectively. 

.. math::
    \begin{equation} 
        E_{B,0} = 0.5C_B
    \end{equation} 
    :label: equ:24

.. math::
    \begin{equation} 
        0.5C_B \leq RE_{B,t}  \leq C_B
    \end{equation}  
    :label: equ:25

.. math::
    \begin{equation} 
        E_{B,t+1} \geq E_{B,t} -0.3C_B
    \end{equation}  
    :label: equ:27

.. math::
    \begin{equation} 
        E_{B,t+1} \leq E_{B,t} +0.3C_B
    \end{equation}  
    :label: equ:28



Diesel generator
---------------------------------

The fuel consumption of a diesel generator is a function of its capacity and output power. This function uses linear or quadratic formulations \cite{13,14}. Reference \cite{Scioletti2017} makes a quadratic fit to estimate :math:`\alpha`, :math:`\beta`, and :math:`\gamma` parameters as a function of the capacity of the generator using manufacturer-provided fuel consumption data. Bukar et al. use a linear approximation to describe the diesel consumption of a Diesel Generator \cite{Bukar2019}. Equation :eq:`equ:diesel_generator_operational_costs` describes the function that \cite{Bukar2019} use.

.. math::
    \begin{equation}
        F_{DG,t}=0.246E_{DG,t}+0.08415C_{DG}
    \end{equation}  
    :label: equ:diesel_generator_operational_costs

where, :math:`E_{DG,t}`, :math:`F_{DG,t}`, and :math:`C_{DG}` denote the generated power (kW), the fuel consumption (L/hour), and the installed capacity (kW) of the diesel generator. 

\subsection{Wind generator}  

The output power of a wind turbine is a function of the wind speed and its rated capacity. Equation :eq:`equ:wind_turbine` presents a well-accepted model to compute the output power of a wind turbine \cite{Ramli2018, Kaabeche2017}. The proposed methodology uses this model.  

.. math::
    \begin{equation}
        E_{WT}=
        \begin{cases}
        0, & V_{A,t}<V_{cut-in},V_{A,t}>V_{cut-out} \\
        V_{A,t}^3\left(\frac{E_{WT,R}}{V^3_{Rated}-V^3_{cut-in}}\right)-E_{WT,R}\left(\frac{V^3_{cut-in}}{V^3_{Rated}-V^3_{cut-in}}\right), & V_{cut-in} \leq V_{A,t} < V_{Rated} \\
        E_{WT,R}, & V_{Rated} \leq V_{A,t} < V_{cut-out}
        \end{cases}
    \end{equation} 
    :label: equ:wind_turbine

where :math:`V_{A,t}` is the wind speed (m/s), :math:`E_{WT,R}` is the rated power (kW), :math:`V_{cut-in}`, :math:`V_{Rated}`, :math:`V_{cut-out}` represent the cut-in, nominal and cut-out speed of the wind turbine (m/s), respectively.  
