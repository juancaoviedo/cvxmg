Methodology formulation
===========================

The problem that aims to solve CVXMG is to study the effects of Demand Side Management (DSM) strategies over the planning of Islanded/Isolated Microgrids (IMGs). The study should consider the impact of the technical and environmental aspects. The study will address regulatory aspects as well. This requires a methodology capable of:  


+ Integrate different energy sources for the IMG.
+ Compute the sizing of the energy sources.
+ Compute the energy dispatch of the energy sources.
+ Consider the effects of the DSM over the lifetime of the project. 
+ Consider business models to recreate the real-life conditions of the development of IMG projects.
+ Set the tariffs of the energy for the customers.
+ Evaluate the impact of the DSM strategies over the planning of IMGs. 


A methodology with the above characteristics does not exist in the reviewed literature. In this regard, CVXMG builds a methodology to solve that. 

Proposed solution
---------------------

The proposed solution implements a multiyear-stochastic analysis using Disciplined Convex Stochastic Programming (DCSP). DCSP builds on principles from stochastic optimization and convex analysis, representing a considerable advantage to build the desired methodology \cite{Ali2015}. Equation :eq:`equ:dcsp_formulation` presents the general formulation of a convex stochastic problem:   

.. math::
    \begin{equation}
        minimize\;E(a_1(x,\xi)) \\
        subject\;to\;E(b_i(x,\xi)) = 0\;\; i=1, \ldots ,B \\
        c_i(x,\xi) \geq 0\;\; i=1, \ldots ,C
    \end{equation}
    :label: equ:dcsp_formulation

where :math:`b_i:\mathbf{R}^n \times \mathbf{R}^q \to \mathbf{R}`, :math:`i=1, \ldots ,B` are convex functions in :math:`x` for each value of the random variable :math:`\xi \in \mathbf{R}^q`, and :math:`c_i:\mathbf{R}^n \to \mathbf{R}`, :math:`i=1 ,\ldots ,C` are (deterministic) affine functions; since expectations preserve convexity, the objective and inequality constraint functions in :eq:`equ:dcsp_formulation` are (also) convex in :math:`x`, making :eq:`equ:dcsp_formulation` a convex optimization problem \cite{Ali2015}, \cite[Chapter~7]{Liberti2008}. 

Main assumptions
--------------------

The formulation of the methodology assumes that the planner can have at least one year of historical data of weather variables and electrical demand. The formulation use this historical data to build the multiyear, and multiyear-stochastic analysis of the methodology by using a scenario construction technique. Section \ref{subsec:math_formulation} presents the information about the scenario building technique.

The methodology assumes that there is no presence of smart or controllable loads in the IMGs. Considering this, it is not possible to apply advanced DSM strategies for IMGs. Due to this limitation, the present study proposes to use price-based DSM strategies and one DSM strategy based on Direct Load Curtailment. Both kinds of DSM strategies offer less technical difficulty as their more sophisticated counterparts. 

The formulation also assumes that the planner can know the price elasticity of the demand of the customers. By using the price elasticity of the customers' demand, it is possible to compute how they will react to different stimuli. Additionally, the price elasticity of the demand intrinsically implies that without any external stimulus, the customers do not have any incentive to modify their consumption patterns. This assumption means that customers will not alter their consumption patterns if the IMG uses a flat tariff.    

Mathematical formulation
---------------------------

The formulation of the problem aims to minimize the total costs of the IMG project. The total costs of the project are Capital Expenditures (:math:`\zeta`), Operational Expenditures (:math:`\vartheta`), Maintenance Expenditures (:math:`\mu`) and Carbon Taxes Expenditures (:math:`\phi`): 

.. math::
    \begin{equation}
        \zeta=\sum_{u=1}^U{C_{u}I_{u}}
    \end{equation}
    :label: equ:capex

.. math::
    \begin{equation}
        \vartheta =\sum_{t=1}^T\sum_{u=1}^U{\lambda_{u,t}E_{u,t}}
    \end{equation}
    :label: equ:opex

.. math::
    \begin{equation}
        \mu =\sum_{t=1}^T\sum_{u=1}^U{\Lambda_{u,t}E_{u,t}}
    \end{equation}
    :label: equ:maintenance

.. math::
    \begin{equation}
        \Phi =\sum_{t=1}^T\sum_{u=1}^U{B_u F_{u,t}}
    \end{equation}
    :label: equ:taxes

and :math:`C_{u}`, :math:`I_{u}`, :math:`\lambda_{u,t}`, :math:`\Lambda_{u,t}`, :math:`E_{u,t}`, :math:`B_u` and :math:`F_{u,t}` represent the installed capacity, unitary investment cost, unitary dispatch costs, unitary maintenance costs, dispatched energy, carbon dioxide production by liter, and fuel consumption of the :math:`u` energy source at time :math:`t`, respectively. :math:`T` represents the horizon of the optimization. 

The mathematical formulation allows the planner to build all kinds of business models by considering that a :math:`i \in I` number of different investors (:math:`\varphi`) can fund the IMG project. These :math:`i \in I` investors can contribute to pay capital (:math:`\varphi_{i,\zeta}`), operational (:math:`\varphi_{i,\vartheta}`) or maintenance (:math:`\varphi_{i,\mu}`) expenditures. The objective function captures the different sources of money to fund the project:   

.. math::
    \begin{equation}
            X_{1} = argmin_{C_{u},E_{u,t}} \; \sum_{i=1}^I{\varphi_{i,\zeta}\zeta+\varphi_{i,\vartheta}\vartheta+\varphi_{i,\mu}\mu+\varphi_{i,\phi}\phi}    
    \end{equation}
    :label: equ:objective_1

The formulation considers the energy prices as the only revenue stream for the investors that aim to recover their investment and have profits. If the business model has private investors ($\varphi^{priv}$) the formulation allows to guarantee an expected Rate of Return ($R$) using the following constraint:

.. math::
    \begin{equation}
        (1+R)\sum_{y=1}^{Y}{(\varphi^{priv,\zeta}\zeta_{y}+\varphi^{priv,\vartheta}\vartheta_{y}+\varphi^{priv,\mu}\mu_{y} + \varphi^{priv,\phi}\phi_{y})} \geq \sum_{t=1}^{YT}{\pi_{x,t}D_{t}^{dr}}
    \end{equation}
    :label: equ:return_of_investment

where $\pi_{n,t}$ is the price of the energy at time $t$ using the $n$ DSM strategy, and $D_{t}^{dr}$ is the electrical demand after the $x$ DSM strategy is applied. However, it is crucial to highlight that the horizon of this constraint is the life time of the project. The life time of the project is measured in years ($Y$) for the sum in the left, and in hours for the sum in the right ($Y$ multiplied by $T$). 

Equation :eq:`equ:elasticity` uses the demand with flat tariff ($D_{t}^{flat}$) as the base demand, the flat tariff ($\pi^{flat}$) as the base price, the $x$ price ($\pi_{x,t}$) as the DSM tariff, and the price-elasticity ($e_{t}$) of the customers to compute the response of the demand $D_{t}^{dr}$. 

.. math::
    \begin{equation}
        e_{t}=\frac{\pi^{flat}(D_{t}^{dr}-D_{t}^{flat})}{D_{t}^{flat}(\pi_{x,t} - \pi^{flat})}
    \end{equation}
    :label: equ:elasticity

The formulation allows defining the changes in the total electrical demand after the introduction of the DSM using factor $\Psi^{c}$ in Equation :eq:`equ:6a`. Factor $\Psi^{c}$ is an input parameter that the planner choose according to the conditions of the IMG project. Values $\Psi^{c} \leq 1$ decreases the total energy consumption, while values $\Psi^{c} \geq 1$ increases the total energy consumption over the optimization horizon. A value  $\Psi^{c}=1$ indicates that the total energy consumption over the optimization horizon remains constant after the introduction of DSM. 

.. math::
    \begin{equation}
        \sum_{t=1}^TD_{t}^{dr} - \Psi^{c}\sum_{t=1}^TD_{t}^{flat} = 0 
    \end{equation}
    :label: equ:6a

The formulation naturally includes the balance Equation:

.. math::
    \begin{equation}
        \sum_{t=1}^{T}\sum_{u=1}^{U} E_{u,t} - EE_t + LE_t - D_{t}^{dr}=0
    \end{equation}
    :label: equ:energy_balance_f2_f1

where $EE_t$ and $LE_t$ are the excess and lack of energy. According to \cite{Chauhan2014,Diaf2008}, the loss of power supply probability (LPSP) is: 

.. math::
    \begin{equation}
        LPSP=\frac{\sum_{t=1}^{T}LE_t}{\sum_{t=1}^{T} D_{t}^{dr}}
    \end{equation}
    :label: equ:19

Similarly, Equation :eq:`equ:20` defines the excess of power supply probability (EPSP) as:  

.. math::
    \begin{equation}
        EPSP=\frac{\sum_{t=1}^{T}EE_t}{\sum_{t=1}^{T} D_{t}^{dr}}
    \end{equation}
    :label: equ:20

By using Equations :eq:`equ:19` and :eq:`equ:20` it is possible to create two constraints to control LPSP :eq:`equ:lack_energy` and EPSP :eq:`equ:excess_energy` over the optimization horizon:   

.. math::
    \begin{equation}
        \sum_{t=1}^{T}LE_t\leq LPSP\sum_{t=1}^{T} D_{t}^{dr}
    \end{equation}
    :label: equ:lack_energy


.. math::
    \begin{equation}
        \sum_{t=1}^{T}EE_t\leq EPSP\sum_{t=1}^{T} D_{t}^{dr}
    \end{equation}
    :label: equ:excess_energy
