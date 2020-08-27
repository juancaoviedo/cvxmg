Stochastic multiyear analysis
================================

The study proposes a stochastic analysis to deal with the uncertainties of electric demand, weather variables, and future prices. The stochastic approach uses a Montecarlo Sampling (MCS) approach (see appendix \ref{app:montecarlo}). The MCS approach creates random samples of the probability distribution functions using Equation~(\ref{equ:generate_distributions}) to build the scenarios. Algorithm \ref{stochastic_multiyear} describes the multiyear stochastic analysis.  

Inputs: Weather, forecasted acquisition prices of energy sources, forecasted fuel prices over the lifetime of the IMG project.
Outputs: Tariffs of energy for the customers, average yearly acquisition, yearly dispatch of energy sources over the life time of the IMG project.

::

    prob_info = Set problem information
    historic_data = Save historic weather and demand data
    synthetic_data = create_synthetic_data(historic_data)

    for scenario in range(scenarios):
        for year in range(lifetime):
            prev_data = Read results of previous years
            act_param = Actualize solver parameters
            resul = yearly_solver(prob_info, synthetic_data[year], prev_data, act_param)
            summary[year] = resul
        total_summary[scenario] = summary


The above algorithm uses the multiyear analysis in its core. The only difference with the multiyear analysis is an additional loop. The stochastic multiyear solves one multiyear problem for each scenario that the MCS approach builds. Variable summary stores the results of installing and operating the IMG each year of the simulations. Variable total_summary stores the results of installing and operating each of the scenarios of the stochastic analysis. In the end, the results are the average of all the simulations, as Equation (\ref{eq:dcsp_formulation}) describes. 


