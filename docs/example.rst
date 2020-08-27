
Example of how to use CVXMG 
=================================

Imagine that a user wants to compute the sizing of an IMG with photovoltaic panels, a diesel generator , a wind generator and a battery energy storage system, using a stochastic analysis. 
To do so, the user must first create the information for "prob_info" and "sources_info". Afterwards, the user must set the constructor and solve the problem. The following lines present a brief example using the multiyear constructor. 


First step: Create the information of "prob_info" and "sources_info"
---------------------------------------------------------------------

To import the information of prob_info dictionary, must do the following:
::

   import  numpy                   as      np              # Library to work with arrays and math
   import  pandas                  as      pd              # Library for date frames handling
   import  cvxpy                   as      cp              # Library for convex optimization
   import  cvxmg                   as      cm              # Library for the planning of Islanded Microgrids
   import  matplotlib.pyplot       as      plt             # Ploting command
   plt.style.use('default')                                # Restore default values for graphs

   #region to read the parameters to intialize the code                                    
   prob_info = {}
   variables_csv   = pd.read_csv('config.csv', sep=';', header=None, skip_blank_lines=True)
   prob_info['project_life_time'], prob_info['interest_rate'], prob_info['scenarios'], prob_info["years"], prob_info["scala"], prob_info["prxo"], prob_info["percentage_yearly_growth"], prob_info["percentage_variation"], prob_info["dlcpercenthour"], prob_info["dlcpercenttotal"], prob_info["sen_ince"], prob_info["sen_ghi"], prob_info["elasticity"], prob_info["curtailment"], prob_info["capex_private"], prob_info["capex_gov"], prob_info["capex_community"], prob_info["capex_ong"], prob_info["opex_private"], prob_info["opex_gov"], prob_info["opex_community"], prob_info["opex_ong"], prob_info["rate_return_private"], prob_info["max_value_tariff"], prob_info['drpercentage'], prob_info['diesel_system'], prob_info['pv_system'], prob_info['battery_system'], prob_info['wind_system'], prob_info['hydro_system'], prob_info['hydrogen_system'], prob_info['gas_system'], prob_info['biomass_system'], prob_info['flat'], prob_info['tou'], prob_info['tou_sun'], prob_info['tou_three'], prob_info['cpp'], prob_info['dadp'], prob_info['shape_tar'], prob_info['ince'], prob_info['dilc'], prob_info['residential'],prob_info['commercial'],prob_info['industrial'],prob_info['community']  = cm.variables(variables_csv)
   #endregion

   #region to read the weather data, the load and resource availability for the community  
   data_csv        = pd.read_csv('resourcedata.csv', sep=';', header=None, skip_blank_lines=True)
   prob_info["ghi"], prob_info["irrdiffuse"], prob_info["temperature"], prob_info["wind"], prob_info["hydro"], prob_info["load_residential"], prob_info["load_commercial"], prob_info["load_industrial"], prob_info["load_community"] = cm.resources_norm(data_csv, years=prob_info["years"], scenarios=prob_info["scenarios"], percentage_yearly_growth=prob_info["percentage_yearly_growth"])
   #endregion

To define the characteristics of the energy sources that the IMG will use the user must do the following: 
:: 

   #region to set the characteristics of the energy sources                    

   sources_info = {                                
      # Battery Energy Storage System info
      "bess_1" : {            
         "life_time"             : 2,
         "investment_cost"       : 420,                                              # USD
         "fuel_function"         : 0,                                                # Fuel function                  
         "fuel_cost"             : 0,                                                # USD
         "maintenance_cost"      : 6,                                                # Percentage of the capacity
         "min_out_power"         : 50,                                               # Percentage of the capacity
         "max_out_power"         : 100,                                              # Percentage of the capacity
         "rate_up"               : 1,                                                # Percentage of the capacity
         "rate_down"             : 1,                                                # Percentage of the capacity    
         "initial_charge"        : 50,                                               # Percentage of the capacity
      },
      
      # Diesel generator info
      "diesel_gen_1" : {      
         "life_time"         : 3,
         "investment_cost"   : 550,                                                  # USD
         "fuel_function"     : np.array([0.246, 0.08415]),                           # Fuel function                  
         # "fuel_function"     : np.array([0.000203636364, 0.224872727, 4.22727273]),  # Fuel function                  
         "fuel_cost"         : 0.8,                                                  # USD
         "maintenance_cost"  : 6,                                                    # Percentage of the capacity
         "min_out_power"     : 0,                                                    # Percentage of the capacity
         "max_out_power"     : 100,                                                  # Percentage of the capacity
         "rate_up"           : 1,                                                    # Percentage of the capacity
         "rate_down"         : 1,                                                    # Percentage of the capacity    
      },

      # Photovoltaic system info
      "pv_gen_1" : {          
         "life_time"         : 25,
         "investment_cost"   : 1300,                                                  # USD
         "maintenance_cost"  : 6,                                                     # Percentage of the capacity
         "rate_up"           : 1,                                                     # Percentage of the capacity
         "rate_down"         : 1,    
         "derat"             : 1,        # Derating factor
         "pstc"              : 0.3,      # Nominal capacity of the ov module                                                     # Percentage of the capacity    
         "Ct"                : -0.0039,  # Termic coefficient of the pv module
      },

      # Wind generator info
      "wind_gen_1" : {        
         "life_time"         : 15,
         "investment_cost"   : 2000,                                                  # USD
         "maintenance_cost"  : 5,                                                     # Percentage of the capacity
         "rate_up"           : 1,                                                     # Percentage of the capacity
         "rate_down"         : 1,
         "rated_speed"       : 13,
         "speed_cut_in"      : 3,
         "speed_cut_out"     : 12.5,
         "nominal_capacity"  : 1,

      },

      # Lack of energy info
      "lack_ene" : {          
         "cost_function"    : 0,                                                     # Cost function  
         "reliability"      : 2,                                                     # Percentage of reliability                                                   # Percentage of the capacity    
      },
      
      # Excess of energy info
      "excess_ene" : {        
         "cost_function"    : 0,                                                     # Cost function  
         "reliability"      : 2,                                                     # Percentage of reliability
      }
      }

   #endregion



Second step: Set the constructor
-----------------------------------

To use the constructor, the user must execute the following: 
::

   MicroGrid = cm.StochasticDSMS(prob_info_input=prob_info, sources_info=sources_info)


To extract the results of the optimization the user must execute: 
::

   summary=MicroGrid.resultsMG()

All the results are stored inside of the summary variable. 


