
Setting the informationf of the IMG 
======================================

A crucial part of the process using CVXMG is defining the information of the project. To do so, CVXMG needs to dictionaries, "prob_info" and "sources_info". 
In the following lines present a brief description of how to create the dictionaries "prob_info" and "sources_info". 


Setting "prob_info"
----------------------

CVXMG provides two .csv files to make easier setting the data of "prob_info".

The file named config.csv contains the information about the architecture of the IMG. The file named resourcedata.csv cointains the information of the primary energy resources. 
CVXMG provides one function to read the data of config.csv and three different functions to read the reorcedata.csv file. 
The function to read config.csv is "variables". The three functions to import the resource data are: resources_norm, resources_all and resources_noise. These functions are crucial for the multyyear analysis and the stochastic analysis. 
For more information about these functions, please refer to the docs of the functions.  

The user must do the following to import the information of prob_info dictionary:
::

    import  pandas                  as      pd              # Library for date frames handling
    import  cvxmg                   as      cm              # Import cvxmg   
                                
    #region to read the parameters to intialize the code                                    
    prob_info = {}
    variables_csv   = pd.read_csv('config.csv', sep=';', header=None, skip_blank_lines=True)
    prob_info['project_life_time'], prob_info['interest_rate'], prob_info['scenarios'], prob_info["years"], prob_info["scala"], prob_info["prxo"], prob_info["percentage_yearly_growth"], prob_info["percentage_variation"], prob_info["dlcpercenthour"], prob_info["dlcpercenttotal"], prob_info["sen_ince"], prob_info["sen_ghi"], prob_info["elasticity"], prob_info["curtailment"], prob_info["capex_private"], prob_info["capex_gov"], prob_info["capex_community"], prob_info["capex_ong"], prob_info["opex_private"], prob_info["opex_gov"], prob_info["opex_community"], prob_info["opex_ong"], prob_info["rate_return_private"], prob_info["max_value_tariff"], prob_info['drpercentage'], prob_info['diesel_system'], prob_info['pv_system'], prob_info['battery_system'], prob_info['wind_system'], prob_info['hydro_system'], prob_info['hydrogen_system'], prob_info['gas_system'], prob_info['biomass_system'], prob_info['flat'], prob_info['tou'], prob_info['tou_sun'], prob_info['tou_three'], prob_info['cpp'], prob_info['dadp'], prob_info['shape_tar'], prob_info['ince'], prob_info['dilc'], prob_info['residential'],prob_info['commercial'],prob_info['industrial'],prob_info['community']  = cm.variables(variables_csv)
    #endregion

    #region to read the weather data, the load and resource availability for the community  
    data_csv        = pd.read_csv('resourcedata.csv', sep=';', header=None, skip_blank_lines=True)
    prob_info["ghi"], prob_info["irrdiffuse"], prob_info["temperature"], prob_info["wind"], prob_info["hydro"], prob_info["load_residential"], prob_info["load_commercial"], prob_info["load_industrial"], prob_info["load_community"] = cm.resources_norm(data_csv, years=prob_info["years"], scenarios=prob_info["scenarios"], percentage_yearly_growth=prob_info["percentage_yearly_growth"])
    #endregion



Setting "sources_info"
------------------------

The information of "sources_info" specify the characteristics of the energy sources. Each energy source expect different parameters. 
"sources_info" is a nested dictionary. To create "sources_info", the user first must initialize the dictionary:
:: 

    sources_info = {}

If the user wants to create the information of a Battery Energy Storage System must execute:
::

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
    }
    }

If the user wants to create the information of a Diesel Generator must execute: 
::

    sources_info = {      
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
    }
    }

If the user wants to create the information of a Photovoltaic System must execute: 
::

    sources_info = { 
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
    }
    }

If the user wants to create the information of the wind generation System must execute: 
::

    sources_info = {
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

    }
    }

It is crucial to specify the information of the lack of energy and the excess of energy in "sources_info". The user can use this information to control the desired level of reliability of the microgrid and to associate a cost to these values. 
To create this information the user must execute: 
::

    sources_info = {
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
