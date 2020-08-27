#%%
#region to import libraries                                                     
import  numpy                   as      np              # Library to work with arrays and math
import  pandas                  as      pd              # Library for date frames handling
import  cvxpy                   as      cp              # Library for convex optimization
from    scipy                   import  stats 
from    scipy.stats             import  exponweib       
from    scipy.optimize          import  curve_fit       # Curve fitting 
from    scipy.optimize          import  fmin
from    time                    import  time            # Library to take the execution time of the optimization problem
import  matplotlib.pyplot       as      plt             # Ploting command
import  matplotlib                                      # Plotting library

plt.style.use('default')                                # Restore default values for graphs

#endregion

#%%

#region for the constructors                                    

#region for the "one-year deterministic" constructor        

class DeterministicDSMS():                                  
    '''
    Deterministic constructor to ensamble the microgrid using Demand Side Management Strategies. The deterministic constructor build the optimization formulation for the microgrid using the information of the problem and the information of the energy sources. The information of the problem is stored in prob_info, the information of the energy sources is stored in sources_info. 

    | **prob_info:** (dictionary) 
    | Contains the information related to the configuration of the optimization formulation. To make easier setting the parameters, a function named variables is created to import the values of problem info from a *.csv file.
    | To execute the function variables do: 
    ::                             
        prob_info = {}
        variables_csv   = pd.read_csv('config.csv', sep=';', header=None, skip_blank_lines=True)
        prob_info['scenarios'], prob_info["years"], prob_info["scala"], prob_info["prxo"], prob_info["percentage_yearly_growth"], prob_info["percentage_variation"], prob_info["dlcpercenthour"], prob_info["dlcpercenttotal"], prob_info["sen_ince"], prob_info["sen_ghi"], prob_info["elasticity"], prob_info["curtailment"], prob_info["capex_private"], prob_info["capex_gov"], prob_info["capex_community"], prob_info["capex_ong"], prob_info["opex_private"], prob_info["opex_gov"], prob_info["opex_community"], prob_info["opex_ong"], prob_info["rate_return_private"], prob_info["max_value_tariff"], prob_info['diesel_system'], prob_info['pv_system'], prob_info['battery_system'], prob_info['wind_system'], prob_info['hydro_system'], prob_info['hydrogen_system'], prob_info['gas_system'], prob_info['biomass_system'], prob_info['flat'], prob_info['tou'], prob_info['cpp'], prob_info['dadp'], prob_info['ince'], prob_info['dilc'], prob_info['residential'],prob_info['commercial'],prob_info['industrial'],prob_info['community']  = cm.variables(variables_csv)

    Where 'config.csv' is the file containing the information of prob_info. For more information refer to the documentation of the function variables. 

    | Additionally, prob_info should include the weather conditions of the geographical region where the IMG is going to be installed. To make easier the process of importing these data, resources_norm, resources_all and resources_noise functions were created. 
    | To execute one of the functions do:
    ::    
        data_csv        = pd.read_csv('resourcedata.csv', sep=';', header=None, skip_blank_lines=True)
        prob_info["ghi"], prob_info["irrdiffuse"], prob_info["temperature"], prob_info["wind"], prob_info["hydro"], prob_info["load_residential"], prob_info["load_commercial"], prob_info["load_industrial"], prob_info["load_community"] = cm.resources_all(data_csv, years=prob_info["years"], scenarios=prob_info['scenarios'], percentage_yearly_growth=prob_info["percentage_yearly_growth"], fit_type="weekly")

    For more information about the different options of how to generate the scenarios for the multiyear and stochastic analysis, please refer to resources_norm, resources_all and resources_noise.  

    | **sources_info:** (dictionary) 
    | Contains the information related to the configuration of the energy sources. The following parameters are expected:

    sources_info = {                                
        | # Battery Energy Storage System info
        | "bess_1" : {            
        |     "life_time"         : 5,                               # Life time of the BESS system
        |     "fuel_function"     : 0,                               # Fuel function                  
        |     "fuel_cost"         : 0,                               # USD
        |     "maintenance_cost"  : 6,                               # Percentage of the capacity
        |     "min_out_power"     : 50,                              # Percentage of the capacity
        |     "max_out_power"     : 90,                              # Percentage of the capacity
        |     "rate_up"           : 1,                               # Percentage of the capacity
        |     "rate_down"         : 1,                               # Percentage of the capacity    
        |     "initial_charge"    : 50,                              # Percentage of the capacity
        | },
         
        | # Diesel generator info
        | "diesel_gen_1" : {      
        |     "life_time"         : 2,                                # Life time of the diesel system
        |     "fuel_function"     : np.array([0.246, 0.08415]),       # Linear fuel function                  
        |     # "fuel_function"     : np.array([0.000203636364, 0.224872727, 4.22727273]), # Quadratic fuel function                  
        |     "fuel_cost"         : 0.8,                              # USD per liter
        |     "maintenance_cost"  : 6,                                # Percentage of the capacity
        |     "min_out_power"     : 0,                                # Percentage of the capacity
        |     "max_out_power"     : 100,                              # Percentage of the capacity
        |     "rate_up"           : 1,                                # Percentage of the capacity
        |     "rate_down"         : 1,                                # Percentage of the capacity    
        | },
 
        | # Photovoltaic system info
        | "pv_gen_1" : {          
        |     "life_time"         : 25,                               # Life time of the PV module                                                   
        |     "maintenance_cost"  : 6,                                # Percentage of the capacity
        |     "rate_up"           : 1,                                # Percentage of the capacity
        |     "rate_down"         : 1,                                # Percentage of the capacity    
        |     "derat"             : 1,                                # Derating factor
        |     "pstc"              : 0.3,                              # Nominal capacity of the PV module                                                     # Percentage of the capacity    
        |     "Ct"                : -0.0039,                          # Termic coefficient of the PV module
        | },
 
        | # Wind generator info
        | "wind_gen_1" : {        
        |     "life_time"         : 15,                               # Life time of the WIND turbine
        |     "maintenance_cost"  : 5,                                # Percentage of the capacity
        |     "rate_up"           : 1,                                # Percentage of the capacity
        |     "rate_down"         : 1,                                # Percentage of the capacity
        |     "rated_speed"       : 12.5,                             # Rated speed in m/s
        |     "speed_cut_in"      : 3,                                # Speed to start generating of the turbine        
        |     "speed_cut_out"     : 13,                               # Speed to stop generating of the turbine            
        |     "nominal_capacity"  : 1,                                # Nominal capacity of the turnine in kW        
        | },
 
        | # Lack of energy info
        | "lack_ene" : {          
        |     "cost_function"    : 0,                                 # Cost function  
        |     "reliability"      : 2,                                 # Percentage of reliability for the lack of energy                                                   # Percentage of the capacity    
        | },
         
        | # Excess of energy info
        | "excess_ene" : {        
        |     "cost_function"    : 0,                                 # Cost function  
        |     "reliability"      : 2,                                 # Percentage of reliability for the excess of energy
        | }   
        | }
    '''
    def __init__(self, prob_info, sources_info):            
        self.simulation_year   = 0
        self.prob_info    = prob_info
        self.sources_info = sources_info
        
        #region to set the electrical demand                
        if self.prob_info["residential"] == True: 
            tempload1 = int(self.prob_info['scala'])*self.prob_info["load_residential"]
        if self.prob_info["commercial"]  == True:
            tempload1 = int(self.prob_info['scala'])*self.prob_info["load_commercial"]
        if self.prob_info["industrial"]  == True:
            tempload1 = int(self.prob_info['scala'])*self.prob_info["load_industrial"] 
        if self.prob_info["community"]   == True:
            tempload1 = int(self.prob_info['scala'])*self.prob_info["load_community"]
        
        tempload2               = np.zeros((8760,1))
        tempload2[:,0]          = tempload1[0,self.simulation_year*8760:(self.simulation_year+1)*8760]
        self.load_ini           = tempload2
        
        #endregion

        #region to create a dictionary for the energy sources                   
        self.sources_dict= {}
        
        # Lack and Excess of energy
        self.sources_dict['lack_energy']        = Lackenergy(cost_function=self.sources_info["lack_ene"]["cost_function"], reliability=self.sources_info["lack_ene"]["reliability"], years=1)                  # Create an object for the lack of energy
        self.sources_dict['excess_energy']      = Excessenergy(cost_function=self.sources_info["excess_ene"]["cost_function"], reliability=self.sources_info["excess_ene"]["reliability"], years=1)              # Create an object for the excess of energy

        if self.prob_info["battery_system"]     == True:    
            self.sources_dict['battery_system'] = EStorage(investment_cost=self.sources_info["bess_1"]["investment_cost"],fuel_function=self.sources_info["bess_1"]["fuel_function"],fuel_cost=self.sources_info["bess_1"]["fuel_cost"],maintenance_cost=self.sources_info["bess_1"]["maintenance_cost"],min_out_power=self.sources_info["bess_1"]["min_out_power"],max_out_power=self.sources_info["bess_1"]["max_out_power"],rate_up=self.sources_info["bess_1"]["rate_up"],rate_down=self.sources_info["bess_1"]["rate_down"], initial_charge = self.sources_info["bess_1"]["initial_charge"], years=1,life_time=self.sources_info["bess_1"]["life_time"]) # Create an object for the battery
        
        if self.prob_info["diesel_system"]      == True:    
            self.sources_dict['diesel_system']  = EGenerator(investment_cost=self.sources_info["diesel_gen_1"]["investment_cost"],fuel_function=self.sources_info["diesel_gen_1"]["fuel_function"],fuel_cost=self.sources_info["diesel_gen_1"]["fuel_cost"],maintenance_cost=self.sources_info["diesel_gen_1"]["maintenance_cost"],min_out_power=self.sources_info["diesel_gen_1"]["min_out_power"],max_out_power=self.sources_info["diesel_gen_1"]["max_out_power"],rate_up=self.sources_info["diesel_gen_1"]["rate_up"],rate_down=self.sources_info["diesel_gen_1"]["rate_down"],years=1,life_time=self.sources_info["diesel_gen_1"]["life_time"])
        
        if self.prob_info["pv_system"]          == True:    
            self.sources_dict['pv_system']      = PVGenerator(investment_cost=self.sources_info["pv_gen_1"]["investment_cost"], maintenance_cost=self.sources_info["pv_gen_1"]["maintenance_cost"], rate_up=self.sources_info["pv_gen_1"]["rate_up"], rate_down=self.sources_info["pv_gen_1"]["rate_down"], ghi=self.prob_info["ghi"], temperature=self.prob_info["temperature"], derat = self.sources_info["pv_gen_1"]["derat"], pstc= self.sources_info["pv_gen_1"]["pstc"], Ct = self.sources_info["pv_gen_1"]["Ct"], years=1, sen_ghi=self.prob_info["sen_ghi"],life_time=self.sources_info["pv_gen_1"]["life_time"])             # Create an object for pv generator
            
        if self.prob_info["wind_system"]        == True:    
            self.sources_dict['wind_system']    = WINDGenerator(investment_cost=self.sources_info["wind_gen_1"]["investment_cost"], maintenance_cost=self.sources_info["wind_gen_1"]["maintenance_cost"], rate_up=self.sources_info["wind_gen_1"]["rate_up"], rate_down=self.sources_info["wind_gen_1"]["rate_down"], wind=self.prob_info["wind"], years=1, rated_speed=self.sources_info["wind_gen_1"]["rated_speed"], speed_cut_in=self.sources_info["wind_gen_1"]["speed_cut_in"], speed_cut_out=self.sources_info["wind_gen_1"]["speed_cut_out"], nominal_capacity=self.sources_info["wind_gen_1"]["nominal_capacity"],life_time=self.sources_info["wind_gen_1"]["life_time"])             # Create an object for pv generator
            
        # Execute the method to create the optimization formulation inside of the energy sources
        for unit in self.sources_dict.keys():
            self.sources_dict[unit].formulate()
        #endregion

        #region to set the formulation                      
        capex_sources_list, opex_list, maintenance_list = [], [], []
        for source in self.sources_dict.keys():
            capex_sources_list          += [self.sources_dict[source].capex]
            opex_list           += [self.sources_dict[source].opex]
            maintenance_list    += [self.sources_dict[source].maintenance]

        self.tcapex             = cp.sum(capex_sources_list)         # Compute the Capex
        self.topex              = cp.sum(opex_list)          # Compute the Opex and the maintenance
        self.tmaintenance       = cp.sum(maintenance_list)   # Compute the maintenance
        self.ttaxes             = cp.sum(self.sources_dict['diesel_system'].tax_tonne)
        self.objective          = self.tcapex + self.topex + self.tmaintenance + self.ttaxes
        #endregion

        #region to compute the Demand Response              
        self.dsms               = ComputeDSMS(loadfile=self.load_ini, elasticity=self.prob_info["elasticity"], initial_price=self.prob_info["prxo"], max_value_tariff = self.prob_info["max_value_tariff"], years = 1, dlcpercenthour = self.prob_info["dlcpercenthour"], dlcpercenttotal= self.prob_info["dlcpercenttotal"], flat = self.prob_info["flat"], tou = self.prob_info["tou"], tou_sun = self.prob_info["tou_sun"], tou_three = self.prob_info["tou_three"], cpp = self.prob_info["cpp"], dadp = self.prob_info["dadp"], shape_tar = self.prob_info["shape_tar"], ince = self.prob_info["ince"], dilc = self.prob_info["dilc"], drpercentage = prob_info['drpercentage'])                            
        #endregion                                     


        #region to set the constraints of the formulation   
        self.constraints        = []  

        for source in self.sources_dict.keys(): 
            for constraint in self.sources_dict[source].constraints:
                self.constraints.append(constraint)

        # Import the constraints of the DSMS
        for constraint in self.dsms.constraints:
            self.constraints.append(constraint)
        
        # Set the general constraints of the formulation
        if self.prob_info["dilc"] == True:
            self.constraints += [   
                                    cp.sum(self.sources_dict['lack_energy'].dispatch)       <= self.sources_dict['lack_energy'].reliability/100*cp.sum(self.dsms.loadDSMS),        # Maximum allowed lack of energy                                   
                                    cp.sum(self.sources_dict['excess_energy'].dispatch)     >= -self.sources_dict['excess_energy'].reliability/100*cp.sum(self.dsms.loadDSMS),     # Maximum allowed excess of energy
                                    # cp.sum(self.dsms.loadDSMS) - self.prob_info["curtailment"]*cp.sum(self.load_ini) >= 0,                       # Allowed curtailment after the application of the tariff
                                    # cp.sum(self.dsms.customer_payments) >= (self.prob_info["capex_private"]*self.tcapex+self.prob_info["opex_private"]*self.topex+self.tmaintenance+self.ttaxes)*self.prob_info["rate_return_private"], # Guarantee to private investors to recover their capital
                                ]                                     
        else:    
            self.constraints += [   
                                    cp.sum(self.sources_dict['lack_energy'].dispatch)       <= self.sources_dict['lack_energy'].reliability/100*cp.sum(self.dsms.loadDSMS),        # Maximum allowed lack of energy                                   
                                    cp.sum(self.sources_dict['excess_energy'].dispatch)     >= -self.sources_dict['excess_energy'].reliability/100*cp.sum(self.dsms.loadDSMS),     # Maximum allowed excess of energy
                                    cp.sum(self.dsms.loadDSMS) - self.prob_info["curtailment"]*cp.sum(self.load_ini) >= 0,                       # Allowed curtailment after the application of the tariff
                                    # cp.sum(self.dsms.customer_payments) >= (self.prob_info["capex_private"]*self.tcapex+self.prob_info["opex_private"]*self.topex+self.tmaintenance+self.ttaxes)*self.prob_info["rate_return_private"], # Guarantee to private investors to recover their capital
                                ]                                    

        # if self.prob_info["wind_system"]        == True:
        #     if self.prob_info["pv_system"]        == True:
        #         self.constraints += [cp.sum(self.sources_dict['excess_energy'].dispatch) - self.sources_dict['pv_system'].curtailment - self.sources_dict['wind_system'].curtailment    >= -self.sources_dict['excess_energy'].reliability/100*cp.sum(self.dsms.loadDSMS)]     # Maximum allowed excess of energy
        #     else:
        #         self.constraints += [cp.sum(self.sources_dict['excess_energy'].dispatch) - self.sources_dict['wind_system'].curtailment    >= -self.sources_dict['excess_energy'].reliability/100*cp.sum(self.dsms.loadDSMS)]     # Maximum allowed excess of energy
        # else:
        #     if self.prob_info["pv_system"]        == True:
        #         self.constraints += [cp.sum(self.sources_dict['excess_energy'].dispatch) - self.sources_dict['pv_system'].curtailment    >= -self.sources_dict['excess_energy'].reliability/100*cp.sum(self.dsms.loadDSMS)]     # Maximum allowed excess of energy
        #     else:
        #         self.constraints += [cp.sum(self.sources_dict['excess_energy'].dispatch) >= -self.sources_dict['excess_energy'].reliability/100*cp.sum(self.dsms.loadDSMS)]     # Maximum allowed excess of energy

        #region to perform the energy balance with the battery without efficiency   
        net_dispatch=[]                                      
        for source in self.sources_dict.keys():
            if source == "battery_system":
                pass
            else:
                net_dispatch  += [self.sources_dict[source].dispatch]
        
        T=8760
        if self.prob_info["battery_system"]  == True:
            self.constraints +=  [ self.sources_dict["battery_system"].dispatch[1:T] == self.sources_dict["battery_system"].dispatch[0:T-1] + cp.sum(net_dispatch)[1:T] - self.dsms.loadDSMS[1:T]]# Energy balance 
        else:
            self.constraints +=  [  cp.sum(net_dispatch) - self.dsms.loadDSMS == 0]# Energy balance
        #endregion

        #endregion

        #region to build the optimization formulation       
        self.formulation = cp.Problem(cp.Minimize(self.objective), self.constraints)
        #endregion
    
    def solveMG(self):  
        '''
        DeterministicDSMS.solveMG():  
        Method created to solve the optimization formulation. 
        '''                                    
        initial_time = time() 
        self.formulation.solve(solver=cp.MOSEK)
        final_time = time() 
        self.execution_time = final_time - initial_time
        return f"The status of the optimization formulation is: {self.formulation.status}"

    def resultsMG(self):
        '''
        DeterministicDSMS.resultsMG():  
        Method created to obtain the results of the optimization. It returns a tuple. summary, and dispatch_results.  
        '''                                      
        #region to create the pandas structures             
        self.summary   = pd.DataFrame(columns=['private_roi','private_npv','battery_capex','diesel_capex','pv_capex','wind_capex','battery_opex','diesel_opex','pv_opex','wind_opex','battery_maintenance','diesel_maintenance','pv_maintenance','wind_maintenance','scenarios','sen_ince','sen_batcost','sen_ghi','sen_pvcost','sen_diecost','sen_gdcost', 'curtailment', 'elasticity', 'total_costs', 'total_capex', 'total_opex','total_maintenance', 'private_investment', 'private_capex','private_opex','customer_payments','private_profits','final_demand','delivered_energy','diesel_consumption','diesel_cost','total_lcoe','government_lcoe','private_lcoe','excess_energy','lack_energy','pv_capacity','dg_capacity','bess_capacity','wind_capacity','solving_time'])
        self.dispatch_results  = pd.DataFrame(columns=['date','tariff','load_flat', 'load_dsms', 'pv_dispatch', 'wind_dispatch', 'dg_dispatch', 'bess_dispatch', 'bess_stored_energy', 'dispatch_ee','dispatch_le'])
        self.dispatch_results.date=pd.date_range('2018-01-01', periods=8760, freq='H')
        self.dispatch_results.set_index('date',inplace=True)
        #endregion

        #region to store the summary                        
        pos = 0

        #region for the capex                       
        capex_sources_list=[]  
        if self.prob_info["battery_system"]  == True:
            self.summary.loc[pos,['bess_capacity']] = self.sources_dict['battery_system'].capacity.value                        # Installed capacity of BESS
            battery_one_time                        = self.sources_dict['battery_system'].capacity.value*self.sources_info['bess_1']['investment_cost']
            battery_times                           = int(self.prob_info['project_life_time']/self.sources_info['bess_1']['life_time'])
            if battery_times < 1:
                battery_capex                           = battery_one_time
            else:
                battery_capex                           = battery_times*battery_one_time

            capex_sources_list.append(battery_capex)
            self.summary.loc[pos,['battery_capex']] = battery_capex
        else:
            self.summary.loc[pos,['battery_capex']] = 0

        if self.prob_info["diesel_system"]   == True:
            self.summary.loc[pos,['dg_capacity']]   = self.sources_dict['diesel_system'].capacity.value                           # Installed capacity of DG
            diesel_one_time                         = self.sources_dict['diesel_system'].capacity.value*self.sources_info['diesel_gen_1']['investment_cost']
            diesel_times                            = int(self.prob_info['project_life_time']/self.sources_info['diesel_gen_1']['life_time'])
            if diesel_times<1:
                diesel_capex                            = diesel_one_time
            else:
                diesel_capex                            = diesel_times*diesel_one_time
            capex_sources_list.append(diesel_capex)
            self.summary.loc[pos,['diesel_capex']]  = diesel_capex
        else:
            self.summary.loc[pos,['diesel_capex']]  = 0

        if self.prob_info["pv_system"]   == True:
            self.summary.loc[pos,['pv_capacity']]   = self.sources_dict['pv_system'].capacity.value                           # Installed capacity of PV
            pv_one_time                             = self.sources_dict['pv_system'].capacity.value*self.sources_info['pv_gen_1']['investment_cost']
            pv_times = int(self.prob_info['project_life_time']/self.sources_info['pv_gen_1']['life_time'])
            if pv_times< 1:
                pv_capex                            = pv_one_time
            else:
                pv_capex                            = pv_times*pv_one_time

            capex_sources_list.append(pv_capex)
            self.summary.loc[pos,['pv_capex']]      = pv_capex
        else:
            self.summary.loc[pos,['pv_capex']]      = 0

        if self.prob_info["wind_system"]   == True:
            self.summary.loc[pos,['wind_capacity']] = self.sources_dict['wind_system'].capacity.value                           # Installed capacity of PV
            wind_one_time                           = self.sources_dict['wind_system'].capacity.value*self.sources_info['wind_gen_1']['investment_cost']
            wind_times                              = int(self.prob_info['project_life_time']/self.sources_info['wind_gen_1']['life_time'])
            if wind_times<1:
                wind_capex                              = wind_one_time
            else:
                wind_capex                              = wind_times*wind_one_time
            
            capex_sources_list.append(wind_capex)
            self.summary.loc[pos,['wind_capex']]    = wind_capex
        else:
            self.summary.loc[pos,['wind_capex']]    = 0
        #endregion 

        #region for the opex                        
        opex_sources_list=[]    
        if self.prob_info["battery_system"]  == True:
            self.summary.loc[pos,['battery_opex']] = self.sources_dict['battery_system'].opex*self.prob_info['project_life_time']            
            opex_sources_list.append(self.summary['battery_opex'])
        else:
            self.summary.loc[pos,['battery_opex']] = 0
            opex_sources_list.append(self.summary['battery_opex'])

        if self.prob_info["diesel_system"]   == True:
            self.summary.loc[pos,['diesel_opex']]   = self.sources_dict['diesel_system'].opex.value*self.prob_info['project_life_time']
            opex_sources_list.append(self.summary['diesel_opex'])
        else:
            self.summary.loc[pos,['diesel_opex']]  = 0
            opex_sources_list.append(self.summary['diesel_opex'])

        if self.prob_info["pv_system"]   == True:
            self.summary.loc[pos,['pv_opex']]   = self.sources_dict['pv_system'].opex*self.prob_info['project_life_time']
            opex_sources_list.append(self.summary['pv_opex'])
        else:
            self.summary.loc[pos,['pv_opex']]      = 0
            opex_sources_list.append(self.summary['pv_opex'])

        if self.prob_info["wind_system"]   == True:
            self.summary.loc[pos,['wind_opex']] = self.sources_dict['wind_system'].opex*self.prob_info['project_life_time']
            opex_sources_list.append(self.summary['wind_opex'])
        else:
            self.summary.loc[pos,['wind_opex']]    = 0
            opex_sources_list.append(self.summary['wind_opex'])
        #endregion 

        #region for the maintenance                 
        maintenance_sources_list=[] 
        if self.prob_info["battery_system"]  == True:
            self.summary.loc[pos,['battery_maintenance']] = self.sources_dict['battery_system'].maintenance.value*self.prob_info['project_life_time']            
            maintenance_sources_list.append(self.summary['battery_maintenance'])
        else:
            self.summary.loc[pos,['battery_maintenance']] = 0
            maintenance_sources_list.append(self.summary['battery_maintenance'])

        if self.prob_info["diesel_system"]   == True:
            self.summary.loc[pos,['diesel_maintenance']]   = self.sources_dict['diesel_system'].maintenance.value*self.prob_info['project_life_time']
            maintenance_sources_list.append(self.summary['diesel_maintenance'])
        else:
            self.summary.loc[pos,['diesel_maintenance']]  = 0
            maintenance_sources_list.append(self.summary['diesel_maintenance'])

        if self.prob_info["pv_system"]   == True:
            self.summary.loc[pos,['pv_maintenance']]   = self.sources_dict['pv_system'].maintenance.value*self.prob_info['project_life_time']
            maintenance_sources_list.append(self.summary['pv_maintenance'])
        else:
            self.summary.loc[pos,['pv_maintenance']]      = 0
            maintenance_sources_list.append(self.summary['pv_maintenance'])

        if self.prob_info["wind_system"]   == True:
            self.summary.loc[pos,['wind_maintenance']] = self.sources_dict['wind_system'].maintenance.value*self.prob_info['project_life_time']
            maintenance_sources_list.append(self.summary['wind_maintenance'])
        else:
            self.summary.loc[pos,['wind_maintenance']]    = 0
            maintenance_sources_list.append(self.summary['wind_maintenance'])
        #endregion

        #region to perform additional operations    
        total_capex             = np.sum(capex_sources_list)                                # Total costs of the CAPEX  
        total_opex              = np.sum(opex_sources_list)                                 # Total costs of the OPEX
        total_maintenance       = np.sum(maintenance_sources_list)                          # Total costs of the maintenance
        total_taxes             = self.ttaxes.value                                         # Total costs of taxes
        total_costs             = total_capex+total_opex+total_maintenance+total_taxes      # Total costs of the project
        private_capex           = self.prob_info['capex_private']*total_capex               # Value of the capex paid by the private investors
        private_opex            = self.prob_info['opex_private']*total_opex                 # Value of the opex paid by the private investors
        private_investment      = private_capex + private_opex + total_maintenance          # Total investment paid by the private investors
        government_capex        = self.prob_info['capex_gov']*total_capex                   # Value of the capex paid by the government
        government_opex         = self.prob_info['opex_gov']*total_opex                     # Value of the opex paid by the government
        government_investment   = government_capex + government_opex                        # Total investment paid by the government
        customer_payments       = np.sum(self.dsms.customer_payments.value)*self.prob_info['project_life_time']
        private_profits         = customer_payments-private_investment 
        rate_of_return          = private_profits/private_investment
        net_present_value       = private_profits
        lack_energy             = np.sum(self.sources_dict['lack_energy'].dispatch.value)*self.prob_info['project_life_time']                
        excess_energy           = np.absolute(np.sum(self.sources_dict['excess_energy'].dispatch.value))*self.prob_info['project_life_time'] 
        final_demand            = np.sum(self.dsms.loadDSMS.value)*self.prob_info['project_life_time']                                                  # Sum of the total final demand
        delivered_energy        = final_demand-excess_energy-lack_energy      # Delivered energy
        total_lcoe              = total_costs/delivered_energy
        private_lcoe            = private_investment/delivered_energy                      
        government_lcoe         = government_investment/delivered_energy                       
        solving_time            = self.execution_time                         
        
        #region for the diesel operation                    
        if self.prob_info["diesel_system"]   == True:
            if type(self.sources_info["diesel_gen_1"]["fuel_function"]) == np.ndarray:  
                if len(self.sources_info["diesel_gen_1"]["fuel_function"]) == 3:
                    diesel_consumption           = self.sources_info["diesel_gen_1"]["fuel_function"][0]*np.sum(self.sources_dict['diesel_system'].dispatch.value**2) + self.sources_info["diesel_gen_1"]["fuel_function"][1]*np.sum(self.sources_dict['diesel_system'].dispatch.value) + self.sources_info["diesel_gen_1"]["fuel_function"][2]                 
                if len(self.sources_info["diesel_gen_1"]["fuel_function"]) == 2:
                    diesel_consumption           = self.sources_info["diesel_gen_1"]["fuel_function"][0]*np.sum(self.sources_dict['diesel_system'].dispatch.value) + self.sources_info["diesel_gen_1"]["fuel_function"][1]*self.sources_dict['diesel_system'].capacity.value                 
                if len(self.sources_info["diesel_gen_1"]["fuel_function"]) == 1:
                    diesel_consumption           = self.sources_info["diesel_gen_1"]["fuel_function"][0]*np.sum(self.sources_dict['diesel_system'].dispatch.value)                 
            else:
                diesel_consumption           = 0
        diesel_cost         = self.sources_info['diesel_gen_1']['fuel_cost']*diesel_consumption
        #endregion   
        
        #endregion   

        #region to store in the pandas structure            
        self.summary.loc[pos,['scenarios']]              = self.prob_info['scenarios']                # Number of escenarios
        self.summary.loc[pos,['sen_ince']]               = self.prob_info['sen_ince']                 # Sensitivity analysis of values of incentives
        # self.summary.loc[pos,['sen_batcost']]            = self.prob_info['sen_batcost']              # Sensitivity analysis of cost of the batteries
        self.summary.loc[pos,['sen_ghi']]                = self.prob_info['sen_ghi']                  # Sensitivity analysis of global horizontal radiation
        # self.summary.loc[pos,['sen_pvcost']]             = self.prob_info['sen_pvcost']               # Sensitivity analysis of costs of the pv
        # self.summary.loc[pos,['sen_diecost']]            = self.prob_info['sen_diecost']              # Sensitivity analysis of costs of the diesel
        # self.summary.loc[pos,['sen_gdcost']]             = self.prob_info['sen_gdcost']               # Sensitivity analysis of costs of the diesel generator
        self.summary.loc[pos,['curtailment']]            = self.prob_info['curtailment']              # Save the curtailment factor
        self.summary.loc[pos,['elasticity']]             = self.prob_info['elasticity']*(-1)          # Save the elasticity
        self.summary.loc[pos,['total_costs']]            = total_costs                           # Total costs
        self.summary.loc[pos,['private_roi']]            = rate_of_return                        # Rate of return
        self.summary.loc[pos,['private_npv']]            = net_present_value                     # Net present value
        self.summary.loc[pos,['total_capex']]            = total_capex                           # Total costs of the capex
        self.summary.loc[pos,['total_opex']]             = total_opex                            # total costs of the opex
        self.summary.loc[pos,['total_maintenance']]      = total_maintenance                     # total costs of the maintenance
        self.summary.loc[pos,['private_investment']]     = private_investment                    # Total payments of the investors
        self.summary.loc[pos,['private_capex']]          = private_capex                         # Payments of the investors for the capex                          
        self.summary.loc[pos,['private_opex']]           = private_opex                          # Payments of the investors for the opex                             
        self.summary.loc[pos,['customer_payments']]      = customer_payments                     # Payments of the customers after the introduction of the dsms         
        self.summary.loc[pos,['private_profits']]        = private_profits                       # Profit of the investors
        self.summary.loc[pos,['final_demand']]           = final_demand                          # Electrical demand after the DSMS
        self.summary.loc[pos,['delivered_energy']]       = delivered_energy                      # Delivered energy
        self.summary.loc[pos,['diesel_consumption']]     = diesel_consumption                    # Diesel consumption
        self.summary.loc[pos,['diesel_cost']]            = diesel_cost                           # Diesel costs
        self.summary.loc[pos,['total_lcoe']]             = total_lcoe                            # Compute the LCOE
        self.summary.loc[pos,['government_lcoe']]        = government_lcoe                       # Compute the LCOE for the goverment
        self.summary.loc[pos,['private_lcoe']]           = private_lcoe                          # Compute the LCOE for the provate investors
        self.summary.loc[pos,['excess_energy']]          = excess_energy                         # Excess of energy  
        self.summary.loc[pos,['lack_energy']]            = lack_energy                           # Lack of energy 
        self.summary.loc[pos,['solving_time']]           = solving_time                          # Time that the solver takes to solve the optimization formulation
        
        
        #endregion
        
        #endregion

        #region to store the dispatch                       
        self.dispatch_results.load_flat                 = self.load_ini              # Initial load (without DSMS)
        self.dispatch_results.load_dsms                 = self.dsms.loadDSMS.value         # Final load (After the DSMS)
        self.dispatch_results.tariff                    = self.dsms.tariff.value     # Final price for the energy
        self.dispatch_results.dispatch_le               = self.sources_dict['lack_energy'].dispatch.value                    # Lack of energy
        self.dispatch_results.dispatch_ee               = self.sources_dict['excess_energy'].dispatch.value                  # Excess of energy
       
        if self.prob_info["battery_system"]:
            T=8760
            self.dispatch_results.bess_stored_energy    = self.sources_dict['battery_system'].dispatch.value                      # Residual energy in the BESS
            temp = np.zeros((8760,1))
            temp[0,0]      = 0                                              # Dispatch of the BESS at position zero
            temp[1:T,0]    = (self.sources_dict['battery_system'].dispatch.value[1:T]-self.sources_dict['battery_system'].dispatch.value[0:T-1]).flatten()                 # Dispatch of the BESS
            self.dispatch_results.bess_dispatch         = np.float64(temp)  # Dispatch of the BESS at position zero
        if self.prob_info["diesel_system"]:
            self.dispatch_results.dg_dispatch           = self.sources_dict['diesel_system'].dispatch.value
        if self.prob_info["pv_system"]:
            self.dispatch_results.pv_dispatch           = self.sources_dict['pv_system'].dispatch.value
        if self.prob_info["wind_system"]:
            self.dispatch_results.wind_dispatch         = self.sources_dict['wind_system'].dispatch.value
        #endregion
        
        #return the pandas structure
        return self.summary, self.dispatch_results

    def plotMG(self):       
        '''
        DeterministicDSMS.plotMG():  
        Method created to make some default plots of the results.   
        '''  

        hora = np.arange(24)
        dias = np.arange(365)
        
        #region to plot the conditions of the study case    
        demand_heatmap=np.transpose(self.dispatch_results.load_flat.values.reshape([365,24]))
        ghi_heatmap=np.transpose(self.prob_info['ghi'][0,0:8760].reshape([365,24]))
        wind_heatmap=np.transpose(self.prob_info['wind'][0,0:8760].reshape([365,24]))
        temperature_heatmap=np.transpose(self.prob_info['temperature'][0,0:8760].reshape([365,24]))
        list_heatmaps = [demand_heatmap,ghi_heatmap,wind_heatmap,temperature_heatmap]
        list_titles = ['Electrical demand','Global horizontal radiation','Wind speed','Temperature']
        list_abreviations = ['Demand','ghi','wind','temperature']
        list_units = ['[$kW$]', '[$kW/m^2$]','[$m/s$]', '[$^{\circ}C$]']
        list_colors = ['viridis','inferno','winter','coolwarm']
        
        for counter in range(len(list_heatmaps)):
            fig, ax = plt.subplots()
            ax.set_ylabel('Hours')
            ax.set_xlabel('Days')
            ax.set_xticks(dias[::30])
            ax.set_yticks(hora[::3])
            im = ax.imshow(list_heatmaps[counter], aspect=5, origin='lower',cmap=f'{list_colors[counter]}')
            cb = fig.colorbar(im,pad=0.02,aspect=11,shrink=0.355)
            cb.set_label(f'{list_abreviations[counter]} {list_units[counter]}')
            cb.ax.locator_params(nbins=9)
            ax.set_title(f'{list_titles[counter]}',fontsize=12)
            plt.show()

        #endregion

        #region to plot the dispatch                        
        means = self.dispatch_results.groupby(self.dispatch_results.index.hour).mean()
        stds  = self.dispatch_results.groupby(self.dispatch_results.index.hour).std()
        
        #region for the tariff plots                        
        flat_tariff_ini = np.ones((24,1))*self.prob_info['prxo']
        fig, ax = plt.subplots()
        ax.step(hora, flat_tariff_ini,  alpha=1, label=u'Flat tariff', linestyle='-', linewidth=2)
        ax.step(hora, means.tariff, alpha=1, label=u'DSMS tariff', linestyle='-', linewidth=2)
        ax.fill_between(hora, flat_tariff_ini.flatten(),    flat_tariff_ini.flatten() ,  alpha=0.3, label=u'Flat tariff STD',  linewidth=1.0)        
        ax.fill_between(hora, means.tariff - stds.tariff,    means.tariff + stds.tariff,  alpha=0.3, label=u'DSMS tariff STD',  linewidth=1.0)        
        ax.set_ylim(0,0.4)
        ax.yaxis.grid(True)
        ax.set_ylabel(u'USD',fontsize=14)
        ax.set_xlabel(u'hours',fontsize=14)
        ax.set_xticks(hora[::2])
        ax.set_xticklabels(hora[::2],fontsize=12)
        ax.legend(loc=7, bbox_to_anchor=(0.93, -0.32), ncol=2, frameon=False, shadow=False, fontsize=14)
        plt.title('Comparison of the tariffs',fontsize=16)


        fig.tight_layout()
        plt.show() 
        #endregion
                  
        #region to compare the demands                      
        fig, ax = plt.subplots()
        ax.step(hora, means.load_flat,            alpha=1, label=u'Demand flat',  linestyle='-',                   linewidth=2)
        ax.fill_between(hora, means.load_flat - stds.load_flat,    means.load_flat + stds.load_flat,     alpha=0.3, label=u'Demand flat STD',   linewidth=1.0)        
        ax.step(hora, means.load_dsms,            alpha=1, label=u'Demand DSMS',  linestyle='-',                   linewidth=2)
        ax.fill_between(hora, means.load_dsms - stds.load_dsms,    means.load_dsms + stds.load_dsms,     alpha=0.3, label=u'Demand DSMS STD',   linewidth=1.0)        

        ax.yaxis.grid(True)
        ax.set_ylabel(u'kW',fontsize=14)
        ax.set_xlabel(u'hours',fontsize=14)
        ax.set_xticks(hora[::2])
        ax.set_xticklabels(hora[::2],fontsize=12)
        ax.legend(loc=7, bbox_to_anchor=(0.98, -0.32), ncol=2, frameon=False, shadow=False, fontsize=14)
        plt.title('Comparison of the Demand',fontsize=16)

        fig.tight_layout()
        plt.show() 

        #endregion

        #region for the dispatch plot                       
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.step(hora, means.load_dsms,            alpha=1, label=u'Demand',  linestyle='-',                   linewidth=2)
        ax.fill_between(hora, means.load_dsms - stds.load_dsms,    means.load_dsms + stds.load_dsms,     alpha=0.3, label=u'Demand STD',   linewidth=1.0)        

        if self.prob_info["diesel_system"]:
            ax.step(hora, means.dg_dispatch,            alpha=1, label=u'DG',  linestyle='-',                   linewidth=2)
            ax.fill_between(hora, means.dg_dispatch - stds.dg_dispatch,    means.dg_dispatch + stds.dg_dispatch,     alpha=0.3, label=u'DG STD',   linewidth=1.0)        
        if self.prob_info["pv_system"]:
            ax.step(hora, means.pv_dispatch,            alpha=1, label=u'PV',  linestyle='-',                   linewidth=2)
            ax.fill_between(hora, means.pv_dispatch - stds.pv_dispatch,    means.pv_dispatch + stds.pv_dispatch,     alpha=0.3, label=u'PV STD',   linewidth=1.0)        
        if self.prob_info["battery_system"]:
            ax.step(hora, -means.bess_dispatch,            alpha=1, label=u'BESS',  linestyle='-',                   linewidth=2)
            ax.fill_between(hora, -means.bess_dispatch + stds.bess_dispatch,    -means.bess_dispatch - stds.bess_dispatch,     alpha=0.3, label=u'BESS STD',   linewidth=1.0)         
        if self.prob_info["wind_system"]:
            ax.step(hora, means.wind_dispatch,            alpha=1, label=u'Wind',  linestyle='-',                   linewidth=2)
            ax.fill_between(hora, means.wind_dispatch - stds.wind_dispatch,    means.wind_dispatch + stds.wind_dispatch,     alpha=0.3, label=u'Wind STD',   linewidth=1.0)        
        
        # Excess of energy
        ax.step(hora, means.dispatch_ee,            alpha=1, label=u'EE',  linestyle='-',                   linewidth=2)
        ax.fill_between(hora, means.dispatch_ee - stds.dispatch_ee,    means.dispatch_ee + stds.dispatch_ee,     alpha=0.3, label=u'EE STD',   linewidth=1.0)        

        # Lack of energy
        ax.step(hora, means.dispatch_le,            alpha=1, label=u'LE',  linestyle='-',                   linewidth=2)
        ax.fill_between(hora, means.dispatch_le - stds.dispatch_le,    means.dispatch_le + stds.dispatch_le,     alpha=0.3, label=u'LE STD',   linewidth=1.0)        


        ax.yaxis.grid(True)
        ax.set_ylabel(u'[kW]',fontsize=14)
        ax.set_xlabel(u'Time [hours]',fontsize=14)
        ax.set_xticks(hora[::3])
        ax.set_xticklabels(hora[::3],fontsize=12)
        ax.legend(loc=7, bbox_to_anchor=(1, -0.42), ncol=3, frameon=False, shadow=False, fontsize=12)
        plt.title('Dispatch',fontsize=16)

        fig.tight_layout()
        plt.show() 
        #endregion

        #region to plot the residual energy in the bess     
        
        if self.prob_info["battery_system"]:
            fig, ax = plt.subplots()
            ax.step(hora, means.bess_stored_energy,            alpha=1, label=u'BESS',  linestyle='-',                   linewidth=2)
            ax.fill_between(hora, means.bess_stored_energy - stds.bess_stored_energy,    means.bess_stored_energy + stds.bess_stored_energy,     alpha=0.3, label=u'BESS STD',   linewidth=1.0)        
            
            ax.yaxis.grid(True)
            ax.set_ylabel(u'[kW]',fontsize=14)
            ax.set_xlabel(u'Time [hours]',fontsize=14)
            ax.set_xticks(hora[::3])
            ax.set_xticklabels(hora[::3],fontsize=12)
            ax.legend()
            plt.title('BESS residual energy', fontsize=16)

            fig.tight_layout()
            plt.show()
        #endregion

        #endregion

#endregion

#region for the "one-year solver"                           

class YearlySolverDSMS():                                   
    '''
    Solver created for the multy year analysis. This solver computes the results of one year of operation of the microgrid. The yearly solver requires the following parameters:

    | **prob_info:** (dictionary) 
    | Contains the information related to the configuration of the optimization formulation. To make easier setting the parameters, a function named variables is created to import the values of problem info from a *.csv file.
    | To execute the function variables do: 
    ::                             
        prob_info = {}
        variables_csv   = pd.read_csv('config.csv', sep=';', header=None, skip_blank_lines=True)
        prob_info['scenarios'], prob_info["years"], prob_info["scala"], prob_info["prxo"], prob_info["percentage_yearly_growth"], prob_info["percentage_variation"], prob_info["dlcpercenthour"], prob_info["dlcpercenttotal"], prob_info["sen_ince"], prob_info["sen_ghi"], prob_info["elasticity"], prob_info["curtailment"], prob_info["capex_private"], prob_info["capex_gov"], prob_info["capex_community"], prob_info["capex_ong"], prob_info["opex_private"], prob_info["opex_gov"], prob_info["opex_community"], prob_info["opex_ong"], prob_info["rate_return_private"], prob_info["max_value_tariff"], prob_info['diesel_system'], prob_info['pv_system'], prob_info['battery_system'], prob_info['wind_system'], prob_info['hydro_system'], prob_info['hydrogen_system'], prob_info['gas_system'], prob_info['biomass_system'], prob_info['flat'], prob_info['tou'], prob_info['cpp'], prob_info['dadp'], prob_info['ince'], prob_info['dilc'], prob_info['residential'],prob_info['commercial'],prob_info['industrial'],prob_info['community']  = cm.variables(variables_csv)

    Where 'config.csv' is the file containing the information of prob_info. For more information refer to the documentation of the function variables. 

    | Additionally, prob_info should include the weather conditions of the geographical region where the IMG is going to be installed. To make easier the process of importing these data, resources_norm, resources_all and resources_noise functions were created. 
    | To execute one of the functions do:
    ::    
        data_csv        = pd.read_csv('resourcedata.csv', sep=';', header=None, skip_blank_lines=True)
        prob_info["ghi"], prob_info["irrdiffuse"], prob_info["temperature"], prob_info["wind"], prob_info["hydro"], prob_info["load_residential"], prob_info["load_commercial"], prob_info["load_industrial"], prob_info["load_community"] = cm.resources_all(data_csv, years=prob_info["years"], scenarios=prob_info['scenarios'], percentage_yearly_growth=prob_info["percentage_yearly_growth"], fit_type="weekly")

    For more information about the different options of how to generate the scenarios for the multiyear and stochastic analysis, please refer to resources_norm, resources_all and resources_noise.  

    **sources_info:** (dictionary) 
    
    Contains the information related to the configuration of the energy sources. 
    
    **simulation_year:** (int)
    Refers to the year to simulate. This parameter is controlled by the MultiyearDSMS constructor. 

    **summary:** (dictionary) 
    Refers to a dictionary that contains the results of simulating the microgrid in the years before to the simulation year. This parameter is controlled by the MultiyearDSMS constructor. 
    '''

    def __init__(self, prob_info, sources_info, simulation_year, summary):            

        #region to set the values of the last years                 

        self.simulation_year   = simulation_year
        self.prob_info    = prob_info
        self.sources_info = sources_info
        if self.simulation_year == 0:
            # Capacities at the initial year are zero
            self.last_capacity_wind_system    = 0
            self.last_capacity_battery_system = 0
            self.last_capacity_diesel_system  = 0
            self.last_capacity_pv_system      = 0
            
            self.year_summary   = {}
            # self.year_summary["yearly_capex"]={}
            horizon = self.prob_info['years'] + self.sources_info['pv_gen_1']['life_time']    
            # self.year_summary["yearly_capex"]['battery_system']   =  np.zeros((self.prob_info['years'],horizon))          
            # self.year_summary["yearly_capex"]['diesel_system']    =  np.zeros((self.prob_info['years'],horizon))
            # self.year_summary["yearly_capex"]['pv_system']        =  np.zeros((self.prob_info['years'],horizon))
            # self.year_summary["yearly_capex"]['wind_system']      =  np.zeros((self.prob_info['years'],horizon))       
            tax_diesel_system       = self.sources_info["diesel_gen_1"]["tax_tonne_price"] 
        else:
            self.year_summary = summary
        
            # Capacity of the last years
            tax_diesel_system           = self.sources_info["diesel_gen_1"]["tax_tonne_price"] 
            # if np.isnan(np.sum(self.year_summary['yearly_capex']['wind_system'][:,simulation_year])):
            #     self.last_capacity_battery_system = np.sum(self.year_summary['yearly_capacity']['battery_system'][:,simulation_year-1])
            #     self.last_capacity_diesel_system  = np.sum(self.year_summary['yearly_capacity']['diesel_system'][:,simulation_year-1] ) 
            #     self.last_capacity_pv_system      = np.sum(self.year_summary['yearly_capacity']['pv_system'][:,simulation_year-1]     ) 
            # else:
            self.last_capacity_wind_system    = np.sum(self.year_summary['yearly_capacity']['wind_system'][:,simulation_year-1]   )
            self.last_capacity_battery_system = np.sum(self.year_summary['yearly_capacity']['battery_system'][:,simulation_year-1])
            self.last_capacity_diesel_system  = np.sum(self.year_summary['yearly_capacity']['diesel_system'][:,simulation_year-1] ) 
            self.last_capacity_pv_system      = np.sum(self.year_summary['yearly_capacity']['pv_system'][:,simulation_year-1]     ) 
        
        #endregion

        #region to set the electrical demand                        
               
        if self.prob_info["residential"] == True: 
            tempload1 = self.prob_info['scala']*self.prob_info["load_residential"]
        if self.prob_info["commercial"]  == True:
            tempload1 = self.prob_info['scala']*self.prob_info["load_commercial"]
        if self.prob_info["industrial"]  == True:
            tempload1 = self.prob_info['scala']*self.prob_info["load_industrial"] 
        if self.prob_info["community"]   == True:
            tempload1 = self.prob_info['scala']*self.prob_info["load_community"]
        
        tempload2               = np.zeros((8760,1))

        # Como crear aqui un condicional para que sirva con y sin analisis estocastico

        # tempload2[:,0]          = tempload1 
        tempload2[:,0]          = tempload1[0,self.simulation_year*8760:(self.simulation_year+1)*8760] 
        self.load_ini           = tempload2
        
        #endregion

        #region to create a dictionary for the energy sources       
        self.sources_dict= {}
        
        # Lack and Excess of energy
        self.sources_dict['lack_energy']        = Lackenergy(cost_function=self.sources_info["lack_ene"]["cost_function"], reliability=self.sources_info["lack_ene"]["reliability"], years=1)                  # Create an object for the lack of energy
        self.sources_dict['excess_energy']      = Excessenergy(cost_function=self.sources_info["excess_ene"]["cost_function"], reliability=self.sources_info["excess_ene"]["reliability"], years=1)              # Create an object for the excess of energy

        if self.prob_info["battery_system"]     == True:    
            self.sources_dict['battery_system'] = EStorage(investment_cost=self.sources_info["bess_1"]["investment_cost"],fuel_function=self.sources_info["bess_1"]["fuel_function"],fuel_cost=self.sources_info["bess_1"]["fuel_cost"],maintenance_cost=self.sources_info["bess_1"]["maintenance_cost"],min_out_power=self.sources_info["bess_1"]["min_out_power"],max_out_power=self.sources_info["bess_1"]["max_out_power"],rate_up=self.sources_info["bess_1"]["rate_up"],rate_down=self.sources_info["bess_1"]["rate_down"], initial_charge = self.sources_info["bess_1"]["initial_charge"], years=1, yearly_total_capacity = self.last_capacity_battery_system)  
        
        if self.prob_info["diesel_system"]      == True:    
            self.sources_dict['diesel_system']  = EGenerator(investment_cost=self.sources_info["diesel_gen_1"]["investment_cost"],fuel_function=self.sources_info["diesel_gen_1"]["fuel_function"],fuel_cost=self.sources_info["diesel_gen_1"]["fuel_cost"],maintenance_cost=self.sources_info["diesel_gen_1"]["maintenance_cost"],min_out_power=self.sources_info["diesel_gen_1"]["min_out_power"],max_out_power=self.sources_info["diesel_gen_1"]["max_out_power"],rate_up=self.sources_info["diesel_gen_1"]["rate_up"],rate_down=self.sources_info["diesel_gen_1"]["rate_down"],years=1, yearly_total_capacity = self.last_capacity_diesel_system, tax_tonne_price=tax_diesel_system )
        
        if self.prob_info["pv_system"]          == True:    
            self.sources_dict['pv_system']      = PVGenerator(investment_cost=self.sources_info["pv_gen_1"]["investment_cost"], maintenance_cost=self.sources_info["pv_gen_1"]["maintenance_cost"], rate_up=self.sources_info["pv_gen_1"]["rate_up"], rate_down=self.sources_info["pv_gen_1"]["rate_down"], ghi=self.prob_info["ghi"], temperature=self.prob_info["temperature"], derat = self.sources_info["pv_gen_1"]["derat"], pstc= self.sources_info["pv_gen_1"]["pstc"], Ct = self.sources_info["pv_gen_1"]["Ct"], years=1, sen_ghi=self.prob_info["sen_ghi"], simulation_year=self.simulation_year, yearly_total_capacity = self.last_capacity_pv_system)
            
        if self.prob_info["wind_system"]        == True:    
            self.sources_dict['wind_system']    = WINDGenerator(investment_cost=self.sources_info["wind_gen_1"]["investment_cost"], maintenance_cost=self.sources_info["wind_gen_1"]["maintenance_cost"], rate_up=self.sources_info["wind_gen_1"]["rate_up"], rate_down=self.sources_info["wind_gen_1"]["rate_down"], wind=self.prob_info["wind"], years=1, rated_speed=self.sources_info["wind_gen_1"]["rated_speed"], speed_cut_in=self.sources_info["wind_gen_1"]["speed_cut_in"], speed_cut_out=self.sources_info["wind_gen_1"]["speed_cut_out"], nominal_capacity=self.sources_info["wind_gen_1"]["nominal_capacity"], simulation_year=self.simulation_year, yearly_total_capacity = self.last_capacity_wind_system)  
            
        # Execute the method to create the optimization formulation inside of the energy sources
        for unit in self.sources_dict.keys():
            self.sources_dict[unit].formulate()
        #endregion

        #region to set the formulation                              
        capex_sources_list, opex_list, maintenance_list = [], [], []
        for source in self.sources_dict.keys():
            capex_sources_list  += [self.sources_dict[source].capex]
            opex_list           += [self.sources_dict[source].opex]
            maintenance_list    += [self.sources_dict[source].maintenance]

        # Capex of the last years
        # if np.isnan(np.sum(self.year_summary['yearly_capex']['wind_system'][:,simulation_year])):
        #     capex_year = np.sum(self.year_summary['yearly_capex']['battery_system'][:,simulation_year]) + np.sum(self.year_summary['yearly_capex']['diesel_system'][:,simulation_year]) + np.sum(self.year_summary['yearly_capex']['pv_system'][:,simulation_year])
        # else:
        #     capex_year = np.sum(self.year_summary['yearly_capex']['battery_system'][:,simulation_year]) + np.sum(self.year_summary['yearly_capex']['diesel_system'][:,simulation_year]) + np.sum(self.year_summary['yearly_capex']['pv_system'][:,simulation_year])+np.sum(self.year_summary['yearly_capex']['wind_system'][:,simulation_year])
        
        # self.tcapex             = cp.sum(capex_sources_list) + capex_year   # Compute the Capex and add the capex of the last years
        self.tcapex             = cp.sum(capex_sources_list)        # Compute the Capex 
        self.topex              = cp.sum(opex_list)                 # Compute the Opex 
        self.tmaintenance       = cp.sum(maintenance_list)          # Compute the maintenance
        self.ttaxes             = cp.sum(self.sources_dict['diesel_system'].tax_tonne)
        self.objective          = self.tcapex + self.topex + self.tmaintenance + self.ttaxes 
        #endregion

        #region to compute the Demand Response                      
        yearly_prices = LookTable()
        cpi = yearly_prices.table_cpi[self.simulation_year]
        self.initial_price_tem = self.prob_info["prxo"]*(1+cpi/100)**(simulation_year)
        self.max_tarrif_tem    = self.initial_price_tem*2
        self.dsms               = ComputeDSMS(loadfile=self.load_ini, elasticity=self.prob_info["elasticity"], initial_price=self.initial_price_tem, max_value_tariff = self.max_tarrif_tem,                years = 1, dlcpercenthour = self.prob_info["dlcpercenthour"], dlcpercenttotal= self.prob_info["dlcpercenttotal"], flat = self.prob_info["flat"], tou = self.prob_info["tou"], tou_sun = self.prob_info["tou_sun"], tou_three = self.prob_info["tou_three"], cpp = self.prob_info["cpp"], dadp = self.prob_info["dadp"], shape_tar = self.prob_info["shape_tar"], ince = self.prob_info["ince"], dilc = self.prob_info["dilc"], drpercentage = prob_info['drpercentage'])
        #endregion 

        #region to set the constraints of the formulation           
        self.constraints        = []  

        for source in self.sources_dict.keys(): 
            for constraint in self.sources_dict[source].constraints:
                self.constraints.append(constraint)

        # Import the constraints of the DSMS
        for constraint in self.dsms.constraints:
            self.constraints.append(constraint)
        
        # Set the general constraints of the formulation
        if self.prob_info["dilc"] == True:
            self.constraints += [   
                                    cp.sum(self.sources_dict['lack_energy'].dispatch)       <= self.sources_dict['lack_energy'].reliability/100*cp.sum(self.dsms.loadDSMS),        # Maximum allowed lack of energy                                   
                                    cp.sum(self.sources_dict['excess_energy'].dispatch)     >= -self.sources_dict['excess_energy'].reliability/100*cp.sum(self.dsms.loadDSMS),     # Maximum allowed excess of energy
                                    # cp.sum(self.dsms.loadDSMS) - self.prob_info["curtailment"]*cp.sum(self.load_ini) >= 0,                       # Allowed curtailment after the application of the tariff
                                    # cp.sum(self.dsms.customer_payments) >= (self.prob_info["capex_private"]*self.tcapex+self.prob_info["opex_private"]*self.topex+self.tmaintenance+self.ttaxes)*self.prob_info["rate_return_private"], # Guarantee to private investors to recover their capital
                                ]                                     
        else:   
            self.constraints += [   
                                    cp.sum(self.sources_dict['lack_energy'].dispatch)       <= self.sources_dict['lack_energy'].reliability/100*cp.sum(self.dsms.loadDSMS),        # Maximum allowed lack of energy                                   
                                    cp.sum(self.sources_dict['excess_energy'].dispatch)     >= -self.sources_dict['excess_energy'].reliability/100*cp.sum(self.dsms.loadDSMS),     # Maximum allowed excess of energy
                                    cp.sum(self.dsms.loadDSMS) - self.prob_info["curtailment"]*cp.sum(self.load_ini) >= 0,                       # Allowed curtailment after the application of the tariff
                                    # cp.sum(self.dsms.customer_payments) >= (self.prob_info["capex_private"]*self.tcapex+self.prob_info["opex_private"]*self.topex+self.tmaintenance+self.ttaxes)*self.prob_info["rate_return_private"], # Guarantee to private investors to recover their capital
                                ]                                     

        #region to perform the energy balance with the battery without efficiency   
        net_dispatch=[]                                      
        for source in self.sources_dict.keys():
            if source == "battery_system":
                pass
            else:
                net_dispatch  += [self.sources_dict[source].dispatch]
        
        T=8760
        if self.prob_info["battery_system"]  == True:
            self.constraints +=  [ self.sources_dict["battery_system"].dispatch[1:T] == self.sources_dict["battery_system"].dispatch[0:T-1] + cp.sum(net_dispatch)[1:T] - self.dsms.loadDSMS[1:T]]# Energy balance 
        else:
            self.constraints +=  [  cp.sum(net_dispatch) - self.dsms.loadDSMS == 0]# Energy balance
        #endregion

        #endregion

        #region to build the optimization formulation               
        self.formulation = cp.Problem(cp.Minimize(self.objective), self.constraints)
        #endregion   
                                      
    def solveMG(self):                                              
        '''
        YearlySolverDSMS.solveMG():  
        Method created to solve the optimization formulation. 
        '''      
        initial_time = time() 
        self.formulation.solve(solver=cp.MOSEK)
        final_time = time() 
        self.execution_time = final_time - initial_time
        return f"The status of the optimization formulation is: {self.formulation.status}"

    def resultsMG(self):
        '''
        YearlySolverDSMS.resultsMG():  
        Method created to obtain the results of the optimization. It returns a tuple. summary, and dispatch_results.  
        '''                                             
        #region to create the pandas structures             
        self.year_summary   = pd.DataFrame(columns=['up_limit_tariff','scenarios','sen_ince','sen_batcost','sen_ghi','sen_pvcost','sen_diecost','sen_gdcost', 'curtailment', 'internal_rate_of_return', 'elasticity', 'total_costs', 'total_capex', 'total_maintenance', 'total_taxes', 'total_opex', 'private_investment', 'private_capex','private_opex','customer_payments','private_profits','final_demand','delivered_energy','diesel_consumption','diesel_cost','total_lcoe','government_lcoe','private_lcoe','excess_energy','lack_energy','pv_capacity_this_year','dg_capacity_this_year','bess_capacity_this_year','wind_capacity_this_year','pv_capacity','dg_capacity','bess_capacity','wind_capacity','solving_time'])
        self.dispatch_results  = pd.DataFrame(columns=['date','tariff','load_flat', 'load_dsms', 'pv_dispatch', 'wind_dispatch', 'dg_dispatch', 'bess_dispatch', 'bess_stored_energy', 'dispatch_ee','dispatch_le'])
        self.dispatch_results.date=pd.date_range('2018-01-01', periods=8760, freq='H')
        self.dispatch_results.set_index('date',inplace=True)
        #endregion

        #region to store the summary                        
        
        #region to perform operations to obtain results     
        pos                     = 0
        total_capex             = self.tcapex.value                                         # Total costs of the CAPEX  
        total_opex              = self.topex.value                                          # Total costs of the OPEX
        total_maintenance       = self.tmaintenance.value                                   # Total costs of the maintenance
        total_taxes             = self.ttaxes.value                                         # Total costs of taxes
        total_costs             = total_capex+total_opex+total_maintenance + total_taxes    # Total costs of the project
        private_capex           = self.prob_info["capex_private"]*total_capex               # Value of the capex paid by the private investors
        private_opex            = self.prob_info["opex_private"]*total_opex                 # Value of the opex paid by the private investors
        private_investment      = private_capex + private_opex + total_maintenance + total_taxes  # Total investment paid by the private investors
        government_capex        = self.prob_info['capex_gov']*total_capex                   # Value of the capex paid by the government
        government_opex         = self.prob_info['opex_gov']*total_opex                     # Value of the opex paid by the government
        government_investment   = government_capex + government_opex                        # Total investment paid by the government
        customer_payments       = np.sum(self.dsms.customer_payments.value)
        private_profits         = customer_payments-private_investment 
        rate_of_return          = private_profits/private_investment    
        final_demand            = np.sum(self.dsms.loadDSMS.value)                                                  # Sum of the total final demand
        total_lcoe              = total_costs/final_demand
        private_lcoe            = private_investment/final_demand                      
        government_lcoe         = government_investment/final_demand                      
        lack_energy             = np.sum(self.sources_dict['lack_energy'].dispatch.value)                
        excess_energy           = np.absolute(np.sum(self.sources_dict['excess_energy'].dispatch.value)) 
        delivered_energy        = final_demand-excess_energy-lack_energy      # Delivered energy
        solving_time            = self.execution_time                         
        
        #region for the diesel operation                    
        if self.prob_info["diesel_system"]   == True:
            if type(self.sources_info["diesel_gen_1"]["fuel_function"]) == np.ndarray:  
                if len(self.sources_info["diesel_gen_1"]["fuel_function"]) == 3:
                    diesel_consumption           = self.sources_info["diesel_gen_1"]["fuel_function"][0]*np.sum(self.sources_dict['diesel_system'].dispatch.value**2) + self.sources_info["diesel_gen_1"]["fuel_function"][1]*np.sum(self.sources_dict['diesel_system'].dispatch.value) + self.sources_info["diesel_gen_1"]["fuel_function"][2]                 
                if len(self.sources_info["diesel_gen_1"]["fuel_function"]) == 2:
                    diesel_consumption           = self.sources_info["diesel_gen_1"]["fuel_function"][0]*np.sum(self.sources_dict['diesel_system'].dispatch.value) + self.sources_info["diesel_gen_1"]["fuel_function"][1]*self.sources_dict['diesel_system'].capacity.value                 
                if len(self.sources_info["diesel_gen_1"]["fuel_function"]) == 1:
                    diesel_consumption           = self.sources_info["diesel_gen_1"]["fuel_function"][0]*np.sum(self.sources_dict['diesel_system'].dispatch.value)                 
            else:
                diesel_consumption           = 0
        diesel_cost         = self.sources_info['diesel_gen_1']['fuel_cost']*diesel_consumption
        #endregion   
        
        #endregion   

        #region to store in the pandas structure            
        self.year_summary.loc[pos,['scenarios']]              = self.prob_info['scenarios']           # Number of escenarios
        self.year_summary.loc[pos,['sen_ince']]               = self.prob_info['sen_ince']            # Sensitivity analysis of values of incentives
        # self.year_summary.loc[pos,['sen_batcost']]            = self.prob_info['sen_batcost']       # Sensitivity analysis of cost of the batteries
        self.year_summary.loc[pos,['sen_ghi']]                = self.prob_info['sen_ghi']             # Sensitivity analysis of global horizontal radiation
        # self.year_summary.loc[pos,['sen_pvcost']]             = self.prob_info['sen_pvcost']        # Sensitivity analysis of costs of the pv
        # self.year_summary.loc[pos,['sen_diecost']]            = self.prob_info['sen_diecost']       # Sensitivity analysis of costs of the diesel
        # self.year_summary.loc[pos,['sen_gdcost']]             = self.prob_info['sen_gdcost']        # Sensitivity analysis of costs of the diesel generator
        self.year_summary.loc[pos,['up_limit_tariff']]        = self.max_tarrif_tem                        # Save the value of the tariff
        self.year_summary.loc[pos,['curtailment']]            = self.prob_info['curtailment']         # Save the curtailment factor
        self.year_summary.loc[pos,['elasticity']]             = self.prob_info['elasticity']*(-1)     # Save the elasticity
        self.year_summary.loc[pos,['total_costs']]            = total_costs                           # Total costs
        self.year_summary.loc[pos,['total_maintenance']]      = total_maintenance                     # Total costs of the maintenance
        self.year_summary.loc[pos,['internal_rate_of_return']]= rate_of_return                     # Rate oif return
        self.year_summary.loc[pos,['total_taxes']]            = total_taxes                           # Total costs of the taxes
        self.year_summary.loc[pos,['total_capex']]            = total_capex                           # Total costs of the capex
        self.year_summary.loc[pos,['total_opex']]             = total_opex                            # total costs of the opex
        self.year_summary.loc[pos,['private_investment']]     = private_investment                    # Total payments of the investors
        self.year_summary.loc[pos,['private_capex']]          = private_capex                         # Payments of the investors for the capex                          
        self.year_summary.loc[pos,['private_opex']]           = private_opex                          # Payments of the investors for the opex                             
        self.year_summary.loc[pos,['customer_payments']]      = customer_payments                     # Payments of the customers after the introduction of the dsms         
        self.year_summary.loc[pos,['private_profits']]        = private_profits                       # Profit of the investors
        self.year_summary.loc[pos,['final_demand']]           = final_demand                          # Electrical demand after the DSMS
        self.year_summary.loc[pos,['delivered_energy']]       = delivered_energy                      # Delivered energy
        self.year_summary.loc[pos,['diesel_consumption']]     = diesel_consumption                    # Diesel consumption
        self.year_summary.loc[pos,['diesel_cost']]            = diesel_cost                           # Diesel costs
        self.year_summary.loc[pos,['total_lcoe']]             = total_lcoe                            # Compute the LCOE
        self.year_summary.loc[pos,['government_lcoe']]        = government_lcoe                       # Compute the LCOE for the goverment
        self.year_summary.loc[pos,['private_lcoe']]           = private_lcoe                          # Compute the LCOE for the provate investors
        self.year_summary.loc[pos,['excess_energy']]          = excess_energy                         # Excess of energy  
        self.year_summary.loc[pos,['lack_energy']]            = lack_energy                           # Lack of energy 
        self.year_summary.loc[pos,['solving_time']]           = solving_time                          # Time that the solver takes to solve the optimization formulation
        #endregion
        
        

        #region to store total and partial capacities       
        if self.prob_info["battery_system"]  == True:
            self.year_summary.loc[pos,['bess_capacity']]         = self.sources_dict['battery_system'].capacity.value                        # Installed capacity of BESS
        if self.prob_info["diesel_system"]   == True:
            self.year_summary.loc[pos,['dg_capacity']]           = self.sources_dict['diesel_system'].capacity.value                           # Installed capacity of DG
        if self.prob_info["pv_system"]   == True:
            self.year_summary.loc[pos,['pv_capacity']]           = self.sources_dict['pv_system'].capacity.value                           # Installed capacity of PV
        if self.prob_info["wind_system"]   == True:
            self.year_summary.loc[pos,['wind_capacity']]         = self.sources_dict['wind_system'].capacity.value                           # Installed capacity of PV
        
        if self.prob_info["battery_system"]  == True:
            self.year_summary.loc[pos,['bess_capacity_this_year']]         = self.sources_dict['battery_system'].capacity.value - self.last_capacity_battery_system                        # Installed capacity of BESS
        if self.prob_info["diesel_system"]   == True:
            self.year_summary.loc[pos,['dg_capacity_this_year']]           = self.sources_dict['diesel_system'].capacity.value - self.last_capacity_diesel_system                           # Installed capacity of DG
        if self.prob_info["pv_system"]   == True:
            self.year_summary.loc[pos,['pv_capacity_this_year']]           = self.sources_dict['pv_system'].capacity.value - self.last_capacity_pv_system                          # Installed capacity of PV
        if self.prob_info["wind_system"]   == True:
            self.year_summary.loc[pos,['wind_capacity_this_year']]         = self.sources_dict['wind_system'].capacity.value - self.last_capacity_wind_system                          # Installed capacity of PV
        #endregion
        
        #endregion

        #region to store the dispatch                       
        self.dispatch_results.load_flat                 = self.load_ini              # Initial load (without DSMS)
        self.dispatch_results.load_dsms                 = self.dsms.loadDSMS.value         # Final load (After the DSMS)
        self.dispatch_results.tariff                    = self.dsms.tariff.value     # Final price for the energy
        self.dispatch_results.dispatch_le               = self.sources_dict['lack_energy'].dispatch.value                    # Lack of energy
        self.dispatch_results.dispatch_ee               = self.sources_dict['excess_energy'].dispatch.value                  # Excess of energy
    
        if self.prob_info["battery_system"]:
            T=8760
            self.dispatch_results.bess_stored_energy    = self.sources_dict['battery_system'].dispatch.value                      # Residual energy in the BESS
            temp = np.zeros((8760,1))
            temp[0,0]      = 0                                              # Dispatch of the BESS at position zero
            temp[1:T,0]    = (self.sources_dict['battery_system'].dispatch.value[1:T]-self.sources_dict['battery_system'].dispatch.value[0:T-1]).flatten()                 # Dispatch of the BESS
            self.dispatch_results.bess_dispatch         = np.float64(temp)  # Dispatch of the BESS at position zero
        if self.prob_info["diesel_system"]:
            self.dispatch_results.dg_dispatch           = self.sources_dict['diesel_system'].dispatch.value
        if self.prob_info["pv_system"]:
            self.dispatch_results.pv_dispatch           = self.sources_dict['pv_system'].dispatch.value
        if self.prob_info["wind_system"]:
            self.dispatch_results.wind_dispatch         = self.sources_dict['wind_system'].dispatch.value
        #endregion
        
        #return the pandas structure
        return self.year_summary, self.dispatch_results

#endregion

#region for the "multiyear constructor"                     

class MultiyearDSMS():                                      
    '''
    Multiyear constructor to ensamble the microgrid using Demand Side Management Strategies. The multy year constructor implements the adaptative method described in [pece2019]_. However, the multy year analysis here does not consider three day each month as the authors propose in the article. The multy-year analysis here considers the full year analysis (8760 hours).
    
    The multy year constructor requires the following parameters:

    | **prob_info:** (dictionary) 
    | Contains the information related to the configuration of the optimization formulation. To make easier setting the parameters, a function named variables is created to import the values of problem info from a *.csv file.
    | To execute the function variables do: 
    ::                             
        prob_info = {}
        variables_csv   = pd.read_csv('config.csv', sep=';', header=None, skip_blank_lines=True)
        prob_info['scenarios'], prob_info["years"], prob_info["scala"], prob_info["prxo"], prob_info["percentage_yearly_growth"], prob_info["percentage_variation"], prob_info["dlcpercenthour"], prob_info["dlcpercenttotal"], prob_info["sen_ince"], prob_info["sen_ghi"], prob_info["elasticity"], prob_info["curtailment"], prob_info["capex_private"], prob_info["capex_gov"], prob_info["capex_community"], prob_info["capex_ong"], prob_info["opex_private"], prob_info["opex_gov"], prob_info["opex_community"], prob_info["opex_ong"], prob_info["rate_return_private"], prob_info["max_value_tariff"], prob_info['diesel_system'], prob_info['pv_system'], prob_info['battery_system'], prob_info['wind_system'], prob_info['hydro_system'], prob_info['hydrogen_system'], prob_info['gas_system'], prob_info['biomass_system'], prob_info['flat'], prob_info['tou'], prob_info['cpp'], prob_info['dadp'], prob_info['ince'], prob_info['dilc'], prob_info['residential'],prob_info['commercial'],prob_info['industrial'],prob_info['community']  = cm.variables(variables_csv)

    Where 'config.csv' is the file containing the information of prob_info. For more information refer to the documentation of the function variables. 

    | Additionally, prob_info should include the weather conditions of the geographical region where the IMG is going to be installed. To make easier the process of importing these data, resources_norm, resources_all and resources_noise functions were created. 
    | To execute one of the functions do:
    ::    
        data_csv        = pd.read_csv('resourcedata.csv', sep=';', header=None, skip_blank_lines=True)
        prob_info["ghi"], prob_info["irrdiffuse"], prob_info["temperature"], prob_info["wind"], prob_info["hydro"], prob_info["load_residential"], prob_info["load_commercial"], prob_info["load_industrial"], prob_info["load_community"] = cm.resources_all(data_csv, years=prob_info["years"], scenarios=prob_info['scenarios'], percentage_yearly_growth=prob_info["percentage_yearly_growth"], fit_type="weekly")

    For more information about the different options of how to generate the scenarios for the multiyear and stochastic analysis, please refer to resources_norm, resources_all and resources_noise.  

    **sources_info:** (dictionary) 
    
    Contains the information related to the configuration of the energy sources. The following parameters are expected:
    
    .. [pece2019] Z. K. Pecenak, M. Stadler, and K. Fahy, “Efficient multi-year economic energy planning in microgrids,” Appl. Energy, vol. 255, no. August, p. 113771, 2019.

    '''
    def __init__(self, prob_info_input, sources_info):            
        self.prob_info = prob_info_input
        # self.prob_info['ghi'] = prob_info_input['ghi'][scenario]
        # self.prob_info['irrdiffuse'] = prob_info_input['irrdiffuse'][scenario]
        # self.prob_info['temperature'] = prob_info_input['temperature'][scenario]
        # self.prob_info['wind'] = prob_info_input['wind'][scenario]
        # self.prob_info['hydro'] = prob_info_input['hydro'][scenario]
        # self.prob_info['load_residential'] = prob_info_input['load_residential'][scenario]
        # self.prob_info['load_commercial'] = prob_info_input['load_commercial'][scenario]
        # self.prob_info['load_industrial'] = prob_info_input['load_industrial'][scenario]
        # self.prob_info['load_community'] = prob_info_input['load_community'][scenario]
        self.sources_info = sources_info

        #region to initialize values for the yearly solver          

        #region to initialize the capacities matrix                 
        self.summary = {}
        self.dispatch=[]
        self.summary["yearly_results"] = []
        self.summary["dispatch_results"] = []
        self.summary["yearly_installed_capacities"] = {}
        self.summary["yearly_installed_capacities"]['battery_system']   = np.zeros((1,self.prob_info['years']))
        self.summary["yearly_installed_capacities"]['diesel_system']    = np.zeros((1,self.prob_info['years']))
        self.summary["yearly_installed_capacities"]['pv_system']        = np.zeros((1,self.prob_info['years']))
        self.summary["yearly_installed_capacities"]['wind_system']      = np.zeros((1,self.prob_info['years']))
        #endregion
        
        #region to create the capex values                          
        self.summary["yearly_capex"]={}
        self.summary["yearly_capex"]['battery_system']  = np.zeros((1,self.prob_info['years']))
        self.summary["yearly_capex"]['diesel_system']   = np.zeros((1,self.prob_info['years']))
        self.summary["yearly_capex"]['pv_system']       = np.zeros((1,self.prob_info['years']))
        self.summary["yearly_capex"]['wind_system']     = np.zeros((1,self.prob_info['years']))
        self.summary["yearly_capex"]['total']           = np.zeros((1,self.prob_info['years']))
        #endregion

        #region to create the opex values                           
        self.summary["yearly_opex"]={}
        self.summary["yearly_opex"]['battery_system']   = np.zeros((1,self.prob_info['years']))
        self.summary["yearly_opex"]['diesel_system']    = np.zeros((1,self.prob_info['years']))
        self.summary["yearly_opex"]['pv_system']        = np.zeros((1,self.prob_info['years']))
        self.summary["yearly_opex"]['wind_system']      = np.zeros((1,self.prob_info['years']))
        self.summary["yearly_opex"]['total']            = np.zeros((1,self.prob_info['years']))
        #endregion

        #region to create the maintenance values                    
        self.summary["yearly_maintenance"]={}
        self.summary["yearly_maintenance"]['battery_system']    = np.zeros((1,self.prob_info['years']))
        self.summary["yearly_maintenance"]['diesel_system']     = np.zeros((1,self.prob_info['years']))
        self.summary["yearly_maintenance"]['pv_system']         = np.zeros((1,self.prob_info['years']))
        self.summary["yearly_maintenance"]['wind_system']       = np.zeros((1,self.prob_info['years']))
        self.summary["yearly_maintenance"]['total']             = np.zeros((1,self.prob_info['years']))
        #endregion

        #region to create the customer payments values              
        self.summary["yearly_customer_payments"]             = np.zeros((1,self.prob_info['years']))
        #endregion

        #region to create the total private investments values      
        self.summary["yearly_private_payments"]              = np.zeros((1,self.prob_info['years']))
        #endregion

        #region to create the capacity matrix                       
        self.summary["yearly_capacity"]={}  
        horizon = self.prob_info['years'] + self.sources_info['pv_gen_1']['life_time']    
        self.summary["yearly_capacity"]['battery_system']   =  np.zeros((self.prob_info['years'],horizon))          
        self.summary["yearly_capacity"]['diesel_system']    =  np.zeros((self.prob_info['years'],horizon))
        self.summary["yearly_capacity"]['pv_system']        =  np.zeros((self.prob_info['years'],horizon))
        self.summary["yearly_capacity"]['wind_system']      =  np.zeros((self.prob_info['years'],horizon))
        #endregion

        #endregion

        yearly_prices = LookTable()
       
        for simulation_year in range(self.prob_info['years']):

            # #region to actualice the cost parameters of the generators for the low scenarios      
            # cpi = yearly_prices.table_cpi_low[simulation_year+2]
            # self.sources_info["bess_1"]["investment_cost"]      = yearly_prices.table_bess_low[simulation_year+2]  *(1+cpi/100)**(simulation_year)   
            # self.sources_info["pv_gen_1"]["investment_cost"]    = yearly_prices.table_pv_low[simulation_year+2]    *(1+cpi/100)**(simulation_year)     
            # self.sources_info["wind_gen_1"]["investment_cost"]  = yearly_prices.table_wind_low[simulation_year+2]  *(1+cpi/100)**(simulation_year)   
            # self.sources_info["diesel_gen_1"]["investment_cost"]= yearly_prices.table_genset_low[simulation_year+2]*(1+cpi/100)**(simulation_year)     
            # self.sources_info["diesel_gen_1"]["fuel_cost"]      = yearly_prices.table_diesel_low[simulation_year+2]*(1+cpi/100)**(simulation_year)            
            # self.sources_info["diesel_gen_1"]["tax_tonne_price"]= yearly_prices.table_ctax_low[simulation_year+2]  *(1+cpi/100)**(simulation_year)   
            # #endregion
            
            #region to actualice the cost parameters of the generators for the standard scenarios      
            cpi = yearly_prices.table_cpi[simulation_year+2]
            self.sources_info["bess_1"]["investment_cost"]      = yearly_prices.table_bess[simulation_year+2]  *(1+cpi/100)**(simulation_year)   
            self.sources_info["pv_gen_1"]["investment_cost"]    = yearly_prices.table_pv[simulation_year+2]    *(1+cpi/100)**(simulation_year)     
            self.sources_info["wind_gen_1"]["investment_cost"]  = yearly_prices.table_wind[simulation_year+2]  *(1+cpi/100)**(simulation_year)   
            self.sources_info["diesel_gen_1"]["investment_cost"]= yearly_prices.table_genset[simulation_year+2]*(1+cpi/100)**(simulation_year)     
            self.sources_info["diesel_gen_1"]["fuel_cost"]      = yearly_prices.table_diesel[simulation_year+2]*(1+cpi/100)**(simulation_year)            
            self.sources_info["diesel_gen_1"]["tax_tonne_price"]= yearly_prices.table_ctax[simulation_year+2]  *(1+cpi/100)**(simulation_year)   
            #endregion

            #region to call the constructor and solve the problem           
            year = YearlySolverDSMS(self.prob_info, self.sources_info, simulation_year, self.summary)
            year.solveMG()
            #endregion

            #region to extract the results and save them                    
            
            #region to import results                                           
            year_summary, year_dispatch_results = year.resultsMG()
            self.summary["yearly_results"].append(year_summary)
            self.dispatch.append(year_dispatch_results)         
            #endregion

            #region for the installed capacities                                
            
            self.summary["yearly_installed_capacities"]['battery_system'][0,simulation_year]   = float(self.summary["yearly_results"][simulation_year].bess_capacity_this_year.values)
            self.summary["yearly_installed_capacities"]['diesel_system'] [0,simulation_year]   = float(self.summary["yearly_results"][simulation_year].dg_capacity_this_year.values  )
            self.summary["yearly_installed_capacities"]['pv_system']     [0,simulation_year]   = float(self.summary["yearly_results"][simulation_year].pv_capacity_this_year.values  )
            self.summary["yearly_installed_capacities"]['wind_system']   [0,simulation_year]   = float(self.summary["yearly_results"][simulation_year].wind_capacity_this_year.values)    

            #endregion

            #region to compute the capex paid each year                         
            if self.prob_info['battery_system'] == True:  
                self.summary["yearly_capex"]['battery_system'][0,simulation_year]   = float(self.summary["yearly_results"][simulation_year].bess_capacity_this_year.values)*self.sources_info["bess_1"]["investment_cost"]*self.prob_info['capex_private']
            else:
                self.summary["yearly_capex"]['battery_system'][0,simulation_year]   = 0

            if self.prob_info['diesel_system'] == True:    
                self.summary["yearly_capex"]['diesel_system' ][0,simulation_year]   = float(self.summary["yearly_results"][simulation_year].dg_capacity_this_year.values  )*self.sources_info["diesel_gen_1"]["investment_cost"]*self.prob_info['capex_private']
            else:    
                self.summary["yearly_capex"]['diesel_system' ][0,simulation_year]   = 0
            
            if self.prob_info['pv_system'] == True:    
                self.summary["yearly_capex"]['pv_system'     ][0,simulation_year]   = float(self.summary["yearly_results"][simulation_year].pv_capacity_this_year.values  )*self.sources_info["pv_gen_1"]["investment_cost"]*self.prob_info['capex_private']
            else:    
                self.summary["yearly_capex"]['pv_system'     ][0,simulation_year]   = 0
            
            if self.prob_info['wind_system'] == True:
                self.summary["yearly_capex"]['wind_system'   ][0,simulation_year]   = float(self.summary["yearly_results"][simulation_year].wind_capacity_this_year.values)*self.sources_info["wind_gen_1"]["investment_cost"]*self.prob_info['capex_private']
            else:
                self.summary["yearly_capex"]['wind_system'   ][0,simulation_year]   = 0
            
            self.summary["yearly_capex"]['total'         ][0,simulation_year]   = self.summary["yearly_capex"]['battery_system'][0,simulation_year]+self.summary["yearly_capex"]['diesel_system'] [0,simulation_year]+self.summary["yearly_capex"]['pv_system']     [0,simulation_year]+self.summary["yearly_capex"]['wind_system']   [0,simulation_year]
            #endregion

            #region to compute the opex paid each year 
            if self.prob_info['battery_system'] == True:                         
                self.summary["yearly_opex"]['battery_system'][0,simulation_year]   = year.sources_dict['battery_system'].opex*self.prob_info['opex_private']
            else:
                self.summary["yearly_opex"]['battery_system'][0,simulation_year]   = 0

            if self.prob_info['diesel_system'] == True:
                self.summary["yearly_opex"]['diesel_system' ][0,simulation_year]   = year.sources_dict['diesel_system' ].opex.value*self.prob_info['opex_private']
            else:
                self.summary["yearly_opex"]['diesel_system' ][0,simulation_year]   = 0
            
            if self.prob_info['pv_system'] == True:
                self.summary["yearly_opex"]['pv_system'     ][0,simulation_year]   = year.sources_dict['pv_system'     ].opex*self.prob_info['opex_private']
            else:
                self.summary["yearly_opex"]['pv_system'     ][0,simulation_year]   = 0

            if self.prob_info['wind_system'] == True:
                self.summary["yearly_opex"]['wind_system'   ][0,simulation_year]   = year.sources_dict['wind_system'   ].opex*self.prob_info['opex_private']
            else:
                self.summary["yearly_opex"]['wind_system'   ][0,simulation_year]   = 0
                
            self.summary["yearly_opex"]['total'         ][0,simulation_year]   = self.summary["yearly_opex"]['battery_system'][0,simulation_year]+ self.summary["yearly_opex"]['diesel_system' ][0,simulation_year]+self.summary["yearly_opex"]['pv_system'     ][0,simulation_year]+self.summary["yearly_opex"]['wind_system'   ][0,simulation_year]
            #endregion

            #region to compute the maintenance paid each year                   
            if self.prob_info['battery_system'] == True:
                self.summary["yearly_maintenance"]['battery_system'][0,simulation_year]   = year.sources_dict['battery_system'].maintenance.value
            else:
                self.summary["yearly_maintenance"]['battery_system'][0,simulation_year]   = 0

            if self.prob_info['diesel_system'] == True:
                self.summary["yearly_maintenance"]['diesel_system' ][0,simulation_year]   = year.sources_dict['diesel_system' ].maintenance.value
            else:
                self.summary["yearly_maintenance"]['diesel_system' ][0,simulation_year]   = 0
            
            if self.prob_info['pv_system'] == True:
                self.summary["yearly_maintenance"]['pv_system'     ][0,simulation_year]   = year.sources_dict['pv_system'     ].maintenance.value
            else:
                self.summary["yearly_maintenance"]['pv_system'     ][0,simulation_year]   = 0
            
            if self.prob_info['wind_system'] == True:
                self.summary["yearly_maintenance"]['wind_system'   ][0,simulation_year]   = year.sources_dict['wind_system'   ].maintenance.value
            else:
                self.summary["yearly_maintenance"]['wind_system'   ][0,simulation_year]   = 0
                
            self.summary["yearly_maintenance"]['total'         ][0,simulation_year]   = self.summary["yearly_maintenance"]['battery_system'][0,simulation_year]+self.summary["yearly_maintenance"]['diesel_system' ][0,simulation_year]+self.summary["yearly_maintenance"]['pv_system'     ][0,simulation_year]+self.summary["yearly_maintenance"]['wind_system'   ][0,simulation_year]
            #endregion

            #region to compute the payments of the private investors each year  
            self.summary["yearly_private_payments"][0,simulation_year]    = self.summary["yearly_capex"]['total'][0,simulation_year]+self.summary["yearly_opex"]['total'][0,simulation_year]+self.summary["yearly_maintenance"]['total'][0,simulation_year]
            #endregion
            
            #region to compute the payments of the customers each year          
            self.summary["yearly_customer_payments"][0,simulation_year]   = np.sum(year.dsms.customer_payments.value)
            #endregion

            #region to fill the capacity matrix                                 
            for counter in range(horizon):                
                if counter >= simulation_year and counter < self.sources_info["bess_1"]["life_time"]       + simulation_year:
                    self.summary["yearly_capacity"]['battery_system'][simulation_year,counter]  = self.summary["yearly_installed_capacities"]['battery_system'][0,simulation_year]               
                if counter >= simulation_year and counter < self.sources_info["pv_gen_1"]["life_time"]     + simulation_year:
                    self.summary["yearly_capacity"]['pv_system'][simulation_year,counter]       = self.summary["yearly_installed_capacities"]['pv_system']     [0,simulation_year]
                if counter >= simulation_year and counter < self.sources_info["wind_gen_1"]["life_time"]   + simulation_year:
                    self.summary["yearly_capacity"]['wind_system'][simulation_year,counter]     = self.summary["yearly_installed_capacities"]['wind_system']   [0,simulation_year]
                if counter >= simulation_year and counter < self.sources_info["diesel_gen_1"]["life_time"] + simulation_year:
                    self.summary["yearly_capacity"]['diesel_system'][simulation_year,counter]   = self.summary["yearly_installed_capacities"]['diesel_system'] [0,simulation_year]

            #endregion       

            #endregion

    def resultsMG(self):   
        return self.summary, self.dispatch

    def financialMG(self):
        yearly_prices = LookTable()

        yearly_private_roi = np.zeros((1,self.prob_info['years']))
        yearly_customer_payments_in_present_value = np.zeros((1,self.prob_info['years']))
        yearly_private_payments_in_present_value  = np.zeros((1,self.prob_info['years']))
        net_cash_flows_in_present_value                    = np.zeros((1,self.prob_info['years']))
        
        for simulation_year in range(self.prob_info['years']):
            cpi = yearly_prices.table_cpi[simulation_year+2]
            yearly_customer_payments_in_present_value[0,simulation_year] = self.summary["yearly_customer_payments"][0,simulation_year]/((1+cpi/100)**(simulation_year))
            yearly_private_payments_in_present_value[0,simulation_year]  = self.summary["yearly_private_payments"][0,simulation_year]/((1+cpi/100)**(simulation_year))
            net_cash_flows_in_present_value[0,simulation_year] = yearly_customer_payments_in_present_value[0,simulation_year]- yearly_private_payments_in_present_value[0,simulation_year]
            

        total_customer_payments_in_present_value    = np.sum(yearly_customer_payments_in_present_value)
        total_private_payments_in_present_value     = np.sum(yearly_private_payments_in_present_value)
        private_npv                                 = total_customer_payments_in_present_value-total_private_payments_in_present_value
        internal_rate_of_return                     = round(np.irr(net_cash_flows_in_present_value.flatten()),4)
        return_of_investment_in_present_value       = (yearly_customer_payments_in_present_value-yearly_private_payments_in_present_value)/(yearly_private_payments_in_present_value)
        net_cash_flows_in_future_value              = self.summary["yearly_customer_payments"]-self.summary["yearly_private_payments"]
        return_of_investment_in_future_value        = net_cash_flows_in_future_value/(self.summary["yearly_private_payments"])

        self.financial_analysis = {}
        self.financial_analysis['private_npv'] = private_npv
        self.financial_analysis['total_customer_payments_in_present_value'] = total_customer_payments_in_present_value
        self.financial_analysis['total_private_payments_in_present_value']  = total_private_payments_in_present_value
        self.financial_analysis['profit_expenditures_ratio']                = total_customer_payments_in_present_value/total_private_payments_in_present_value
        self.financial_analysis['net_cash_flows_in_present_value']          = net_cash_flows_in_present_value
        self.financial_analysis['net_cash_flows_in_future_value']           = net_cash_flows_in_future_value
        self.financial_analysis['internal_rate_of_return']                  = internal_rate_of_return
        self.financial_analysis['return_of_investment_in_present_value']    = return_of_investment_in_present_value
        self.financial_analysis['return_of_investment_in_future_value']     = return_of_investment_in_future_value
        self.financial_analysis['yearly_customer_payments_in_present_value']= yearly_customer_payments_in_present_value  
        self.financial_analysis['yearly_private_payments_in_present_value'] = yearly_private_payments_in_present_value    
        self.financial_analysis['yearly_customer_payments_in_future_value'] = self.summary["yearly_customer_payments"]     
        self.financial_analysis['yearly_private_payments_in_future_value']  = self.summary["yearly_private_payments"]     


        self.summary['financial_analysis']={}
        self.summary['financial_analysis']['private_npv']=private_npv
        self.summary['financial_analysis']['internal_rate_of_return']=internal_rate_of_return
        self.summary['financial_analysis']['return_of_investment_in_present_value']=return_of_investment_in_present_value



        return self.financial_analysis

    def plotMG(self):                                       
        '''
        MultiyearDSMS.plotMG():  
        Method created to make some default plots of the results.   
        '''  
        #region to plot the rate of return
        xnum=np.arange(1,self.prob_info['years']+1,1)
        xnames=[]
        for counter in range(len(xnum)):
            xnames+=[f'{xnum[counter]}']

        plt.plot(self.financial_analysis['return_of_investment_in_present_value'][0],label='Yearly Rate of Return (present value)')
        plt.xticks(ticks=np.arange(0,self.prob_info['years']+1,1),labels=xnames)
        plt.grid(axis='y')
        plt.legend()
        plt.show() 
        #endregion
        
        #region to plot the upper limits of the tariffs

        res=np.zeros(self.prob_info['years'])
        for counter in range(self.prob_info['years']):
            res[counter]=self.summary['yearly_results'][counter].up_limit_tariff
        plt.plot(res,label='Upper limit of the tariffs')
        plt.xticks(ticks=np.arange(0,self.prob_info['years']+1,1),labels=xnames)
        plt.grid(axis='y')
        plt.legend()
        plt.show() 
        #endregion        
        
        #region to plot the total capacities  
        xnum=np.arange(1,self.prob_info['years']+1,1)
        xnames=[]
        for counter in range(len(xnum)):
            xnames+=[f'{xnum[counter]}']

        if self.prob_info['years'] ==1:
            last=0
            counter=-1
        width=0.1

        fig, ax = plt.subplots()
        for counter in range(self.prob_info['years']-1):
            y0=self.summary["yearly_results"][counter].pv_capacity.values       
            y1=self.summary["yearly_results"][counter].dg_capacity.values       
            y2=self.summary["yearly_results"][counter].bess_capacity.values
            y3=self.summary["yearly_results"][counter].wind_capacity.values

            ax.bar(counter-3*width/2,    y0,   zorder=3, width=width, color = 'tab:green')    
            ax.bar(counter-width/2,      y1,   zorder=3, width=width, color = 'tab:orange')   
            ax.bar(counter+width/2,      y2,   zorder=3, width=width, color = 'tab:brown')
            ax.bar(counter+3*width/2,    y3,   zorder=3, width=width, color = 'tab:blue')
            last=counter+1

        ax.bar(last-3*width/2,    self.summary["yearly_results"][counter+1].pv_capacity.values,    zorder=3,  width=width, color = 'tab:green', label='PV')    
        ax.bar(last-width/2,      self.summary["yearly_results"][counter+1].dg_capacity.values,    zorder=3,  width=width, color = 'tab:orange', label='DG')   
        ax.bar(last+width/2,      self.summary["yearly_results"][counter+1].bess_capacity.values,  zorder=3,  width=width, color = 'tab:brown', label='BESS')
        ax.bar(last+3*width/2,    self.summary["yearly_results"][counter+1].wind_capacity.values,  zorder=3,  width=width, color = 'tab:blue', label='WIND')

        ax.set_ylabel(u'Capacities [kW/kWh]')
        ax.set_xlabel(u'Years')

        plt.xticks(ticks=np.arange(0,self.prob_info['years']+1,1),labels=xnames)
        plt.grid(axis='y',zorder=0)
        plt.legend(loc=7, bbox_to_anchor=(0.88, -0.16), ncol=4, frameon=False, shadow=False)

        plt.show()
        fig.savefig('total_installed_capacities.pdf', bbox_inches='tight',transparent=True)

        #endregion

        #region to plot the yearly installed capacities         
        width=0.1
        xnum=np.arange(1,self.prob_info['years']+1,1)
        xnames=[]
        for counter in range(len(xnum)):
            xnames+=[f'{xnum[counter]}']
        
        if self.prob_info['years'] ==1:
            last=0
            counter=-1

        y0=self.summary["yearly_installed_capacities"]['pv_system']     .flatten()
        y1=self.summary["yearly_installed_capacities"]['diesel_system'] .flatten()       
        y2=self.summary["yearly_installed_capacities"]['battery_system'].flatten()    
        y3=self.summary["yearly_installed_capacities"]['wind_system']   .flatten()

        fig, ax = plt.subplots()
        for counter in range(self.prob_info['years']-1):
            ax.bar(counter-3*width/2,    y0[counter],    zorder=3, width=width, color = 'tab:green')    
            ax.bar(counter-width/2,      y1[counter],    zorder=3, width=width, color = 'tab:orange')   
            ax.bar(counter+width/2,      y2[counter],    zorder=3, width=width, color = 'tab:brown')
            ax.bar(counter+3*width/2,    y3[counter],    zorder=3, width=width, color = 'tab:blue')
            last=counter+1

        ax.bar(last-3*width/2,    y0[counter+1],   zorder=3, width=width, color = 'tab:green', label='PV')    
        ax.bar(last-width/2,      y1[counter+1],   zorder=3, width=width, color = 'tab:orange', label='DG')   
        ax.bar(last+width/2,      y2[counter+1],   zorder=3, width=width, color = 'tab:brown', label='BESS')
        ax.bar(last+3*width/2,    y3[counter+1],   zorder=3, width=width, color = 'tab:blue', label='WIND')

        ax.set_ylabel(u'Capacities [kW/kWh]')
        ax.set_xlabel(u'Years')

        plt.xticks(ticks=np.arange(0,self.prob_info['years']+1,1),labels=xnames)
        plt.grid(axis='y',zorder=0)
        plt.legend(loc=7, bbox_to_anchor=(0.88, -0.16), ncol=4, frameon=False, shadow=False)

        plt.show()
        fig.savefig('yearly_installed_capacities.pdf', bbox_inches='tight',transparent=True)

        #endregion

#endregion

#region for the "multiyear stochastic" constructor          

class StochasticDSMS():                                     
    '''
    Stochastic multiyear constructor to ensamble the microgrid using Demand Side Management Strategies. This constructor perform an sochastic analysis

    | **prob_info:** (dictionary) 
    | Contains the information related to the configuration of the optimization formulation. To make easier setting the parameters, a function named variables is created to import the values of problem info from a *.csv file.
    | To execute the function variables do: 
    ::                             
        prob_info = {}
        variables_csv   = pd.read_csv('config.csv', sep=';', header=None, skip_blank_lines=True)
        prob_info['scenarios'], prob_info["years"], prob_info["scala"], prob_info["prxo"], prob_info["percentage_yearly_growth"], prob_info["percentage_variation"], prob_info["dlcpercenthour"], prob_info["dlcpercenttotal"], prob_info["sen_ince"], prob_info["sen_ghi"], prob_info["elasticity"], prob_info["curtailment"], prob_info["capex_private"], prob_info["capex_gov"], prob_info["capex_community"], prob_info["capex_ong"], prob_info["opex_private"], prob_info["opex_gov"], prob_info["opex_community"], prob_info["opex_ong"], prob_info["rate_return_private"], prob_info["max_value_tariff"], prob_info['diesel_system'], prob_info['pv_system'], prob_info['battery_system'], prob_info['wind_system'], prob_info['hydro_system'], prob_info['hydrogen_system'], prob_info['gas_system'], prob_info['biomass_system'], prob_info['flat'], prob_info['tou'], prob_info['cpp'], prob_info['dadp'], prob_info['ince'], prob_info['dilc'], prob_info['residential'],prob_info['commercial'],prob_info['industrial'],prob_info['community']  = cm.variables(variables_csv)

    Where 'config.csv' is the file containing the information of prob_info. For more information refer to the documentation of the function variables. 

    | Additionally, prob_info should include the weather conditions of the geographical region where the IMG is going to be installed. To make easier the process of importing these data, resources_norm, resources_all and resources_noise functions were created. 
    | To execute one of the functions do:
    ::    
        data_csv        = pd.read_csv('resourcedata.csv', sep=';', header=None, skip_blank_lines=True)
        prob_info["ghi"], prob_info["irrdiffuse"], prob_info["temperature"], prob_info["wind"], prob_info["hydro"], prob_info["load_residential"], prob_info["load_commercial"], prob_info["load_industrial"], prob_info["load_community"] = cm.resources_all(data_csv, years=prob_info["years"], scenarios=prob_info['scenarios'], percentage_yearly_growth=prob_info["percentage_yearly_growth"], fit_type="weekly")

    For more information about the different options of how to generate the scenarios for the multiyear and stochastic analysis, please refer to resources_norm, resources_all and resources_noise.  

    **sources_info:** (dictionary) 
    
    Contains the information related to the configuration of the energy sources. The following parameters are expected:
    
    sources_info = {                                
        | # Battery Energy Storage System info
        | "bess_1" : {            
        |     "life_time"         : 5,                               # Life time of the BESS system
        |     "fuel_function"     : 0,                               # Fuel function                  
        |     "fuel_cost"         : 0,                               # USD
        |     "maintenance_cost"  : 6,                               # Percentage of the capacity
        |     "min_out_power"     : 50,                              # Percentage of the capacity
        |     "max_out_power"     : 90,                              # Percentage of the capacity
        |     "rate_up"           : 1,                               # Percentage of the capacity
        |     "rate_down"         : 1,                               # Percentage of the capacity    
        |     "initial_charge"    : 50,                              # Percentage of the capacity
        | },
         
        | # Diesel generator info
        | "diesel_gen_1" : {      
        |     "life_time"         : 2,                                # Life time of the diesel system
        |     "fuel_function"     : np.array([0.246, 0.08415]),       # Linear fuel function                  
        |     # "fuel_function"     : np.array([0.000203636364, 0.224872727, 4.22727273]), # Quadratic fuel function                  
        |     "fuel_cost"         : 0.8,                              # USD per liter
        |     "maintenance_cost"  : 6,                                # Percentage of the capacity
        |     "min_out_power"     : 0,                                # Percentage of the capacity
        |     "max_out_power"     : 100,                              # Percentage of the capacity
        |     "rate_up"           : 1,                                # Percentage of the capacity
        |     "rate_down"         : 1,                                # Percentage of the capacity    
        | },
 
        | # Photovoltaic system info
        | "pv_gen_1" : {          
        |     "life_time"         : 25,                               # Life time of the PV module                                                   
        |     "maintenance_cost"  : 6,                                # Percentage of the capacity
        |     "rate_up"           : 1,                                # Percentage of the capacity
        |     "rate_down"         : 1,                                # Percentage of the capacity    
        |     "derat"             : 1,                                # Derating factor
        |     "pstc"              : 0.3,                              # Nominal capacity of the PV module                                                     # Percentage of the capacity    
        |     "Ct"                : -0.0039,                          # Termic coefficient of the PV module
        | },
 
        | # Wind generator info
        | "wind_gen_1" : {        
        |     "life_time"         : 15,                               # Life time of the WIND turbine
        |     "maintenance_cost"  : 5,                                # Percentage of the capacity
        |     "rate_up"           : 1,                                # Percentage of the capacity
        |     "rate_down"         : 1,                                # Percentage of the capacity
        |     "rated_speed"       : 12.5,                             # Rated speed in m/s
        |     "speed_cut_in"      : 3,                                # Speed to start generating of the turbine        
        |     "speed_cut_out"     : 13,                               # Speed to stop generating of the turbine            
        |     "nominal_capacity"  : 1,                                # Nominal capacity of the turnine in kW        
        | },
 
        | # Lack of energy info
        | "lack_ene" : {          
        |     "cost_function"    : 0,                                 # Cost function  
        |     "reliability"      : 2,                                 # Percentage of reliability for the lack of energy                                                   # Percentage of the capacity    
        | },
         
        | # Excess of energy info
        | "excess_ene" : {        
        |     "cost_function"    : 0,                                 # Cost function  
        |     "reliability"      : 2,                                 # Percentage of reliability for the excess of energy
        | }   
        | }

    '''
    def __init__(self, prob_info_input, sources_info):            
        self.prob_info_total = prob_info_input
        self.prob_info_scenario = prob_info_input
        self.sources_info = sources_info                       
        self.summary = {}
        self.summary['summary']             = []
        self.summary['dispatch']            = []
        self.summary['financial_analysis']  = []

        for scenario in range(prob_info_input['scenarios']):
            
            self.prob_info_scenario["ghi"]             [0,:] = self.prob_info_total["ghi"]             [scenario,:]    # Enviarlo como un vector desde aquí
            self.prob_info_scenario["irrdiffuse"]      [0,:] = self.prob_info_total["irrdiffuse"]      [scenario,:]       
            self.prob_info_scenario["temperature"]     [0,:] = self.prob_info_total["temperature"]     [scenario,:]       
            self.prob_info_scenario["wind"]            [0,:] = self.prob_info_total["wind"]            [scenario,:]   
            self.prob_info_scenario["hydro"]           [0,:] = self.prob_info_total["hydro"]           [scenario,:]   
            self.prob_info_scenario["load_residential"][0,:] = self.prob_info_total["load_residential"][scenario,:]        
            self.prob_info_scenario["load_commercial"] [0,:] = self.prob_info_total["load_commercial"] [scenario,:]        
            self.prob_info_scenario["load_industrial"] [0,:] = self.prob_info_total["load_industrial"] [scenario,:]        
            self.prob_info_scenario["load_community"]  [0,:] = self.prob_info_total["load_community"]  [scenario,:]       

            MicroGrid = MultiyearDSMS(prob_info_input=self.prob_info_scenario, sources_info=self.sources_info)
            yearly_summary, dispatch = MicroGrid.resultsMG()
            financial_analysis = MicroGrid.financialMG()

            #region to save results
            self.summary['summary'].append(yearly_summary)
            self.summary['dispatch'].append(dispatch)
            self.summary['financial_analysis'].append(financial_analysis)
            #endregion

    def resultsMG(self):   
        return self.summary

#endregion

#endregion


#%%

#region for the energy sources class                            

class EStorage():                       
    '''
    General class for storage (Batteries, CAES, HydroPump, Flywheel, Hydrogen, etc.). This class will create an object that represents a electric energy storage system. The datails of this model can be found in [ov2020]_. This class requires the following attributes:

    | **investment_cost:** (int, float)
    | Initial investment cost paid for installing a kW of the selected technology. 
    | If any value is passed the class will take the default value = 0, which means that the investment costs will be assumed to be zero.
    
    | **fuel_function:** (numpy array of dimension less than three)
    | The fuel function can take a cuadratic or linear input arguments. The dimension of the numpy array will determine the fuel function. 
    | if the input have a numpy array of dimension three ([a, b, c]), a cuadratic function will be used for the operational costs: 
    | :math:`aQ^2+bQ+c`, where Q is the power that flows (in/out) in the storage system.
    | if the input have a numpy array of dimension two ([a, b]), a linear function will be used for the operational costs: 
    | :math:`aQ+b`, where Q is the power that flows (in/out) in the storage system.
    | if the input have a numpy array of dimension one ([a]), a linear function will be used for the operational costs: 
    | :math:`aQ`, where Q is the power that flows (in/out) in the storage system.

    If any value is passed the class will take the default value = 0, which means that the operational costs will be assumed to be zero.

    | **fuel_cost:** (int, float)
    | This attribute takes the fuel cost expresed in measure unity by money unity (L/USD, g/CAD, etc.). It is worth it to mention that the fuel cost units must be consistent with the fuel function units. 
    | fuel_function units multiplied by fuel_cost units = money units (USD, CAD, etc.)
    | If any value is passed the class will take the default value = 0, which means that the operational costs will be zero even if a fuel function cost was passed. 

    | **maintenance_cost:** (int, float, numpy.ndarray of dimension less than three)
    | If a int or float value is passed, the maintenance cost will be expressed as the yearly cost. The input value is considered as a percentage of the installation costs. 
    | if the value is 15, then a 15% of the installation costs are taken as the yearly maintenance costs of the storage system. If a numpy array is passed, a function will be built using the same principle of the fuel function.
    | if the input have a numpy array of dimension three ([a, b, c]), a cuadratic function will be used for the maintenance costs: 
    | :math:`aQ^2+bQ+c`, where Q is the power that flows (in/out) in the storage system. 
    | if the input have a numpy array of dimension two ([a, b]), a linear function will be used for the maintenance costs: 
    | :math:`aQ+b`, where Q is the power that flows (in/out) in the storage system.
    | if the input have a numpy array of dimension one ([a]), a linear function will be used for the maintenance costs: 
    | :math:`aQ`, where Q is the power that flows (in/out) in the storage system.
   
    If any value is passed the class will take the default value = 5, which means that the maintenance costs will be assumed to be 5% of the investment costs.

    | **min_out_power:** (int, float)
    | Minimum output power of the storage system. If the storage system has a lower generation limit, this value should be specified here. The input value must be a percentage [0,100]. 
    | If 30 is passed, it is assumed that the storage system can dispatch values only above the 30% of its nominal capacity.  
    | If any value is passed the class will take the default value = 0, which means that the the storage system can dispatch any value above zero. 

    | **max_out_power:** (int, float)
    | Maximum output power of the storage system. If the storage system has a superior generation limit, this value should be specified here. The input value must be a percentage [0,100]. 
    | If 80 is passed, it is assumed that the storage system can dispatch values only below the 80% of its nominal capacity.  
    | If any value is passed the class will take the default value = 100, which means that the the storage system can dispatch any value below the 100% of its nominal capacity. 

    | **rate_up:** (int, float)
    | Maximum rate to increase the output power in one time step. If the storage system has a limit in the variation of output power this value should be specified here. The input value is considered as a percentage of the installed capacity. 
    | If any value is passed the class will take the default value = 1, which means that the storage system will consider that is able to increase the output power from zero to the installed capacity in one time step. 

    | **rate_down:** (int, float)
    | Maximum rate to decrease the output power in one time step. If the storage system has a limit in the variations of output power this value should be specified here. The input value is considered as a percentage of the installed capacity. 
    | If any value is passed the class will take the default value = 1, which means that the storage system will consider that is able to decrease the output power from the installed capacity to zero in one time step. 

    | **initial_charge:** (int, float)
    | Initial value of charge at the beggining of the optimization horizon. This value is considered as a percentage of the installed capacity. It is wort it to mention that the passed value must be inside of the limits of the lower and upper limit of charge of the storage system.
    | If any value is passed the class will take the default value = 50, which means that the initial charge of the storage is considered to be 50% of its nominal capacity.

    | **years:** (int)
    | Number of years of optimization. 
    | If any value is passed the class will take the default value = 1.

    | **yearly_total_capacity:** (int, float)
    | Installed capacity in the previous year. This attribute only works for multy year analysis. The default value is zero. 

    | **tax_tonne_price:** (int, float)
    | Price paid for each tonne of CO2 produced for the storage. 

    '''    

    def __init__(self, investment_cost=0,fuel_function=0,fuel_cost=0,maintenance_cost=5,min_out_power=0, max_out_power=100,rate_up=1,rate_down=1, initial_charge = 50, years=1, yearly_total_capacity=0, tax_tonne_price=0,life_time=4):
        self.investment_cost    = investment_cost
        self.fuel_cost          = fuel_cost
        self.fuel_function      = fuel_function    
        self.maintenance_cost   = maintenance_cost      
        self.min_out_power      = min_out_power       
        self.max_out_power      = max_out_power       
        self.rate_up            = rate_up       
        self.rate_down          = rate_down
        self.initial_charge     = initial_charge
        self.years              = years
        self.tax_tonne_price    = tax_tonne_price
        self.yearly_total_capacity = yearly_total_capacity  
        self.life_time          = life_time
    
    def formulate(self):                        
        self.capacity           = cp.Variable((1,1),       nonneg=True) + self.yearly_total_capacity                       
        self.dispatch           = cp.Variable((8760,1),  nonneg=True)
        self.capex              = cp.multiply(self.capacity, self.investment_cost/self.life_time)                                                              
        self.max_dispatch       = self.max_out_power/100*self.capacity                                  
        self.min_dispatch       = self.min_out_power/100*self.capacity                                     
                             
        if type(self.fuel_function) == np.ndarray:  
            if len(self.fuel_function) == 3:
                self.opex           = self.fuel_cost*(self.fuel_function[0]*cp.sum(self.dispatch**2) + self.fuel_function[1]*cp.sum(self.dispatch) + self.fuel_function[2])                 
                self.tax_tonne      = self.tax_tonne_price/1000*(self.fuel_function[0]*cp.sum(self.dispatch**2) + self.fuel_function[1]*cp.sum(self.dispatch) + self.fuel_function[2])                 
            if len(self.fuel_function) == 2:
                self.opex           = self.fuel_cost*(self.fuel_function[0]*cp.sum(self.dispatch) + self.fuel_function[1])                 
                self.tax_tonne      = self.tax_tonne_price/1000*(self.fuel_function[0]*cp.sum(self.dispatch) + self.fuel_function[1])                 
            if len(self.fuel_function) == 1:
                self.opex           = self.fuel_cost*(self.fuel_function[0]*cp.sum(self.dispatch))                 
                self.tax_tonne      = self.tax_tonne_price/1000*(self.fuel_function[0]*cp.sum(self.dispatch))                 
        else:
            self.opex               = 0
            self.tax_tonne          = 0

        if type(self.maintenance_cost) == np.ndarray: 
            if len(self.maintenance_cost) == 3:
                self.maintenance    = self.fuel_cost*(self.maintenance_cost[0]*cp.sum(self.dispatch**2) + self.maintenance_cost[1]*cp.sum(self.dispatch) + self.maintenance_cost[2])                 
            if len(self.maintenance_cost) == 2:
                self.maintenance    = self.fuel_cost*(self.maintenance_cost[0]*cp.sum(self.dispatch) + self.maintenance_cost[1])                 
            if len(self.maintenance_cost) == 1:
                self.maintenance    = self.fuel_cost*(self.maintenance_cost[0]*cp.sum(self.dispatch))                 
        else:
            self.maintenance        = self.maintenance_cost/100*self.investment_cost*self.capacity

        self.constraints            = []                                                             
        
        # Initial state of charge of the storage
        self.constraints            += [self.dispatch[0] == self.initial_charge/100*self.capacity]
        
        if self.max_out_power == 100:
            self.constraints        += [self.dispatch <= self.capacity]
        else:       
            self.constraints        += [self.dispatch <= self.max_dispatch]
        
        if self.min_out_power == 0:
            pass # It has a nonnegativity constraint, this is not need it.         
        else:           
            self.constraints        += [self.dispatch >= self.min_dispatch]
             
        T=8760
        if self.rate_up != 1: 
            self.constraints += [self.dispatch[1:T] <= self.dispatch[0:T-1] + self.rate_up*self.capacity  ]     # Restriction of discharge of the pv generator 
        if self.rate_down != 1:
            self.constraints += [self.dispatch[1:T] >= self.dispatch[0:T-1] - self.rate_down*self.capacity]     # Restriction of charge of the pv generator 
        
    def __str__(self):
        return f"This storage device has the following attributes: \n \n Initial investment costs = {self.investment_cost} \n Operational costs = {self.fuel_function} \n Maintenance costs = {self.maintenance_cost} \n Max output power = {self.max_out_power} \n Min output power = {self.min_out_power} \n Max up rate power change = {self.rate_up} \n Max down rate power change = {self.rate_down} \n Years of simulation = {self.years} \n \n Additionally, this generator has the following CVXPY variables: \n One variable for the installed capacity of length = ({self.years},1) \n One dispatch variable of length = ({self.years},1)"

class EGenerator():                     
    '''
    General class for generators (Diesel, Gas, Biomass, etc.). This class will create an object that represents an electric energy generator. 
    This class requires the following attributes:

    | **investment_cost:** (int, float)
    | Initial investment cost paid for installing a kW of the selected technology. 
    | If any value is passed the class will take the default value = 0, which means that the investment costs will be assumed to be zero.
    
    | **fuel_function:** (numpy.ndarray of dimension less than three)
    | The fuel function can take a cuadratic or linear input arguments. The dimension of the numpy array will determine the fuel function. 
    | if the input has a numpy array of dimension three ([a, b, c]), a cuadratic function will be used for the operational costs. The datails of this approximation can be found in [ov2020]_: 
    | :math:`aQ^2+bQ+c`, where Q is the output power of the generator. 
    | if the input has a numpy array of dimension two ([a, b]), a linear function will be used for the operational costs. However, to better aproximate the fuel consumption, the second term is multiplied by the installed capacity. The datails of this approximation can be found in [bu2019]_: 
    | :math:`aQ+bC`, where Q is the output power of the generator and C is the installed capacity. 
    | if the input has a numpy array of dimension one ([a]), a linear function will be used for the operational costs: 
    | :math:`aQ`, where Q is the output power of the generator. 

    If any value is passed the class will take the default value = 0, which means that the operational costs will be assumed to be zero.

    | **fuel_cost:** (int, float)
    | This attribute takes the fuel cost expresed in measure unity by money unity (L/USD, g/CAD, etc.). It is worth it to mention that the fuel cost units must be consistent with the fuel function units. 
    | fuel_function units multiplied by fuel_cost units = money units (USD, CAD, etc.)
    | If any value is passed the class will take the default value = 0, which means that the operational costs will be zero even if a fuel function cost was passed. 

    | **maintenance_cost:** (int, float, numpy.ndarray of dimension less than three)
    | If an int or float value is passed, the maintenance cost of the generator will be expressed as the yearly cost. The input value is considered as a percentage of the installation costs. 
    | if the value is 6, then a 6% of the installation costs are taken as the yearly maintenance costs of the generator. If a numpy array is passed then a function will be built using the same principle of the fuel function.
    | if the input has a numpy array of dimension three ([a, b, c]), a cuadratic function will be used for the maintenance costs: 
    | :math:`aQ^2+bQ+c`, where Q is the output power of the generator. 
    | if the input has a numpy array of dimension two ([a, b]), a linear function will be used for the maintenance costs: 
    | :math:`aQ+b`, where Q is the output power of the generator. 
    | if the input has a numpy array of dimension one ([a]), a linear function will be used for the maintenance costs: 
    | :math:`aQ`, where Q is the output power of the generator. 
   
    If any value is passed the class will take the default value = 5, which means that the maintenance costs will be assumed to be 5% of the investment costs.

    | **min_out_power:** (int, float)
    | Minimum output power of the generator. If the generator has a lower generation limit, this value should be specified here. The input value must be a percentage [0,100]. 
    | If 30 is passed, it is assumed that the generator can dispatch values only above the 30% of its nominal capacity.  
    | If any value is passed the class will take the default value = 0, which means that the the generator can dispatch any value above zero. 

    | **max_out_power:** (int, float)
    | Maximum output power of the generator. If the generator has a superior generation limit, this value should be specified here. The input value must be a percentage [0,100]. 
    | If 80 is passed, it is assumed that the generator can dispatch values only below the 80% of its nominal capacity.  
    | If any value is passed the class will take the default value = 100, which means that the the generator can dispatch any value below the 100% of its nominal capacity. 

    | **rate_up:** (int, float)
    | Maximum rate to increase the output power in one time step. If the generator has a limit in the variation of output power this value should be specified here. The input value is considered as a percentage of the installed capacity. 
    | If any value is passed the class will take the default value = 1, which means that the generator will consider that is able to increase the output power from zero to the installed capacity in one time step. 

    | **rate_down:** (int, float)
    | Maximum rate to decrease the output power in one time step. If the generator has a limit in the variations of output power this value should be specified here. The input value is considered as a percentage of the installed capacity. 
    | If any value is passed the class will take the default value = 1, which means that the generator will consider that is able to decrease the output power from the installed capacity to zero in one time step. 

    | **years:** (int)
    | Number of years of optimization. 
    | If any value is passed the class will take the default value = 1.

    | **yearly_total_capacity:** (int, float)
    | Installed capacity in the previous year. This attribute only works for multy year analysis. The default value is zero. 

    | **tax_tonne_price:** (int, float)
    | Price paid for each tonne of CO2 produced for the generator. The consumed liters of diesel are multiplied by 2.65 (Diesel type B5) to obtain the kilograms of CO2.
    
    '''    
    
    def __init__(self, investment_cost=0,fuel_function=0,fuel_cost=0,maintenance_cost=5,min_out_power=0,max_out_power=100,rate_up=1,rate_down=1,years=1,yearly_total_capacity=0,tax_tonne_price=0,life_time=3):
        self.investment_cost    = investment_cost
        self.fuel_cost          = fuel_cost
        self.fuel_function      = fuel_function    
        self.maintenance_cost   = maintenance_cost      
        self.min_out_power      = min_out_power       
        self.max_out_power      = max_out_power       
        self.rate_up            = rate_up       
        self.rate_down          = rate_down
        self.years              = years
        self.tax_tonne_price    = tax_tonne_price  
        self.yearly_total_capacity = yearly_total_capacity
        self.life_time          = life_time

    def formulate(self):                        
        # Variables CVXPY
        self.capacity           = cp.Variable((1,1),       nonneg=True) + self.yearly_total_capacity                       
        self.dispatch           = cp.Variable((8760,1),  nonneg=True)                     
        
        # Compute the CAPEX
        self.capex              = cp.multiply(self.capacity, self.investment_cost/self.life_time)                                
        
        # # Compute the OPEX and the taxes
        if type(self.fuel_function) == np.ndarray:  
            if len(self.fuel_function) == 3:
                self.opex           = cp.sum(self.fuel_cost*(self.fuel_function[0]*(self.dispatch**2) + self.fuel_function[1]*self.dispatch + self.fuel_function[2]))                
                self.tax_tonne      = 2.65*self.tax_tonne_price/1000*(self.fuel_function[0]*cp.sum(self.dispatch**2) + self.fuel_function[1]*cp.sum(self.dispatch) + self.fuel_function[2])                 
            if len(self.fuel_function) == 2:
                self.opex           = cp.sum(self.fuel_cost*(self.fuel_function[0]*(self.dispatch) + self.fuel_function[1]*self.capacity))                 
                self.tax_tonne      = 2.65*self.tax_tonne_price/1000*(self.fuel_function[0]*cp.sum(self.dispatch) + self.fuel_function[1]*self.capacity)                 
            if len(self.fuel_function) == 1:
                self.opex           = cp.sum(self.fuel_cost*(self.fuel_function[0]*(self.dispatch)))                
                self.tax_tonne      = 2.65*self.tax_tonne_price/1000*(self.fuel_function[0]*cp.sum(self.dispatch))                 
        else:
            self.opex               = 0
            self.tax_tonne          = 0

        # Compute the maintenance costs
        if type(self.maintenance_cost) == np.ndarray: 
            if len(self.maintenance_cost) == 3:
                self.maintenance    = cp.sum(self.fuel_cost*(self.maintenance_cost[0]*(self.dispatch**2) + self.maintenance_cost[1]*(self.dispatch) + self.maintenance_cost[2]))                 
            if len(self.maintenance_cost) == 2:
                self.maintenance    = cp.sum(self.fuel_cost*(self.maintenance_cost[0]*(self.dispatch) + self.maintenance_cost[1]))                 
            if len(self.maintenance_cost) == 1:
                self.maintenance    = cp.sum(self.fuel_cost*(self.maintenance_cost[0]*(self.dispatch)))                 
        else:
            self.maintenance        = self.maintenance_cost/100*self.investment_cost*self.capacity

        self.max_dispatch       = self.max_out_power/100*self.capacity                                  
        self.min_dispatch       = self.min_out_power/100*self.capacity                                     
        
        # Constraints CVXPY
        self.constraints        = []                                                             

        if self.max_out_power == 100:
            self.constraints        += [self.dispatch <= self.capacity]
        else:       
            self.constraints        += [self.dispatch <= self.max_dispatch]
        
        if self.min_out_power == 0:
            pass # It has a nonnegativity constraint, this is not need it.         
        else:           
            self.constraints        += [self.dispatch >= self.min_dispatch]
        
        T=8760
        if self.rate_up != 1: 
            self.constraints += [self.dispatch[1:T] <= self.dispatch[0:T-1] + self.rate_up*self.capacity  ]     # Restriction of discharge of the pv generator 
        if self.rate_down != 1:
            self.constraints += [self.dispatch[1:T] >= self.dispatch[0:T-1] - self.rate_down*self.capacity]     # Restriction of charge of the pv generator 
        
    def __str__(self):
        return f"This generator has the following attributes: \n \n Initial investment costs = {self.investment_cost} \n Fuel function = {self.fuel_function} \n Fuel cost = {self.fuel_cost} \n Maintenance costs = {self.maintenance_cost} \n Max output power = {self.max_out_power} \n Min output power = {self.min_out_power} \n Max up rate power change = {self.rate_up} \n Max down rate power change = {self.rate_down} \n \n Additionally, this generator has the following CVXPY variables: \n One variable for the installed capacity of length = (1,1) \n One dispatch variable of length = ({8760},1) \n \n This generator also have constraints that can be accessed by using .constraints"

class PVGenerator():                    
    '''
    General class for the photovoltaic generator. The model of the pv generator can be found in [ov2020]_ and [bu2019]_.
    This class requires the following attributes:

    | **investment_cost:** (int, float)
    | Initial investment cost paid for installing a kW of the selected technology. 
    | If any value is passed the class will take the default value = 0, which means that the investment costs will be assumed to be zero.
    
    | **maintenance_cost:** (int, float, numpy.ndarray of dimension less than three)
    | If an int or float value is passed, the maintenance cost of the partially dispatchable generator will be expressed as the yearly cost. The input value is considered as a percentage of the installation costs. 
    | if the value is 6, then a 6% of the installation costs are taken as the yearly maintenance costs of the partially dispatchable generator. If a numpy array is passed then a function will be built using the same principle of the fuel function.
    | if the input has a numpy array of dimension three ([a, b, c]), a cuadratic function will be used for the maintenance costs: 
    | :math:`aQ^2+bQ+c`, where Q is the output power of the partially dispatchable generator. 
    | if the input has a numpy array of dimension two ([a, b]), a linear function will be used for the maintenance costs: 
    | :math:`aQ+b`, where Q is the output power of the partially dispatchable generator. 
    | if the input has a numpy array of dimension one ([a]), a linear function will be used for the maintenance costs: 
    | :math:`aQ`, where Q is the output power of the partially dispatchable generator. 
   
    If any value is passed the class will take the default value = 5, which means that the maintenance costs will be assumed to be 5% of the investment costs.

    | **rate_up:** (int, float)
    | Maximum rate to increase the output power in one time step. If the partially dispatchable generator has a limit in the variation of output power this value should be specified here. The input value is considered as a percentage of the installed capacity. 
    | If any value is passed the class will take the default value = 1, which means that the partially dispatchable generator will consider that is able to increase the output power from zero to the installed capacity in one time step. 

    | **rate_down:** (int, float)
    | Maximum rate to decrease the output power in one time step. If the partially dispatchable generator has a limit in the variations of output power this value should be specified here. The input value is considered as a percentage of the installed capacity. 
    | If any value is passed the class will take the default value = 1, which means that the partially dispatchable generator will consider that is able to decrease the output power from the installed capacity to zero in one time step. 

    | **ghi:** (numpy.ndarray)
    | A numpy array with the Global Horizontal Radiation. This array should be passed for at least one year, with hourly time intervals. The default value is zero. However, if this value is not passed the pv generator will not work.  

    | **temperature:** (numpy.ndarray)
    | A numpy array with the Temperature values. This array should be passed for at least one year, with hourly time intervals. The default value is zero. However, if this value is not passed the pv generator will not work.  

    | **derat:** (int, float)
    | Derating factor. By default is one. Any value different from one will scale the pv generation. 

    | **pstc:** (int, float)
    | Output power of the selected pv module. This value should be passed in kW. By default this value is 0.3 kW.

    | **Ct:** (int, float)
    | Termic coeficient of the selected pv module. By default this value is -0.0039.

    | **years:** (int)
    | Number of years of optimization. 
    | If any value is passed the class will take the default value = 1.

    | **sen_ghi:** (int, float)
    | Value to perform sensitivity analysis over the global horizontal radiation. This value scale all the array of global horizontal radiation. The default value is = 1.
    
    | **simulation_year:** (int)        
    | Year of simulation. By default this parameter is zero. It is only used when a multiyear analysis is applied. 
    
    ''' 

    def __init__(self, investment_cost=0,maintenance_cost=5, rate_up=1, rate_down=1, ghi=0, temperature=0, derat = 1, pstc= 0.3, Ct = -0.0039, years=1, sen_ghi=1,simulation_year=0,yearly_total_capacity=0,life_time=25):
        self.investment_cost    = investment_cost
        self.sen_ghi            = float(sen_ghi)
        self.maintenance_cost   = maintenance_cost           
        self.rate_up            = rate_up       
        self.rate_down          = rate_down
        self.derat              = float(derat)
        self.pstc               = float(pstc)
        self.Ct                 = float(Ct)
        self.years              = years  
        tempghi                 = np.zeros((8760,1))
        tempghi[:,0]            = ghi[0,simulation_year*8760:(simulation_year+1)*8760]
        self.ghi                = tempghi
        temptem                 = np.zeros((8760,1))
        temptem[:,0]            = ghi[0,simulation_year*8760:(simulation_year+1)*8760]
        self.temperature        = temptem
        self.yearly_total_capacity = yearly_total_capacity 
        self.life_time          = life_time 
    def formulate(self):    
        # Variables
        self.capacity           = cp.Variable((1,1), nonneg=True) + self.yearly_total_capacity                    
        
        working_temperature     = self.temperature+0.0275*self.ghi                                     # Working Temperature of the PV cell as afunction of the ambient temperature
        self.pv_generation      = cp.multiply(self.sen_ghi*self.derat*self.pstc*self.ghi, (1+self.Ct*(working_temperature-25)))      # Generation of a PV module in kW       
        # self.max_curtailment    = self.pv_generation*self.capacity

        if type(self.maintenance_cost) == np.ndarray: 
            if len(self.maintenance_cost) == 3:
                self.maintenance    = self.fuel_cost*(self.maintenance_cost[0]*cp.sum(self.dispatch**2) + self.maintenance_cost[1]*cp.sum(self.dispatch) + self.maintenance_cost[2])                 
            if len(self.maintenance_cost) == 2:
                self.maintenance    = self.fuel_cost*(self.maintenance_cost[0]*cp.sum(self.dispatch) + self.maintenance_cost[1])                 
            if len(self.maintenance_cost) == 1:
                self.maintenance    = self.fuel_cost*(self.maintenance_cost[0]*cp.sum(self.dispatch))                 
        else:
            self.maintenance        = self.maintenance_cost/100*self.investment_cost*self.capacity

        self.capex              = self.capacity*self.investment_cost/self.life_time  
                                        
        self.opex               = 0                                  
        self.dispatch           = cp.multiply(self.capacity,self.pv_generation)
        
        self.constraints        = []
        # self.curtailment        = cp.Variable((8760,1), nonneg=True)                     
        # self.dispatch           = self.capacity*self.pv_generation-self.curtailment
        # self.constraints        += [self.curtailment <= self.max_curtailment]                                                             

        T=8760
        if self.rate_up != 1: 
            self.constraints += [self.dispatch[1:T] <= self.dispatch[0:T-1] + self.rate_up*self.capacity  ]     # Restriction of discharge of the pv generator 
        if self.rate_down != 1:
            self.constraints += [self.dispatch[1:T] >= self.dispatch[0:T-1] - self.rate_down*self.capacity]     # Restriction of charge of the pv generator 
        

    def __str__(self):
        return f"This partially dispatchable renewable energy generator has the following attributes: \n \n Initial investment costs = {self.investment_cost} \n Operational costs = {self.fuel_function} \n Maintenance costs = {self.maintenance_cost} \n Max up rate power change = {self.rate_up} \n Max down rate power change = {self.rate_down}  \n \n Additionally, this generator has the following CVXPY variables: \n One variable for the installed capacity of length = ({self.years},1)" # \n One dispatch variable of length = ({self.years},1)" # It is possible to add this for the curtailment down

class WINDGenerator():                  
    '''
    General class for Partially Dispatchable Renewable Wind generator. The model of the wind generator can be found in [bu2019]_. This class requires the following attributes:

    | **investment_cost:** (int, float)
    | Initial investment cost paid for installing a kW of the selected technology. 
    | If any value is passed the class will take the default value = 0, which means that the investment costs will be assumed to be zero.
    
    | **maintenance_cost:** (int, float, numpy.ndarray of dimension less than three)
    | If an int or float value is passed, the maintenance cost of the partially dispatchable generator will be expressed as the yearly cost. The input value is considered as a percentage of the installation costs. 
    | if the value is 6, then a 6% of the installation costs are taken as the yearly maintenance costs of the partially dispatchable generator. If a numpy array is passed then a function will be built using the same principle of the fuel function.
    | if the input has a numpy array of dimension three ([a, b, c]), a cuadratic function will be used for the maintenance costs: 
    | :math:`aQ^2+bQ+c`, where Q is the output power of the partially dispatchable generator. 
    | if the input has a numpy array of dimension two ([a, b]), a linear function will be used for the maintenance costs: 
    | :math:`aQ+b`, where Q is the output power of the partially dispatchable generator. 
    | if the input has a numpy array of dimension one ([a]), a linear function will be used for the maintenance costs: 
    | :math:`aQ`, where Q is the output power of the partially dispatchable generator. 
   
    If any value is passed the class will take the default value = 5, which means that the maintenance costs will be assumed to be 5% of the investment costs.

    | **rate_up:** (int, float)
    | Maximum rate to increase the output power in one time step. If the partially dispatchable generator has a limit in the variation of output power this value should be specified here. The input value is considered as a percentage of the installed capacity. 
    | If any value is passed the class will take the default value = 1, which means that the partially dispatchable generator will consider that is able to increase the output power from zero to the installed capacity in one time step. 

    | **rate_down:** (int, float)
    | Maximum rate to decrease the output power in one time step. If the partially dispatchable generator has a limit in the variations of output power this value should be specified here. The input value is considered as a percentage of the installed capacity. 
    | If any value is passed the class will take the default value = 1, which means that the partially dispatchable generator will consider that is able to decrease the output power from the installed capacity to zero in one time step. 

    | **wind:** (numpy.ndarray)
    | A numpy array with the wind speeds. This array should be passed for at least one year, with hourly time intervals. The default value is zero. However, if this value is not passed the wind generator will not work.  

    | **years:** (int)
    | Number of years of optimization. 
    | If any value is passed the class will take the default value = 1.

    | **rated_speed:** (int, float)
    | Rated speed of the wind turbine. The default value is 13 m/s.

    | **speed_cut_in:** (int, float)
    | Speed at which the wind turbine start to generate. The default value is 3 m/s.
    
    | **speed_cut_out:** (int, float)
    | Speed at which the wind turbine stop to generate. The default value is 12.5 m/s.

    | **nominal_capacity:** (int, float)
    | Nominal capacity of the wind turbine. The default value is 1 kW.

    | **simulation_year:** (int)        
    | Year of simulation. By default this parameter is zero. It is only used when a multiyear analysis is applied. 
    
    ''' 

    def __init__(self, investment_cost=0, maintenance_cost=5, rate_up=1, rate_down=1, wind=0, years=1, rated_speed=13, speed_cut_in=3, speed_cut_out=12.5, nominal_capacity=1, simulation_year=0, yearly_total_capacity=0,life_time=15):
        self.investment_cost    = investment_cost
        self.maintenance_cost   = maintenance_cost           
        self.rate_up            = rate_up       
        self.rate_down          = rate_down
        self.years              = years  
        self.rated_speed        = rated_speed           
        self.speed_cut_in       = speed_cut_in
        self.speed_cut_out      = speed_cut_out
        self.nominal_capacity   = nominal_capacity
        tempwind                = np.zeros((8760,1))
        tempwind[:,0]           = wind[0,simulation_year*8760:(simulation_year+1)*8760]
        self.wind               = tempwind
        self.yearly_total_capacity = yearly_total_capacity
        self.life_time          = life_time

    def formulate(self):    
        # Compute wind output power 
        # % Wind turbine characteristics
        self.wind_generation = np.zeros((8760,1)); #% Wind power at 1 kW basis
        for counter in range(8760):
            if self.wind[counter,0] >= self.rated_speed and self.wind[counter,0] <= self.speed_cut_out:
                self.wind_generation[counter,0]=self.nominal_capacity
            
            if self.wind[counter,0] >= self.speed_cut_in and self.wind[counter,0] <= self.rated_speed:
                self.wind_generation[counter,0] = (self.wind[counter]**3)*(self.nominal_capacity/(self.rated_speed**3-self.speed_cut_in**3)) - self.nominal_capacity*(self.speed_cut_in**3/(self.rated_speed**3-self.speed_cut_in**3))
            else:
                self.wind_generation[counter,0]=0
 
        
        # Variables
        self.capacity           = cp.Variable((1,1), nonneg=True) + self.yearly_total_capacity                    
        # self.max_curtailment    = self.wind_generation*self.capacity

        if type(self.maintenance_cost) == np.ndarray: 
            if len(self.maintenance_cost) == 3:
                self.maintenance    = self.fuel_cost*(self.maintenance_cost[0]*cp.sum(self.dispatch**2) + self.maintenance_cost[1]*cp.sum(self.dispatch) + self.maintenance_cost[2])                 
            if len(self.maintenance_cost) == 2:
                self.maintenance    = self.fuel_cost*(self.maintenance_cost[0]*cp.sum(self.dispatch) + self.maintenance_cost[1])                 
            if len(self.maintenance_cost) == 1:
                self.maintenance    = self.fuel_cost*(self.maintenance_cost[0]*cp.sum(self.dispatch))                 
        else:
            self.maintenance        = self.maintenance_cost/100*self.investment_cost*self.capacity

        self.capex              = self.capacity*self.investment_cost/self.life_time                                   
        self.opex               = 0                                  
        self.dispatch           = cp.multiply(self.capacity,self.wind_generation)
        
        self.constraints        = []
        
        # self.curtailment        = cp.Variable((8760,1), nonneg=True)                           
        # self.dispatch           = self.capacity*self.wind_generation-self.curtailment
        # self.constraints        += [self.curtailment <= self.max_curtailment]                                                             
        
        T=8760
        if self.rate_up != 1: 
            self.constraints += [self.dispatch[1:T] <= self.dispatch[0:T-1] + self.rate_up*self.capacity  ]     # Restriction of discharge of the pv generator 
        if self.rate_down != 1:
            self.constraints += [self.dispatch[1:T] >= self.dispatch[0:T-1] - self.rate_down*self.capacity]     # Restriction of charge of the pv generator 
        

    def __str__(self):
        return f"This partially dispatchable renewable energy generator has the following attributes: \n \n Initial investment costs = {self.investment_cost} \n Operational costs = {self.fuel_function} \n Maintenance costs = {self.maintenance_cost} \n Max up rate power change = {self.rate_up} \n Max down rate power change = {self.rate_down}  \n \n Additionally, this generator has the following CVXPY variables: \n One variable for the installed capacity of length = ({self.years},1)" # \n One dispatch variable of length = ({self.years},1)" # It is possible to add this for the curtailment down

class Lackenergy():                     
    '''
    General class for Lack of Energy. The details of the model can be found in [ov2020]_. This class requires the following attributes:

    | **cost_function:** (numpy.ndarray of dimension less than three)
    | The cost function can take a cuadratic or linear input arguments. The dimension of the numpy array will determine the cost function. 
    | if the input has a numpy array of dimension three ([a, b, c]), a cuadratic function will be used for the operational costs: 
    | :math:`aQ^2+bQ+c`, where Q is the lack of energy. 
    | if the input has a numpy array of dimension two ([a, b]), a linear function will be used for the operational costs: 
    | :math:`aQ+b`, where Q is the lack of energy 
    | if the input has a numpy array of dimension one ([a]), a linear function will be used for the operational costs: 
    | :math:`aQ`, where Q is the lack of energy. 

    If any value is passed the class will take the default value = 0, which means that the operational costs will be assumed to be zero.

    | **reliability:** (int, float)
    | Desired value of reliability for the lack of energy. This value is expressed as a percentage. A value of 5 means that is allowed a lack of energy of 5% to supply the demand.  

    | **years:** (int)
    | Number of years of optimization. 
    | If any value is passed the class will take the default value = 1.
        
    .. [ov2020]     J. C. Oviedo Cepeda et al. (in press), “Design of an Incentive-based Demand Side Management Strategy for Stand-Alone Microgrids Planning,” Int. J. Sustain. Energy Plan. Manag., vol. 28, pp. 1–21, 2020.
    .. [bu2019]     A. L. Bukar, C. W. Tan, and K. Y. Lau, “Optimal sizing of an autonomous photovoltaic/wind/battery/diesel generator microgrid using grasshopper optimization algorithm,” Sol. Energy, vol. 188, no. March, pp. 685–696, 2019.
    
    '''    
    def __init__(self, cost_function=0, reliability=5, years=1):
        self.cost_function      = cost_function
        self.reliability        = reliability           
        self.years              = years  
    
    def formulate(self):                       
        self.capex              = 0
        self.maintenance        = 0
        self.dispatch           = cp.Variable((8760,1), nonneg=True)                     
        self.constraints        = []                                                                                              

        if type(self.cost_function) == np.ndarray:  
            if len(self.cost_function) == 3:
                self.opex           = self.cost_function[0]*cp.sum(self.dispatch**2) + self.cost_function[1]*cp.sum(self.dispatch) + self.cost_function[2]                
            if len(self.cost_function) == 2:
                self.opex           = self.cost_function[0]*cp.sum(self.dispatch) + self.cost_function[1]                 
            if len(self.cost_function) == 1:
                self.opex           = self.cost_function[0]*cp.sum(self.dispatch)                
        else:
            self.opex           = 0

    def __str__(self):
        return f"This partially dispatchable renewable energy generator has the following attributes: \n \n Initial investment costs = {self.investment_cost} \n Operational costs = {self.fuel_function} \n Maintenance costs = {self.maintenance_cost} \n Max up rate power change = {self.rate_up} \n Max down rate power change = {self.rate_down}  \n \n Additionally, this generator has the following CVXPY variables: \n One variable for the installed capacity of length = ({self.years},1)" # \n One dispatch variable of length = ({self.years},1)" # It is possible to add this for the curtailment down

class Excessenergy():                   
    '''
    General class for excess of Energy. The details of this model can be found in [ov2020]_. This Class requires the following attributes:

    | **cost_function:** (numpy.ndarray of dimension less than three)
    | The cost function can take a cuadratic or linear input arguments. The dimension of the numpy array will determine the cost function. 
    | if the input has a numpy array of dimension three ([a, b, c]), a cuadratic function will be used for the operational costs: 
    | :math:`aQ^2+bQ+c`, where Q is the excess of energy. 
    | if the input has a numpy array of dimension two ([a, b]), a linear function will be used for the operational costs: 
    | :math:`aQ+b`, where Q is the excess of energy 
    | if the input has a numpy array of dimension one ([a]), a linear function will be used for the operational costs: 
    | :math:`aQ`, where Q is the excess of energy. 

    If any value is passed the class will take the default value = 0, which means that the operational costs will be assumed to be zero.

    | **reliability:** (int, float)
    | Desired value of reliability for the excess of energy. This value is expressed as a percentage. A value of 5 means that is allowed a excess of energy of 5% to supply the demand.  

    | **years:** (int)
    | Number of years of optimization. 
    | If any value is passed the class will take the default value = 1.
    
    '''    
    def __init__(self, cost_function=0, reliability=5, years=1):
        self.cost_function      = cost_function
        self.reliability        = reliability           
        self.years              = years  
    
    def formulate(self):                                     
        self.capex              = 0
        self.maintenance        = 0
        self.dispatch           = cp.Variable((8760,1), nonpos=True)                     
        self.constraints        = []                                                                                              

        if type(self.cost_function) == np.ndarray:  
            if len(self.cost_function) == 3:
                self.opex           = self.cost_function[0]*cp.sum(self.dispatch**2) + self.cost_function[1]*cp.sum(self.dispatch) + self.cost_function[2]                
            if len(self.cost_function) == 2:
                self.opex           = self.cost_function[0]*cp.sum(self.dispatch) + self.cost_function[1]                 
            if len(self.cost_function) == 1:
                self.opex           = self.cost_function[0]*cp.sum(self.dispatch)                 
        else:
            self.opex           = 0

    def __str__(self):
        return f"This partially dispatchable renewable energy generator has the following attributes: \n \n Initial investment costs = {self.investment_cost} \n Operational costs = {self.fuel_function} \n Maintenance costs = {self.maintenance_cost} \n Max up rate power change = {self.rate_up} \n Max down rate power change = {self.rate_down}  \n \n Additionally, this generator has the following CVXPY variables: \n One variable for the installed capacity of length = ({self.years},1)" # \n One dispatch variable of length = ({self.years},1)" # It is possible to add this for the curtailment down

#endregion 


#%%

#region for data manipulation                                   

#region for the variables data import function                  

def variables(variables_csv):              
    '''
    Function to import some variables for the dictionary 'prob_info' from an csv file. This function is used only for the provided example. 
    The input parameters are:

    | **variables(variables_csv):** (csv file)
    | variables_csv is a csv file that contains the following information: 
    
    | project_life_time;20          # Set the life time of the project in years
    | interest_rate;4               # Set the interest rate   
    | esce;3                        # Set the number of scenarios (only for the stochastic analysis)                                                                                                          
    | years;3                       # Set the number of years (only for the multiyear analysis)                                                              
    | scala;1                       # Set the scala to adjust the peak load                                                                                                                                  
    | prxo;0.17                     # Set the price of the flat tariff (initial price)                                                                                                                            
    | percentage_yearly_growth;5    # Set the percentage of the yearly growth of the electric demand                                                                            
    | percentage_variation;50       # Set the percentage variation (only for generating data with noise)                                                                     
    | dlcpercenthour;30             # Set the percentage of allowed load curtailment in the DLC strategy                                                           
    | dlcpercenttotal;15            # Set the percentage of allowed load curtailment in the DLC strategy                                                           
    | sen_ince;0.1                  # Set the variable for sensitivity analysis of the values of the incentives                                                   
    | sen_ghi;1                     # Set the variable for sensitivity analysis of the values of the global horizontal radiation                               
    | elasticity;0.3                # Set the value of the elasticity of the customers                                                                              
    | curtailment;1                 # Set the value of allowed curtailment after the introduction of the DSM                                                       
    | capex_private;0.2             # Set the capex paid by the private investors                                                                                      
    | capex_gov;0.8                 # Set the capex paid by the govrnment                                                                                          
    | capex_community;0             # Set the capex paid by the community                                                                                              
    | capex_ong;0                   # Set the capex paid by the ONGs                                                                                             
    | opex_private;1                # Set the opex paid by the private investors                                                                                    
    | opex_gov;0                    # Set the opex paid by the govrnment                                                                                        
    | opex_community;0              # Set the opex paid by the community                                                                                              
    | opex_ong;0                    # Set the opex paid by the ONGs                                                                                             
    | rate_return_private;1.15      # Set the rate of return for private investors                                                                                            
    | max_value_tariff;0.34         # Set the maximum value of the tariff                                           
    | drpercentage;30               # Set the percentage of the demand that responds to the price incentive                                         
    | diesel_system;True            # Include a diesel generator system                                                             
    | pv_system;True                # Include a photovoltaic system                                                    
    | battery_system;True           # Include a battery energy storage system                                               
    | wind_system;True              # Include a wind generator system                                                    
    | hydro_system;False            # Set False, energy source not implemented yet                                         
    | hydrogen_system;False         # Set False, energy source not implemented yet                                            
    | gas_system;False              # Set False, energy source not implemented yet                                       
    | biomass_system;False          # Set False, energy source not implemented yet                                           
    | flat;False                    # Set True to select flat tariff, otherwise set False                          
    | tou;False                     # Set True to select ToU tariff, otherwise set False                          
    | tou_sun;False                 # Set True to select ToU_SUN tariff, otherwise set False                          
    | tou_three;False               # Set True to select ToU_Three tariff, otherwise set False                          
    | cpp;True                      # Set True to select CPP tariff, otherwise set False                         
    | dadp;False                    # Set True to select DADP tariff, otherwise set False                          
    | shape;False                   # Set True to select Shape tariff, otherwise set False                          
    | ince;False                    # Set True to select incentive-based DSMS, otherwise set False                 
    | dilc;False                    # Set True to select direct load curtailment DSMS, otherwise set False         
    | residential;False             # Set True to select this type of load, otherwise set False                           
    | commercial;False              # Set True to select this type of load, otherwise set False                          
    | industrial;False              # Set True to select this type of load, otherwise set False                          
    | community;True                # Set True to select this type of load, otherwise set False                        

    '''
    variables_csv=variables_csv.T
    variables_csv.columns = variables_csv.iloc[0]
    variables_csv.drop(variables_csv.index[0],inplace=True)

    project_life_time           = int(variables_csv.project_life_time[1])              # Set the life time of the project
    esce                        = int(variables_csv.esce[1])                           # Define number of escenarios 
    years                       = int(variables_csv.years[1])                          # Horizon of optimization in number of years
    interest_rate               = float(variables_csv.interest_rate[1]            )    # Set the interest rate 
    scala                       = float(variables_csv.scala[1]                    )    # How many times the demand is scaled
    prxo                        = float(variables_csv.prxo[1]                     )    # Initial flat tariff
    percentage_yearly_growth    = float(variables_csv.percentage_yearly_growth[1] )    # To define the percentage of variation of the demand
    percentage_variation        = float(variables_csv.percentage_variation[1]     )    # To define the percentage of variation of the demand
    dlcpercenthour              = float(variables_csv.dlcpercenthour[1]           )    # Percentage of direct load control each hour
    dlcpercenttotal             = float(variables_csv.dlcpercenttotal[1]          )    # Percentage of direct load control total 
    sen_ince                    = float(variables_csv.sen_ince[1]                 )    # Sensitivity analysis of incentives
    sen_ghi                     = float(variables_csv.sen_ghi[1]                  )    # Sensitivity analysis of global horizontal radiation     
    elasticity                  = float(variables_csv.elasticity[1]               )    # Parameter of elasticity   
    curtailment                 = float(variables_csv.curtailment[1]              )    # Parameter of curtailment    
    capex_private               = float(variables_csv.capex_private[1]            )    # Parameter of percentage of the capex paid by the investor   
    capex_gov                   = float(variables_csv.capex_gov[1]                )    # Parameter of percentage of the capex paid by the goverment    
    capex_community             = float(variables_csv.capex_community[1]          )    # Parameter of percentage of the capex paid by the community     
    capex_ong                   = float(variables_csv.capex_ong[1]                )    # Parameter of percentage of the capex paid by the ongs          
    opex_private                = float(variables_csv.opex_private[1]             )    # Parameter of percentage of the opex paid by the investor        
    opex_gov                    = float(variables_csv.opex_gov[1]                 )    # Parameter of percentage of the opex paid by the goverment   
    opex_community              = float(variables_csv.opex_community[1]           )    # Parameter of percentage of the opex paid by the community       
    opex_ong                    = float(variables_csv.opex_ong[1]                 )    # Parameter of percentage of the opex paid by the ongs        
    rate_return_private         = float(variables_csv.rate_return_private[1]      )    # Parameter of the rate of return of the private investors        
    max_value_tariff            = float(variables_csv.max_value_tariff[1]         )    # Parameter of the maximum value of the tariff
    drpercentage                = float(variables_csv.drpercentage[1]             )    # Parameter to control the percentage of the demand that responds to the price incentives
    diesel_system       	    = variables_csv.diesel_system  [1] == 'True'           # Include a diesel generator system                                                                                        
    pv_system           	    = variables_csv.pv_system      [1] == 'True'           # Include a photovoltaic system                                                                                       
    battery_system      	    = variables_csv.battery_system [1] == 'True'           # Include a battery energy storage system                                                                        
    wind_system         	    = variables_csv.wind_system    [1] == 'True'           # Include a wind generator system                                               
    hydro_system        	    = variables_csv.hydro_system   [1] == 'True'           # Set False, energy source not implemented yet                                             
    hydrogen_system     	    = variables_csv.hydrogen_system[1] == 'True'           # Set False, energy source not implemented yet                                          
    gas_system          	    = variables_csv.gas_system     [1] == 'True'           # Set False, energy source not implemented yet                                               
    biomass_system      	    = variables_csv.biomass_system [1] == 'True'           # Set False, energy source not implemented yet                                           
    flat                	    = variables_csv.flat           [1] == 'True'           # Set True to select flat tariff, otherwise set False                                                                      
    tou                 	    = variables_csv.tou            [1] == 'True'           # Set True to select ToU tariff, otherwise set False                                                                        
    tou_sun              	    = variables_csv.tou_sun        [1] == 'True'           # Set True to select ToU tariff, otherwise set False                                                                        
    tou_three             	    = variables_csv.tou_three      [1] == 'True'           # Set True to select ToU tariff, otherwise set False                                                                        
    cpp                 	    = variables_csv.cpp            [1] == 'True'           # Set True to select CPP tariff, otherwise set False                                                                        
    dadp                	    = variables_csv.dadp           [1] == 'True'           # Set True to select DADP tariff, otherwise set False                                                                      
    shape_tar            	    = variables_csv.shape_tar      [1] == 'True'           # Set True to select shape tariff, otherwise set False                                                                      
    ince                	    = variables_csv.ince           [1] == 'True'           # Set True to select incentive-based DSMS, otherwise set False                                                             
    dilc                	    = variables_csv.dilc           [1] == 'True'           # Set True to select direct load curtailment DSMS, otherwise set False                                                     
    residential         	    = variables_csv.residential    [1] == 'True'           # Set True to select this type of load, otherwise set False                                              
    commercial          	    = variables_csv.commercial     [1] == 'True'           # Set True to select this type of load, otherwise set False                                               
    industrial          	    = variables_csv.industrial     [1] == 'True'           # Set True to select this type of load, otherwise set False                                               
    community           	    = variables_csv.community      [1] == 'True'           # Set True to select this type of load, otherwise set False                                  

                                    
    return project_life_time, interest_rate, esce, years, scala, prxo, percentage_yearly_growth, percentage_variation, dlcpercenthour, dlcpercenttotal, sen_ince, sen_ghi, elasticity, curtailment, capex_private, capex_gov, capex_community, capex_ong, opex_private, opex_gov, opex_community, opex_ong, rate_return_private, max_value_tariff, drpercentage, diesel_system, pv_system, battery_system, wind_system, hydro_system, hydrogen_system, gas_system, biomass_system, flat,tou,tou_sun,tou_three,cpp,dadp,shape_tar,ince,dilc,residential,commercial,industrial,community

 
#endregion

#region for the data creator                                    

#region for the normal generator                                

def data_years_norm(data_in,years):
    '''
    Function designed to create synthetic data, using hourly normal distribution aproximated from the input data. Input parameters: 
    
    | **data_in:** (pandas dataframe) 
    | A pandas series of data indexed with an hourly time stamp over a period of one year.  

    | **years:** (int) 
    | Number of years to generate the data 

    This function returns a numpy array of dimension (1,8760*years), with the synthetic generated data.  
    '''

    #region to initialize data                          
    # data_frame      =   np.log(data_in.to_frame()*10000+2)
    data_frame      =   data_in.to_frame()
    data_process    =   np.zeros((1,8760*int(years)))
    days_per_month  =   np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    #endregion

    meses = data_frame.groupby(data_frame.index.month)
    months_list = []
    for mon, df_mes  in meses:
        months_list.append(np.array(df_mes.groupby(df_mes.index.hour)))

    params_list = []

    for con_months in range(len(months_list)):
        for con_hours in range(24):
            temp = stats.distributions.norm.fit(months_list[con_months][con_hours][1].values)
            params_list.append(temp)

    hour_one_by_one = 0
    for con_years in range(years):
        for con_months in range(len(months_list)):
            for con_days in range(days_per_month[con_months]):
                for con_hours in range(24):
                    data_process[0,hour_one_by_one] = np.random.normal(loc=params_list[24*con_months+con_hours][0], scale=params_list[24*con_months+con_hours][1]) # loc=mean, scale=standard deviation
                    if data_process[0,hour_one_by_one] < 0:
                        data_process[0,hour_one_by_one] = 0
                    hour_one_by_one+=1

    # data_out =np.exp(data_process)/10000
    data_out = data_process

    return data_out

def yearly_data_generator_norm(data_in, years=1, scenarios=1, percentage_yearly_growth=0):
    '''
    Function designed to create synthetic data, adding gaussian noise to the input data. Input parameters: 
    
    | **data_in:** 
    | A numpy array of data with hourly intervals over a period of one year.  
    
    | **years:** 
    | Number of years to generate the data 
    
    | **scenarios:** 
    | Number of scenarios to generate the data (stochastic analysis)
    
    | **percentage_variation:**
    | Percentage of the noise 
    
    | **percentage_yearly_growth:** 
    | Percentage of desired yearly growth of the output data (geometric growth). This function returns a numpy array of dimension (escenarios, 8760*years), with the synthetic generated data.  
    '''

    data_out    = np.zeros((scenarios,8760*years))

    if scenarios > 1:
        if years >1:
            #following scenarios
            for conscena in range(scenarios):
                for cyears in range(years):                   
                    data_out[conscena][8760*cyears:8760*(cyears+1)]=data_years_norm(data_in,1)*(1+cyears*percentage_yearly_growth/100)
            # Year zero
            data_out[0][0:8760] =   data_in.values
        else:
            # Following scenarios 
            for conscena in range(scenarios-1):                     
                data_out[conscena+1][:]=data_years_norm(data_in,years)
                # Year zero
                data_out[0][0:8760] =   data_in.values

    else:
        if years >1:
            for cyears in range(years):
                data_out[0][8760*cyears:8760*(cyears+1)] = data_years_norm(data_in,1)*(1+cyears*percentage_yearly_growth/100)
            # Year zero
            data_out[0][0:8760] =   data_in.values  

        else:
            # Year zero
            data_out[0][0:8760] =   data_in.values  

    return data_out

#endregion

#region for the beta generator                                  

def data_years_beta(data_in,years):
    '''
    Function designed to create synthetic data, using hourly normal distribution aproximated from the input data. Input parameters:

    | **data_in:** (pandas dataframe) 
    | A pandas series of data indexed with an hourly time stamp over a period of one year.  
    
    | **years:** (pandas dataframe) 
    | Number of years to generate the data 

    This function returns a numpy array of dimension (1,8760*years), with the synthetic generated data.  
    '''

    # if fit_type == "weekly":            
    #     #region for weekly fitting                  
    #     #region to initialize data for weekly fitting
    #     data_frame      =   data_in.to_frame()
    #     data_process    =   np.zeros((1,8760*int(years)))
    #     days_per_month  =   np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    #     #endregion
    #     weeks = data_frame.groupby(data_frame.index.week)
    #     weeks_list = []
    #     for mon, df_mes  in weeks:
    #         weeks_list.append(np.array(df_mes.groupby(df_mes.index.hour)))

    #     params_list = []

    #     for con_weeks in range(len(weeks_list)):
    #         for con_hours in range(24):
    #             temp = stats.distributions.beta.fit(weeks_list[con_weeks][con_hours][1].values+0.2)
    #             params_list.append(temp)

    #     hour_one_by_one = 0
    #     for con_years in range(years):
    #         for con_weeks in range(len(weeks_list)):    
    #             for con_days in range(7): 
    #                 for con_hours in range(24):
    #                     data_process[0,hour_one_by_one] = stats.beta.rvs(a=params_list[24*con_weeks+con_hours][0], b=params_list[24*con_weeks+con_hours][1] , loc=params_list[24*con_weeks+con_hours][2], scale=params_list[24*con_weeks+con_hours][3]) 
    #                     if data_process[0,hour_one_by_one] < 0:
    #                         data_process[0,hour_one_by_one] = 0
    #                     hour_one_by_one+=1
    #                     print(hour_one_by_one)
    #     #endregion

    # if fit_type == "monthly": 
    #region to initialize data                          
    # data_frame      =   np.log(data_in.to_frame()*10000+2)
    data_frame      =   data_in.to_frame()
    data_process    =   np.zeros((1,8760*int(years)))
    days_per_month  =   np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    #endregion

    meses = data_frame.groupby(data_frame.index.month)
    months_list = []
    for mon, df_mes  in meses:
        months_list.append(np.array(df_mes.groupby(df_mes.index.hour)))

    params_list = []

    for con_months in range(len(months_list)):
        for con_hours in range(24):
            temp = stats.distributions.beta.fit(months_list[con_months][con_hours][1].values)
            params_list.append(temp)

    hour_one_by_one = 0
    for con_years in range(years):
        for con_months in range(len(months_list)):
            for con_days in range(days_per_month[con_months]):
                for con_hours in range(24):
                    data_process[0,hour_one_by_one] = stats.beta.rvs(a=params_list[24*con_months+con_hours][0], b=params_list[24*con_months+con_hours][1],loc=params_list[24*con_months+con_hours][2], scale=params_list[24*con_months+con_hours][3]) # loc=mean, scale=standard deviation
                    if data_process[0,hour_one_by_one] < 0:
                        data_process[0,hour_one_by_one] = 0
                    hour_one_by_one+=1

    # data_out =np.exp(data_process)/10000
    data_out = data_process

    return data_out

def yearly_data_generator_beta(data_in, years=1, scenarios=1, percentage_yearly_growth=0):
    '''
    Function designed to create synthetic data, using a Beta distribution. Input parameters:

    | **data_in:** (pandas dataframe)
    | A numpy array of data with hourly intervals over a period of one year.  
    
    | **years:** (int) 
    | Number of years to generate the data 
    
    | **scenarios:** (int)
    | Number of scenarios to generate the data (stochastic analysis)
    
    | **percentage_variation:** (int, float) 
    | Percentage of the noise 
    
    | **percentage_yearly_growth:** (int, float)
    | Percentage of desired yearly growth of the output data (geometric growth)
    
    This function returns a numpy array of dimension (escenarios, 8760*years), with the synthetic generated data.  
    '''

    data_out    = np.zeros((scenarios,8760*years))

    if scenarios > 1:
        if years >1:
            #following scenarios
            for conscena in range(scenarios):
                for cyears in range(years):                   
                    data_out[conscena][8760*cyears:8760*(cyears+1)]=data_years_beta(data_in,1)*(1+cyears*percentage_yearly_growth/100)
            # Year zero
            data_out[0][0:8760] =   data_in.values
        else:
            # Following scenarios 
            for conscena in range(scenarios-1):                     
                data_out[conscena+1][:]=data_years_beta(data_in,years)
                # Year zero
                data_out[0][0:8760] =   data_in.values

    else:
        if years >1:
            for cyears in range(years):
                data_out[0][8760*cyears:8760*(cyears+1)] = data_years_beta(data_in,1)*(1+cyears*percentage_yearly_growth/100)
            # Year zero
            data_out[0][0:8760] =   data_in.values  

        else:
            # Year zero
            data_out[0][0:8760] =   data_in.values  

    return data_out

#endregion

#region for the weibull generator                               

def fitweibull(x): 
    '''
    Function designed to fit a weibull distribution.
    '''                                             
    def optfun(theta):
        return -np.sum(np.log(exponweib.pdf(x, 1, theta[0], scale = theta[1], loc = 0)))
    logx = np.log(x)
    for a in range(len(x)):
        if logx[a] is np.nan: 
            logx[a] == 2
    shape = 1.2 / np.std(logx)
    scale = np.exp(np.mean(logx) + (0.572 / shape))
    return fmin(optfun, [shape, scale], xtol = 0.01, ftol = 0.01, disp = 0)

def data_years_weibull(data_in,years=1,fit_type="weekly"):                 
    '''
    Function designed to create synthetic data, using a hourly weibull distribution aproximated from the input data. Input parameters: 
    
    | **data_in:** (pandas dataframe)
    | A pandas series of data indexed with an hourly time stamp over a period of one year.  
    
    | **years:** (int) 
    | Number of years to generate the data 

    This function returns a numpy array of dimension (1,8760*years), with the synthetic generated data.  
    '''
    if fit_type == "weekly":            
        #region for weekly fitting                  
        #region to initialize data for weekly fitting
        data_frame      =   data_in.to_frame()
        data_process    =   np.zeros((1,8760*int(years)))
        days_per_month  =   np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        #endregion
        weeks = data_frame.groupby(data_frame.index.week)
        weeks_list = []
        for mon, df_mes  in weeks:
            weeks_list.append(np.array(df_mes.groupby(df_mes.index.hour)))

        params_list = []

        for con_weeks in range(len(weeks_list)):
            for con_hours in range(24):
                temp = fitweibull(weeks_list[con_weeks][con_hours][1].values+0.1)
                params_list.append(temp)

        hour_one_by_one = 0
        for con_years in range(years):
            for con_weeks in range(len(weeks_list)):     
                for con_days in range(7):
                    for con_hours in range(24):
                        data_process[0,hour_one_by_one] = stats.weibull_min.rvs(params_list[24*con_weeks+con_hours][0], loc=0, scale=params_list[24*con_weeks+con_hours][1], size=1) # (a=params_list[24*con_months+con_hours][0]) # loc=mean, scale=standard deviation
                        if data_process[0,hour_one_by_one] < 0:
                            data_process[0,hour_one_by_one] = 0
                        hour_one_by_one+=1
        #endregion

    if fit_type == "monthly":           
        #region for mothly fitting                  
        #region to initialize data for monthly fitting                         
        data_frame      =   data_in.to_frame()
        data_process    =   np.zeros((1,8760*int(years)))
        days_per_month  =   np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        #endregion

        meses = data_frame.groupby(data_frame.index.month)
        months_list = []
        for mon, df_mes  in meses:
            months_list.append(np.array(df_mes.groupby(df_mes.index.hour)))
        params_list = []
        for con_months in range(len(months_list)):
            for con_hours in range(24):
                temp = fitweibull(months_list[con_months][con_hours][1].values+0.1)
                params_list.append(temp)
        hour_one_by_one = 0
        for con_years in range(years):
            for con_months in range(len(months_list)):
                for con_days in range(days_per_month[con_months]):
                    for con_hours in range(24):
                        data_process[0,hour_one_by_one] = stats.weibull_min.rvs(params_list[24*con_months+con_hours][0], loc=0, scale=params_list[24*con_months+con_hours][1], size=1) # (a=params_list[24*con_months+con_hours][0]) # loc=mean, scale=standard deviation
                        if data_process[0,hour_one_by_one] < 0:
                            data_process[0,hour_one_by_one] = 0
                        hour_one_by_one+=1
        #endregion

    data_out = data_process

    return data_out

def yearly_data_generator_weibull(data_in, years=1, scenarios=1, percentage_yearly_growth=0,fit_type="weekly"):
    '''
    Function designed to create synthetic data, using a Weibull distribution. Input parameters:

    | **data_in:** (numpy ndarray) 
    | A numpy array of data with hourly intervals over a period of one year.  
    
    | **years:** (int) 
    | Number of years to generate the data 
    
    | **scenarios:** (int) 
    | Number of scenarios to generate the data (stochastic analysis)
    
    | **percentage_variation:** (int, float) 
    | Percentage of the noise 
    
    | **percentage_yearly_growth:** (int, float) 
    | Percentage of desired yearly growth of the output data (geometric growth)
    
    This function returns a numpy array of dimension (escenarios, 8760*years), with the synthetic generated data.  
    '''

    data_out    = np.zeros((scenarios,8760*years))

    if scenarios > 1:
        if years >1:
            #following scenarios
            for conscena in range(scenarios):
                for cyears in range(years):                   
                    data_out[conscena][8760*cyears:8760*(cyears+1)]=data_years_weibull(data_in=data_in,years=1,fit_type=fit_type)*(1+cyears*percentage_yearly_growth/100)
            # Year zero
            data_out[0][0:8760] =   data_in.values
        else:
            # Following scenarios 
            for conscena in range(scenarios-1):                     
                data_out[conscena+1][:]=data_years_weibull(data_in=data_in,years=years,fit_type=fit_type)
                # Year zero
                data_out[0][0:8760] =   data_in.values

    else:
        if years >1:
            for cyears in range(years):
                data_out[0][8760*cyears:8760*(cyears+1)] = data_years_weibull(data_in=data_in,years=1,fit_type=fit_type)*(1+cyears*percentage_yearly_growth/100)
            # Year zero
            data_out[0][0:8760] =   data_in.values  

        else:
            # Year zero
            data_out[0][0:8760] =   data_in.values  

    return data_out

#endregion

#region for the resources import with normal fitting            

def resources_norm(data_input,years=1,scenarios=1, percentage_yearly_growth=0):
    '''
    Function designed to create synthetic data, using only normal distributions. Input parameters:

    | **data_input:** (numpy ndarray) 
    | A numpy array of data with hourly intervals over a period of one year.  
    
    | **years:** (int) 
    | Number of years to generate the data 
    
    | **scenarios:** (int) 
    | Number of scenarios to generate the data (stochastic analysis)
       
    | **percentage_yearly_growth:** (int, float) 
    | Percentage of desired yearly growth of the output data (geometric growth)
    
    This function returns a numpy array of dimension (escenarios, 8760*years), with the synthetic generated data.  
    '''    
        
    data_input.columns = data_input.iloc[0]
    data_input.drop(labels=[0,1], axis=0, inplace=True)
    data_input["date"]    = pd.date_range('2018-01-01', periods=8760, freq='H')
    data_input.set_index('date',inplace=True)

    ghi                 = yearly_data_generator_norm(data_input.ghi.astype(float),                years=years, scenarios=scenarios, )            
    irrdiffuse          = yearly_data_generator_norm(data_input.irrdiffuse.astype(float),         years=years, scenarios=scenarios, )      
    temperature         = yearly_data_generator_norm(data_input.temperature.astype(float),        years=years, scenarios=scenarios, )     
    wind                = yearly_data_generator_norm(data_input.wind.astype(float),               years=years, scenarios=scenarios, )            
    hydro               = yearly_data_generator_norm(data_input.hydro.astype(float),              years=years, scenarios=scenarios, )           
    load_residential    = yearly_data_generator_norm(data_input.load_residential.astype(float),   years=years, scenarios=scenarios, percentage_yearly_growth = int(percentage_yearly_growth))
    load_commercial     = yearly_data_generator_norm(data_input.load_commercial.astype(float),    years=years, scenarios=scenarios, percentage_yearly_growth = int(percentage_yearly_growth)) 
    load_industrial     = yearly_data_generator_norm(data_input.load_industrial.astype(float),    years=years, scenarios=scenarios, percentage_yearly_growth = int(percentage_yearly_growth)) 
    load_community      = yearly_data_generator_norm(data_input.load_community.astype(float),     years=years, scenarios=scenarios, percentage_yearly_growth = int(percentage_yearly_growth))       

    return ghi, irrdiffuse, temperature, wind, hydro, load_residential, load_commercial, load_industrial, load_community

#endregion 

#region for the resources import with normal weibull fittings   

def resources_all(data_input,years=1,scenarios=1, percentage_yearly_growth=0, fit_type="weekly"):
    '''
    Function designed to create synthetic data, using normal, beta and weibull distributions. The normal distribution is used for the temperature, hydro, and the different types of load. The beta distribution is used for the GHI and the diffuse irradaition. The weibull distribution is used ofr the wind. The input parameters are:

    | **data_input:** (numpy ndarray) 
    | A numpy array of data with hourly intervals over a period of one year.  
    
    | **years:** (int) 
    | Number of years to generate the data 
    
    | **scenarios:** (int) 
    | Number of scenarios to generate the data (stochastic analysis)
       
    | **percentage_yearly_growth:** (int, float) 
    | Percentage of desired yearly growth of the output data (geometric growth)
    
    This function returns a numpy array of dimension (escenarios, 8760*years), with the synthetic generated data.  
    '''   

    data_input.columns = data_input.iloc[0]
    data_input.drop(labels=[0,1], axis=0, inplace=True)
    data_input["date"]    = pd.date_range('2018-01-01', periods=8760, freq='H')
    data_input.set_index('date',inplace=True)

    ghi                 = yearly_data_generator_beta(data_input.ghi.astype(float),                years=years, scenarios=scenarios )            
    irrdiffuse          = yearly_data_generator_beta(data_input.irrdiffuse.astype(float),         years=years, scenarios=scenarios )      
    temperature         = yearly_data_generator_norm(data_input.temperature.astype(float),        years=years, scenarios=scenarios )   
    wind                = yearly_data_generator_weibull(data_input.wind.astype(float),            years=years, scenarios=scenarios, fit_type = fit_type)            
    hydro               = yearly_data_generator_norm(data_input.hydro.astype(float),              years=years, scenarios=scenarios )           
    load_residential    = yearly_data_generator_norm(data_input.load_residential.astype(float),   years=years, scenarios=scenarios, percentage_yearly_growth = percentage_yearly_growth)
    load_commercial     = yearly_data_generator_norm(data_input.load_commercial.astype(float),    years=years, scenarios=scenarios, percentage_yearly_growth = percentage_yearly_growth) 
    load_industrial     = yearly_data_generator_norm(data_input.load_industrial.astype(float),    years=years, scenarios=scenarios, percentage_yearly_growth = percentage_yearly_growth) 
    load_community      = yearly_data_generator_norm(data_input.load_community.astype(float),     years=years, scenarios=scenarios, percentage_yearly_growth = percentage_yearly_growth)       

    return ghi, irrdiffuse, temperature, wind, hydro, load_residential, load_commercial, load_industrial, load_community

#endregion 

#region for the noise generator                                 

def yearly_data_generator_noise(data_in, years=1, scenarios=1, percentage_variation=10, percentage_yearly_growth=0):
    '''
    Function designed to create synthetic data, adding gaussian noise. Input parameters:

    | **data_in:** (numpy ndarray) 
    | A numpy array of data with hourly intervals over a period of one year.  
    
    | **years:** (int) 
    | Number of years to generate the data 
    
    | **scenarios:** (int) 
    | Number of scenarios to generate the data (stochastic analysis)

    | **percentage_variation:** (int, float)
    | Variation of the gaussian noise. 

    | **percentage_yearly_growth:** (int, float) 
    | Percentage of desired yearly growth of the output data (geometric growth)
    
    This function returns a numpy array of dimension (escenarios, 8760*years), with the synthetic generated data.  
    '''   
    desvstd     = 0.002*percentage_variation 
    data_out    = np.zeros((scenarios,8760*years))

    if scenarios > 1:
        if years >1:
            # # Year zero
            # data_out[0][0:8760] =   data_in[0][0:8760]

            #following scenarios
            for conscena in range(scenarios):
                for cyears in range(years):
                    for chour in range(8760):                        
                        data_out[conscena][chour+cyears*8760]=data_in[0][chour]*np.random.normal(1,desvstd,1)*(1+cyears*percentage_yearly_growth/100)
        else:
            # Year zero
            data_out[0][0:8760] =   data_in[0][0:8760]
            
            #following scenarios 
            for conscena in range(scenarios-1):
                for chour in range(8760):                        
                    data_out[conscena+1][chour]=data_in[0][chour]*np.random.normal(1,desvstd,1)

    else:
        if years >1:
            # Year zero
            data_out[0][0:8760] =   data_in[0][0:8760]  
            
            #following years
            year=1
            for cyears in range(years-1):
                for chour in range(8760):                        
                    data_out[0][chour+year*8760]=data_in[0][chour]*np.random.normal(1,desvstd,1)*(1+year*percentage_yearly_growth/100)
                year+=1
        else:
            # Year zero
            data_out[0][0:8760] =   data_in[0][0:8760]  

    return data_out

#endregion

#region for the resources import with noise                     

def resources_noise(data_input,years=1,scenarios=1, percentage_variation=10, percentage_yearly_growth=0):
    '''
    Function designed to create synthetic data, adding gaussian noise. Input parameters:

    | **data_in:** (numpy ndarray) 
    | A numpy array of data with hourly intervals over a period of one year.  
    
    | **years:** (int) 
    | Number of years to generate the data 
    
    | **scenarios:** (int) 
    | Number of scenarios to generate the data (stochastic analysis)

    | **percentage_variation:** (int, float)
    | Variation of the gaussian noise. 

    | **percentage_yearly_growth:** (int, float) 
    | Percentage of desired yearly growth of the output data (geometric growth)
    
    This function returns a numpy array of dimension (escenarios, 8760*years), with the synthetic generated data.  
    '''     
    data_input.columns = data_input.iloc[0]
    data_input.drop(labels=[0,1], axis=0, inplace=True)
    data_input["date"]    = pd.date_range('2018-01-01', periods=8760, freq='H')
    data_input.set_index('date',inplace=True)

    ghi                 = yearly_data_generator_noise(data_input.ghi.values.astype(float).reshape(1,8760),                years=years, scenarios=scenarios, percentage_variation=percentage_variation)            
    irrdiffuse          = yearly_data_generator_noise(data_input.irrdiffuse.values.astype(float).reshape(1,8760),         years=years, scenarios=scenarios, percentage_variation=percentage_variation)      
    temperature         = yearly_data_generator_noise(data_input.temperature.values.astype(float).reshape(1,8760),        years=years, scenarios=scenarios, percentage_variation=percentage_variation)     
    wind                = yearly_data_generator_noise(data_input.wind.values.astype(float).reshape(1,8760),               years=years, scenarios=scenarios, percentage_variation=percentage_variation)            
    hydro               = yearly_data_generator_noise(data_input.hydro.values.astype(float).reshape(1,8760),              years=years, scenarios=scenarios, percentage_variation=percentage_variation)           
    load_residential    = yearly_data_generator_noise(data_input.load_residential.values.astype(float).reshape(1,8760),   years=years, scenarios=scenarios, percentage_variation=percentage_variation, percentage_yearly_growth = percentage_yearly_growth)
    load_commercial     = yearly_data_generator_noise(data_input.load_commercial.values.astype(float).reshape(1,8760),    years=years, scenarios=scenarios, percentage_variation=percentage_variation, percentage_yearly_growth = percentage_yearly_growth) 
    load_industrial     = yearly_data_generator_noise(data_input.load_industrial.values.astype(float).reshape(1,8760),    years=years, scenarios=scenarios, percentage_variation=percentage_variation, percentage_yearly_growth = percentage_yearly_growth) 
    load_community      = yearly_data_generator_noise(data_input.load_community.values.astype(float).reshape(1,8760),     years=years, scenarios=scenarios, percentage_variation=percentage_variation, percentage_yearly_growth = percentage_yearly_growth)       

    return ghi, irrdiffuse, temperature, wind, hydro, load_residential, load_commercial, load_industrial, load_community

#endregion 

#endregion

#endregion


#%%

#region for the class of the DSMS                               

class ComputeDSMS():                    
    '''
    This class computes the response of the customers to the incentive of the DSMS. Additionally, this class provides the customer payments, the demand after the application of the DSMS, and the tariff for the optimization horizon.
    This class takes the following parameters as inputs: 
    
    | **loadfile:** (numpy.ndarray)
    | File vector that contains the electrical demand. 

    | **elasticity:** (int, float)
    | Value of elasticity of the customers.

    | **initial_price:** (int, float)
    | Initial price of the electric energy. Value of the flat tariff. 

    | **years:** (int)
    | Number of years for the optimization horizon. 

    | **dlcpercenthour:** (int, float)
    | Percentage of load curtailment each hour for the Direct Load Curtailment DSMS. 

    | **dlcpercenttotal:** (int, float)
    | Percentage of load curtailment for the day for the Direct Load Curtailment DSMS. 

    | **flat:** (True or False)
    | Assign True to select flat tariff, otherwise assign False. 

    | **tou:** (True or False)
    | Assign True to select Time of Use tariff, otherwise assign False. 

    | **tou_sun:** (True or False)
    | Assign True to select Time of Use tariff for sun and off sun hours, otherwise assign False. 

    | **tou_three:** (True or False)
    | Assign True to select Time of Use tariff of three levels, otherwise assign False. 

    | **cpp:** (True or False)
    | Assign True to select Critical Peak Pricing tariff, otherwise assign False. 

    | **dadp:** (True or False)
    | Assign True to select Day Ahead Dynamic Pricing tariff, otherwise assign False. 

    | **shape:** (True or False)
    | Assign True to select Shape tariff, otherwise assign False. 

    | **ince:** (True or False)
    | Assign True to select Incentive-Based Pricing tariff, otherwise assign False. Incentive-based DSMS assumes a flat tariff for the base. Additionally to the base tariff (flat) the incentives are computed. 

    | **dilc:** (True or False)
    | Assign True to select the Direct Load Curtailment DSMS, otherwise assign False. Direct Load Curtailment DSMS assumes a flat tariff. 

    | **drpercentage:** (int or float)
    | Define the percentage of the demand that is sensible to the variations in price. This value must be a percentage (25%, 35%, etc.).

    | **simulation_year:** (int)        
    | Year of simulation. By default this parameter is zero. It is only used when a multiyear analysis is applied. 
    
    '''


    def __init__(self, loadfile, elasticity, initial_price, max_value_tariff, years, dlcpercenthour, dlcpercenttotal, flat, tou, tou_sun, tou_three, cpp, dadp, shape_tar, ince, dilc, drpercentage, simulation_year=0):                               

        #region to initialice parameters
        T=8760
        Load = loadfile
        dias=365
        self.constraints = []
        #endregion

        #region for the creation of variables and self.constraints for the fares     
        
        # Creation of the variables of the flat fare  
        if flat == True:                
            pflat = cp.Variable((1,1), nonneg=True)           # Price of the flat fare
            # self.constraints for the flat fare
            self.constraints += [pflat <= initial_price] # Maximum value of the flat fare
            pfs=np.ones((T,1))                                         # Modulation signal
            self.tariff=pflat*pfs                                          # File vector of prices

        # Creation of the variables of the ToU fare
        if tou == True:                 
            pon         = cp.Variable((1,1), nonneg=True)           # Peak price of the ToU fare
            poff        = cp.Variable((1,1), nonneg=True)           # Off peak price of the ToU fare

            vtoou=np.zeros(T)
            b=0
            for a in range(dias):
                entryy = 18 # Entrancy of the fare Pon
                exitt  = 22 # Exit of the fare Pon
                duration=exitt-entryy
                for t in range(entryy):
                    b+=1

                for t in range(duration):
                    vtoou[b]=1
                    b+=1

                for t in range(24-exitt):
                    b+=1

            temp1=pon*vtoou+poff
            self.tariff=temp1.T

        # Creation of the variables of the ToU_sun fare
        if tou_sun == True:             
            psun           = cp.Variable((1,1), nonneg=True)           # Peak price of the ToU fare
            poffsun        = cp.Variable((1,1), nonneg=True)           # Off peak price of the ToU fare

            vsunmod=np.zeros(T)
            b=0
            for a in range(dias):
                entryy = 9 # Entrancy of the fare psun
                exitt  = 17 # Exit of the fare psun
                duration=exitt-entryy
                for t in range(entryy):
                    b+=1

                for t in range(duration):
                    vsunmod[b]=1
                    b+=1

                for t in range(24-exitt):
                    b+=1

            temp2=-psun*vsunmod+poffsun
            self.constraints += [-psun+poffsun >=0]
            self.tariff=temp2.T
 
        # Creation of the variables of the ToU_three levels fare
        if tou_three == True:           
            psun           = cp.Variable((1,1), nonneg=True)           # Peak price of the ToU fare
            pon            = cp.Variable((1,1), nonneg=True)           # Off peak price of the ToU fare
            poff           = cp.Variable((1,1), nonneg=True)           # Off peak price of the ToU fare
            
            vonmod=np.zeros(T)
            b=0
            for a in range(dias):
                entryy = 17 # Entrancy of the fare Pon
                exitt  = 22 # Exit of the fare Pon
                duration=exitt-entryy
                for t in range(entryy):
                    b+=1

                for t in range(duration):
                    vonmod[b]=1
                    b+=1

                for t in range(24-exitt):
                    b+=1

            vsunmod=np.zeros(T)
            b=0
            for a in range(dias):
                entryy = 9 # Entrancy of the fare psun
                exitt  = 17 # Exit of the fare psun
                duration=exitt-entryy
                for t in range(entryy):
                    b+=1

                for t in range(duration):
                    vsunmod[b]=1
                    b+=1

                for t in range(24-exitt):
                    b+=1

            temp3=-vsunmod*psun+vonmod*pon+poff
            self.constraints += [-psun+poff >=0]
            self.tariff=temp3.T

        # Creation of the variables of the CPP fare
        if cpp == True:                 
            pbs = cp.Variable((1,1),    nonneg=True)           # Base price of the CPP fare
            pcr = cp.Variable((T,1),    nonneg=True)           # Peak price
            self.tariff=pbs+pcr

            # self.constraints for the CPP fare
            self.constraints += [
                cp.sum(pcr) <= int(0.0018*T*5)*initial_price,               # Time restriction of the peak price
                pcr <= 5*initial_price,
                ]

        # Creation of the variables of the DADP fare
        if dadp == True:                
            self.tariff = cp.Variable((T,1), nonneg=True)           # Energy fare

        # Creation of the variables of the SHAPE fare
        if shape_tar == True:               
            vars_shape ={}
            for counter in range(24):
                vars_shape[f'h{counter}'] = cp.Variable((1,1),    nonneg=True)        

            mods=[]
            for bigcounter in range(24):
                temphour = np.zeros((8760,1))
                for dias in range(365):
                    temphour[dias*24+bigcounter,0]=1
                mods.append(temphour)

            multip = []
            for counter in range(24):
                multip.append(cp.multiply(mods[counter],vars_shape[f'h{counter}']))

            self.tariff = cp.sum(multip)

        # Creation of the variables of the IBP fare
        if ince == True:                
            price_ince = cp.Variable((T,1))           # Energy fare
            pbase=np.ones((T,1))
            self.tariff=initial_price*pbase+price_ince
            # Creation of the self.constraints
            self.constraints += [
                price_ince         <= 0.5,                # Maximum value of the incentive
                price_ince         >= -0.1,              # Minimum value of the incentive
                ]

        # Creation of the variables of the DLC fare
        if dilc == True:                
            self.dlc    = cp.Variable((T,1), nonneg=True)       # Quantity of energy to curtail    
            maxdem      = np.multiply(dlcpercenthour/100, Load) # Maximum allowed curtailed demand
            pflat       = initial_price                         # Price of the flat fare
            pfs         = np.ones((T,1))                        # Modulation signal
            self.tariff = cp.multiply(pflat, pfs)
            self.constraints += [ 
                                self.dlc <= maxdem,
                                cp.sum(self.dlc) <= dlcpercenttotal/100*cp.sum(Load)
                                ] # Maximum value of the flat fare
            
 
            # pflat       = cp.Variable((1,1), nonneg=True)       # Price of the flat fare
            # self.tariff = cp.multiply(pflat, pfs)
            # self.constraints += [pflat <= initial_price, 
            #                     self.dlc <= maxdem
            #                     ] # Maximum value of the flat fare
            
        #endregion

        #region to compute the demand response                                  
        # Linear response of the demand (Equation 5)
        responsible_load    = drpercentage/100*Load
        no_responsible_load = Load-responsible_load
        
        if dilc == True: 
            self.loadDSMS    = cp.multiply(Load,(-elasticity/initial_price*self.tariff+elasticity+1)) - self.dlc 
            self.customer_payments = cp.multiply(self.loadDSMS, self.tariff) # Payments of the customers
            # self.constraints += [cp.sum(self.dlc) <= dlcpercenttotal/100*cp.sum(Load)]
        else:
            # self.customer_payments  = cp.multiply(responsible_load,(-elasticity*self.tariff**2/initial_price+self.tariff*(1+elasticity)))    # Payments of the customers
            self.demand_response    = cp.multiply(responsible_load,(-elasticity/initial_price*self.tariff+elasticity+1))    # Final load (after DSMS)
            self.loadDSMS           = no_responsible_load + self.demand_response               
            self.customer_payments  = cp.multiply(self.loadDSMS, self.tariff)                                               # Payments of the customers
        #endregion

        self.constraints += [self.tariff <=  max_value_tariff] # Maximum price

#endregion

#%%

#region to create a look up table for the prices of the technologies    

class LookTable():                              
    '''
    LookTable class creates an iterable object that contains the capex values of the energy sources until the year 2050. All source data for the CAPEX prices can be found in https://atb.nrel.gov/electricity/2019/data.html and the source data for the diesel price can be foun in https://www.eia.gov/outlooks/aeo/tables_ref.php
    '''
    #region to create the fitting of the yearly data 
    # All source data can be found in https://atb.nrel.gov/electricity/2019/data.html
    # [1] W. Cole and A. W. Frazier, “Cost Projections for Utility- Scale Battery Storage Cost Projections for Utility- Scale Battery Storage,” Natl. Renew. Energy Lab., no. June, p. NREL/TP-6A20-73222, 2019.
    # https://www.eia.gov/outlooks/aeo/tables_ref.php
    def __init__(self):            

        #region for the standard case           

        #region for the BESS [NREL]             
        self.table_bess=np.array([371,346,321,305,289,274,258,242,234,226,218,210,203,200,198,195,192,190,187,185,182,180,177,175,172,170,167,165,162,160,157,154,152])
        #endregion

        #region for the PV                      
        self.table_pv=np.array([1810,1723,1623,1600,1563,1526,1489,1452,1415,1378,1341,1304,1267,1255,1243,1231,1219,1208,1196,1184,1172,1160,1148,1134,1121,1108,1094,1081,1067,1054,1041,1027,1014])
        #endregion

        #region for the WIND [NREL]             
        self.table_wind=np.array([1583,1555,1528,1500,1472,1445,1417,1390,1362,1335,1307,1280,1252,1239,1225,1212,1198,1184,1171,1157,1144,1130,1116,1102,1089,1075,1061,1047,1033,1020,1006,992,978])
        #endregion

        #region for the DIESEL [EIA]             
        self.table_diesel=np.array([3.2,3.04,2.93,2.95,2.99,3.01,3.07,3.08,3.14,3.15,3.20,3.23,3.29,3.33,3.36,3.41,3.44,3.47,3.50,3.53,3.56,3.59,3.59,3.61,3.66,3.69,3.71,3.76,3.77,3.80,3.84,3.86,3.88])/4
        #endregion
        
        #region for the GENSET
        self.table_genset=np.array([600, 582.79, 567.05, 552.68, 539.54, 527.53, 516.55, 506.52, 497.35, 488.97, 481.31, 474.32, 467.92, 462.07, 456.73, 451.85, 447.39, 443.31, 439.58, 436.17, 433.06, 430.21, 427.61, 425.24, 423.07, 421.08, 419.27, 417.61, 416.09, 414.71, 413.44, 412.28, 411.23])
        #endregion

        #region for the carbon tax (CTAX)              
        self.table_ctax=np.array([0,20,30,40,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50])
        #endregion

        self.table_cpi=np.array([3.18,3.8,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2])

        #endregion

        #region for the low case                

        #region for the BESS [NREL]             
        self.table_bess_low=np.array([371,331,290,268,246,224,202,180,168,156,145,133,121,119,117,114,112,110,107,105,102,100,98,95,93,91,88,86,83,81,79,76,74])
        #endregion

        #region for the PV [NREL]                      
        self.table_pv_low=np.array([1810,1333,1248,1123,1085,1047,1008,970,932,894,856,818,780,761,742,723,704,684,665,646,627,608,589,584,579,574,568,543,542,541,540,538,537])
        #endregion

        #region for the WIND [NREL]             
        self.table_wind_low=np.array([1610,1573,1535,1498,1461,1423,1386,1349,1311,1274,1237,1199,1162,1125,1109,1094,1079,1064,1049,1034,1018,1003,987,972,956,941,925,909,893,877,861,845,829,813])
        #endregion

        #region for the DIESEL [EIA]             
        self.table_diesel_low=np.array([0.64 , 0.608, 0.586, 0.59 , 0.598, 0.602, 0.614, 0.616, 0.628,0.63 , 0.64 , 0.646, 0.658, 0.666, 0.672, 0.682, 0.688, 0.694,0.7  , 0.706, 0.712, 0.718, 0.718, 0.722, 0.732, 0.738, 0.742,0.752, 0.754, 0.76 , 0.768, 0.772, 0.776])
        #reduction of 20%
        #endregion
        
        #region for the GENSET
        self.table_genset_low=np.array([480.   , 466.232, 453.64 , 442.144, 431.632, 422.024, 413.24 ,405.216, 397.88 , 391.176, 385.048, 379.456, 374.336, 369.656,365.384, 361.48 , 357.912, 354.648, 351.664, 348.936, 346.448,344.168, 342.088, 340.192, 338.456, 336.864, 335.416, 334.088,332.872, 331.768, 330.752, 329.824, 328.984])
        #reduction of 20%
        #endregion

        #region for the carbon tax (CTAX)              
        self.table_ctax_low=np.array([0,20,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30])
        #endregion

        #region for the Consumer Prices Index (CPI)
        self.table_cpi_low=np.array([3.28,3.8,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6])
        #reduction of 20%
        #endregion

        #endregion

        #region for the high case               

        #region for the BESS [NREL]             
        self.table_bess_high=np.array([371,346,321,305,289,274,258,242,234,226,218,210,203,200,198,195,192,190,187,185,182,180,177,175,172,170,167,165,162,160,157,154,152])
        #endregion

        #region for the PV                      
        self.table_pv_high=np.array([1810,1723,1623,1600,1563,1526,1489,1452,1415,1378,1341,1304,1267,1255,1243,1231,1219,1208,1196,1184,1172,1160,1148,1134,1121,1108,1094,1081,1067,1054,1041,1027,1014])
        #endregion

        #region for the WIND [NREL]             
        self.table_wind_high=np.array([1583,1555,1528,1500,1472,1445,1417,1390,1362,1335,1307,1280,1252,1239,1225,1212,1198,1184,1171,1157,1144,1130,1116,1102,1089,1075,1061,1047,1033,1020,1006,992,978])
        #endregion

        #region for the DIESEL [EIA]             
        self.table_diesel_high=np.array([3.2,3.04,2.93,2.95,2.99,3.01,3.07,3.08,3.14,3.15,3.20,3.23,3.29,3.33,3.36,3.41,3.44,3.47,3.50,3.53,3.56,3.59,3.59,3.61,3.66,3.69,3.71,3.76,3.77,3.80,3.84,3.86,3.88])/4
        #endregion
        
        #region for the GENSET
        self.table_genset_high=np.array([600, 582.79, 567.05, 552.68, 539.54, 527.53, 516.55, 506.52, 497.35, 488.97, 481.31, 474.32, 467.92, 462.07, 456.73, 451.85, 447.39, 443.31, 439.58, 436.17, 433.06, 430.21, 427.61, 425.24, 423.07, 421.08, 419.27, 417.61, 416.09, 414.71, 413.44, 412.28, 411.23])
        #endregion

        #region for the carbon tax (CTAX)              
        self.table_ctax_high=np.array([0,20,30,40,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50])
        #endregion

        self.table_cpi_high=np.array([3.18,3.8,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2])

        #endregion

    #region for the plots

    def plotBESS(self):
        '''
        Method to plot the BESS CAPEX prices until the year 2050. All source data can be found in https://atb.nrel.gov/electricity/2019/data.html
        '''
        years_axis=[]
        for counter in range(33):
            years_axis.append(f'{counter+2018}')
        
        fig, ax = plt.subplots()
        ax.scatter(np.linspace(0,32,33), self.table_bess)
        ax.plot(np.linspace(0,32,33), self.table_bess, 'r-', label="BESS capex prices")
        plt.xticks(ticks=np.linspace(0,32,33)[::4],labels=years_axis[::4])
        ax.grid(which='both',axis='y')
        plt.legend()
        plt.show()
        fig.savefig('prices_bess.pdf', bbox_inches='tight',transparent=True)

    def plotPV(self):
        '''
        Method to plot the PV CAPEX prices until the year 2050. All source data can be found in https://atb.nrel.gov/electricity/2019/data.html
        '''               
        years_axis=[]
        for counter in range(33):
            years_axis.append(f'{counter+2018}')
        fig, ax = plt.subplots()
        ax.scatter(np.linspace(0,32,33), self.table_pv)
        ax.plot(np.linspace(0,32,33), self.table_pv, 'r-', label="PV capex prices")
        plt.xticks(ticks=np.linspace(0,32,33)[::4],labels=years_axis[::4])
        ax.grid(which='both',axis='y')
        plt.legend()
        plt.show()
        fig.savefig('prices_pv.pdf', bbox_inches='tight',transparent=True)

    def plotWIND(self):
        '''
        Method to plot the WIND CAPEX prices until the year 2050. All source data can be found in https://atb.nrel.gov/electricity/2019/data.html
        '''               
        years_axis=[]
        for counter in range(33):
            years_axis.append(f'{counter+2018}')
        fig, ax = plt.subplots()
        ax.scatter(np.linspace(0,32,33), self.table_wind)
        ax.plot(np.linspace(0,32,33), self.table_wind, 'r-', label="WIND capex prices")
        plt.xticks(ticks=np.linspace(0,32,33)[::4],labels=years_axis[::4])
        ax.grid(which='both',axis='y')
        plt.legend()
        plt.show()
        fig.savefig('prices_wind.pdf', bbox_inches='tight',transparent=True)

    def plotDIESEL(self):
        '''
        Method to plot the DIESEL prices until the year 2050. All source data can be found in https://www.eia.gov/outlooks/aeo/tables_ref.php
        '''               
        years_axis=[]
        for counter in range(33):
            years_axis.append(f'{counter+2018}')
        fig, ax = plt.subplots()
        ax.scatter(np.linspace(0,32,33), self.table_diesel)
        ax.plot(np.linspace(0,32,33), self.table_diesel, 'r-', label="Diesel prices")
        plt.xticks(ticks=np.linspace(0,32,33)[::4],labels=years_axis[::4])
        ax.grid(which='both',axis='y')
        plt.legend(loc=1)
        plt.ylim(0.7,1)
        plt.show()
        fig.savefig('prices_diesel.pdf', bbox_inches='tight',transparent=True)
  
    def plotGENSET(self):
        '''
        Method to plot the diesel GENSET CAPEX prices until the year 2050. Due to the lack of data, it is assumed a linear reduction in the prices. 
        '''               
        years_axis=[]
        for counter in range(33):
            years_axis.append(f'{counter+2018}')
        fig, ax = plt.subplots()
        ax.scatter(np.linspace(0,32,33), self.table_genset)
        ax.plot(np.linspace(0,32,33), self.table_genset, 'r-', label="Diesel genset capex prices")
        plt.xticks(ticks=np.linspace(0,32,33)[::4],labels=years_axis[::4])
        ax.grid(which='both',axis='y')
        plt.legend()
        plt.show()
        fig.savefig('prices_genset.pdf', bbox_inches='tight',transparent=True)
  
    def plotCTAX(self):
        '''
        Method to plot the TAX prices until the year 2050. All source data can be found in the Greenhouse Gas Pollution Pricing Act of the Canadian Government https://www.canlii.org/en/ca/laws/stat/sc-2018-c-12-s-186/139160/sc-2018-c-12-s-186.html
        '''               
        years_axis=[]
        for counter in range(33):
            years_axis.append(f'{counter+2018}')
        fig, ax = plt.subplots()
        ax.scatter(np.linspace(0,32,33), self.table_ctax)
        ax.plot(np.linspace(0,32,33), self.table_ctax, 'r-', label="Carbon taxes")
        plt.xticks(ticks=np.linspace(0,32,33)[::4],labels=years_axis[::4])
        ax.grid(which='both',axis='y')
        plt.legend(loc=1)
        plt.ylim(0,60)
        plt.show()
        fig.savefig('prices_ctax.pdf', bbox_inches='tight',transparent=True)

    def plotCPI(self):
        '''
        Method to plot the Consumer Index Prices until the year 2050. Due to the lack of data, it is assumed a stable price for the following years.
        '''               
        years_axis=[]
        for counter in range(33):
            years_axis.append(f'{counter+2018}')
        fig, ax = plt.subplots()
        ax.scatter(np.linspace(0,32,33), self.table_cpi)
        ax.plot(np.linspace(0,32,33), self.table_cpi, 'r-', label="Consumer Index Prices")
        plt.xticks(ticks=np.linspace(0,32,33)[::4],labels=years_axis[::4])
        ax.grid(which='both',axis='y')
        plt.legend(loc=1)
        plt.show()
        fig.savefig('prices_ctax.pdf', bbox_inches='tight',transparent=True)

    
    #endregion

    #endregion

#endregion












