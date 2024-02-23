from lib_lognormal_regression import *


# PROPOSAL:  uncomment one function-call at a time, and run the modul,  in successive steps



# ============================================================================ #
# # OVERVIEW OF EPIDEMIC CURVES
# ============================================================================ #

# fn_compare_johns_hopkins('France')           # OWID versus Johns Hopkins
# fn_corrs_country_data()                         # correlation heatmap

# Fn_overview(outtype='common',G=14,normalize=False)                  # Overview of the epidemic curves
# Fn_overview(outtype='scatter gauss',G=14,countries=['Hungary'])       # One country
# Fn_overview(outtype='common_normalized',G=20,countries=['Hungary','Germany','Austria','Croatia'])  # more country, shaded
# Fn_overview(outtype='common',G=14,countries=['Lithuania','Ukraine'])   # more country shaded, not normalized

# Fn_overview_firstwave(G=35,fit_erlang=True,fit_lognormal=True,fit_gauss=False)  # Overview of the first wave in Europe
# fn_compare_country_curves('Netherlands,Greece')       # Simple comparison


# ============================================================================ #
# # OVERVIEW OF COUNTRIES
# ============================================================================ #

# fn_country_stats(lower_limit=0.05)                      # calculations, runtime 2-3 minutes
# fn_plot_country_stats(out='losses')                     # bar-plot (area)      Base losses by countries 
# fn_plot_country_stats(out='decompose_loss')             # bar_plot (area)    Loss if lognormal decomposition by countries
# fn_plot_country_stats(out='stats')                      # bar_plot (area)   Maximum peak, Relative scatter, Weekly relative scatter


# ============================================================================ #
# # LOSS ANALYSIS
# ============================================================================ #

# fn_plot_base_MAPE_MDAPE_by_daysafter()                  # Base losses, MAPE and MDAPE
# fn_plot_compare_base_and_lognorm_MAPE_by_daysafter()      # Comparison of lognormal and base predictions, MAPE and MDAPE
# fn_plot_compare_base_and_lognorm_MAPEmax_by_daysafter()     # Comparison of lognormal and base predictions, APEmax and SMAPE
# fn_plot_lognorm_scores()                                # Score of lognormal prediction relative to the base predictions
# fn_plot_lognorm_scores(phases=[2,3])                  # ...  in phase [2,3]
# fn_loss_distrbutions(outliers=False)                    # Loss distribution, without outliers
# fn_loss_distrbutions(outliers=True)                    # Loss distribution, with outliers
# fn_plot_score_by_countries()                            # Score to base model by countries (barplot)
# fn_plot_compare_base_loss('Russia,South Korea')          # illustration of base loss

# fn_MedianTAPE_comparable()                                            # calculate and print
# fn_MedianTAPE_base_comparable(countries='Europe+',johnshopkins=False)  # calculate and print
# fn_MedianTAPE_base_comparable(countries='nway',johnshopkins=False)    # calculate and print
# fn_MedianTAPE_base_comparable(countries='Europe+',johnshopkins=True)    # calculate and print
# fn_MedianTAPE_base_comparable(countries='nway',johnshopkins=True)     # calculate and print
# fn_median_total_absolute_percentage_error_by_countries()                # barplot


# ============================================================================ #
# # LOSS BY PHASE AND TIME
# ============================================================================ #

# fn_loss_plot_by_phase(APE='APEmax',afterdays=14,score=False)        # Loss by phase, lognormal, base, linear
# fn_loss_plot_by_phase(APE='APEmax',afterdays=14,score=True)         # Score by phase,  to base, to linear
# fn_loss_plot_by_time()                                              # Change of loss by time

# fn_loss_by_weekday()                                                 # calculations and printing
# fn_plot_underestimation_percent_by_daysafter()                       # lognormal, base, linear 


# ============================================================================ #
# # # PROPERTIES OF ERLANG and LOGNORMAL FUNCTIONS
# ============================================================================ #

# fn_erlang_exponent_plot()                           # Erlang functions with different powers         
# Fn_erlang_grad_plot()                                 # Erlang function and its derivatives
# fn_plot_erlang_superposition()                        # Superposition of two elementary Erlang-surges with different offsets

# fn_compare_erlang_lognormal(fix='leftwidth')          # Curve-fitting of log-normal function with Erlang (plot)

# fn_plot_lognormal(variants='modus')                   # Lognormal functions with different modus
# fn_lognorm_grad_plot()                                # Lognormal function and its derivatives
# fn_plot_lognorm_superposition()                       # Superposition of two elementary log-normal surges with different offsets



# ============================================================================ #
# # GAUSSIAN MOVING AVERAGE
# ============================================================================ #

# fn_plot_gauss_illustration()                            # Illustration of Gaussisan moving average
# fn_plot_gauss_timescales('United States')               # Illustration of timescales



# ============================================================================ #
# DECOMPOSE
# ============================================================================ #

# fn_decompose(fn_countries('Europe+'),G=[14,21,28,35],to_csv=True,leftwidth_type='free',ser_flatten=False)   # runtime: 10 minutes
# fn_decompose(fn_countries('Europe+'),G=[14,21,28,35],to_csv=True,leftwidth_type='free',corefunction='gauss')  # whth Gauss function

# fn_decompose('Hungary',G=14,plot=True,scatter=False,plot_grad2=True,to_csv=False,leftwidth_type='free',ser_flatten=False)              # calculation and plot, 20 seconds
# fn_decompose('Germany',G=[14,21,28,35],plot=True,scatter=False,plot_grad2=False,to_csv=False,leftwidth_type='free',ser_flatten=False)  # calculation and plot, with different G values, 30 seconds
# fn_decompose('Italy',G=[14,21,28,35],plot=True,scatter=False,plot_grad2=False,to_csv=False,leftwidth_type='free',ser_flatten=False,corefunction='gauss')  # calculation and plot, with Gauss core-function, 30 seconds

# fn_decompose_loss()             # print losses


# ============================================================================ #
# # RETROSPECTIVE TESTING
# ============================================================================ #

# Fn_country_test('Portugal',G=[14,21,28,35],plot=True,interval=8,flatten=True,scatter=False)  # plot all predictions for a country, testing days every 8 days
# Fn_country_test('Germany',G=14,plot=True,interval=8,flatten=True,scatter=False)  # plot all predictions for a country, specific time scale, testing days every 8 days
# Fn_country_test('Spain',G=35,plot=True,flatten=False,xmax='2021-12-23',count_back=1)  # testing one prediction
# Fn_country_tests('group:Europe+')         # Save plots into png files (by countries)

# fn_test_all('group:Europe+',G=[[14,21,28,35]],interval=8,flatten=True)    # runtime: 30 minutes  ToCSV
# fn_test_all('group:Europe+',G=[14,21,28,35],interval=8,flatten=True)      # multiG variant for G_weights train  ToCSV

# fn_test_all('group:Europe+',G=[[14,21,28,35]],interval=1,flatten=True,dayfirst=datefloat('2021-01-09')-14,daylast=datefloat('2021-01-31')-14,above=0)    # for LSTM comparison


# ============================================================================ #
# # PREDICTIONS
# ============================================================================ #

# fn_plot_predictions('Germany','2022-01-21',weeks=8,optimalday=False)     # calculate and plot, 1-2 minutes
# fn_plot_predictions('Austria','2022-01-31',weeks=6) 
# fn_plot_predictions('France','2021-12-30',weeks=8,optimalday=False,flatten=True)
# fn_plot_predictions('Hungary','2022-01-09',weeks=8,optimalday=False) 
# fn_plot_predictions('United States','2021-12-25',weeks=8)     

# fn_plot_live_prediction('Hungary',weeks=1, flatten=False)
# fn_plot_live_prediction('United Kingdom',weeks=4,flatten=True)          # live prediction
# fn_plot_live_prediction('South Korea',weeks=4,flatten=True)          # live prediction
# fn_plot_live_prediction('Italy',weeks=2)



# ============================================================================ #
# # KERAS MAXPOS MODEL
# ============================================================================ #

# fn_train_boostwidth(train_count=2,verbose=1,fname_fix='predict maxpos by firsthalf lognorm',leftwidth_type='tune',flatten=True,val=0.1,xmin_train=None,xmax_train='2021-06-30',xmin_test='2021-08-20',xmax_test=None)
# fn_train_boostwidth(train_count=10,verbose=0,fname_fix='predict maxpos by secondhalf lognorm',leftwidth_type='tune',flatten=True,val=0.1,xmin_train='2021-08-20',xmax_train='2022-12-30',xmin_test=None,xmax_test='2021-06-30')
#       - runtime: 40-50 minutes;    creates keras model and saves history plot (as png)

# fn_depend_plot_boostwidth()             # dependancy plots


# ============================================================================ #
# # TIMESCALE WEIGHTS
# ============================================================================ #

# fn_normalized_loss_by_G_and_daysafter(plot_weighted=False)              # plot,    MAPE, MDAPE, MAPEmax
# fn_normalized_loss_by_country_G_and_daysafter('Portugal,Sweden')        # plot     for tow country
# fn_normalized_loss_by_period_G_and_daysafter()                          # plot by periods
# fn_compare_weighted()                                # plot,  Comparison of the wheighted prediction with the original predictions

# fn_G_weights_new(plot=False,csv=True)                # creates G_whiegts dataset


# ============================================================================ #
# LEFTWIDTH BY COUNTRIES AND TIMESCALES
# ============================================================================ #

# fn_plot_leftwidth_bytime(country='all',G='all')         # plot,  Change of the average leftwidth during the pandemic
# fn_save_leftwidth_bytime(flatten=True)                  # creates leftwidth dataset


# ============================================================================ #
# # OPTIMAL DAYOFWEEK
# ============================================================================ #

# fn_save_optimal_dayofweek_by_country_and_time()         # creates optimal_dayofweek dataset,  runtime: 40-50 minutes
# fn_optimal_dayofweek(fn_countries('Europe+'))






