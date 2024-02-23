import numpy as np
import pandas as pd

from ezcharts import *
from ezplot import *
from ez_sklearn import *
matplot_init()

from sklearn.metrics import r2_score,mean_absolute_percentage_error,mean_absolute_error

from sklearn.preprocessing import StandardScaler            # nullára centrálás és skálázás a szórással ([-1,1]-be kerül a többség)
from sklearn.preprocessing import MinMaxScaler              # [0-1] tartományba transzformálja
from sklearn.preprocessing import MaxAbsScaler              # csak skálázás
from sklearn.preprocessing import RobustScaler              # a pontok fele kerüljön a [-1,1] tartományba

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

from tensorflow.keras.models import Sequential,load_model,save_model
from tensorflow.keras.layers import Dense


from scipy.optimize import curve_fit

from pandasql import sqldf

import json
import warnings     
    # ne jelenjen meg figyelmeztetés: warnings.filterwarnings("ignore")
    # kezelje hibaként:    warnings.filterwarnings("error", category=RuntimeWarning)




# warnings.filterwarnings("error", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)  #,message="*power*")
warnings.filterwarnings("ignore",message="FixedFormatter*")
# - "invalid value encountered in power" elnémítása
# - UserWarning: "FixedFormatter should only be used together with FixedLocator"


# ============================================================================ #
# # GLOBAL PARAMETERS  (can be modified)
# ============================================================================ #

# IMPORTANT
data_path = r".\\data"                # data in the "data" subdirectory under the program directory
# data_path = dir_downloads()         # data in the personal downloads dir


# Developer parameters
bVersion=True           # for version handling
version_params={}
bDebug=False            # for debug  (conditional breakpoints)


def Datapath(filename,ext):
    if ext: filename=filename + '.' + ext
    return os.path.join(data_path,filename)



# ============================================================================ #
# # READING COVID DATA
# ============================================================================ #

def Fn_read_coviddata(closed='2023'):
    if closed=='2023':  # 2023.09
        tbl=pd.read_csv(Datapath('owid-covid-data 2023-09','csv'))  
    elif closed=='2024': # 2024.01
        tbl=pd.read_csv(Datapath('owid-covid-data 2024-01','csv'))  
        # - it has less data retroactively. They switched to one data per week
        # - it is no longer guaranteed for every country to have a record every day
    tbl['date'] = pd.to_datetime(tbl['date'],format='%Y-%m-%d')
    tbl.set_index('date',inplace=True,drop=True)
    return tbl

def fn_read_johns_hopkins():
    tbl = Read_csv(Datapath('JOHNS HOPKINS time_series_covid19_confirmed_global','csv'),format='en')
    return tbl


def fn_read_and_compare():
    tbl=pd.read_csv(Datapath('owid-covid-data 2023-09','csv'))   
    tbl2=pd.read_csv(Datapath('owid-covid-data 2024-01','csv'))  

    tblinfo(tbl,toexcel=True)
    tblinfo(tbl2,toexcel=True)

    tblinfo(tbl,'minmax',toexcel=True)
    tblinfo(tbl2,'minmax',toexcel=True)
# fn_read_and_compare()
# exit()

def Fn_coviddata_country(tbl,country,col='new_cases_per_million',backfill=True,flatten=False):
    '''
    flatten: averaging every 7 days, then entering the average value for all seven days
         - elimination of systematic swings between the days of the week
         - in principle, even in the case of flattening, a daily updated forecast can be given, if the closing day of the flattening is continuous
             changes
    '''
    
    ser=serfromtbl(tbl,col,group='location:"' + country + '"',orderby='index')

    if flatten:
        ser = ser.resample('D').mean()      # hányzó dátumok hozzáadása

        ser=ser.sort_index()
        ser.fillna(value=0,inplace=True)        # nem lehetnek benne üres értékek

        # The only-0 part must be cut from the right
        ser_notnull = ser[ser>0] 
        lastday_ok = ser_notnull.index.max()
        ser=ser.loc[(ser.index<=lastday_ok)]

        # distribution of subsequently reported values to the missing-data days before it
        ser=fillna_back(ser)

        # Weekly entry of the weekly average for each day of the week
        y=ser.values
        days=array(ser.index)

        # The fractional week at the end must be omitted
        nWeeks = len(y)//7
        y = y[:nWeeks*7]
        days = days[:nWeeks*7]

        for i_week in range(nWeeks):
            mean=y[i_week*7:(i_week+1)*7].mean()
            y[i_week*7:(i_week+1)*7] = mean

        ser = pd.Series(y,days)

    else:
        ser=ser.sort_index()
        ser.fillna(value=0,inplace=True)        # nem lehetnek benne üres értékek

        # The only-0 part must be cut from the left and right
        ser_notnull = ser[ser>0] 
        lastday_ok = ser_notnull.index.max()
        firstday_ok = ser_notnull.index.min()

        # ser=ser.loc[(ser.index>=firstday_ok) & (ser.index<=lastday_ok)]
        ser=ser.loc[(ser.index<=lastday_ok)]

        # distribution of subsequently reported values to the missing-data days before it
        if backfill: ser=fillna_back(ser)

    return ser    

def fn_coviddata_country_js(tbl,country,backfill=True):
    if country == 'South Korea': country='Korea, South'
    elif country == 'United States': country='US'
    elif country == 'Palestine': country = 'West Bank and Gaza'

    tblL = tbl.loc[tbl['Country/Region']==country]

    # For some countries, there are only provincial data (for other countries, there are also provinces, but there is also a summary line)
    if country in ['Canada','Australia','China']:
        ser = tblL.groupby(by='Country/Region').sum().iloc[0]
        ser = ser.iloc[2:]  # omitting the lat, long fields at the beginning
    # Some countries have provincial and country level records
    elif country in ['Denmark','France','Netherlands','New Zealand','United Kingdom']:
        ser = tblL.loc[tblL['Province/State'].isna()].iloc[0]
        ser = ser.iloc[4:]  # omit the two text fields at the beginning and the lat,long fields
    else:
        ser = tblL.iloc[0]
        ser = ser.iloc[4:]  # omit the two text fields at the beginning and the lat,long fields

    ser = ser.diff()    # switch to differences from cumulative data
    ser.index=pd.to_datetime(ser.index,format='%m/%d/%y')   # string to date

    ser=ser.sort_index()
    ser.fillna(value=0,inplace=True)        # it cannot contain empty values

    # Balról és jobbról levágandó a csak-0 rész
    ser_notnull = ser[ser>0] 
    lastday_ok = ser_notnull.index.max()
    firstday_ok = ser_notnull.index.min()
    ser=ser.loc[(ser.index>=firstday_ok) & (ser.index<=lastday_ok)]

    # distribution of subsequently reported values to the missing-data days before it
    if backfill: ser=fillna_back(ser,nullvalue_max=1)   # elszórva 1 értékek is előfordulnak 0 helyett

    return ser


def fn_compare_johns_hopkins(country):
    tbl=Fn_read_coviddata()
    ser=Fn_coviddata_country(tbl,country,'new_cases')

    tbl_js=fn_read_johns_hopkins()
    ser_js=fn_coviddata_country_js(tbl_js,country)

    pltinit(suptitle='Comparison of OWID and Johns Hopkins COVID time series',
            title=country,top=0.85)
    FvPlot(ser,'original',label='owid',annot='max last')
    FvPlot(ser_js,'original',label='johns_hopkins',annot='max last',alpha=0.5)
    pltshow()
# fn_compare_johns_hopkins('France')
# exit()




# ============================================================================ #
# # OVERVIEW OF COUNTRIES AND EPIDEMIC CURVES
# ============================================================================ #

def fn_countries(countrygroup='Europe'):
    '''
    If there is no such group name, it returns the input
    '''

        # countries=serfromtbl(tbl,'population',indexcol='location',
        #                         query='population>1000000 and continent=="Europe"',
        #                         orderby='population desc',aggfunc='mean').index

    countries_europe=["Portugal","Spain","Ireland","United Kingdom","Netherlands","Belgium","Denmark",
        "Norway","Sweden","Finland","Germany","France","Switzerland","Italy","Austria","Slovenia",
        "Czechia","Slovakia","Poland","Croatia","Hungary","Romania","Moldova","Bosnia and Herzegovina",
        "Serbia","North Macedonia","Albania","Greece","Bulgaria","Lithuania","Latvia","Estonia",
        "Ukraine","Russia"]
        # "Kosovo","Belarus"]
    countries_other=['United States','Canada',
        'South Africa','Qatar','Japan','South Korea','Australia']

    if countrygroup=='Europe':
        return countries_europe
    elif countrygroup=='Europe+':
        return countries_europe + countries_other

    elif countrygroup=='firstwave':
        countries=fn_countries('Europe')
        anomalies = ('Poland,United Kingdom,Sweden,France,Belarus,' +
            'North Macedonia,Russia,Ukraine,Romania,' +
            'Bosnia and Herzegovina,Albania,Kosovo,Bulgaria,Moldova')   # nem volt valódi maxhelyük   
        countries=[country for country in countries if country not in anomalies]
        return countries

    elif countrygroup=='nway':
        # tbl = Fn_read_coviddata()
        # tbl=tbl.loc[tbl.continent.notnull()]
        # tbl=tbl.loc[(tbl.index>='2021-01-01') & (tbl.index<='2021-01-31')]
        # ser_countries = serfromtbl(tbl,'new_cases_per_million',
        #                         indexcol='location',
        #                         query='population>1000000',
        #                         orderby='values desc',
        #                         aggfunc='sum')
        # print(','.join(list(ser_countries.index)[:84]))

        countries = (
            'Portugal,Czechia,Slovenia,Israel,Lebanon,Ireland,Spain,United Kingdom,United States,Panama,Latvia,' +
            'Lithuania,Estonia,Slovakia,Sweden,Netherlands,United Arab Emirates,France,Colombia,Serbia,Georgia,' +
            'Switzerland,Italy,Brazil,Argentina,Bahrain,South Africa,Albania,Germany,Uruguay,Denmark,Chile,' +
            'Austria,Croatia,Tunisia,Belarus,Poland,Eswatini,Canada,Belgium,North Macedonia,Costa Rica,Romania,' +
            'Kosovo,Ukraine,Moldova,Russia,Palestine,Bolivia,Hungary,Namibia,Dominican Republic,' +
            'Paraguay,Mexico,Peru,Bosnia and Herzegovina,Kuwait,Turkey,Bulgaria,Malaysia,Jordan,Armenia,Qatar,' +
            'Botswana,Lesotho,Libya,Norway,Honduras,Iran,Ecuador,Greece,Finland,Kazakhstan,Zambia,Azerbaijan,' +
            'Japan,El Salvador,Cuba,Guatemala,Indonesia,Zimbabwe,Oman,Jamaica'
        )
        # Puerto Rico
        return countries.split(',')

    return countrygroup



def Fn_overview(col='new_cases_per_million',outtype='common',G=14,countries=None,       # járványgörbék áttekintése
                xmin=None,xmax=None,normalize=None):       
    '''
    col:  'new_cases_per_million',  'new_deaths_per_million'
    outtype:  'subs', 'common_normalized', 'common'
    G:
    countries:   if None, "Europe+"
    '''

    tbl=Fn_read_coviddata()

    if outtype=='scatter gauss':
        pltinit('Overview of epidemic curves')
        for country in countries:
            ser=Fn_coviddata_country(tbl,'"' + country + '"')
            FvPlot(ser,plttype='scatter gauss',G=14,label=country)
        pltshow()

        return


    if not countries: 
        countries=fn_countries('Europe+')
        # countries=list(countries) + ['Europe']

        # countries=serfromtbl(tbl,'population',indexcol='location',
        #                         query='population>1000000 and continent=="Europe"',
        #                         orderby='population desc',aggfunc='mean').index
        # countries=["Portugal","Spain","Ireland","United Kingdom","Netherlands","Belgium","Denmark",
        # "Norway","Sweden","Finland","Germany","France","Switzerland","Italy","Austria","Slovenia",
        # "Czechia","Slovakia","Poland","Croatia","Hungary","Romania","Moldova","Bosnia and Herzegovina",
        # "Serbia","Kosovo","North Macedonia","Albania","Greece","Bulgaria","Lithuania","Latvia","Estonia",
        # "Belarus","Ukraine","Russia"]

        # countries=list(countries) + ['united states','Canada','Japan','South Korea','Israel','Turkey','Iran']

        # countries=[country for country in countries if country not in ['England','Scotland','Wales','Northern Ireland']]    # hiányzó adatok


    points=False
    serBaseline=None

    tbl_mean=pd.DataFrame()
    for country in countries:
        ser=Fn_coviddata_country(tbl,'"' + country + '"')

        ser=FvGaussAvg(ser,G)
        if xmax: ser=ser.loc[:xmax]
        if xmin: ser=ser.loc[xmin:]

        if normalize: ser=normalizeSer(ser)

        # Átlagoláshoz és plot-hoz
        tbl_mean[country]=ser

    ser_mean=tbl_mean.mean(axis=1)
    # ser_mean=normalizeSer(ser_mean)
    
    # plot
    pltinit('Overview of the epidemic curves',
            title='The epidemic curves of the 41 examined countries, 2020 February - 2023 September',
            height=0.5,left=0.07,right=0.96,top=0.83,bottom=0.07)
    for country in countries:
        FvPlot(tbl_mean[country],plttype='original',colors=False,
            label=country,annot='localmax',annotcaption='label',
            area=True,baseline=ser_mean.values)
    # FvPlot(ser_mean,plttype='original',colors=ddef(color='blue'))
    
    pltshow(annot_fontsize=7,annot_baseline=ser_mean,
            xtickerstep='year',ylabel='Daily COVID cases per million inhabitants') #(xmin=-60,xmax=60)
# Fn_overview('new_cases_per_million','common',G=14,normalize=False)
# Fn_overview('new_cases_per_million','scatter gauss',G=14,countries=['Hungary'])
# Fn_overview(outtype='common_normalized',G=20,countries=['Hungary','Germany','Austria','Croatia'])
# Fn_overview(outtype='common',G=14,countries=['Lithuania','Ukraine'])
# exit()

def Fn_overview_firstwave(G=14,fit_erlang=False,fit_lognormal=False,fit_gauss=False):
    tbl=Fn_read_coviddata()
    
    pltinit('Overview of the first wave in Europe',
            title='Normalized and peak-adjusted epidemic curves of "one-peak" countries in Europe, 2020 February - June',
            left=0.067,right=0.96,height=0.5,top=0.84)
    # The maximum location of the first wave is required per country (it will appear with offsets)
    countries=fn_countries('firstwave')

    tbl_mean=pd.DataFrame()
    for country in countries:
        ser=Fn_coviddata_country(tbl,'"' + country + '"')
        ser=ser.loc['2020-02-01':'2020-06-30']
        ser=datefloat(ser)
        # Már itt kell egy gauss, mert a ser-ben ismétlődő értékek lethetnek (backfill volt az nan értékekre)
        ser=FvGaussAvg(ser,gausswidth=G)      # két hetes simítás
        maxpos=ser.idxmax()
        maxpos=fn_maxpos_precise(ser,maxpos)
        ser.index = ser.index - maxpos
        # resample - mindegyik ser ugyanazokban a pontokban
        ser=FvLinearResample(ser,X_out=Range(-60,70,add=0.2))
        #  - az x_tartományon túlnyúló pontok np.nan értéket kapnak
        ser=normalizeSer(ser)

        # Átlagoláshoz és plot-hoz
        tbl_mean[country]=ser

    ser_mean=tbl_mean.mean(axis=1)
    # ser_mean=normalizeSer(ser_mean)
    
    # plot
    annot='baseline10'
    alpha=None
    area=True
    if fit_erlang or fit_lognormal: 
        annot=''    # túlzottan sok lenne a felirat
        area='noline'
        alpha=0.5
    for country in countries:
        FvPlot(tbl_mean[country],plttype='original',colors=False,
               label=country,annot=annot,annotcaption='label',
               area=area,baseline=ser_mean,alpha=alpha)
    # FvPlot(ser_mean,plttype='original',colors=ddef(color='blue'))

    # Check: exponential growth at ramp start
    factors=[]
    last=servalue(ser_mean,-40,assume_sorted=True)
    for day in Range(-38,-10,add=2):
        value=servalue(ser_mean,day,assume_sorted=True)
        factors.append(strnum(value/last,'4g'))
        last=value
    print('Mean-curve daily factors: ' + ', '.join(factors))            



    ser_fit=ser_mean.loc[ser_mean.index<18]
    if fit_erlang:
        X=list(ser_fit.index)
        Y=ser_fit.values
        params0 = [9,0,1,30]        # kitevo,maxpos,maxvalue,leftwidth
        lower = [2,-1,0,10] 
        upper = [150,1,2,100]
        scale = [10,10,1,10] 

        params = fn_erlang_fit(X,Y,params0,lower_bounds=lower,upper_bounds=upper,x_scale=scale,kitevo='fit_all')
        X_out=X   #list(ser_mean.index)
        ser = pd.Series(fn_erlang_multi(X_out,*params,kitevo='fit_all'),X_out)
        FvPlot(ser,'original',label='Erlang fitted',color='blue',annot='-30 18')

        # Loss értékek
        mape_max=(abs(ser_fit-ser)).mean()      # csúcsérték=1
        mape = (abs(ser_fit-ser)/ser).mean()      
        print('MAPEmax=' + strnum(mape_max,'2%') + '   MAPE=' + strnum(mape,'2%'))

        # Check: exponential growth at ramp start
        factors=[]
        last=servalue(ser,-40,assume_sorted=True)
        for day in Range(-38,-10,add=2):
            value=servalue(ser,day,assume_sorted=True)
            factors.append(strnum(value/last,'4g'))
            last=value
        print('Erlang daily factors: ' + ', '.join(factors))            


    if fit_lognormal:
        X=list(ser_fit.index)
        Y=ser_fit.values
        params0 = [0,1,30,0.9]        # maxpos,maxvalue,leftwidth,modus
        lower = [-1,0,10,0.6] 
        upper = [1,2,100,0.99]
        scale = [10,1,10,1] 

        lognormalparams = fn_lognormal_fit(X,Y,params0,lower=lower,upper=upper,scale=scale)
        X_out=X    #list(ser_mean.index)
        ser = pd.Series(fn_lognormal_multi(X_out,*lognormalparams),X_out)
        FvPlot(ser,'original',label='Lognormal_fitted',color='green',annot='-30 18')

        # Loss értékek
        mape_max=(abs(ser_fit-ser)).mean()      # csúcsérték=1
        mape = (abs(ser_fit-ser)/ser).mean()      
        print('MAPEmax=' + strnum(mape_max,'2%') + '   MAPE=' + strnum(mape,'2%'))

        # Check: exponential growth at ramp start
        factors=[]
        last=servalue(ser,-40,assume_sorted=True)
        for day in Range(-38,-10,add=2):
            value=servalue(ser,day,assume_sorted=True)
            factors.append(strnum(value/last,'4g'))
            last=value
        print('Lognormal daily factors: ' + ', '.join(factors))            

    if fit_gauss:
        X=list(ser_fit.index)
        Y=ser_fit.values
        params0 = [0,1,30]        # maxpos,maxvalue,G
        lower = [-1,0,10] 
        upper = [1,2,100]
        scale = [10,1,10] 

        params = fn_gauss_fit(X,Y,params0,lower=lower,upper=upper,scale=scale)
        X_out=X    #list(ser_mean.index)
        ser = pd.Series(fn_gauss_multi(X_out,*params),X_out)
        FvPlot(ser,'original',label='Gauss_fitted',color='orange',annot='-30 18',alpha=0.6)

        # Loss értékek
        mape_max=(abs(ser_fit-ser)).mean()      # csúcsérték=1
        mape = (abs(ser_fit-ser)/ser).mean()      
        print('MAPEmax=' + strnum(mape_max,'2%') + '   MAPE=' + strnum(mape,'2%'))

        # Check: exponential growth at ramp start
        factors=[]
        last=servalue(ser,-40,assume_sorted=True)
        for day in Range(-38,-10,add=2):
            value=servalue(ser,day,assume_sorted=True)
            factors.append(strnum(value/last,'4g'))
            last=value
        print('Gauss daily factors: ' + ', '.join(factors))            

    
    pltshow(ylabel='COVID cases (normalized by country peak)',
            xlabel='Distance from peak (days)',
            commenttopright=strint(G) + '-day width centered Gauss moving average' + '\n' + 'on the country-level curves',
            annot_fontsize=8,annot_baseline=ser_mean, 
            annot_count='localmin:12//localmax:12//other:10')        #(xmin=-60,xmax=60)
# Fn_overview_firstwave(G=30,fit_erlang=True,fit_lognormal=True,fit_gauss=True)
# exit()

def fn_compare_country_curves(countries):
    tbl=Fn_read_coviddata()

    if type(countries)==str: countries=countries.split(',')

    pltinit('Comparison of epidemic curves',ncols=1,nrows=len(countries),height=0.8)
    for country in countries:
        ser = Fn_coviddata_country(tbl,country)
        pltinitsub('next',title=country)
        FvPlot(ser,'scatter',label='original',annot='upper1',color='blue') #,alpha=0.7)
        FvPlot(ser,'gauss',G=14,label='Gauss14',annot='max last')
        pltshowsub(xtickerstep='year')
    pltshow()
# fn_compare_country_curves('Netherlands,Greece')
# exit()


def fn_ser_cases_mean():        # can be used as background plot
    tbl=Fn_read_coviddata()
    tbl_mean=pd.DataFrame()
    countries=fn_countries('Europe+')
    for country in countries:
        ser=Fn_coviddata_country(tbl,'"' + country + '"')
        ser=FvGaussAvg(ser,14)
        # Átlagoláshoz és plot-hoz
        tbl_mean[country]=ser
    ser_mean=tbl_mean.mean(axis=1)
    return ser_mean

def fn_country_peaks():         # obsolote
    tbl=Fn_read_coviddata()

    countries = fn_countries('Europe+')
    recs=[]
    for country in countries:
        ser=Fn_coviddata_country(tbl,country)
        serGauss=FvGaussAvg(ser,gausswidth=14)
        peak=serGauss.max()
        recs.append([country,peak])
    tbl_out=pd.DataFrame(recs,columns='country,peak'.split(','))
    Tocsv(tbl_out,'country_peaks',format='hu')
# fn_country_peaks()
# exit()

def fn_base_loss(ser,lower_limit=0.05,daysafter_max=50,plot=False):     # calculation of BASE and LINEAR loss-values
    '''
    Base and linear prediction for all days for which number of cases>lower_limit
    The benchmark is the actual curve averaged on the G14 time scale.

    lower_limit:   compared to the peak value of the epidemic curve.
         If the daily number of cases is below this, no prediction is needed
    daysafter_max: duration of each prediction
    plot: predictions can be displayed for testing purposes (with 14-day intervals)
    '''

    days=datefloat(ser.index.values)

    ser14 = FvGaussAvg(ser,gausswidth=14)
    ser_max14 = ser14.max()

    if plot:
        pltinit()
        FvPlot(ser,'scatter',annot='')
        FvPlot(ser14,'original',annot='',colors=ddef(color='gray'))

    APE_peak_base,APE_peak_linear,APE_peak_base2,APE_base,APE_linear,APE_base2 = [],[],[],[],[],[]
    for i_day,day in enumerate(days[14:-daysafter_max]):
        # - az elején legalább 14 előzmény-nap kell a lineáris predikcióhoz
        # - a végén daysafter_max nap szükséges

        day_last_known=day-1  # a day már ne legyen benne (tegnapig ismertek az adatokat)

        if lower_limit:
            if ser[day_last_known] < ser_max14*lower_limit: continue

        # A tesztnapon ismert görbe
        ser_known = ser.loc[:day_last_known]   

        X_pred=Range(day,count=daysafter_max)   # predikciós napok
        Y_true=ser14.loc[X_pred].values         # a G14-es időskála a viszonyítási alap

        # heti átlagértékes becslés hibája (BASE)
        mean_7days = ser_known.iloc[-7:].mean()  # az utolsó 7 nap átlaga
        Y_pred_base = [mean_7days  for x in X_pred]     # az átlagértéke viszi tovább a predikciós napokra
        Y_diff_base=abs(Y_pred_base - Y_true)   
        APE_peak_base.append(Y_diff_base/ser_max14)  # két dimenziós tömb
        APE_base.append(Y_diff_base/Y_true)      # két dimenziós tömb    APE[i_testday][i_dayafter]


        # átlagértékes becslés hibája (BASE2)    
        # mean = ser_known.mean()  # átlag a teljes futamidőre
        # Y_pred_base2 = [mean  for x in X_pred]     # az átlagértéket viszi tovább a predikciós napokra
        # Y_diff_base2=abs(Y_pred_base2 - Y_true)   
        # APE_peak_base2.append(Y_diff_base2/ser_max14)  # két dimenziós tömb
        # APE_base2.append(Y_diff_base2/Y_true)      # két dimenziós tömb    APE[i_testday][i_dayafter]


        # lineáris becslés hibája
        mean_7days_2 = ser_known.iloc[-7:].mean()       # utolsó 7 nap átlaga
        mean_7days_1 = ser_known.iloc[-14:-7].mean()    # előző 7 nap átlaga
        grad1=(mean_7days_2 - mean_7days_1) / 7
        Y_pred_linear=array([mean_7days_2 + grad1 * (4 + (x-day))  for x in X_pred])   # in range(int(G/2)+1,int(G/2)+1+nLossDays)]
            # - az utolsó hét közepétől indul az illesztés
        Y_pred_linear[Y_pred_linear<0] = 0
            # - ha 0 alá menne az eredmény, akkor legyen 0 (erre szolgál a max) 
        Y_diff_linear=abs(Y_pred_linear-Y_true)
        APE_peak_linear.append(Y_diff_linear/ser_max14)  # két dimenziós tömb
        APE_linear.append(Y_diff_linear/Y_true)   # két dimenziós tömb    APE[i_testday][i_dayafter]

        # Kis mértékben eltérő számolás a lineráris predikcióra
        # day_last_knownG=day_last_known-7    # az utolsó biztos adat a G14-es időskálán
        # grad1=ser14.loc[day_last_knownG] - ser14.loc[day_last_knownG-1]
        # Y_pred_linear2=[max(ser14.loc[day_last_knownG] + grad1 * (x-day), 0)  for x in X_pred]   # in range(int(G/2)+1,int(G/2)+1+nLossDays)]

        # Plot minden 14. napra
        if plot and day%14==0:
            FvPlot(pd.Series(Y_pred_linear,X_pred),'original',annot='',colors=ddef(color='blue'))
            # FvPlot(pd.Series(Y_pred_linear2,X_pred),'original',annot='',colors=ddef(color='red'))
    if plot: pltshow()

    APE_base = array(APE_base)                  # APE_base[i_test][i_dayafter]
    APE_peak_base = array(APE_peak_base)
    APE_base2 = array(APE_base2)                  # APE_base2[i_test][i_dayafter]
    APE_peak_base2 = array(APE_peak_base2)
    APE_linear = array(APE_linear)              # APE_linear[i_test][i_dayafter]
    APE_peak_linear = array(APE_peak_linear)

    return APE_base, APE_linear, APE_base2, APE_peak_base, APE_peak_linear, APE_peak_base2


def fn_plot_compare_base_loss(countries,out='base'):       # NEM KELL   több ország base predikciójának összehasonlítása (járványgörbék)
    '''
    out:  'base'    vagy     'linear'
    '''
    
    tbl=Fn_read_coviddata()

    lower_limit=0.05

    if type(countries)==str: countries=countries.split(',')

    pltinit(suptitle='Illustration of base loss',
            nrows=len(countries),ncols=1,sharey=False,sharex=False)

    for country in countries:
        pltinitsub('next',title=country)

        ser = Fn_coviddata_country(tbl,country,flatten=False)

        ser=datefloat(ser)

        days=datefloat(ser.index.values)

        ser14 = FvGaussAvg(ser,gausswidth=14)
        ser_max14 = ser14.max()

        FvPlot(ser,'scatter',annot='')
        FvPlot(ser14,'original',annot='',colors=ddef(color='gray'))

        for day in days[14:-14:14]:
            # - az elején és a végén is 14 nap elhagyandó

            day_last_known=day-1  # a day már ne legyen benne (tegnapig ismertek az adatokat)

            if lower_limit:
                if ser[day_last_known] < ser_max14*lower_limit: continue

            # A tesztnapon ismert görbe
            ser_known = ser.loc[:day_last_known]   

            X_pred=Range(day,count=14)   # predikciós napok
            Y_true=ser14.loc[X_pred].values         # a G14-es időskála a viszonyítási alap

            # heti átlagértékes becslés hibája (BASE)
            if out=='base':
                mean_7days = ser_known.iloc[-7:].mean()  # az utolsó 7 nap átlaga
                Y_pred = [mean_7days  for x in X_pred]     # az átlagértéket viszi tovább a predikciós napokra
            elif out=='linear':
                mean_7days_2 = ser_known.iloc[-7:].mean()       # utolsó 7 nap átlaga
                mean_7days_1 = ser_known.iloc[-14:-7].mean()    # előző 7 nap átlaga
                grad1=(mean_7days_2 - mean_7days_1) / 7
                Y_pred=array([mean_7days_2 + grad1 * (4 + (x-day))  for x in X_pred])   # in range(int(G/2)+1,int(G/2)+1+nLossDays)]
                    # - az utolsó hét közepétől indul az illesztés
                Y_pred[Y_pred<0] = 0
                    # - ha 0 alá menne az eredmény, akkor legyen 0 (erre szolgál a max) 

            FvPlot(pd.Series(Y_pred,X_pred),'original',annot='',colors=ddef(color='blue'),area=True,
                   baseline=Y_true)
        pltshowsub()
    pltshow()
# fn_plot_compare_base_loss('Russia,South Korea')
# exit()

def fn_country_stats(tocsv='countries,baseloss',lower_limit=0.05):    # országok járványgörbéinek statisztikai paraméterei (to_csv)
    '''
    out:
         firstday           The first data reporting day
         lastday            Last data reporting day
         curve_length       Length of date range (last_day - first_day +1)
         days_count         Number of records
         nan_proportion     Proportion of vomiting values
         null_proportion    Proportion of 0 values
         peak               peak value
         peak14             peak value, on a time scale of 14
         median             median (for full date range)
         mean               average (for the entire date range)
         peakedness         peak14 / median
         surge_proportion   peak14 Number of days with more than 5% of cases
         maxpos_count_G14   number of local max positions on G35 time scale
         maxpos_count_G35   number of local max positions on G35 time scale
         base_mape_14       base prediction median MAPE value for day 14
         linear_mape_14     linear prediction median MAPE value for day 14
         base_mape_peak_1_14        average of base prediction mape_max values (dayafter: 1-14)
         linear_mape_peak_1_14      average of linear prediction mape_max (dayafter: 1-14)
         scatter_rel_G14            relative scatter to the G14 curve (weekly relative spread)
         variance
         std_rel                    standard deviation / mean

         decompose_loss             value of decomposition loss (for all time scales; APE / country_peak)
         leftwidth_mean             the average leftwdith for the surges (for all G)
         leftwidth_opt              optimized leftwdith value (for all G)
         lognorm_count_number       count of elementary surges, obtained by G14 decomposition

         prediction mdape14         lognorm prediction Median APE value on day 14
         prediction mape14          lognorm prediction MAPE value on day 14
         prediction score           lognorm prediction score value

        
     tocsv: enumeration with comma     e.g. "countries,baseloss"
    '''
    
    print('Country stats')

    tbl=Fn_read_coviddata()

    # Dekompozíció adataihoz
    tbl_decomp_countries = Read_csv(Datapath('decompose_countries lognorm G_all leftw_free','csv'))
    tbl_decomp_tune = Read_csv(Datapath('decompose_countries lognorm G_all leftw_tune flatten','csv'))
    tbl_decomp_lognorms = Read_csv(Datapath('decompose_lognorms G_all leftw_free','csv'))

    # Prediction loss-hoz


    countries = fn_countries('Europe+')
    # countries = ['Hungary']
    recs=[]
    APE_base_all, APEmax_base_all, APE_base2_all, APEmax_base2_all, APE_linear_all, APEmax_linear_all = None,None,None,None,None,None
    for country in countries:
        progress(country)

        decompose_loss = Query_tbl(tbl_decomp_countries,country=country)['loss'].mean()  # minden G-re
        leftwidth_mean = Query_tbl(tbl_decomp_lognorms,country=country)['leftwidth'].mean()  # minden G-re
        leftwidth_opt = Query_tbl(tbl_decomp_tune,country=country)['leftwidth'].mean()  # minden G-re
        lognorm_count_G14 = Query_tbl(tbl_decomp_countries,country=country,G=14)['count_lognorm'].mean()  # elvileg egyetlen rekord

        # pred_mdape14 =
        # pred_mape14 =
        # pred_score = 


        ser=serfromtbl(tbl,'new_cases_per_million',group='location:' + country,orderby='index')

        ser=datefloat(ser)
        ser=ser.sort_index()

        # Balról és jobbról levágandó a csak-0 rész
        ser_notnull = ser[ser>0] 
        lastday = ser_notnull.index.max()
        firstday = ser_notnull.index.min()
        curve_length = lastday - firstday + 1

        ser=ser.loc[(ser.index>=firstday) & (ser.index<=lastday)]
        ser=ser.loc[(ser.index<=lastday)]
        days_count = len(ser)

        null_count = Count_nan(ser) + Count(ser==0)     # nan értékek csak elvétve fordulnak elő 
        null_proportion = null_count / len(ser)   

        ser.fillna(value=0,inplace=True)        
        ser=fillna_back(ser)    # kumulatív adatközlések esetén a közölt értéket szétosztja az előtte lévő üres vagy nullás cellákra

        ser14 = FvGaussAvg(ser,gausswidth=14)
        ser35 = FvGaussAvg(ser,gausswidth=35)

        # A hosszfüggés kiküszöbölésére lecsökkentem az időtartományt a legnagyobb közös tartományra
        dayfirst_global = datefloat('2020-03-21')
        daylast_global = datefloat('2023-03-26')
        ser=ser.loc[(ser.index>=dayfirst_global) & (ser.index<=daylast_global)]
        ser14=ser14.loc[(ser14.index>=dayfirst_global) & (ser14.index<=daylast_global)]
        ser35=ser35.loc[(ser35.index>=dayfirst_global) & (ser35.index<=daylast_global)]

        peak=ser.max()
        median=ser.median()
        mean=ser.mean()
        std_rel = ser.std() / mean

        peak14 = ser14.max()
        scatter_rel_G14 = ((ser - ser14)/ser14).std()       # RSD jellegű adat, G14-es időskálán

        peakedness = peak14/median
        active_proportion = len(ser14.loc[ser14>peak14*0.05]) / len(ser14)

        # lognormparams14=fn_regressions_back(country,G=14,kitevo=9,bFixLeftwidth=False,verbose=0,bPlot=False,tbl_covid=tbl,ser=ser)
        # surge_count_G14 = lognormparams14//4
        
        # lognormparams35=fn_regressions_back(country,G=35,kitevo=9,bFixLeftwidth=False,verbose=0,bPlot=False,tbl_covid=None,ser=ser)
        # surge_count_G35 = lognormparams35//4

        peak_count_G14 = len(serlocalmax(ser14,endpoints=False))
        peak_count_G35 = len(serlocalmax(ser35,endpoints=False))

        
        APE_base, APE_linear, APE_base2, APEmax_base, APEmax_linear, APEmax_base2 = \
            fn_base_loss(ser,lower_limit=lower_limit,daysafter_max=50,plot=False)
        
        base_mdape14 = np.median(APE_base[:,13])            # utolsó heti átlaggal való becslés
        linear_mdape14 = np.median(APE_linear[:,13])        # utolsó két hét trendjének lineáris folytatása
        # base2_mdape14 = np.median(APE_base2[:,13])          # teljes átlaggal való becslés
        base_mape_peak_1_14 = APEmax_base[:,:14].mean()
        linear_mape_peak_1_14 = APEmax_linear[:,:14].mean()
        # base2_mape_peak_1_14 = APEmax_base2[:,:14].mean()

        APE_base_all = Concat(APE_base_all,APE_base)            # APE[i_test][i_dayafter]
        # APE_base2_all = Concat(APE_base2_all,APE_base2)            # APE[i_test][i_dayafter]
        APE_linear_all = Concat(APE_linear_all,APE_linear)

        APEmax_base_all = Concat(APEmax_base_all,APEmax_base)            # APE[i_test][i_dayafter]
        # APEmax_base2_all = Concat(APEmax_base2_all,APEmax_base2)            # APE[i_test][i_dayafter]
        APEmax_linear_all = Concat(APEmax_linear_all,APEmax_linear)


        recs.append([country,floatdate(firstday),floatdate(lastday),curve_length,days_count,null_count,null_proportion,
                     peak,peak14,median,mean,std_rel,peakedness,scatter_rel_G14,active_proportion,
                     peak_count_G14,peak_count_G35,lognorm_count_G14,
                     base_mdape14,0,linear_mdape14,
                     base_mape_peak_1_14,0,linear_mape_peak_1_14,
                     leftwidth_mean,leftwidth_opt,decompose_loss])
    
        columns=('country,firstday,lastday,curve_length,days_count,null_count,null_proportion,' +
                    'peak,peak14,median,mean,std_rel,peakedness,scatter_rel_G14,active_proportion,' +
                    'maxpos_count_G14,maxpos_count_G35,lognorm_count_G14,' +
                    'base_mdape_14,base2_mdape_14,linear_mdape_14,' +
                    'base_mape_peak_1_14,base2_mape_peak_1_14,linear_mape_peak_1_14,' +
                    'leftwidth_mean,leftwidth_opt,decompose_loss')


    print(
        'Count of predictions=' + strint(len(APE_base_all)) + '\n' +
        'base_mdape14=' + strnum(np.median(APE_base_all[:,13]),'%') + '\n'
        'linear_mdape14=' + strnum(np.median(APE_linear_all[:,13]),'%') + '\n'
        'base_mape14=' + strnum(APE_base_all[:,13].mean(),'%') + '\n'
        'linear_mape14=' + strnum(APE_linear_all[:,13].mean(),'%')
    )

    if 'baseloss' in tocsv:
        # MAPE és MDAPE függése a daysafter-től     Átlagolás az összes predikcióra
        tbl_out=pd.DataFrame()
        tbl_out['daysafter'] = Range(1,count=50)    # az utolsó ismert nap utáni hányadik napra vonatkozik a predikció

        tbl_out['MAPE_base'] = np.mean(APE_base_all,axis=0)        # MAPE_base[i_daysafter]
        tbl_out['MDAPE_base'] = np.median(APE_base_all,axis=0)     # MDAPE_base[i_daysafter]
        tbl_out['MAPEmax_base'] = np.mean(APEmax_base_all,axis=0)     # MDAPE_base[i_daysafter]

        tbl_out['MAPE_base2'] = np.mean(APE_base2_all,axis=0)        # MAPE_base2[i_daysafter]
        tbl_out['MDAPE_base2'] = np.median(APE_base2_all,axis=0)     # MDAPE_base2[i_daysafter]
        tbl_out['MAPEmax_base2'] = np.mean(APEmax_base2_all,axis=0)        # MAPE_base2[i_daysafter]

        tbl_out['MAPE_linear'] = np.mean(APE_linear_all,axis=0)        # MAPE_linear[i_daysafter]
        tbl_out['MDAPE_linear'] = np.median(APE_linear_all,axis=0)     # MDAPE_linear[i_daysafter]
        tbl_out['MAPEmax_linear'] = np.mean(APEmax_linear_all,axis=0)        # MAPE_linear[i_daysafter]

        Tocsv(tbl_out,'base_losses lognorm',format='hu')

    if 'countries' in tocsv:
        tbl_out=pd.DataFrame(recs,columns=columns.split(','))
        Tocsv(tbl_out,'country_stats lognorm',format='hu')
# fn_country_stats(lower_limit=0.05)
# exit()

def fn_plot_country_stats(fname='country_stats lognorm',out='stats'):           # Country-level statistical data,  barplot
    '''
    out: 'stats'  'losses'  'decompose_loss'
    '''

    path=Datapath(fname,'csv')
    tbl=Read_csv(path,index_col='country')

    # tblinfo(tbl,'hist')

    captionfilter=['South Korea','Denmark','Belgium','Poland','Russia','United States','Canada',
                   'Hungary','Germany','France','Spain','United Kingdom']
    height=0.9

    if out=='stats':
        cols='peak14,std_rel,scatter_rel_G14'
        suptitle='Country-level statistical data'
        title='Value distributions'
        # annotbyindex=20
    elif out=='losses':
        cols='base_mdape_14,linear_mdape_14'
        suptitle='Base losses by countries'
        title = 'Distribution of loss values'
        height=0.6
        # annotbyindex=41
    elif out=='decompose_loss':
        cols='decompose_loss'
        suptitle='Loss of lognormal decomposition by countries'
        title = 'Distribution of loss values'
        height=0.4

    subparams = ddef(
        titles = ddef(
            peak14='Maximum peak',
            maxpos_count_G35 = 'Count of peaks',
            std_rel = 'Relative scatter',
            scatter_rel_G14 = 'Weekly relative scatter',
            null_proportion = 'Proportion of missing data',
            base_mdape_14 = 'Median APE of base predictions',
            linear_mdape_14 = 'Median APE of linear predictions',
            decompose_loss = 'Decompose loss (MAPEmax)'
        ),
        comments = ddef(
            peak14='Daily COVID cases / 1 million population, at 14 days timescale',
            maxpos_count_G35 = 'at 35 days timescale',
            # scatter_rel_G14='at 14-days timescale',
            base_mdape_14 = 'on the 14th day',
            linear_mdape_14 = 'on the 14th day',
            std_rel='std / mean'
        ),
        numformats = ddef(
            peak14='0f',
            std_rel='0%',
            scatter_rel_G14 = '0%',
            null_proportion = '0%',
            base_mdape_14 = '0%',
            linear_mdape_14 = '0%',
            decompose_loss = '2%'
        )
    )


    titles,comments,numformats = dgets(subparams,'titles,comments,numformats')
    cols=cols.split(',')

    pltinit(suptitle=suptitle,nrows=len(cols),ncols=1,sharey=False,sharex=False,
            height=height,bottom=0.03,left=0.08,right=0.88,hspace=0.3)
    for col in cols:
        pltinitsub('next',title=dget(titles,col))
        FvPlotAreaBars(tbl[col],captionfilter=captionfilter,numformat=dget(numformats,col,'3g'))
        pltshowsub(ynumformat=dget(numformats,col,'3g'),
                   commenttopright_out=dget(comments,col,col),
                   annot_fontsize=8)
    pltshow()



    # hist_plot(tbl,G=0.02,cols=cols,annotbyindex=captionfilter,
    #           suptitle=suptitle,title=title,
    #           left=0.06,right=0.85,width=0.75,bottom=0.074,top=0.84,height=height,
    #           subparams=subparams)

# fn_plot_country_stats('country_stats',out='losses')
# fn_plot_country_stats('country_stats lognorm',out='decompose_loss')
# exit()


def fn_plot_country_base_losses(fname='country_stats lognorm'):         # FvPlotBarH variant (see also fn_plot_country stats)
    path=Datapath(fname,'csv')
    tbl=Read_csv(path,index_col='country')

    pltinit(suptitle='Base losses by countries',ncols=2,nrows=1,sharey=False,sharex=False)

    captionfilter=['South Korea','Denmark','Belgium','Poland','Russia','United States','Canada',
                   'Hungary','Germany','France','Spain','United Kingdom']

    pltinitsub('next',title='MAPE of base prediction')
    ser = tbl['base_mdape_14'].sort_values()
    FvPlotBarH(ser,shorten=15,annotformat='%',captionfilter=captionfilter)
    pltshowsub(commenttopright_out='on the 14th day')

    pltinitsub('next',title='MAPE of linear prediction')
    ser = tbl['linear_mdape_14'].sort_values()
    FvPlotBarH(ser,shorten=15,annotformat='%',captionfilter=captionfilter)
    pltshowsub(commenttopright_out='on the 14th day')

    pltshow()
# fn_plot_country_base_losses('country_stats lognorm')
# exit()

def fn_corrs_country_data(fname='country_stats lognorm'):
    path=Datapath(fname,'csv')
    tbl=Read_csv(path,index_col='country')

    cols=['curve_length','null_proportion','peak14','median','mean','std_rel','peakedness','active_proportion',
          'maxpos_count_G14','base_mdape_14','linear_mdape_14','scatter_rel_G14',
          'leftwidth_mean','leftwidth_opt','decompose_loss']

    tblinfo(tbl[cols],'corrplot')
# fn_corrs_country_data('country_stats')
# exit()


def fn_plot_base_MAPE_MDAPE_by_daysafter():
    path=Datapath('base_losses lognorm','csv')
    tbl=Read_csv(path)                  
    count = len(tbl)//50

    pltinit(suptitle='Base losses, MAPE and MDAPE',
            title = 'Losses os base and linear predictions',
            left=0.09,right=0.93,wspace=0.24,top=0.79,bottom=0.1,
            ncols=2,nrows=1,sharey=False)

    v_lines = [ddef(x=14,caption='day 14'),ddef(x=35,caption='day 35')]


    pltinitsub('next',
               title='Mean APE (MAPE)')

    ser = serfromtbl(tbl,'MAPE_base','daysafter')
    FvPlot(ser,'original',label='base',annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')

    ser = serfromtbl(tbl,'MAPE_linear','daysafter')
    FvPlot(ser,'original',label='linear',annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')

    pltshowsub(ymin=0,xlabel='days after last known data',ylabel='MAPE',ynumformat='0%',v_lines=v_lines,
               commenttopright_out='outlier-sensitive view',
               area_under=1)



    pltinitsub('next',
               title='Median APE (MDAPE)')

    ser = serfromtbl(tbl,'MDAPE_base','daysafter')
    FvPlot(ser,'original',label='base',annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')

    ser = serfromtbl(tbl,'MDAPE_linear','daysafter')
    FvPlot(ser,'original',label='linear',annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')

    pltshowsub(ymin=0,xlabel='days after last known data',ylabel='MDAPE',ynumformat='0%',v_lines=v_lines,
               commenttopright_out='bottom view, ignoring outliers',
               area_under=1)

    pltshow()     
# fn_plot_base_MAPE_MDAPE_by_daysafter()
# exit()


def fn_plot_compare_base_and_lognorm_MAPE_by_daysafter(out='all'):      # base, linear, lognorm összehasonlítása
    '''
    out:  'all'  'only_mdape'
    '''

    tbl_lognorm=Read_csv(Datapath('loss_train lognorm FINAL','csv'))

    count = len(tbl_lognorm)//50

    # daysafter=0  1 nappal az utolsó adat napja után van
    tbl_lognorm['dayafter'] = tbl_lognorm['dayafter']+1
    tbl_lognorm = tbl_lognorm.loc[tbl_lognorm.dayafter<=40]

    tbl_lognorm['APE_base'] = abs(tbl_lognorm['y_base_n'] - tbl_lognorm['y_true_n']) / tbl_lognorm['y_true_n']
    tbl_lognorm['APEmax_base'] = abs(tbl_lognorm['y_base_n'] - tbl_lognorm['y_true_n'])     # eleve osztva van a maxértékkel

    tbl_lognorm['APE_linear'] = abs(tbl_lognorm['y_linear_n'] - tbl_lognorm['y_true_n']) / tbl_lognorm['y_true_n']
    tbl_lognorm['APEmax_linear'] = abs(tbl_lognorm['y_linear_n'] - tbl_lognorm['y_true_n'])     # eleve osztva van a maxértékkel


    tbl_lognorm=Read_csv(Datapath('loss_train lognorm FINAL','csv'))

    count = len(tbl_lognorm)//50

    # daysafter=0  1 nappal az utolsó adat napja után van
    tbl_lognorm['dayafter'] = tbl_lognorm['dayafter']+1
    tbl_lognorm = tbl_lognorm.loc[tbl_lognorm.dayafter<=40]

    tbl_lognorm['APE_base'] = abs(tbl_lognorm['y_base_n'] - tbl_lognorm['y_true_n']) / tbl_lognorm['y_true_n']
    tbl_lognorm['APEmax_base'] = abs(tbl_lognorm['y_base_n'] - tbl_lognorm['y_true_n'])     # eleve osztva van a maxértékkel

    tbl_lognorm['APE_linear'] = abs(tbl_lognorm['y_linear_n'] - tbl_lognorm['y_true_n']) / tbl_lognorm['y_true_n']
    tbl_lognorm['APEmax_linear'] = abs(tbl_lognorm['y_linear_n'] - tbl_lognorm['y_true_n'])     # eleve osztva van a maxértékkel


    if out!='only_mdape':
        pltinit(suptitle='Comparison of lognormal and base predictions, MAPE and MDAPE',
                title = 'Lognorm, base and linear predictions ' +
                            '(altogether ' + strint(count) + ' predictions for 41 country)',
                left=0.09,right=0.93,wspace=0.24,top=0.79,bottom=0.1,
                ncols=2,nrows=1,sharey=False)
    elif out=='only_mdape':
        pltinit(suptitle='Comparison of models, MDAPE',
                title='Zoomed-in diagram',
                left=0.09,right=0.93,wspace=0.24,top=0.79,bottom=0.1,width=0.4,
                ncols=1,nrows=1,sharey=False)

    if out=='only_mdape': v_lines = [ddef(x=14,caption='day 14')]
    else: v_lines = [ddef(x=14,caption='day 14'),ddef(x=35,caption='day 35')]


    if out!='only_mdape':
        pltinitsub('next',
                title='Mean APE (MAPE)')

        # ser = serfromtbl(tbl,'MAPE_base','daysafter')
        # FvPlot(ser,'original',label='base',annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')

        ser = serfromtbl(tbl_lognorm,'APE_base','dayafter',aggfunc='mean')
        FvPlot(ser,'original',label='base',annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')

        # ser = serfromtbl(tbl,'MAPE_linear','daysafter')
        # FvPlot(ser,'original',label='linear',annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')

        ser = serfromtbl(tbl_lognorm,'APE_linear','dayafter',aggfunc='mean')
        FvPlot(ser,'original',label='linear',annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')

        ser = serfromtbl(tbl_lognorm,'mape','dayafter',aggfunc='mean')
        FvPlot(ser,'original',label='Lognorm',annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')

        pltshowsub(ymin=0,xlabel='days after last known data',ylabel='MAPE',ynumformat='0%',v_lines=v_lines,
                commenttopright_out='top view, considering outliers',
                area_under=1,annot_fontsize=9)



    pltinitsub('next',
               title='Median APE (MDAPE)')

    # ser = serfromtbl(tbl,'MDAPE_base','daysafter')
    # FvPlot(ser,'original',label='base',annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')

    interval=None
    if out=='only_mdape': 
        interval=(0,17)

    ser = serfromtbl(tbl_lognorm,'APE_base','dayafter',aggfunc='median')
    FvPlot(ser,'original',label='base',annotcaption=ddef(last='label',left='label',xpos='y%'),annot='last left 1 14 35',interval=interval)

    # ser = serfromtbl(tbl,'MDAPE_linear','daysafter')
    # FvPlot(ser,'original',label='linear',annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')

    ser = serfromtbl(tbl_lognorm,'APE_linear','dayafter',aggfunc='median')
    FvPlot(ser,'original',label='linear',annotcaption=ddef(last='label',left='label',xpos='y%'),annot='last left 1 14 35',interval=interval)

    ser = serfromtbl(tbl_lognorm,'mape','dayafter',aggfunc='median')
    FvPlot(ser,'original',label='Lognorm',annotcaption=ddef(last='label',left='label',xpos='y%'),annot='last left 1 14 35',interval=interval)

    if out=='only_mdape':
        FvAnnotateAdd(x=14,y=0.33,caption='LSTM 33%',position='left bottom',color='red')
        FvAnnotateAdd(x=14,y=0.32,caption='ARIMA 32%',position='left bottom',color='red')

    pltshowsub(ymin=0,xlabel='days after last known data',ylabel='MDAPE',ynumformat='0%',v_lines=v_lines,
               commenttopright_out='bottom view, ignoring outliers',
               area_under=1,annot_fontsize=9)

    pltshow()     
# fn_plot_compare_base_and_lognorm_MAPE_by_daysafter()
# fn_plot_compare_base_and_lognorm_MAPE_by_daysafter('only_mdape')
# exit()

def fn_plot_compare_base_and_lognorm_MAPEmax_by_daysafter():      # MAPEmax és SMAPE
    '''
    '''

    tbl_lognorm=Read_csv(Datapath('loss_train lognorm FINAL','csv'))

    count = len(tbl_lognorm)//50

    # daysafter=0 v1 nappal az utolsó adat napja után van
    tbl_lognorm['dayafter'] = tbl_lognorm['dayafter']+1
    tbl_lognorm = tbl_lognorm.loc[tbl_lognorm.dayafter<=40]

    tbl_lognorm['APEmax_base'] = abs(tbl_lognorm['y_base_n'] - tbl_lognorm['y_true_n'])     # eleve osztva van a maxértékkel
    tbl_lognorm['SAPE_base'] = ((abs(tbl_lognorm['y_base_n'] - tbl_lognorm['y_true_n']) /
                               ((tbl_lognorm['y_base_n']) + tbl_lognorm['y_true_n']) / 2)     )

    tbl_lognorm['APEmax_linear'] = abs(tbl_lognorm['y_linear_n'] - tbl_lognorm['y_true_n'])     # eleve osztva van a maxértékkel
    tbl_lognorm['SAPE_linear'] = ((abs(tbl_lognorm['y_linear_n'] - tbl_lognorm['y_true_n']) /
                                 ((tbl_lognorm['y_linear_n']) + tbl_lognorm['y_true_n']) / 2)     )

    tbl_lognorm['APEmax_lognorm'] = abs(tbl_lognorm['y_pred_n'] - tbl_lognorm['y_true_n'])     # eleve osztva van a maxértékkel
    tbl_lognorm['SAPE_lognorm'] = ((abs(tbl_lognorm['y_pred_n'] - tbl_lognorm['y_true_n']) /
                                 ((tbl_lognorm['y_pred_n']) + tbl_lognorm['y_true_n']) / 2)     )


    pltinit(suptitle='Comparison of lognormal and base predictions, APEmax and SMAPE',
            title = 'Lognorm, base and linear predictions ' +
                        '(altogether ' + strint(count) + ' predictions for 41 country)',
            left=0.09,right=0.93,wspace=0.24,top=0.79,bottom=0.1,
            ncols=2,nrows=1,sharey=False)

    v_lines = [ddef(x=14,caption='day 14'),ddef(x=35,caption='day 35')]


    pltinitsub('next',
               title='MAPEmax')

    ser = serfromtbl(tbl_lognorm,'APEmax_base','dayafter',aggfunc='mean')
    FvPlot(ser,'original',label='base',annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')

    ser = serfromtbl(tbl_lognorm,'APEmax_linear','dayafter',aggfunc='mean')
    FvPlot(ser,'original',label='linear',annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')

    ser = serfromtbl(tbl_lognorm,'APEmax_lognorm','dayafter',aggfunc='mean')
    FvPlot(ser,'original',label='Lognorm',annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')

    pltshowsub(ymin=0,ymax=0.3,xlabel='days after last known data',ylabel='MAPEmax',ynumformat='0%',v_lines=v_lines,
               commenttopright_out='Relative to the all-time highest value',annot_fontsize=9)


    pltinitsub('next',
               title='SMAPE')

    ser = serfromtbl(tbl_lognorm,'SAPE_base','dayafter',aggfunc='mean')
    FvPlot(ser,'original',label='base',annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')

    ser = serfromtbl(tbl_lognorm,'SAPE_linear','dayafter',aggfunc='mean')
    FvPlot(ser,'original',label='linear',annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')

    ser = serfromtbl(tbl_lognorm,'SAPE_lognorm','dayafter',aggfunc='mean')
    FvPlot(ser,'original',label='Lognorm',annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')

    pltshowsub(ymin=0,ymax=0.3,xlabel='days after last known data',ylabel='SMAPE',ynumformat='0%',v_lines=v_lines,
               commenttopright_out='Symmetric Mean Absolute Percentage Error',annot_fontsize=9)


    pltshow()     
# fn_plot_compare_base_and_lognorm_MAPEmax_by_daysafter()
# exit()


def fn_plot_lognorm_scores(phases=None):      # Base loss-hoz viszonyított loss MAPE és MAPEmax
    '''
    Score of lognormal prediction relative to base predictions

    phases:   only in the gives phases (list, e.g. [2,3])
    '''


    tbl_lognorm=Read_csv(Datapath('loss_train lognorm FINAL','csv'))

    # daysafter=0 valójában 1 nappal az utolsó adat napja után van
    tbl_lognorm['dayafter'] = tbl_lognorm['dayafter']+1
    tbl_lognorm = tbl_lognorm.loc[tbl_lognorm.dayafter<=40]

    tbl_lognorm['APE_base'] = abs(tbl_lognorm['y_base_n'] - tbl_lognorm['y_true_n']) / tbl_lognorm['y_true_n']
    tbl_lognorm['APEmax_base'] = abs(tbl_lognorm['y_base_n'] - tbl_lognorm['y_true_n'])     # eleve osztva van a maxértékkel

    tbl_lognorm['APE_linear'] = abs(tbl_lognorm['y_linear_n'] - tbl_lognorm['y_true_n']) / tbl_lognorm['y_true_n']
    tbl_lognorm['APEmax_linear'] = abs(tbl_lognorm['y_linear_n'] - tbl_lognorm['y_true_n'])     # eleve osztva van a maxértékkel

    tbl_lognorm['APE_lognorm'] = abs(tbl_lognorm['y_pred_n'] - tbl_lognorm['y_true_n']) / tbl_lognorm['y_true_n']
    tbl_lognorm['APEmax_lognorm'] = abs(tbl_lognorm['y_pred_n'] - tbl_lognorm['y_true_n'])     # eleve osztva van a maxértékkel


    count = len(tbl_lognorm)//50
   
    ymax=2.7
    title = 'Loss base / loss Lognorm (based on ' + strint(count) + ' predictions for 41 country)',
    if phases is not None:
        tbl_lognorm = tbl_lognorm.loc[tbl_lognorm.phase_ok.isin(phases)]
        count = len(tbl_lognorm)//50
        ymax=5.5
        title='In the 2. and 3. phase (based on ' + strint(count) + ' predictions for 41 country)'


    pltinit(suptitle='Score of lognormal prediction relative to the base predictions',
            title = title,
            left=0.09,right=0.93,wspace=0.24,top=0.79,bottom=0.1,
            ncols=2,nrows=1,sharey=False)

    v_lines = [ddef(x=14,caption='day 14'),ddef(x=35,caption='day 35')]


    pltinitsub('next',
               title='MAPE SCORE')

    ser_base = serfromtbl(tbl_lognorm,'APE_base','dayafter',aggfunc='mean')
    ser_linear = serfromtbl(tbl_lognorm,'APE_linear','dayafter',aggfunc='mean')
    ser_lognorm = serfromtbl(tbl_lognorm,'APE_lognorm','dayafter',aggfunc='mean')

    FvPlot(ser_base / ser_lognorm,'original',label='base',annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')
    FvPlot(ser_linear / ser_lognorm,'original',label='linear',annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')

    pltshowsub(ymin=0,ymax=ymax,xlabel='days after last known data',ylabel='MAPE SCORE',ynumformat='0%',v_lines=v_lines,
               commenttopright_out='Loss base / loss Lognorm', area_under=1,annot_fontsize=9)


    pltinitsub('next',
               title='MAPEmax SCORE')

    ser_base = serfromtbl(tbl_lognorm,'APEmax_base','dayafter',aggfunc='mean')
    ser_linear = serfromtbl(tbl_lognorm,'APEmax_linear','dayafter',aggfunc='mean')
    ser_lognorm = serfromtbl(tbl_lognorm,'APEmax_lognorm','dayafter',aggfunc='mean')

    FvPlot(ser_base / ser_lognorm,'original',label='base',annotcaption=ddef(last='label',max='y%',xpos='y%'),annot='last max 1 14 35')
    FvPlot(ser_linear / ser_lognorm,'original',label='linear',annotcaption=ddef(last='label',max='y%',xpos='y%'),annot='last 1 14 35')

    pltshowsub(ymin=0,ymax=ymax,xlabel='days after last known data',ylabel='MAPEmax SCORE',ynumformat='0%',v_lines=v_lines,
               commenttopright_out='Loss base / loss Lognorm', area_under=1,annot_fontsize=9)


    pltshow()     
# fn_plot_lognorm_scores(phases=[2,3])
# exit()


def fn_loss_distrbutions(outliers=True):         # loss értékek hisztogramja   (Lognorm, base, linear)
    '''
    outliers: If false, only displays the distribution up to loss_max=150% and displays the median, mean, iqr values
   
    '''

    tbl_lognorm=Read_csv(Datapath('loss_train lognorm FINAL','csv'))
    # daysafter=0 valójában 1 nappal az utolsó adat napja után van
    tbl_lognorm['dayafter'] = tbl_lognorm['dayafter']+1
    tbl_lognorm = tbl_lognorm.loc[tbl_lognorm.dayafter<=40]

    tbl_lognorm=tbl_lognorm.loc[tbl_lognorm.dayafter==14]

    tbl_lognorm['APE_base'] = abs(tbl_lognorm['y_base_n'] - tbl_lognorm['y_true_n']) / tbl_lognorm['y_true_n']
    tbl_lognorm['APE_linear'] = abs(tbl_lognorm['y_linear_n'] - tbl_lognorm['y_true_n']) / tbl_lognorm['y_true_n']
    tbl_lognorm['APE_lognorm'] = abs(tbl_lognorm['y_pred_n'] - tbl_lognorm['y_true_n']) / tbl_lognorm['y_true_n']

    if not outliers: 
        suptitle='Loss distributions, without outliers'
        xmax=1.5
        mean,median,iqr=True,True,True
        outliers_=''
    else: 
        suptitle='Loss distributions, with outliers'
        xmax=8
        mean,median,iqr=False,False,False
        outliers_='show_and_annot'


    subparams = ddef(
        titles = ddef(                      # subdiagram felett megjelenő címfelirat
            APE_base='APE base 14th day',
            APE_linear='APE linear 14th day',
            APE_lognorm='APE Lognorm 14th day'
        ),
        numformats='1%',xmax=xmax, xmin=0, comments='Absolute Percentage Error',
        mean=mean,median=median,iqr=iqr
    )


    hist_plot(tbl_lognorm,'APE_lognorm,APE_base,APE_linear',
              suptitle=suptitle,height=0.8,width=0.5,left=0.07,right=0.93,outliers=outliers_,
              subparams=subparams)
# fn_loss_distrbutions(outliers=False)
# exit()

def fn_plot_score_by_countries():
    tbl=Read_csv(Datapath('Lognorm test FINAL','csv'),index_col='country')
    tbl['score'] = 1/tbl['lognorm_base']

    captionfilter=['Denmark','Belgium','Poland','Russia','United States','Canada',
                   'Hungary','Germany','France','Spain','United Kingdom','Sweden']


    pltinit(suptitle='Score by countries',
            title='Score to base model by countries',
            height=0.4,bottom=0.08,top=0.82)
    FvPlotAreaBars(tbl['score'],sort='descending',numformat='0%',captionfilter=captionfilter,shorten=15,gap=0.2)
    pltshow(ynumformat='0%',ymin=1,ymax=2.2,annot_fontsize=8,
            commenttopright='loss_base / loss_lognorm' + '\n' +
                            'where loss = MAPEmax 1-14 days')




    # subparams = ddef(
    #     titles = 'Lognorm model score to base model',
    #     comments = 'loss_base / loss_lognorm      where loss = MAPEmax 1-14 days',
    #     numformats='0%',
    #     xmin=1
    # )
    # hist_plot(tbl,G=0.01,cols='score',annotbyindex=captionfilter,
    #           suptitle='Score distribution by countries',
    #           left=0.06,right=0.85,width=0.75,bottom=0.074,top=0.84,height=0.5,
    #           subparams=subparams)
# fn_plot_score_by_countries()
# exit()







# ============================================================================ #
# # LOSS BY PHASE AND TIME
# ============================================================================ #

def fn_loss_by_phase(fname='loss_train lognorm FINAL'):
    path=Datapath(fname,'csv')
    tbl=Read_csv(path)

    tbl=tbl.loc[tbl.day>datefloat('2020-07-01')]   # az induló szakaszon van néhány nagyon magas érték
    # tbl=tbl.loc[tbl.day<datefloat('2022-12-31')]   # az induló szakaszon van néhány nagyon magas érték

    tbl['mape_peak_base'] = tbl['mape_peak'] * tbl['score_factor']
    
    sub_print_losses(tbl,'phase_all')
    for phase in range(1,5):  sub_print_losses(tbl.loc[tbl.phase_ok==phase],'phase=' + str(phase))
# fn_loss_by_phase()
# exit()


def fn_loss_plot_by_phase(fname='loss_train lognorm FINAL',APE='APEmax',afterdays=14,score=False):       # MDAPE14 fázisonként (lognorm, base, linear)
    '''
    APE:   'APE'  vagy  'APEmax'
    afterdays:   hány predikciós napra átlagoljon
    score:  True esetén loss_base / loss_lognorm,  két subplot (base/lognorm, linear/lognorm)
            False esetén három subplot:   lognorm, base, linear
    '''

    path=Datapath(fname,'csv')
    tbl=Read_csv(path)

    tbl=tbl.loc[tbl.day>datefloat('2020-07-01')]   # az induló szakaszon van néhány nagyon magas érték
    # tbl=tbl.loc[tbl.day<datefloat('2022-12-31')]   # az induló szakaszon van néhány nagyon magas érték

    # tbl['mape_peak_base'] = tbl['mape_peak'] * tbl['score_factor']

    tbl['APE_lognorm'] = abs(tbl['y_pred_n'] - tbl['y_true_n']) / tbl['y_true_n']
    tbl['APE_base'] = abs(tbl['y_base_n'] - tbl['y_true_n']) / tbl['y_true_n']
    tbl['APE_linear'] = abs(tbl['y_linear_n'] - tbl['y_true_n']) / tbl['y_true_n']

    tbl['APEmax_lognorm'] = abs(tbl['y_pred_n'] - tbl['y_true_n'])     # eleve osztva van a maxértékkel
    tbl['APEmax_base'] = abs(tbl['y_base_n'] - tbl['y_true_n'])     # eleve osztva van a maxértékkel
    tbl['APEmax_linear'] = abs(tbl['y_linear_n'] - tbl['y_true_n'])     # eleve osztva van a maxértékkel



    # globális átlag
    loss_lognorm_all = tbl.loc[tbl.dayafter<afterdays,APE + '_lognorm'].mean()
    loss_base_all = tbl.loc[tbl.dayafter<afterdays,APE + '_base'].mean()
    loss_linear_all = tbl.loc[tbl.dayafter<afterdays,APE + '_linear'].mean()

    # mape14_mean_all,lognorm_per_base_all,count_all = sub_print_losses(tbl,'phase_all')

    if score: suptitle = 'Score by phase'
    else: suptitle = 'Loss by phase'

    if score: ncols=2
    else: ncols=3

    pltinit(suptitle=suptitle,nrows=1,ncols=ncols,sharex=False,sharey=False,
            height=0.5,width=0.95,left=0.05,right=0.96,bottom=0.03,top=0.84,wspace=0.17)

    X=Range(0,1000)
    leftwidth=400
    modus=0.95

    # Loss számolás a 4 fázisra (lognorm,base,linear)   és a lépcsős görbe előállítása
    ser_lognorm = pd.Series([1]*1001,X)
    ser_base = pd.Series([1]*1001,X)
    ser_linear = pd.Series([1]*1001,X)
    ser_base_lognorm = pd.Series([1]*1001,X)
    ser_linear_lognorm = pd.Series([1]*1001,X)
    aLognorm,aBase,aLinear,counts=[],[],[],[]
    for phase in range(1,5):  
        tblL=tbl.loc[tbl.phase_ok==phase]

        lognorm = tblL.loc[tblL.dayafter<afterdays,APE + '_lognorm'].mean()
        base = tblL.loc[tblL.dayafter<afterdays,APE + '_base'].mean()
        linear = tblL.loc[tblL.dayafter<afterdays,APE + '_linear'].mean()
        count = len(tblL.loc[tblL.dayafter==0])
       
       
        if phase ==1: 
            ser_lognorm.loc[ser_lognorm.index<0.7*leftwidth]  = lognorm
            ser_base.loc[ser_base.index<0.7*leftwidth]  = base
            ser_linear.loc[ser_linear.index<0.7*leftwidth]  = linear
            ser_base_lognorm.loc[ser_base_lognorm.index<0.7*leftwidth]  = base / lognorm
            ser_linear_lognorm.loc[ser_linear_lognorm.index<0.7*leftwidth]  = linear / lognorm
        elif phase ==2: 
            ser_lognorm.loc[(ser_lognorm.index>=0.7*leftwidth) & (ser_lognorm.index<leftwidth)]  = lognorm
            ser_base.loc[(ser_base.index>=0.7*leftwidth) & (ser_base.index<leftwidth)]  = base
            ser_linear.loc[(ser_linear.index>=0.7*leftwidth) & (ser_linear.index<leftwidth)]  = linear
            ser_base_lognorm.loc[ser_base_lognorm.index>=0.7*leftwidth]  = base / lognorm
            ser_linear_lognorm.loc[ser_linear_lognorm.index>=0.7*leftwidth]  = linear / lognorm
        elif phase ==3: 
            ser_lognorm.loc[(ser_lognorm.index>=leftwidth) & (ser_lognorm.index<1.3*leftwidth)]  = lognorm
            ser_base.loc[(ser_base.index>=leftwidth) & (ser_base.index<1.3*leftwidth)]  = base
            ser_linear.loc[(ser_linear.index>=leftwidth) & (ser_linear.index<1.3*leftwidth)]  = linear
            ser_base_lognorm.loc[(ser_base_lognorm.index>=leftwidth) & (ser_base_lognorm.index<1.3*leftwidth)]  = base / lognorm
            ser_linear_lognorm.loc[(ser_linear_lognorm.index>=leftwidth) & (ser_linear_lognorm.index<1.3*leftwidth)]  = linear / lognorm
        elif phase ==4: 
            ser_lognorm.loc[ser_lognorm.index>=1.3*leftwidth]  = lognorm
            ser_base.loc[ser_base.index>=1.3*leftwidth]  = base
            ser_linear.loc[ser_linear.index>=1.3*leftwidth]  = linear
            ser_base_lognorm.loc[ser_base_lognorm.index>=1.3*leftwidth]  = base / lognorm
            ser_linear_lognorm.loc[ser_linear_lognorm.index>=1.3*leftwidth]  = linear / lognorm
        aLognorm.append(lognorm)
        aBase.append(base)
        aLinear.append(linear)
        counts.append(count)

    # Három subplot egymás mellett
    
    for i_losstype in range(3):         
        if score and i_losstype==0: continue    # score esetén csak két subplot

        comment = 'M' + APE + ' 1-' + str(afterdays)

        if i_losstype==0: 
            title='Lognorm model'
        elif i_losstype==1: 
            if score: title='Score to base'
            else: title='Base model'
        elif i_losstype==2: 
            if score: title='Score to linear'
            else: title='Linear model'

        pltinitsub('next',title = title)

        if score: maxvalue = 2
        else:
            if APE=='APE': maxvalue=0.4
            elif APE=='APEmax': maxvalue=0.1

        area_under=None
        if score: area_under=1

        # A háttérben egy idealizált lognorm görbe jelenik meg, a négy fázissal (exponent=9)
        ser_background = pd.Series(fn_lognormal_multi(X,leftwidth,maxvalue,leftwidth,modus),X)

        annotplus=None
        # annotplus={0.7*leftwidth:'inflexion',
        #         1.15*leftwidth:'An illustrated surge',
        #         1.3*leftwidth:'inflexion',
        #         }
        FvPlot(ser_background,'original',area=True,annot='',annotplus=annotplus)
        # két sáv
        ser_band=ser_background.loc[(ser_background.index>0.7*leftwidth) & (ser_background.index<leftwidth)]
        FvPlot(ser_band,'original',area='noline',annot='')
        ser_band=ser_background.loc[ser_background.index>1.3*leftwidth]
        FvPlot(ser_band,'original',area='noline',annot='')

        if i_losstype==0:
            aLoss = aLognorm
            ser_loss= ser_lognorm
            loss_all = loss_lognorm_all
        elif i_losstype==1:
            if score:
                aLoss=array(aBase)/array(aLognorm)
                ser_loss=ser_base_lognorm
                loss_all = loss_base_all/loss_lognorm_all
            else:
                aLoss=aBase
                ser_loss=ser_base
                loss_all = loss_base_all
        elif i_losstype==2:
            if score:
                aLoss=array(aLinear)/array(aLognorm)
                ser_loss=ser_linear_lognorm
                loss_all = loss_linear_all/loss_lognorm_all
            else:
                aLoss=aLinear
                ser_loss=ser_linear
                loss_all = loss_linear_all


        annotplus={0.481*leftwidth:strnum(aLoss[0],'%'),
                0.831*leftwidth:strnum(aLoss[1],'%'),
                1.164*leftwidth:strnum(aLoss[2],'%'),
                1.51*leftwidth:strnum(aLoss[3],'%')
                }
        
        # Vonal és árnyékolás
        FvPlot(ser_loss,'gauss',G=0.01,annotplus=annotplus,annot='',gaussside='')
        FvPlot(ser_loss,'gauss',G=0.01,annot='',gaussside='',area='noline')

        h_lines = [ddef(y=loss_all,caption='mean=' + strnum(loss_all,'%'))]

        xy_texts=[
            ddef(x=0.125,y=0.05,caption='1. PHASE',ha='center',fontsize=8),
            ddef(x=0.375,y=0.05,caption='2. PHASE',ha='center',fontsize=8),
            ddef(x=0.625,y=0.05,caption='3. PHASE',ha='center',fontsize=8),
            ddef(x=0.875,y=0.05,caption='4. PHASE',ha='center',fontsize=8)
            ]

        pltshowsub(xmin=leftwidth*0.4,xmax=leftwidth*1.6,ymin=0,
                xticklabels=False,ynumformat='0%',
                h_lines=h_lines,commenttopright_out=comment,
                xy_texts=xy_texts,area_under=area_under,annot_fontsize=9)
        
        plt.gca().grid(visible=False,axis='x')        
    pltshow()
# fn_loss_plot_by_phase(APE='APEmax',afterdays=14,score=False)
# fn_loss_plot_by_phase(APE='APEmax',afterdays=14,score=True)
# exit()



def fn_loss_plot_by_time(fname='loss_train lognorm FINAL'):
    path=Datapath(fname,'csv')
    tbl=Read_csv(path)

    tbl=tbl.loc[tbl.day>datefloat('2020-07-01')]   # az induló szakaszon van néhány nagyon magas érték
    # tbl=tbl.loc[tbl.day<datefloat('2022-12-31')]   # az induló szakaszon van néhány nagyon magas érték

    tbl['mape_peak_base'] = tbl['mape_peak'] * tbl['score_factor']
    
    tbl = tbl.loc[tbl.dayafter==13]

    median = tbl['mape'].median()


    ser = serfromtbl(tbl,'mape','day',aggfunc='median')
    ser=floatdate(ser)
    # - mindegyik országra ugyanazok a napok, 8 napos lépésközzel, ezért nem kell resample

    # ser = serfromtbl(tbl,'mape','day')
    # ser=floatdate(ser)
    # ser=ser.resample('W').median()

    pltinit('Change of loss by time')
    
    interval=None
    # interval=(Date('2021-01-09'),Date('2021-01-31'))

    # Háttérinformációként az átlagos esetszám idősora
    FvPlot(fn_ser_cases_mean(),'original',label='cases_normalized',annot='max last',normalize=1,area=True,
           interval=interval)
    

    FvPlot(ser,'scatter regauss',G=0.08,label='Median APE',annot='gaussabs12 last',
            annotcaption=ddef(gaussabs='y%',last='label'),interval=interval)
    pltshow(ylabel='Median APE',ynumformat='%',xtickerstep='year',
            h_lines=[ddef(y=median,caption=strnum(median,'%'))],annot_fontsize=9)
# fn_loss_plot_by_time('loss_train lognorm FINAL')
# exit()

def fn_MedianTAPE_comparable():     # Nway cikk, ugyanarra az időszakra
    tbl_flatten=Read_csv(Datapath('loss_train lognorm for MedianTAPE flatten','csv'))
    tbl_flatten=tbl_flatten.loc[tbl_flatten.dayafter==13]

    # Az MDAPE számításakor a mozgóátlaghoz mérendő a hiba
    print('Lognormal MDAPE=' + strnum((abs(tbl_flatten.y_pred_n-tbl_flatten.y_true_n)/tbl_flatten.y_true_n).median(),'%'))
    print('Base MDAPE=' + strnum((abs(tbl_flatten.y_base_n-tbl_flatten.y_true_n)/tbl_flatten.y_true_n).median(),'%'))
    print('Linear MDAPE=' + strnum((abs(tbl_flatten.y_linear_n-tbl_flatten.y_true_n)/tbl_flatten.y_true_n).median(),'%'))


    tbl=Read_csv(Datapath('loss_train lognorm for MedianTAPE','csv'))
    tbl=tbl.loc[tbl.dayafter==13]

    countries=fn_countries('Europe+')
    # countries=['Hungary']
    recs=[]
    for country in countries:
        tblL = tbl.loc[tbl.country==country]
        sum_true = tblL.y_raw_n.sum()
        # sum_true = tblL.y_true_n.sum()
        tape = abs(tblL.y_pred_n.sum() - sum_true) / sum_true
        tape_base = abs(tblL.y_base_n.sum() - sum_true) / sum_true
        tape_linear = abs(tblL.y_linear_n.sum() - sum_true) / sum_true
        recs.append([country,tape,tape_base,tape_linear])
    tbl_out=TblFromRecords(recs,'country,tape,tape_base,tape_linear')

    print()
    print('Lognormal MDTAPE=' + strnum(tbl_out.tape.median(),'%') +
          '   IQR=(' + strnum(np.quantile(tbl_out.tape.values,0.25),'%') + ',' + strnum(np.quantile(tbl_out.tape.values,0.75),'%') + ')')
    print('Base MDTAPE=' + strnum(tbl_out.tape_base.median(),'%') + 
          '   IQR=(' + strnum(np.quantile(tbl_out.tape_base.values,0.25),'%') + ',' + strnum(np.quantile(tbl_out.tape_base.values,0.75),'%') + ')')
    print('Linear MDTAPE=' + strnum(tbl_out.tape_linear.median(),'%') +
          '   IQR=(' + strnum(np.quantile(tbl_out.tape_linear.values,0.25),'%') + ',' + strnum(np.quantile(tbl_out.tape_linear.values,0.75),'%') + ')')


    tbl_out=tbl_out.sort_values('tape')
    # print(tbl_out)
# fn_MedianTAPE_comparable()
# exit()

def fn_MedianTAPE_base_comparable(countries='Europe+',johnshopkins=False):   # Nature cikk, 84 ország, ugyanarra az időszakra
    if johnshopkins:
        tbl=fn_read_johns_hopkins()
    else:
        tbl = Fn_read_coviddata()

    countries_in=countries
    countries = fn_countries(countries)
    # countries = fn_countries('Europe+')
    # countries = ['Hungary']

    recs=[]
    for country in countries:
        if johnshopkins: ser = fn_coviddata_country_js(tbl,country)
        else: ser = Fn_coviddata_country(tbl,country)

        ser = datefloat(ser)

        ser_14 = FvGaussAvg(ser,gausswidth=14)
        max = ser_14.max()

        day0 = datefloat('2021-01-09')

        sum_true,sum_linear,sum_base = 0,0,0
        for day in Range(day0,day0+20,add=1):
 
            # A tesztnapon ismert görbe. 
            ser_known = ser.loc[:day-14]   

            y_true=ser.loc[day]         
            sum_true = sum_true + y_true/max

            # heti átlagértékes becslés hibája (BASE)
            mean_7days = ser_known.iloc[-7:].mean()  # az utolsó 7 nap átlaga
            y_pred_base = mean_7days     # az átlagértéket viszi tovább a predikciós napokra
            sum_base = sum_base + y_pred_base/max

            # lineáris becslés hibája
            mean_7days_2 = ser_known.iloc[-7:].mean()       # utolsó 7 nap átlaga
            mean_7days_1 = ser_known.iloc[-14:-7].mean()    # előző 7 nap átlaga
            grad1=(mean_7days_2 - mean_7days_1) / 7
            y_pred_linear=mean_7days_2 + grad1 * (3 + 14)   # in range(int(G/2)+1,int(G/2)+1+nLossDays)]
                # - az utolsó hét közepétől indul az illesztés
            if y_pred_linear<0:  y_pred_linear=0
                # - ha 0 alá menne az eredmény, akkor legyen 0 (erre szolgál a max) 
            sum_linear = sum_linear + y_pred_linear/max

        tape_base = abs(sum_base - sum_true) / sum_true
        tape_linear = abs(sum_linear - sum_true) / sum_true
        
        recs.append([country,tape_base,tape_linear])
    tbl_out=TblFromRecords(recs,'country,tape_base,tape_linear')

    print()
    print('Countries=' + countries_in + '   Johns Hopkins=' + str(johnshopkins))
    print('Base MDTAPE=' + strnum(tbl_out.tape_base.median(),'%') + 
          '   IQR=(' + strnum(np.quantile(tbl_out.tape_base.values,0.25),'%') + ',' + strnum(np.quantile(tbl_out.tape_base.values,0.75),'%') + ')')
    print('Linear MDTAPE=' + strnum(tbl_out.tape_linear.median(),'%') +
          '   IQR=(' + strnum(np.quantile(tbl_out.tape_linear.values,0.25),'%') + ',' + strnum(np.quantile(tbl_out.tape_linear.values,0.75),'%') + ')')

    tbl_out=tbl_out.sort_values('tape_base')
    # print(tbl_out)
# fn_MedianTAPE_base_comparable(countries='Europe+',johnshopkins=False)
# fn_MedianTAPE_base_comparable(countries='nway',johnshopkins=False)
# fn_MedianTAPE_base_comparable(countries='Europe+',johnshopkins=True)
# fn_MedianTAPE_base_comparable(countries='nway',johnshopkins=True)
# exit()

def  fn_median_total_absolute_percentage_error_by_countries():   # Az Nway et al. cikkben szereplő loss-függvény
    '''
    total absolute precentage error by countries      abs( sum(Y_fact) - sum(Y_pred) ) / sum(Y_pred)
    Median for then country-level losses
    '''

    # tbl=Read_csv(Datapath('loss_train lognorm FINAL','csv'))
    tbl=Read_csv(Datapath('loss_train lognorm for MedianTAPE','csv'))

    tbl=tbl.loc[tbl.dayafter==13]       # 14th day  (dayafter is 0-based)
    
    # Total APE by countries
    recs,recs_linear = [],[]
    for country in fn_countries('Europe+'):
        tblL = tbl.loc[tbl.country==country]
        # sum_true = tblL['y_true_n'].sum()      # normálva van az ország csúcsértékével, de az osztáskor egyébként is kiesne
        sum_true = tblL['y_raw_n'].sum()        # raw értékekkel
        sum_pred = tblL['y_pred_n'].sum()
        sum_linear = tblL['y_linear_n'].sum()
        tape = abs(sum_true-sum_pred) / sum_true
        tape_linear = abs(sum_true-sum_linear) / sum_true
        # print('TAPE ' + country + ' = ' + strnum(tape,'%'))
        recs.append([country,tape])
        recs_linear.append([country,tape_linear])
    
    ser_tape = SerFromRecords(recs)
    ser_tape_linear = SerFromRecords(recs_linear)

    captionfilter=['South Korea','Denmark','Belgium','Poland','Russia','United States','Canada',
                   'Hungary','Germany','France','Spain','United Kingdom']

    pltinit(suptitle='Total Absolute Precentage Error by countries',
            title='Lognorm regression, TAPE on the 14th day',
            height=0.4,bottom=0.08,top=0.75)
    FvPlotAreaBars(ser_tape,captionfilter=captionfilter,numformat='0%')
    pltshow(annot_fontsize=8,ynumformat='0%',
            commenttopright='TAPE = abs( sum(Y_true) - sum(Y_pred) )  /  sum(Y_true)',
            commenttopright_in='Median TAPE = ' + strnum(ser_tape.median(),'%') )

    print()
    print('MDTAPE Lognorm = ' + strnum(ser_tape.median(),'%'))
    print('MDTAPE linear = ' + strnum(ser_tape_linear.median(),'%'))
# fn_median_total_absolute_percentage_error()
# exit()






def fn_loss_by_weekday(fname='loss_train lognorm FINAL'):
    path=Datapath(fname,'csv')
    tbl=Read_csv(path)

    tbl=tbl.loc[tbl.day>datefloat('2020-07-01')]   # az induló szakaszon van néhány nagyon magas érték
    # tbl=tbl.loc[tbl.day<datefloat('2022-12-31')]   # az induló szakaszon van néhány nagyon magas érték

    tbl['mape_peak_base'] = tbl['mape_peak'] * tbl['score_factor']
    
    tbl['day'] = floatdate(tbl['day'].values)

    sub_print_losses(tbl,'weekday_all')
    for weekday in range(7):  sub_print_losses(tbl.loc[tbl.day.dt.dayofweek==weekday],'weekday=' + str(weekday))

    # országonként a legjobb weekday, mape14 illetve lognorm/base alapján
    countries=fn_countries('Europe+')
    for country in countries:
        tblL=tbl.loc[(tbl.country==country) & (tbl.dayafter==13)]
        ser=serfromtbl(tblL,'mape','day')
        # ser=floatdate(ser)
        rec = []
        for weekday in range(7):
            serL= ser.loc[ser.index.dayofweek==weekday]
            mape = serL.median()
            rec.append([weekday,mape])
        ser_out=SerFromRecords(rec).sort_values()
        print(country + '    ' +
                        str(ser_out.index[0]) + ':' + strnum(ser_out.iloc[0],'%') + '   ' +
                        str(ser_out.index[1]) + ':' + strnum(ser_out.iloc[1],'%') + '   ' +
                        '...    ' +
                        str(ser_out.index[6]) + ':' + strnum(ser_out.iloc[6],'%'))
# fn_loss_by_weekday()
# exit()


def fn_plot_underestimation_percent_by_daysafter():    # felülbecslések aránya a daysafter függvényében 
    path=Datapath('loss_train lognorm FINAL','csv')
    tbl=Read_csv(path)


    recs=[]
    for dayafter in range(0,50):
        tblL = tbl.loc[tbl.dayafter==dayafter]
        count_all = len(tblL)

        lognorm_under = len(tblL.loc[tblL.y_pred_n>tblL.y_true_n]) / count_all
        base_under = len(tblL.loc[tblL.y_base_n>tblL.y_true_n]) / count_all
        linear_under = len(tblL.loc[tblL.y_linear_n>tblL.y_true_n]) / count_all
        recs.append([dayafter+1,lognorm_under,base_under,linear_under])
    tbl_out = TblFromRecords(recs,'dayafter,lognorm_under,base_under,linear_under','dayafter')

    pltinit(suptitle = 'Overestimation percent by daysafter',
            title='Proportion of overestimations in the Lognorm and base models',
            top=0.85)

    annotcaption=ddef(middle='label',xpos='y%')

    FvPlot(tbl_out['lognorm_under'],'original',label='Lognorm',annot='middle 1 50',annotcaption=annotcaption)
    FvPlot(tbl_out['base_under'],'original',label='base',annot='middle 1 50',annotcaption=annotcaption)
    FvPlot(tbl_out['linear_under'],'original',label='linear',annot='middle 1 50',annotcaption=annotcaption)

    pltshow(ymin=0,ymax=1,ynumformat='0%',area_under=0.5,xlabel='days after last known data')
# fn_plot_underestimation_percent_by_daysafter()
# exit()




# ============================================================================ #
# # # PROPERTIES OF ERLANG and LOGNORMAL FUNCTIONS
# ============================================================================ #


def fn_erlang_exponent_plot():    # Lognorm functions with different exponents
    X=Range(0,3.5,add=0.001)

    pltinit('Erlang functions with different powers',
            left=0.05,bottom=0.08,height=0.4,right=0.97)
    for kitevo in (4,9,18):
        Y=fn_erlang(X,0,1,kitevo)
        ser=pd.Series(Y,X)

        FvPlot(ser,'original',label='Power=' + str(kitevo),annot='0.5 1.76',area=True)
    pltshow()
# fn_erlang_exponent_plot()
# exit()

def Fn_erlang_grad_plot():          # Erlang és gradienseinek általános szemléletetése

    X=Range(0,3.5,add=0.001)

    pltinit('Erlang function and its derivatives',nrows=3,ncols=1,width=0.6,height=0.9,bottom=0.04)
    for kitevo in (4,9,18):
        Y=fn_erlang(X,0,1,kitevo)
        ser=pd.Series(Y,X)

        serGrad1=FvGradient(ser)
        serGrad2=FvGradient(serGrad1)
        # serGrad3=FvGradient(serGrad2)

        pltinitsub('next',title='Power=' + strint(kitevo))

        FvPlot(ser,'original',label='Erlang peak',annot='localmax',area=True)
        FvPlot(serGrad1,'original',label='grad1',normalize=0.5,annot='localmax1 localmin1')
        FvPlot(serGrad2,'original',label='grad2',normalize=0.5,annot='localmax2 localmin1',annotcaption='grad2 x={x}')
        # FvPlot(serGrad3,'original',label='grad3',normalize=1,annot='localmax localmin',annotcaption='grad3 x={x}')

        x_start = serGrad2.idxmax()
        v_lines=[ddef(x=0,caption='nullpos'),ddef(x=x_start,caption='startpos')]

        pltshowsub(v_lines=v_lines)
    pltshow()
# Fn_erlang_grad_plot()
# exit()

def Fn_erlang_phase(ser,grad_max=5,G=20,Gfaktor=1.4,bPlot=False):      # return serPhase
    '''
    grad_max:  in case of 5  goes until grad5
    '''
    
    ser=ser.sort_index()
    ser.fillna(value=0,inplace=True)        # nem lehetnek benne üres értékek
    ser=fillna_back(ser)      # utólagosan közölt értékek szétosztása az előtte lévő nullás napokra

    if bPlot:
        pltinit()

    serG=FvGaussAvg(ser,G)
    if bPlot: FvPlot(serG,plttype='original',label='serG',annot='max last', normalize=1)

    # gradiens-görbék előállítása 
    aGrad=[None]*(grad_max+1)
    aY = [None]*(grad_max+1)
    aGrad[0]=serG
    aY[0] = serG.values
    for i_grad in range(1,grad_max+1):
        aGrad[i_grad] = FvGaussAvg(FvGradient(aGrad[i_grad-1]))
        if bPlot: FvPlot(aGrad[i_grad],plttype='original',label='grad' + str(i_grad),annot='max last',
                        normalize=1,colors=ddef(alpha=0.2),area=True)
        aY[i_grad] = aGrad[i_grad].values
    
    aY=array(aY)

    # ciklus az összes pontra
    X=list(serG.index)
    aPhase=[None]*len(X)
    for i_x in range(len(serG)):
        phase=0
        for i_grad in range(1,grad_max+1):
            if aY[i_grad,i_x]>0: phase += grad_max - (i_grad-1) 
            else: phase -= grad_max - (i_grad-1)

            # if i_grad==1: 
            #     if aY[1,i_x]>0: phase += 0.2
            #     else: phase -= 0.2
            # else:
            #     if aY[i_grad,i_x]>0: phase += 1
            #     else: phase -= 1
        aPhase[i_x]=phase
    
    serPhase=pd.Series(aPhase,X)

    if bPlot:
        FvPlot(serPhase,plttype='gauss',G=int(G*Gfaktor),label='phase' + str(i_grad),annot='max last',
                        normalize=1)
        pltshow()

    return 

def fn_erlang_boostwidth_samples():        # generate samples
    '''
    Samples:
        Superposition of 2-5 elementary lognorm curves
        The distance between them is [leftwidth*0.01:leftwidth] uniformly distributed
        The value of the first maxvalue is 1, the others randomly with multiplier [0.1:10].
    '''
    leftwidth=200
    kitevo=11
    
    aOut=[]
    for i in range(100):
        progress('Lognorm_multi ' + str(i))
        count=2   #  np.random.randint(3,7)
        maxpos=200
        maxvalue=1
        params=[]
        maxposs,maxvalues=[],[]
        for j in range(count):
            if j>0: 
                maxpos += leftwidth*(0.6 + np.random.rand()*0.4)     # (np.random.rand() + 0.1)*leftwidth
                maxvalue = 0.3     # 0.1 + np.random.rand()*1.9
            params.append(maxpos)
            params.append(maxvalue)
            params.append(leftwidth)

            maxposs.append(maxpos)
            maxvalues.append(maxvalue)

        X=Range(0,maxposs[-1] + 2*leftwidth,add=1)
        ser=pd.Series(fn_erlang_multi(X, *params, kitevo=kitevo),X)
        serGrad1=FvGradient(ser)
        serGrad2=FvGradient(serGrad1)
        # serGrad3=FvGradient(serGrad2)

        points_min,points_max = serlocalminmax(serGrad2,endpoints=False)        # a szélső pontokat ne tekintse szélsőérték helynek
        X_max,Y_max=unzip(points_max)
        X_max=list(X_max)
        Y_max=list(Y_max)
        
        X_min,Y_min=unzip(points_min)
        X_min=list(X_min)
        Y_min=list(Y_min)

        y_bands=[]
        maxposs=array(maxposs)
        for i_max in range(len(X_max)):
            boostwidth=np.nan       # np.nan jelzi, ha nem kell új részlognormot indítani a grad2 maxhelytől
            # Az első olyan részlognorm kell, ami felfutóban van
            #  - az összeolvadások miatt az i_max nem feltétlenül jelöli ki 
            arr=maxposs[maxposs>X_max[i_max]]
            if len(arr)>0: 
                boostwidth = (arr[0]-X_max[i_max])/leftwidth
                if boostwidth>1: boostwidth=np.nan

            gradfaktor=serGrad1[X_max[i_max]]/serGrad2[X_max[i_max]]

            lastmin_distance = np.nan
            if i_max>0: lastmin_distance=X_max[i_max] - X_min[i_max-1]

            # minden maxhelyhez írd fel a grad1/grad2 faktort
            aOut.append([maxposs,maxvalues,X_max[i_max],Y_max[i_max],gradfaktor,lastmin_distance,boostwidth])

            if pd.notna(boostwidth):
                caption=(strnum(boostwidth))
                y_bands.append(ddef(x1=X_max[i_max],x2=X_max[i_max]+boostwidth*leftwidth,color='green',alpha=0.07,
                        caption=caption,align='topcenter'))


        pltinit(suptitle='Lognorm superposition')

        FvPlot(ser,plttype='original',annot='localmax',annotcaption='x',colors=ddef(color='navy'))
        for j in range(count):
            paramsL=params[j*3:j*3+3]
            ser1=pd.Series(fn_erlang_multi(X, *paramsL, kitevo=kitevo),X)
            FvPlot(ser1,plttype='original',annot='',colors=ddef(color='gray',alpha=0.5))

        maxabs=abs(serGrad2).max()
        serGrad2=serGrad2/maxabs * 0.5  - 0.5
        FvPlot(serGrad2,plttype='original',annot='localmax localmin',annotcaption='x',colors=ddef(color='brown'))

        pltshow(
            commenttopright='Leftwidth=' + strint(leftwidth) + '\n'
                            'Exponent=' + strint(kitevo),
            y_bands=y_bands)


    tbl = pd.DataFrame(aOut,columns='maxposs,maxvalues,x_max,y_max,grad2_per_grad1,lastmin,boostwidth'.split(','))
    Tocsv(tbl,'boostwidth faktor',format='hu')
# fn_erlang_boostwidth_samples()
# exit()

def fn_plot_erlang_superposition():        # két számított lognorm görbe szuperpozíciója
    X=Range(0,6000)
    leftwidth=200       # az elemi felfutások letwidth értéke
    kitevo=9
    # maxpos1s = [200,1200,2200,3200,4200,5200]
    maxpos_diff_faktors=[0,0.25,0.5,0.75,1,1.8]

    maxvalue1 = 1
    maxvalue2 = 0.5

    aOut=[]
    y_bands=[]
    pltinit(suptitle='Lognorm superposition',
            title='Superposition of two elementary surges with different offsets')
    for i in range(len(maxpos_diff_faktors)):   # Range(0.5,2.1,add=0.5):
        X=Range(i*1000,(i+1)*1000)
        maxpos_diff=leftwidth*maxpos_diff_faktors[i]
        maxpos1=i*1000+200
        maxpos2=maxpos1 + maxpos_diff

        maxposs=array([maxpos1,maxpos2])

        ser=pd.Series(fn_erlang_multi(X, maxpos1,maxvalue1,leftwidth, maxpos2,maxvalue2,leftwidth, kitevo=kitevo),X)
        serGrad1=FvGradient(ser)
        serGrad2=FvGradient(serGrad1)
        # serGrad3=FvGradient(serGrad2)

        points_min,points_max = serlocalminmax(serGrad2,endpoints=False)        # a szélső pontokat ne tekintse szélsőérték helynek
        X_max,Y_max=unzip(points_max)
        X_max=list(X_max)
        Y_max=list(Y_max)
        
        X_min,Y_min=unzip(points_min)
        X_min=list(X_min)
        Y_min=list(Y_min)

        for i_max in range(len(X_max)):
            boostwidth=np.nan       # np.nan jelzi, ha nem kell új részlognormot indítani a grad2 maxhelytől
            # Az első olyan részlognorm kell, ami felfutóban van
            #  - az összeolvadások miatt az i_max nem feltétlenül jelöli ki 
            arr=maxposs[maxposs>X_max[i_max]]
            if len(arr)>0: 
                boostwidth = (arr[0]-X_max[i_max])/leftwidth
                if boostwidth>1: boostwidth=np.nan

            gradfaktor=serGrad1[X_max[i_max]]/serGrad2[X_max[i_max]]

            if pd.notna(boostwidth):
                caption=(strnum(boostwidth) + '\n' + 
                            strnum(gradfaktor))
                y_bands.append(ddef(x1=X_max[i_max],x2=X_max[i_max]+boostwidth*leftwidth,color='green',alpha=0.07,
                        caption='',align='topcenter'))
            
            # minden maxhelyhez írd fel a grad1/grad2 faktort
            aOut.append([X_max[i_max],Y_max[i_max],gradfaktor,boostwidth])


        # boostwidth=(X_min[0] - X_max[0])
        # leftwidth_faktor1=leftwidth/boostwidth
        # leftwidth_faktor2=np.nan
        # if len(X_min)>1:
        #     leftwidth_faktor2=leftwidth/(X_min[1] - X_max[1])


        FvPlot(ser,plttype='original',label='offset=' + strnum(maxpos_diff),annot='localmax',colors=ddef(color='navy'))
        ser1=pd.Series(fn_erlang_multi(X, maxpos1,maxvalue1,leftwidth, kitevo=kitevo),X)
        FvPlot(ser1,plttype='original',annot='',colors=ddef(color='gray',alpha=0.5))
        ser2=pd.Series(fn_erlang_multi(X, maxpos2,maxvalue2,leftwidth, kitevo=kitevo),X)
        FvPlot(ser2,plttype='original',annot='',colors=ddef(color='gray',alpha=0.5))

        annot=''
        if i==len(maxpos_diff_faktors)-1: annot='last'
        maxabs=abs(serGrad2).max()
        serGrad2=serGrad2/maxabs * 0.5  - 0.5
        FvPlot(serGrad2,plttype='original',label='second derivative',annot=annot,colors=ddef(color='brown'))

        # maxabs=abs(serGrad1).max()
        # serGrad1=serGrad1/maxabs * 0.5  - 0.5
        # FvPlot(serGrad1,plttype='original',label=strnum(maxpos2),annot='localmax',colors=ddef(color='gray'))

    # tbl = pd.DataFrame(aOut,columns='grad2_x,grad2_y,gradfaktor,boostwidth'.split(','))
    # Tocsv(tbl,'boostwidth faktor',format='hu')

    pltshow(
        commenttopright='Leftwidth=' + strint(leftwidth) + '\n'
                        'Exponent=' + strint(kitevo),
        y_bands=y_bands)
# fn_plot_erlang_superposition()
# exit()


def fn_plot_lognormal(variants='leftwidth'):
    X=Range(0,8,add=0.01)
    
    if variants=='leftwidth':
        pltinit()
        for leftwidth in [2,2.5,3]:
            modus=0.9
            ser_lognormal=pd.Series(fn_lognormal(X,maxpos=3,maxvalue=1,leftwidth=leftwidth,modus=0.9),X)
            FvPlot(ser_lognormal,plttype='original',label='leftwidth=' + strnum(leftwidth),annot='last')

    elif variants=='mu':
        pltinit()
        for mu in [0,0.1,0.2]:
            modus=0.9
            ser_lognormal=pd.Series(fn_lognormal(X,maxpos=3,maxvalue=1,leftwidth=2,modus=modus,mu=mu),X)
            FvPlot(ser_lognormal,plttype='original',label='mu=' + strnum(mu),annot='last')

    elif variants=='modus':
        pltinit('Log-normal functions with different modus',
                left=0.05,bottom=0.08,height=0.4,right=0.97)
        for modus in [0.8,0.9,0.95]:      #  [0.3,0.5,0.7]:
            ser_lognormal=pd.Series(fn_lognormal(X,maxpos=3,maxvalue=1,leftwidth=3,modus=modus),X)
            FvPlot(ser_lognormal,plttype='original',label='modus=' + strnum(modus),annot='4',area='colorline')

    pltshow()
# fn_plot_lognormal(variants='modus')
# exit()

def fn_compare_erlang_lognormal(fix='leftwidth'):
    '''
    matching of power and formfactor parameterization.
    Based on the fits, the approximate conversions are:
         power = 1 / ((1-mode)*2.34)
         mode = 1 - 1/(2.34*power)

         leftwidth_lognorm = leftwidth_erlang / 0.7 # depends a little on the modus
         leftwidth_erlang = leftwidth_lognorm * 0.7

    fixed: "modus" or "leftwidth"
    '''
    X=Range(0,8,add=0.01)
    pltinit(suptitle='Curve-fitting of log-normal function with Erlang')


    maxpos=2
    maxvalue=1

    if fix=='modus':
        modus=0.95         # kb power=9-nek felel meg
        for leftwidth in [1.5,2,2.5]:
            ser_lognormal=pd.Series(fn_lognormal(X,maxpos=maxpos,maxvalue=maxvalue,leftwidth=leftwidth,modus=modus),X)
            label='Lognormal lw=' + strnum(leftwidth) + ' modus=' + strnum(modus)
            
            annotplus=None
            if modus==0.8:  annotplus=[[0.6,'Lognormal'],[1.8,'Lognormal']]

            FvPlot(ser_lognormal,'original',label=label,color='green',annot='3',annotplus=annotplus,annotposplus='top left')

            power = 1 / ((1-modus)*2.34)       # közelítő számítás az illesztések alapján
            # leftwidth_lognorm = leftwidth *0.7     # közelítő számítás az illesztések alapján

            params0 = [power,maxpos,maxvalue,leftwidth]        # kitevo,maxpos,maxvalue,leftwidth
            lower = [2,maxpos-1,0,leftwidth/2] 
            upper = [100,maxpos+1,maxvalue+1,leftwidth*2]
            scale = [10,maxpos,maxvalue,leftwidth] 

            Y=ser_lognormal.values
            params=fn_erlang_fit(X,Y,params0,lower,upper,scale,kitevo='fit_all')
            ser = pd.Series(fn_erlang_multi(X,*params,kitevo='fit_all'),X)
            label='Erlang lw=' + strnum(params[3]) + ' power=' + strnum(params[0])
            FvPlot(ser,'original',label=label,color='blue',annot='3')
        pltshow()


    elif fix=='leftwidth':    
        leftwidth=2
        for modus in [0.8,0.9,0.98]:
            ser_lognormal=pd.Series(fn_lognormal(X,maxpos=2,maxvalue=1,leftwidth=leftwidth,modus=modus),X)
            label='Lognormal lw=' + strnum(leftwidth) + ' modus=' + strnum(modus)
            
            annotplus=None
            if modus==0.8:  annotplus=[[0.6,'Lognormal'],[1.8,'Lognormal']]

            FvPlot(ser_lognormal,'original',label=label,color='green',annot='3',annotplus=annotplus,annotposplus='top left')

            params0 = [9,3,1,leftwidth]        # kitevo,maxpos,maxvalue,leftwidth
            lower = [2,-3,0,leftwidth-1] 
            upper = [100,4,2,leftwidth+1]
            scale = [10,10,1,10] 

            Y=ser_lognormal.values
            params=fn_erlang_fit(X,Y,params0,lower,upper,scale,kitevo='fit_all')
            ser = pd.Series(fn_erlang_multi(X,*params,kitevo='fit_all'),X)
            label='Erlang lw=' + strnum(params[3]) + ' power=' + strnum(params[0])
            FvPlot(ser,'original',label=label,color='blue',annot='3')
        pltshow()
# fn_compare_erlang_lognormal(fix='leftwidth')
# exit()

def fn_plot_lognorm_superposition():        # két számított lognorm görbe szuperpozíciója
    X=Range(0,6000)
    leftwidth=300       # az elemi felfutások letwidth értéke
    kitevo=9
    # maxpos1s = [200,1200,2200,3200,4200,5200]
    maxpos_diff_faktors=[0,0.25,0.5,0.75,1,1.8]

    maxvalue1 = 1
    maxvalue2 = 0.5

    modus=0.95

    aOut=[]
    y_bands=[]
    pltinit(suptitle='Lognormal superposition',
            title='Superposition of two elementary surges with different offsets')
    for i in range(len(maxpos_diff_faktors)):   # Range(0.5,2.1,add=0.5):
        X=Range(i*1000,(i+1)*1000)
        maxpos_diff=leftwidth*maxpos_diff_faktors[i]
        maxpos1=i*1000+200
        maxpos2=maxpos1 + maxpos_diff

        maxposs=array([maxpos1,maxpos2])

        ser=pd.Series(fn_lognormal_multi(X, maxpos1,maxvalue1,leftwidth,modus, 
                                         maxpos2,maxvalue2,leftwidth,modus),X)
        serGrad1=FvGradient(ser)
        serGrad2=FvGradient(serGrad1)
        # serGrad3=FvGradient(serGrad2)

        points_min,points_max = serlocalminmax(serGrad2,endpoints=False)        # a szélső pontokat ne tekintse szélsőérték helynek
        X_max,Y_max=unzip(points_max)
        X_max=list(X_max)
        Y_max=list(Y_max)
        
        X_min,Y_min=unzip(points_min)
        X_min=list(X_min)
        Y_min=list(Y_min)

        for i_max in range(len(X_max)):
            boostwidth=np.nan       # np.nan jelzi, ha nem kell új részlognormot indítani a grad2 maxhelytől
            # Az első olyan részlognorm kell, ami felfutóban van
            #  - az összeolvadások miatt az i_max nem feltétlenül jelöli ki 
            arr=maxposs[maxposs>X_max[i_max]]
            if len(arr)>0: 
                boostwidth = (arr[0]-X_max[i_max])/leftwidth
                if boostwidth>1: boostwidth=np.nan

            gradfaktor=serGrad1[X_max[i_max]]/serGrad2[X_max[i_max]]

            if pd.notna(boostwidth):
                caption=(strnum(boostwidth) + '\n' + 
                            strnum(gradfaktor))
                y_bands.append(ddef(x1=X_max[i_max],x2=X_max[i_max]+boostwidth*leftwidth,color='green',alpha=0.07,
                        caption='',align='topcenter'))
            
            # minden maxhelyhez írd fel a grad1/grad2 faktort
            aOut.append([X_max[i_max],Y_max[i_max],gradfaktor,boostwidth])


        # boostwidth=(X_min[0] - X_max[0])
        # leftwidth_faktor1=leftwidth/boostwidth
        # leftwidth_faktor2=np.nan
        # if len(X_min)>1:
        #     leftwidth_faktor2=leftwidth/(X_min[1] - X_max[1])


        FvPlot(ser,plttype='original',label='offset=' + strnum(maxpos_diff),annot='localmax',colors=ddef(color='navy'))
        ser1=pd.Series(fn_lognormal_multi(X, maxpos1,maxvalue1,leftwidth,modus),X)
        FvPlot(ser1,plttype='original',annot='',colors=ddef(color='gray',alpha=0.5))
        ser2=pd.Series(fn_lognormal_multi(X, maxpos2,maxvalue2,leftwidth,modus),X)
        FvPlot(ser2,plttype='original',annot='',colors=ddef(color='gray',alpha=0.5))

        annot=''
        if i==len(maxpos_diff_faktors)-1: annot='last'
        maxabs=abs(serGrad2).max()
        serGrad2=serGrad2/maxabs * 0.5  - 0.5
        FvPlot(serGrad2,plttype='original',label='second derivative',annot=annot,colors=ddef(color='brown'))

        # maxabs=abs(serGrad1).max()
        # serGrad1=serGrad1/maxabs * 0.5  - 0.5
        # FvPlot(serGrad1,plttype='original',label=strnum(maxpos2),annot='localmax',colors=ddef(color='gray'))

    # tbl = pd.DataFrame(aOut,columns='grad2_x,grad2_y,gradfaktor,boostwidth'.split(','))
    # Tocsv(tbl,'boostwidth faktor',format='hu')

    pltshow(
        commenttopright='Leftwidth=' + strint(leftwidth) + '\n'
                        'Modus=' + strnum(modus),
        y_bands=y_bands)
# fn_plot_lognorm_superposition()
# exit()

def fn_lognorm_grad_plot():          # Lognormal és gradienseinek általános szemléletetése

    X=Range(0,3.5,add=0.001)

    pltinit('Lognormal function and its derivatives',nrows=3,ncols=1,width=0.6,height=0.9,bottom=0.04)
    for modus in (0.9,0.95,0.97):
        Y=fn_lognormal(X,1,1,1,modus)
        ser=pd.Series(Y,X)

        serGrad1=FvGradient(ser)
        serGrad2=FvGradient(serGrad1)
        # serGrad3=FvGradient(serGrad2)

        pltinitsub('next',title='Modus=' + strnum(modus))

        FvPlot(ser,'original',label='Lognormal peak',annot='localmax',area=True)
        FvPlot(serGrad1,'original',label='grad1',normalize=0.5,annot='localmax1 localmin1',annotcaption='grad1 x={x}')
        FvPlot(serGrad2,'original',label='grad2',normalize=0.5,annot='localmax2 localmin1',annotcaption='grad2 x={x}')
        # FvPlot(serGrad3,'original',label='grad3',normalize=1,annot='localmax localmin',annotcaption='grad3 x={x}')

        x_start = serGrad2.idxmax()
        v_lines=[ddef(x=0,caption='nullpos'),ddef(x=x_start,caption='startpos')]

        pltshowsub(v_lines=v_lines)
    pltshow()
# fn_lognorm_grad_plot()
# exit()



# GAUSST TEST
    
def fn_plot_gauss_illustration():
    pltinit(suptitle='Illustration of Gauss moving average',
            height=0.4,width=0.9,left=0.06,right=0.95,bottom=0.1,top=0.85,
            )

    # Keskeny kiugrás
    X=Range(0,300)
    Y = []
    for x in X: 
        if x==50: Y.append(1)
        elif x in [49,51]: Y.append(0.5)
        elif x>200: Y.append((x-200)/100)
        else: Y.append(0)
    ser=pd.Series(Y,X)
    FvPlot(ser,'original',color='0.8',annot='max 200 last')
    FvPlot(ser,'gauss',G=28,label='G=28',annot='localmax 200 last',color='blue')

    # serN=fn_moving_avg(ser,count=28)
    # FvPlot(serN,'original',label='normal average',annot='last',color='green')

    # X=Range(200,300)
    # Y = []
    # for x in X: Y.append((x-200)/100)
    # ser=pd.Series(Y,X)
    # FvPlot(ser,'original',color='0.8')
    # FvPlot(ser,'gauss',G=35,label='G=35',annot='last first',color='blue')

    tbl=Fn_read_coviddata()
    ser=Fn_coviddata_country(tbl,'Hungary')

    serG=FvGaussAvg(ser,gausswidth=28)['2021-09-01':'2022-02-20']
    serG=datefloat(serG)
    serG.index=serG.index - serG.index.min() + 300

    # serN=fn_moving_avg(ser,count=28)['2021-09-01':'2022-02-20']
    # serN=datefloat(serN)
    # serN.index=serN.index - serN.index.min() + 300


    ser = ser['2021-09-01':'2022-02-20']
    ser=datefloat(ser)
    ser.index=ser.index - ser.index.min() + 300
    FvPlot(serG,'original',label='final Gauss',annot='last',normalize=1,color='0.8')
    FvPlot(ser,'scatter',normfaktor='last',label='original',annot='max localmin',color='0.6')
    FvPlot(ser,'gauss',G=28,label='G=28',annot='max last',normalize=1,color='blue')
    # FvPlot(serN,'original',label='normal average',annot='max last',normfaktor='last',color='green')
    # FvPlot(ser,'gauss',G=14,label='G=28',annot='max',normalize=1,color='green')

    pltshow()
# fn_plot_gauss_illustration()
# exit()

def fn_plot_gauss_timescales(country):
    tbl=Fn_read_coviddata()
    ser=Fn_coviddata_country(tbl,country)

    pltinit(suptitle='Illustration of timescales',
            title=country,height=0.5,top=0.82)

    # FvPlot(ser,'scatter',label='original',annot='max',color='0.6')
    for G in [14,112]:
        FvPlot(ser,'gauss',G=G,label='G=' + str(G),annot='localmax5')
    pltshow(xtickerstep='year')
# fn_plot_gauss_timescales('United States')
# exit()

def sub_gausst_plot(country,days,G=20,Gfaktor=1.4,xmin=None,xmax=None):
    '''
    country: for which country (loose syntax)
    days: one or more closing days
         list or comma-separated enumeration
    G
    max_count: 300, 500, 1000
    xmin,xmax: limits of the entire curve (can be started from a rest period)

    '''
    tbl=Fn_read_coviddata()
    ser=serfromtbl(tbl,'new_cases_per_million',group='location:' + country,orderby='index',
               xmin=xmin,xmax=xmax)

    pltinit(suptitle='Lognorm illesztés',
            title=ser.name)

    # Eredeti görbe
    serGauss=FvGaussAvg(ser,G)
    FvPlot(serGauss,'original',annotate='',colors=ddef(color='gray'),normalize=1)

    normfaktor=config.normfaktorlastG

    # grad2 plot
    # maxabs=sub_grad2_shifted_plot(serGauss,G,Gfaktor)

    max_y=serGauss.max()
    v_lines=[]

    if type(days)==str: days=tokenize(days,',',['strip'])
    for i,day in enumerate(days):
        ser_fit=ser.loc[:day]
        serGauss=FvGaussAvg(ser_fit,G)
        FvPlot(serGauss.loc[fn_dateadd(day,-int(G/2)):],'original',normfaktor=normfaktor,
                annotate='',colors=ddef(color='green'))
        serGaussT=FvGaussTAvg(ser_fit,G,leftright='right',positive=True,bMirror=False)
        FvPlot(serGaussT.loc[fn_dateadd(day,-int(G/2)):],'original',normfaktor=normfaktor,
                annotate='last',label='T',colors=ddef(color='blue'))
        # serGaussTM=FvGaussTAvg(ser_fit,G,leftright='right',positive=True,bMirror=True)
        # FvPlot(serGaussTM.loc[fn_dateadd(day,-int(G/2)):],'original',normfaktor=normfaktor,
        #         annotate='last',label='TM',colors=ddef(color='navy'))
        # serGaussTE=FvGaussTAvg(ser_fit,G,leftright='right',fit_method='lognorm',
        #                        lognorm_params=ddef(leftwidth=100,leftwidth_delta=10,kitevo=20))
        # FvPlot(serGaussTE.loc[fn_dateadd(day,-int(G/2)):],'original',normfaktor=normfaktor,
        #         annotate='last',label='TE',colors=ddef(color='brown'))

        v_lines.append(ddef(x=day)) #,caption=day,align='top'))

    pltshow(annot_fontsize=8,
            v_lines=v_lines,
            )
# sub_gausst_plot('hungary',Range('2021-10-01',add=10,count=20),G=20,Gfaktor=1.4,xmin='2021-07-06',xmax='2022-06-12')
# sub_gausst_plot('hungary','2022-08-10',G=20,Gfaktor=1.4,xmin='2022-06-13',xmax='2023-07-09')
# exit()









# ============================================================================ #
# # KERAS MAXPOS MODELL
# ============================================================================ #
    
def fn_boostwidth_input(i_start,serGrad2,X_start,X_peak,params,serGauss,leftwidth_0,corefunction='lognormal'):  # input a KERAS modellhez
    '''
    Input data for the boostwidth train and for calculating the next boostwidth

     serGrad2: second derivative of serGauss
     X_start, X_peak:  max and min positions of serGrad2
     i_start:          in case of forward calculation, len(X_start)-1
     params:   if there is also a lognorm for the last start position (backward calculation), then the boostwidth_factor is also returned
         - the parameters are needed for every surges until i_start
     serGauss:  moving averaged input (resample is also recommended to make the max locations more accurate)
         - when teaching, for the entire epidemic period;   in the case of prediction, for the current day
         - in the case of prediction, the end is not yet final
     leftwidth_0: initial value of leftwidth
    '''

    p_len=4
    if corefunction=='gauss': p_len=3

    start_x = X_start[i_start]

    ser_max=serGauss.max()
    # ser_max=(serGauss.loc[serGauss.index<=start_x]).max()   # tanításkor is csak a felfutás indulásáig nezzük a max-értéket

    start_y = serGrad2[start_x] / ser_max
    start_y_gauss = serGauss[int(start_x)] / ser_max
    peak_x =np.nan
    if i_start<len(X_peak): peak_x = X_peak[i_start]

    nStarts=len(X_start)



    nSurges = len(params)//p_len
    maxpos,maxvalue,modus,power,leftwidth,boostwidth,boost_ratio = np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
    if i_start<nSurges:       # only at backward calculation (after the last start there is konwn another start)
        maxpos=params[p_len*i_start]
        maxvalue=params[p_len*i_start+1] / ser_max
        leftwidth=params[p_len*i_start+2]             # tuned data, too
        if corefunction=='lognormal': modus=params[p_len*i_start+3]
        elif corefunction=='erlang': power=params[p_len*i_start+3]

        boostwidth = maxpos - start_x
        boost_ratio = boostwidth / leftwidth        #  / leftwidth_mean    / leftwidth_0

        maxpos=Round(maxpos,2)      # informational data
        boostwidth=Round(boostwidth,2)      # informational data

    maxpos_last_x,descent_ratio,formfaktor,maxvalue_last=np.nan,np.nan,np.nan,np.nan
    start_last_y,peak_last_y,start_last_x,peak_last_x = np.nan,np.nan,np.nan,np.nan
    if i_start>0: 
        start_last_x = (start_x - X_start[i_start-1]) / leftwidth_0     # G-független skálázás miatt kell, ezért jó az átlagos leftwidth
        peak_last_x = (start_x - X_peak[i_start-1]) / leftwidth_0
        start_last_y = serGrad2[X_start[i_start-1]] / ser_max
        peak_last_y = serGrad2[X_peak[i_start-1]] / ser_max   

        # It may happen that the maxvalue of the last surge is 0. In this case, the one before it is the starting point
        i_last=i_start
        while i_last>=0:
            i_last -= 1
            maxpos_last=params[p_len*i_last]
            maxvalue_last=params[p_len*i_last+1]
            if maxvalue_last>0: break 
        maxpos_last_x = (start_x - maxpos_last) / leftwidth_0    # korábban G volt a nevezőben
        # - a maxpos_last lehet későbbi a start_x-nél, de mindenképpen ismert
        #   (predikciós esetben bizonytalanságot jelenthet, hogy az utolsó lezárt görbe
        #    illesztési görbéjének zárópontja még változhat és eleve egy tetőzés előtti
        #    görbéről van szó)
        descent_ratio=servalue(serGauss,X_start[i_start]) / maxvalue_last   #  serGauss[X_start[i_start]] / maxvalue_last
        formfaktor =(serGrad2[X_start[i_start]] - serGrad2[X_peak[i_last]])      \
                    / (serGrad2[X_start[i_last]] - serGrad2[X_peak[i_last]])
        maxvalue_last=maxvalue_last / ser_max


    return ddef2('grad2_start,grad2_peak,maxpos,maxvalue,modus,power,leftwidth,' +
                    'boostwidth,boost_ratio,' +
                    'maxpos_last_x,descent_ratio,formfaktor,maxvalue_last,' +
                    'start_y_gauss,start_y,start_last_y,peak_last_y,start_last_x,peak_last_x',
                 [start_x,peak_x,maxpos,maxvalue,modus,power,leftwidth,
                    boostwidth,boost_ratio,
                    maxpos_last_x,descent_ratio,formfaktor,maxvalue_last,
                    start_y_gauss,start_y,start_last_y,peak_last_y,start_last_x,peak_last_x])

def fn_maxpos_from_model(model,i_start,serGrad2,X_start,X_peak,lognormparams,serGauss,leftwidth,G): # maxpos számítása a KERAS modellel
    '''
    Returns the maxpos calculated based on the model (x_start + boostwidth_from_model)
         maxpos: the expected peak time of the elementary surge (float)

     model: if it is empty, the function reads it
     i_start:   for which surge
     serGrad2,X_start,X_peak,serGauss: calculated with fn_gauss and grad2() function
     lognormparams: in the case of forward calculation, an empty list can be entered
     G:         what G was calculated with serGauss, serGrad2, X_start, X_peak
     leftwidth: may vary slightly from country to country. Approximate value for Erlang model: 10 * sqrt(G) 

    '''

    # if bVersion:
    #     maxpos = X_start[i_start] + 0.377 * leftwidth     # átlagos boost_ratio
    #     return maxpos 

    inputs = fn_boostwidth_input(i_start,serGrad2,X_start,X_peak,lognormparams,serGauss,leftwidth)
    maxpos_last_x,descent_ratio,formfaktor,start_y,peak_last_x =  \
            dgets(inputs,'maxpos_last_x,descent_ratio,formfaktor,start_y,peak_last_x')
    if np.isnan(descent_ratio): boost_ratio=0.5
    else:
        if descent_ratio>20: descent_ratio=20
        if formfaktor>50: formfaktor=50
        rec = [G,maxpos_last_x,descent_ratio,formfaktor,start_y,peak_last_x]   # egy-rekordos
        boost_ratio = f_keras(rec,model)       
    maxpos = X_start[i_start] + boost_ratio * leftwidth

    return maxpos

def fn_train_boostwidth(train_count=3,verbose=1,leftwidth_type=None,flatten=True,
                        fname_fix='predict maxpos',val=0.1,testmodel_fname=None,
                        xmin_train=None,xmax_train=None,xmin_test=None,xmax_test=None):   # KERAS modell a következő tetőzési időpont előrjelzésére
    '''
    input: decomposition dataset
    
    leftwidth_type: defines the decomposition method

    train_count:  the number of trains, select and save the best, then test with the best
    '''

    

    # input minták:   decompose
    tbl_lognorms=Read_csv(Datapath('decompose_surges lognorm G_all leftw_' + leftwidth_type + 
                                        stringif(' flatten',flatten),'csv'))

    fname_fix += ' leftwidth_' + leftwidth_type +  stringif(' flatten',flatten)

    # NYERTES VÁLTOZAT
    #       weighted_loss=0.0661   loss=0.0883     (sample_weights=sqrt(maxvalue))
    # inputcols:  extended2
    # anomáliák:  felső értékkel limitálva
    # G:  all

    # for G in [14,28,56,112]:        # eredetileg egyben volt

    # hist_title = 'G = 14,21,28,35  overlap,  inputcols=extended_sub, weighted loss (sqrt(maxvalue)),  validation: Italy, Portugal'
    # title = 'G=' + str(G) + '  inputcols=extended_sub, weighted loss (sqrt(maxvalue)),  validation: Italy, Portugal'

    tbl=tbl_lognorms.copy()
    
    # tbl=tbl.loc[tbl.G==14]      # próbálkozás
    
    tbl = tbl.loc[pd.notna(tbl['descent_ratio'])]

    # tbl = tbl.loc[tbl.descent_ratio<20]     # a felette lévők 0.5 faktort kapnak
    # tbl = tbl.loc[tbl.formfaktor<50]        # a felette lévők 0.5 faktort kapnak
    tbl.loc[tbl.descent_ratio>20,['descent_ratio']]=20
    tbl.loc[tbl.formfaktor>50,['formfaktor']]=50

    tbl=tbl.loc[tbl.country!='South Africa']    # kiugróan rossz volt a dekompozíció
    # tbl=tbl.loc[tbl.start_y_gauss>0.05]    

    tbl['weight'] = np.sqrt(tbl['maxvalue'])        # sample_weight adatok minden rekordhoz


    tbl_all=tbl.copy()
    
    tbl_val=tbl.copy()
    if xmin_train or xmax_train: tbl_val=pd.DataFrame()       # konkatenációval kap értéket
    
    if xmin_train: 
        tbl=tbl.loc[tbl.day_maxpos>=xmin_train]
        tbl_val = Concat(tbl_val,tbl_all.loc[tbl_all.day_maxpos<xmin_train])
    if xmax_train: 
        tbl=tbl.loc[tbl.day_maxpos<=xmax_train]
        tbl_val = Concat(tbl_val,tbl_all.loc[tbl_all.day_maxpos>xmax_train])

    if val=='countries':
        tbl_val=tbl_val.loc[(tbl_val.country=='Italy') | (tbl_val.country=='Portugal')]
        tbl_train=tbl.loc[(tbl.country!='Italy') & (tbl.country!='Portugal') & (tbl.country!='South Africa')]
        validation=tbl_val
    else:
        tbl_train=tbl.loc[(tbl.country!='South Africa')]
        # - South Africa kilóg a sorból
        validation=val

    hist_subtitle = 'Train length = ' + str(len(tbl_train)) + '     Period = ' + str(xmin_train) + ' - ' + str(xmax_train)

    inputcols = ['G','maxpos_last_x','descent_ratio','formfaktor','start_y','peak_last_x']     # extended2, legjobb 
        # maxpos_last_x:  (start_x - maxpos_lastlognorm) / leftwidth       Az aktuális felfutás indulási helye az előző részlognorm csúcsához képest (negatív is lehet)
        # descent_ratio:  serGauss[start_x] / maxvalue_lastlognorm    Az előző felfutás csúcsértékének hányadrésze az aktuális felfutás starthelyének esetszáma
        # formfaktor:  (grad2[start_x] - grad2[peak_x_last])  / (grad2[start_x_last] - grad2[peak_x_last])
        # start_y:      grad2[start_x] / gauss.y_max        Az y_max a start_x-ig értendő (egy korábbi verzióban a járvány teljes időszakára vonakozott)    
        # peak_last_x:  (start_x - grad2[peak_x_last] / leftwidth   a grad2[peak_x_last] nem pontosan egyezik a maxpos_lastlognorm-gal (az utóbbi az illesztett részlognorm adata)
    # inputcols = ['G','maxpos_last_x','descent_ratio','formfaktor','start_y']   # standard
    # inputcols = ['G','maxpos_last_x','descent_ratio','formfaktor','start_y',
    #               'start_last_y','peak_last_y','start_last_x','peak_last_x']   # extended
    # inputcols = ['G','formfaktor','start_y',
    #               'start_last_y','peak_last_y','start_last_x','peak_last_x']   # extended_sub
    # inputcols = ['G','maxpos_last_x','start_y','start_last_y','peak_last_y','start_last_x','peak_last_x']
    # inputcols = ['start_y','start_last_y','peak_last_y','start_last_x','peak_last_x']

    def sub_sample_weights(tbl):
        # return tbl['maxvalue'].values
        return np.sqrt(tbl['maxvalue'].values)

    if train_count==0:
        model=load_model(Datapath(testmodel_fname,'keras'))

        tbl_history=None

    else:
        model,tbl_history = f_keras_train_and_save(tbl_train,inputcols,'boost_ratio',
                        train_count=train_count,
                        fname_fix=fname_fix,hist_subtitle=hist_subtitle,
                        validation=validation,weightcol='weight',          # validation=tbl_val
                        verbose=verbose,epochs=2000,patience=200)


    tbl_test=tbl_train
    X_test=tbl_test[inputcols].to_numpy()
    # X_test=tbl_test[['maxpos_last_x','descent_ratio','formfaktor','grad2_start_y']].to_numpy()
    Y_true=tbl_test['boost_ratio'].to_numpy()

    Y_pred = f_keras(X_test,model)

    sample_weights_train=sub_sample_weights(tbl_test)   # a nagyobb felfutások fontosabbak a hangoláskor, a nullások egyáltalán nem kellenek
    weighted_loss=mean_absolute_error(Y_true,Y_pred,sample_weight=sample_weights_train)
    loss=mean_absolute_error(Y_true,Y_pred)
    # print('Összes országra   G=' + str(G) + '   weighted_loss=' + strnum(weighted_loss) + '   loss=' + strnum(loss))
    print('Összes országra    weighted_loss=' + strnum(weighted_loss) + '   loss=' + strnum(loss))


    if xmin_test or xmax_test:
        for i in range(2):
            tbl_test=tbl_all.copy()
            if i==0:
                if xmin_train: tbl_test=tbl_test.loc[tbl_test.day_maxpos>=xmin_train]
                if xmax_train: tbl_test=tbl_test.loc[tbl_test.day_maxpos<=xmax_train]
                caption='On train (len=' + str(len(tbl_test)) + ' ' + str(xmin_train) + ' - ' + str(xmax_train) + ')'  
            elif i==1: 
                if xmin_test: tbl_test=tbl_test.loc[tbl_test.day_maxpos>=xmin_test]
                if xmax_test: tbl_test=tbl_test.loc[tbl_test.day_maxpos<=xmax_test]
                caption='On test (len=' + str(len(tbl_test)) + ' ' + str(xmin_test) + ' - ' + str(xmax_test) + ')'

            X_test=tbl_test[inputcols].to_numpy()
            Y_true=tbl_test['boost_ratio'].to_numpy()
            sample_weights=sub_sample_weights(tbl_test)   # a nagyobb felfutások fontosabbak a hangoláskor, a nullások egyáltalán nem kellenek

            Y_pred = f_keras(X_test,model)
            weighted_loss=mean_absolute_error(Y_true,Y_pred,sample_weight=sample_weights)
            print(caption + '   weighted_loss=' + strnum(weighted_loss))

    else:    
        for country in fn_countries('Europe+'):
            tbl_test=tbl_all.loc[tbl_all.country==country]    #  Italy és Portugal is
            if len(tbl_test)==0: continue   # South Africa
            X_test=tbl_test[inputcols].to_numpy()
            # X_test=tbl_test[['maxpos_last_x','descent_ratio','formfaktor','grad2_start_y']].to_numpy()
            # X_test=tbl_test[['maxpos_last_x','formfaktor']].to_numpy()
            Y_true=tbl_test['boost_ratio'].to_numpy()
            # sample_weights=tbl_test['maxvalue'].values   # a nagyobb felfutások fontosabbak a hangoláskor, a nullások egyáltalán nem kellenek
            sample_weights=sub_sample_weights(tbl_test)   # a nagyobb felfutások fontosabbak a hangoláskor, a nullások egyáltalán nem kellenek

            Y_pred = f_keras(X_test,model)
            weighted_loss=mean_absolute_error(Y_true,Y_pred,sample_weight=sample_weights)
            print(country + '   weighted_loss=' + strnum(weighted_loss))

    for G in [14,21,28,35]:
        tbl_test=tbl_all.copy()
        tbl_test=tbl_test.loc[tbl_test.G==G]
        if len(tbl_test)==0: continue
        caption='On all (len=' + str(len(tbl_test)) + '  G=' + str(G) + ')'  

        X_test=tbl_test[inputcols].to_numpy()
        Y_true=tbl_test['boost_ratio'].to_numpy()
        sample_weights=sub_sample_weights(tbl_test)   # a nagyobb felfutások fontosabbak a hangoláskor, a nullások egyáltalán nem kellenek

        Y_pred = f_keras(X_test,model)
        weighted_loss=mean_absolute_error(Y_true,Y_pred,sample_weight=sample_weights)
        print(caption + '   weighted_loss=' + strnum(weighted_loss))

# fn_train_boostwidth(train_count=3,verbose=0,fname_fix='predict maxpos by firsthalf lognorm',
#                     leftwidth_type='free',flatten=True,val=0.1,
#                     xmin_train=None,xmax_train='2021-06-30',xmin_test='2021-07-01',xmax_test=None)
# fn_train_boostwidth(train_count=3,verbose=0,fname_fix='predict maxpos by secondhalf lognorm',
#                     leftwidth_type='free',flatten=True,val=0.1,
#                     xmin_train='2021-07-01',xmax_train='2022-12-30',xmin_test=None,xmax_test='2021-06-30')


# fn_train_boostwidth(train_count=1,verbose=1,leftwidth_type='free',flatten=False,
#                     xmin_train=None,xmax_train='2021-06-30',xmin_test='2021-07-01',xmax_test=None)
# fn_train_boostwidth(train_count=5,verbose=0,fname_fix='predict maxpos, leftwidth tuned',
#                     xmin_train=None,xmax_train='2021-06-30',xmin_test='2021-07-01',xmax_test=None)
# fn_train_boostwidth(train_count=5,verbose=0,fname_fix='predict maxpos, leftwidth free',
#                     xmin_train=None,xmax_train='2021-06-30',xmin_test='2021-07-01',xmax_test=None)
# exit()

maxposmodel_firsthalf,maxposmodel_secondhalf = None,None
def fn_get_maxpos_model(day):
    global maxposmodel_firsthalf,maxposmodel_secondhalf
    if day is None: day=datefloat('2022-01-01')    # ha nincs megadva, akkor használja a második félidős modellt

    day=floatdate(day)
    # Az első félidőben a második félidő alapján előállított modellt kell használni
    if day<Date('2021-06-30'): 
        if maxposmodel_secondhalf is None:      
            maxposmodel_secondhalf=load_model(Datapath('predict maxpos by secondhalf lognorm leftwidth_tune flatten','keras'))
        return maxposmodel_secondhalf
    # A második félidőben az első félidős modell kell
    else:
        if maxposmodel_firsthalf is None:      
            maxposmodel_firsthalf=load_model(Datapath('predict maxpos by firsthalf lognorm leftwidth_tune flatten','keras'))
        return maxposmodel_firsthalf

def fn_depend_plot_boostwidth():
    tbl=Read_csv(Datapath('decompose_surges lognorm G_all leftw_tune flatten','csv'))

    depend_plot(tbl,'boost_ratio', 'maxpos_last_x,descent_ratio,formfaktor',style='ordered',scatter=True)
# fn_depend_plot_boostwidth()
# exit()




# ============================================================================ #
# # OPTIMAL DAYOFWEEK
# ============================================================================ #

def fn_save_optimal_dayofweek_by_country_and_time():    # lásd még: fn_get_optimal_dayofweek
    '''
    calculate the average loss for 8-day time-windows
    output:  weekly, the optimal day of the time window ending one week earlier  
         - a file is created that contains the optimal prediction day per week and per country

    Relatively time-consuming (requests a prediction for every country and every day, on a G14 time scale)
    '''
    tbl_covid=Fn_read_coviddata()
    # countries=fn_countries('Europe+')
    countries=['Germany']
    
    recs=[]     # out: [country,day,weekday_opt]
    print('Start')
    for country in countries:
        progress(country)
        ser_orig = Fn_coviddata_country(tbl_covid,country,backfill=False)

        dayfirst = datefloat(fn_monday('2020-03-01'))   
        daylast = datefloat(fn_monday('2022-12-31'))

        # loss számolása az összes lehetséges predikciós napra (a nyugalmi időszakokra is)
        APE,APE_peak,APE_peak_base,_ = \
                Fn_country_test(country,G=14,verbose=2,dayfirst=dayfirst,daylast=daylast,
                        dayofweek=None,interval=1,above=0,
                        tbl=tbl_covid,ser_orig=ser_orig,outtype='APE_all')

        losses,losses_base = [],[]         # naponként a 14-napos lognorm/base loss
        for i_testday in range(len(APE_peak)-14):    # a -14 ahhoz kell, hogy a loss értékek 14 napra rendelkezésre álljanak
            mape_peak = APE_peak[i_testday,:14].mean()
            mape_peak_base = APE_peak_base[i_testday,:14].mean()
            # lognorm_base=mape_peak / mape_peak_base
            losses.append(mape_peak)
            losses_base.append(mape_peak_base)
        losses=array(losses)
        losses_base=array(losses_base)

        # 8 hetes időablak, heti lépésekkel
        week_count = 8
        for i_week in range(len(losses)//7):        # ciklus az időablakokra (heti eltolásokkal)
            i_dayfirst = i_week*7    # a nap indexe a tesztidőszakon belül
            i_daylast = i_dayfirst + week_count*7   # az ablak utoló napja utáni nap indexe
            weekday_losses = []
            for weekday in range(7):
                loss=losses[i_dayfirst + weekday:i_daylast:7].sum()    # 7 naponként az ablak első napjától
                loss_base=losses_base[i_dayfirst + weekday:i_daylast:7].sum()
                weekday_losses.append(loss/loss_base)    
            weekday_opt = np.argmin(weekday_losses)
            # beírandó az ablak utáni második hét napjaihoz
            recs.append([country,dayfirst + i_daylast + 7,weekday_opt])

    tbl_out=TblFromRecords(recs,colnames='country,day,weekday_opt')
    Tocsv(tbl_out,'optimal_dayofweek_by_country_and_time',format='hu')
# fn_save_optimal_dayofweek_by_country_and_time()
# exit()

fname_optimal_dayofweek = 'optimal_dayofweek_by_country_and_time'
tbl_optimal_dayofweek = None
def fn_get_optimal_dayofweek(country,day):
    global tbl_optimal_dayofweek, fname_optimal_dayofweek
    if tbl_optimal_dayofweek is None:
        path=Datapath(fname_optimal_dayofweek,'csv')
        tbl_optimal_dayofweek = Read_csv(path)
    ser = serfromtbl(tbl_optimal_dayofweek,'weekday_opt',indexcol='day',group='country:"' + country + '"',orderby='index')
    i_before = ser.index.searchsorted(day)
    if i_before==0: return ser.iloc[0]      # ha a bal szél előtti nap, akkor a bal szélső érték
    elif i_before == len(ser):  return ser.iloc[-1]   # ha a jobb szél utáni nap, akkor a jobb szélső érték
    else: return ser.iloc[i_before-1]

def fn_optimal_dayofweek(countries,method='loss14'):    # OLD METHOD
    '''
    method:
        'loss14':   based on the loss on G14
        'gauss':    based on the greatest distance form the Gaussian moving average
    '''
    if type(countries)==str: countries = countries.split(',')
    
    tbl_covid=Fn_read_coviddata()

    recs=[]
    for country in countries:
        if method=='loss14': 
            ser_orig = Fn_coviddata_country(tbl_covid,country,backfill=False)
            count_null = len(ser_orig.loc[ser_orig==0])
            nullpercent = count_null / len(ser_orig)
            print(country + '   nullpercent=' + strnum(nullpercent,'2%'))

            losses=[]
            for dayofweek in range(7):
                progress('dayofweek=' + str(dayofweek))
                lognorm_base = Fn_country_test(country,G=14,verbose=0,interval=14,dayofweek=dayofweek,
                                                         tbl=tbl_covid,ser_orig=ser_orig,outtype='lognorm_base')
                losses.append(lognorm_base)
            losses=array(losses)
            dayofweek_min = np.argmin(losses)
            dayofweek_max = np.argmax(losses)
            loss_min=losses[dayofweek_min]
            loss_max=losses[dayofweek_max]
            print(country + '   dayofweek_min=' + str(dayofweek_min) +
                '   lognorm_base_G14_min=' + strnum(loss_min,'2%') +
                '   lognorm_base_G14_max=' + strnum(loss_max,'2%')
                )
            recs.append([country,dayofweek_min,loss_min,loss_max,nullpercent,
                         datestr(ser_orig.index[0]),datestr(ser_orig.index[-1])])
        
        elif method=='gauss':
            ser = Fn_coviddata_country(tbl_covid,country,backfill=True)
            serGauss = FvGaussAvg(ser,gausswidth=14)
            diffmin=1000
            for dayofweek in range(7):
                serL = ser.loc[ser.index.weekday==dayofweek]
                serGaussL = serGauss.loc[serGauss.index.weekday==dayofweek]
                diff = (abs(serGaussL - serL)/serGaussL).mean()
                if diff<diffmin: 
                    optimal = dayofweek
                    diffmin = diff
                if len(countries)==1: print(country + '  dayofweek=' + str(dayofweek) + '   diff=' + strnum(diff,'2%'))
            print(country + '   Optimális: ' + str(optimal))
    if len(recs)>0:
        tbl_out=pd.DataFrame(recs,columns=
            'country,dayofweek_min,lognorm_base_G14_min,lognorm_base_G14_max,nullpercent,firstday,lastday'.split(','))
        Tocsv(tbl_out,'optimal dayofweek by countries',format='hu')
# fn_optimal_dayofweek(fn_countries('Europe+'))
# fn_optimal_dayofweek('South Korea')
# exit()

def fn_dayofweek_countries():
    '''
    usage:  
        countryToDayofweek = fn_dayofweek_countries()
        countryToDayofweek[country]
    '''
    tbl=Read_csv(Datapath('optimal dayofweek by countries','csv'),index_col='country')
    return tbl['dayofweek_min']

def fn_plot_dayofweek_losses(country):      # NEM TÚL HASZNOS   Hét napjaira eső mape eloszlások, árnyalásos diagrammal
    tbl_covid=Fn_read_coviddata()
    ser_orig=Fn_coviddata_country(tbl_covid,country,backfill=False)
    
    ser = fillna_back(ser_orig)
    ser_gauss14 = FvGaussAvg(ser,gausswidth=14) 
    ser_max14=ser_gauss14.max()


    pltinit()

    dayfirst='2020-06-01'
    dayfirst=datestr(fn_monday(Date(dayfirst)))

    for day in Range(dayfirst,'2022-12-31',add=7):
        # csak azok a hetek kellenek, amelyeknek a zárónapján kellően magas volt az esetszám
        if ser_gauss14[day]<ser_max14*0.1: continue

        APE = Fn_country_test(country,G=14,xmax=day,
                            interval=1,count_back=7,verbose=1,plot=False,outtype='APE',
                            tbl=tbl_covid,ser_orig=ser_orig)
        dayofweek_means = np.mean(APE[:,:14],axis=1)   # APE_peak[i_testday][i_dayafter]

        FvPlot(pd.Series(dayofweek_means),'regauss',G=0.1,resample=10,area=True,annot='')

    pltshow()
# fn_plot_dayofweek_losses('United States')
# exit()



# ============================================================================ #
# # TIMESCALE WEIGHTS
# ============================================================================ #

def fn_normalized_loss_by_G_and_daysafter(plot_weighted=False):
    path=Datapath('base_losses lognorm','csv')
    tbl_base = Read_csv(path)
    tbl_base = tbl_base.loc[tbl_base.daysafter<=40]

    # Súlyozott Lognorm
    tbl_lognorm=Read_csv(Datapath('loss_train lognorm FINAL','csv'))
    # daysafter=0 valójában 1 nappal az utolsó adat napja után van
    tbl_lognorm['dayafter'] = tbl_lognorm['dayafter']+1
    # tbl_lognorm = tbl_lognorm.loc[tbl_lognorm.dayafter<=40]

    # multiG Lognorms
    tbl_lognorm_multi=Read_csv(Datapath('loss_train lognorm multiG FINAL','csv'))
    # daysafter=0 valójában 1 nappal az utolsó adat napja után van
    tbl_lognorm_multi['dayafter'] = tbl_lognorm_multi['dayafter']+1

    count_pred = len(tbl_lognorm)/50

    pltinit(suptitle='Normalized loss by G and daysafter',
            title = 'The base of the normalization is the mean of the four curve',
            left=0.06,right=0.95,wspace=0.23,top=0.79,bottom=0.1,width=0.9,
            ncols=3,nrows=1,sharey=False)

    v_lines = [ddef(x=14,caption='day 14'),ddef(x=35,caption='day 35')]


    pltinitsub('next',
               title='Mean APE (MAPE)')

    ser_allG = serfromtbl(tbl_lognorm_multi,'mape','dayafter',aggfunc='mean')
    # FvPlot(ser_allG,'original',label='Lognorm allG',annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')

    # ser_base = serfromtbl(tbl_base,'MAPE_base','daysafter')
    # FvPlot(ser_base,'original',label='base',annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')


    for G in [14,21,28,35]:
        tbl_lognorm_G=tbl_lognorm_multi.loc[tbl_lognorm_multi['G']==G]
        serG = serfromtbl(tbl_lognorm_G,'mape','dayafter',aggfunc='mean')
        FvPlot(serG/ser_allG,'original',label='G=' + str(G),annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')

    if plot_weighted:
        ser_weighted = serfromtbl(tbl_lognorm,'mape','dayafter',aggfunc='mean')
        FvPlot(ser_weighted/ser_allG,'original',label='Lognorm weighted',annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')


    pltshowsub(ymin=0.5,ymax=1.5,xlabel='days after last known data',ynumformat='0%',v_lines=v_lines,
               commenttopright_out='outlier-sensitive view',area_under=1)


    pltinitsub('next',
               title='Median APE (MDAPE)')

    ser_allG = serfromtbl(tbl_lognorm_multi,'mape','dayafter',aggfunc='median')
    # FvPlot(ser_allG,'original',label='Lognorm allG',annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')

    # ser_base = serfromtbl(tbl_base,'MDAPE_base','daysafter')
    # FvPlot(ser_base,'original',label='base',annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')

    for G in [14,21,28,35]:
        tbl_lognorm_G=tbl_lognorm_multi.loc[tbl_lognorm_multi['G']==G]
        serG = serfromtbl(tbl_lognorm_G,'mape','dayafter',aggfunc='median')
        FvPlot(serG/ser_allG,'original',label='G=' + str(G),annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')

    if plot_weighted:
        ser_weighted = serfromtbl(tbl_lognorm,'mape','dayafter',aggfunc='median')
        FvPlot(ser_weighted/ser_allG,'original',label='Lognorm weighted',annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')

    pltshowsub(ymin=0.5,ymax=1.5,xlabel='days after last known data',ynumformat='0%',v_lines=v_lines,
               commenttopright_out='bottom view, ignoring outliers',area_under=1)



    # tbl_lognorm_multi['APE_base'] = abs(tbl_lognorm_multi['y_base_n'] - tbl_lognorm_multi['y_true_n']) / tbl_lognorm_multi['y_true_n']
    # tbl_lognorm_multi['APEmax_base'] = abs(tbl_lognorm_multi['y_base_n'] - tbl_lognorm_multi['y_true_n'])     # eleve osztva van a maxértékkel

    # tbl_lognorm_multi['APE_linear'] = abs(tbl_lognorm_multi['y_linear_n'] - tbl_lognorm_multi['y_true_n']) / tbl_lognorm_multi['y_true_n']
    # tbl_lognorm_multi['APEmax_linear'] = abs(tbl_lognorm_multi['y_linear_n'] - tbl_lognorm_multi['y_true_n'])     # eleve osztva van a maxértékkel

    # tbl_lognorm_multi['APE_lognorm'] = abs(tbl_lognorm_multi['y_pred_n'] - tbl_lognorm_multi['y_true_n']) / tbl_lognorm_multi['y_true_n']

    tbl_lognorm['APEmax_lognorm'] = abs(tbl_lognorm['y_pred_n'] - tbl_lognorm['y_true_n'])     # eleve osztva van a maxértékkel
    tbl_lognorm_multi['APEmax_lognorm'] = abs(tbl_lognorm_multi['y_pred_n'] - tbl_lognorm_multi['y_true_n'])     # eleve osztva van a maxértékkel


    pltinitsub('next',
               title='MAPEmax')

    ser_allG = serfromtbl(tbl_lognorm_multi,'APEmax_lognorm','dayafter',aggfunc='mean')

    for G in [14,21,28,35]:
        tbl_lognorm_G=tbl_lognorm_multi.loc[tbl_lognorm_multi['G']==G]

        serG = serfromtbl(tbl_lognorm_G,'APEmax_lognorm','dayafter',aggfunc='mean')
        FvPlot(serG/ser_allG,'original',label='G=' + str(G),annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')

    if plot_weighted:
        ser_weighted = serfromtbl(tbl_lognorm,'APEmax_lognorm','dayafter',aggfunc='mean')
        FvPlot(ser_weighted/ser_allG,'original',label='Lognorm weighted',annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')

    pltshowsub(ymin=0.5,ymax=1.5,xlabel='days after last known data',ynumformat='0%',v_lines=v_lines,
               commenttopright_out='top view',area_under=1)


    pltshow()     
# fn_normalized_loss_by_G_and_daysafter(plot_weighted=False)
# exit()    


def fn_normalized_loss_by_country_G_and_daysafter(countries):
    if type(countries)==str: countries=countries.split(',')

    # multiG Lognorms
    tbl_loss=Read_csv(Datapath('loss_train lognorm multiG FINAL','csv'))
    # daysafter=0 valójában 1 nappal az utolsó adat napja után van
    tbl_loss['dayafter'] = tbl_loss['dayafter']+1

    count_pred = len(tbl_loss)/50

    pltinit(suptitle='Normalized loss by G and daysafter',
            title = 'The base of the normalization is the mean of the four curve',
            left=0.06,right=0.95,wspace=0.23,top=0.79,bottom=0.1,width=0.9,
            subcount=len(countries),sharey=False)

    v_lines = [ddef(x=14,caption='day 14'),ddef(x=35,caption='day 35')]

    for country in countries:
        pltinitsub('next',title=country)

        tbl_loss_country = tbl_loss.loc[tbl_loss.country==country]

        ser_allG = serfromtbl(tbl_loss_country,'mape','dayafter',aggfunc='mean')

        for G in [14,21,28,35]:
            tbl_loss_G=tbl_loss_country.loc[tbl_loss_country['G']==G]
            serG = serfromtbl(tbl_loss_G,'mape','dayafter',aggfunc='mean')
            FvPlot(serG/ser_allG,'original',label='G=' + str(G),annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')

        pltshowsub(ymin=0.5,ymax=1.5,xlabel='days after last known data',ynumformat='0%',v_lines=v_lines,
                commenttopright_out='by MAPE (outlier-sensitive view)',area_under=1)

    pltshow()
# fn_normalized_loss_by_country_G_and_daysafter('Portugal,Sweden')
# exit()    


def fn_normalized_loss_by_period_G_and_daysafter():
    # multiG Lognorms
    tbl_loss=Read_csv(Datapath('loss_train lognorm multiG FINAL','csv'))
    # daysafter=0 valójában 1 nappal az utolsó adat napja után van
    tbl_loss['dayafter'] = tbl_loss['dayafter']+1

    count_pred = len(tbl_loss)/50

    untils = ['2020-12-31','2021-06-30','2021-12-31','2022-06-30','2022-12-31','2023-06-30','last']
    #        # - ha az utolsó helyen "all" áll, akkor egyre növekvő időszak

    pltinit(suptitle='Normalized loss by G and daysafter',
            title = 'The base of the normalization is the mean of the four curve',
            left=0.06,right=0.95,wspace=0.23,top=0.82,bottom=0.075,width=0.65,height=0.95,
            nrows=3,ncols=2,sharey=False)

    v_lines = [ddef(x=14,caption='day 14'),ddef(x=35,caption='day 35')]

    for i_until,until in enumerate(untils):     
        if until=='last':  
            tbl_period = tbl_loss.loc[tbl_loss.day>datefloat(untils[i_until-1])]
            title = 'Period:  ' + datestr(datefloat(untils[i_until-1])+1) + ' -'
        else:
            if i_until==0: 
                tbl_period = tbl_loss.loc[tbl_loss.day<=datefloat(until)]
                title = 'Period:  until ' + datestr(untils[i_until])
            else: 
                tbl_period=tbl_loss.loc[(tbl_loss.day>datefloat(untils[i_until-1])) & (tbl_loss.day<=datefloat(until))]
                title = 'Period:  ' + datestr(datefloat(untils[i_until-1])+1) + ' - ' + datestr(datefloat(untils[i_until]))[5:]

        try:
            pltinitsub('next',title=title)
        except: break

        ser_allG = serfromtbl(tbl_period,'mape','dayafter',aggfunc='mean')

        for G in [14,21,28,35]:
            tbl_loss_G=tbl_period.loc[tbl_period['G']==G]
            serG = serfromtbl(tbl_loss_G,'mape','dayafter',aggfunc='mean')
            FvPlot(serG/ser_allG,'original',label='G=' + str(G),annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')

        pltshowsub(ymin=0.5,ymax=1.5,xlabel='days after last known data',ynumformat='0%',v_lines=v_lines,
                commenttopright_out='by MAPE',area_under=1)

    pltshow()
# fn_normalized_loss_by_period_G_and_daysafter()
# exit()    

def fn_compare_weighted():
    # Weighted lognorm
    tbl_lognorm=Read_csv(Datapath('loss_train lognorm FINAL','csv'))
    # daysafter=0   is one day after the last known input data
    tbl_lognorm['dayafter'] = tbl_lognorm['dayafter']+1
    # tbl_lognorm = tbl_lognorm.loc[tbl_lognorm.dayafter<=40]

    # multiG Lognorms
    tbl_lognorm_multi=Read_csv(Datapath('loss_train lognorm multiG FINAL','csv'))
    # daysafter=0   is one day after the last known input data
    tbl_lognorm_multi['dayafter'] = tbl_lognorm_multi['dayafter']+1

    count_pred = len(tbl_lognorm)/50

    pltinit(suptitle='Comparison of weighted prediction with the original predictions',
            title = 'The base of the normalization is the mean of the four original curve',
            left=0.06,right=0.95,wspace=0.23,top=0.79,bottom=0.1,width=0.9,
            ncols=2,nrows=1,sharey=False,width_ratios=(1,2))

    v_lines = [ddef(x=14,caption='day 14'),ddef(x=35,caption='day 35')]


    pltinitsub('next',
               title='MAPE, normalized view')

    ser_allG = serfromtbl(tbl_lognorm_multi,'mape','dayafter',aggfunc='mean')
    # FvPlot(ser_allG,'original',label='Lognorm allG',annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')

    # ser_base = serfromtbl(tbl_base,'MAPE_base','daysafter')
    # FvPlot(ser_base,'original',label='base',annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')


    for G in [14,21,28,35]:
        tbl_lognorm_G=tbl_lognorm_multi.loc[tbl_lognorm_multi['G']==G]
        serG = serfromtbl(tbl_lognorm_G,'mape','dayafter',aggfunc='mean')
        FvPlot(serG/ser_allG,'original',label='G=' + str(G),annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')

    ser_weighted = serfromtbl(tbl_lognorm,'mape','dayafter',aggfunc='mean')
    FvPlot(ser_weighted/ser_allG,'original',label='Lognorm weighted',color='black',
           annotcaption=ddef(last='middle',xpos='y%'),annot='middle 1 14 35')


    pltshowsub(ymin=0.5,ymax=1.5,xlabel='days after last known data',ynumformat='0%',v_lines=v_lines,
               area_under=1)


    pltinitsub('next',
               title='MAPE')


    for G in [14,21,28,35]:
        tbl_lognorm_G=tbl_lognorm_multi.loc[tbl_lognorm_multi['G']==G]
        serG = serfromtbl(tbl_lognorm_G,'mape','dayafter',aggfunc='mean')
        FvPlot(serG,'original',label='G=' + str(G),annotcaption=ddef(last='label',xpos='y%'),annot='last 1 14 35')

    ser_weighted = serfromtbl(tbl_lognorm,'mape','dayafter',aggfunc='mean')
    FvPlot(ser_weighted,'original',label='Lognorm weighted',color='black',
           annotcaption=ddef(middle='label',xpos='y%'),annot='middle 1 14 35')

    pltshowsub(ymin=0,xlabel='days after last known data',ynumformat='0%',v_lines=v_lines,
               area_under=1)


    pltshow()     
# fn_compare_weighted()
# exit()




def fn_G_weights_new(lossfile='loss_train lognorm multiG FINAL',plot=False,csv=True, lossfield='mape'):  # G_weights fájl létrehozása, plot
    '''
    lossfile:    created with fn_test_all function  (for separate G values)
    lossfield:  'mape',  'APEmax_lognorm'
    '''
    path=Datapath(lossfile,'csv')
    tbl=Read_csv(path)

    tbl['APEmax_lognorm'] = abs(tbl['y_pred_n'] - tbl['y_true_n'])     # eleve osztva van a maxértékkel


    recs=[]

    untils = ['2020-12-31','2021-06-30','2021-12-31','2022-06-30','2022-12-31','2023-06-30','last']
    #        # - ha az utolsó helyen "all" áll, akkor egyre növekvő időszak

    # untils = ['2021-06-30','last']      
    # - ha az utolsó helyen "last" áll, akkor diszjunkt időszakok

    # untils = ['all']      

    for i_until,until in enumerate(untils):     

        # # 0: all  1: firsthalf,  2: secondhalf
        # if period==0: tbl_period=tbl.copy()
        # elif period==1: tbl_period=tbl.loc[tbl.day<=day_half]
        # elif period==2: tbl_period=tbl.loc[tbl.day>day_half]

        if untils[-1]=='last':   # diszjunkt időszakok
            if until=='last':  tbl_period = tbl.loc[tbl.day>datefloat(untils[i_until-1])]
            else:
                if i_until==0: tbl_period = tbl.loc[tbl.day<=datefloat(until)]
                else: tbl_period=tbl.loc[(tbl.day>datefloat(untils[i_until-1])) & (tbl.day<=datefloat(until))]

        elif untils[-1]=='all':      # egyre növekvő időszakok
            if until=='all': tbl_period=tbl.copy()
            else: tbl_period=tbl.loc[tbl.day<=datefloat(until)]
        
        countries=fn_countries('Europe+') + ['all']     # összes országra is számolandó
        for country in countries:
            progress(country + '  until=' + str(until))
            if country == 'all': tbl1=tbl_period.copy()        # összes ország
            else: tbl1 = tbl_period.loc[tbl_period.country==country]
            for dayafter in range(50):
                tbl2=tbl1.loc[tbl1.dayafter==dayafter]
                mean = tbl2[lossfield].mean()   # adott országra és dayafter-re
                weight_sum=0
                for G in [14,21,28,35]:
                    tbl3=tbl2.loc[tbl2.G==G]
                    weight=( mean / tbl3[lossfield].mean() )**3
                    recs.append([country,until,dayafter,G,weight])
                    weight_sum += weight
                recs.append([country,until,dayafter,0,weight_sum])    # a G==0 rekord tartalmazza a súlyok összegét

    tbl_out=TblFromRecords(recs,colnames='country,until,dayafter,G,weight')
    if csv: Tocsv(tbl_out,'G_weights',format='hu')
    
    if plot:
        # countries=['Hungary']
        # countries = ['South Africa','Australia']
        countries = ['all']
        for country in countries:
            tbl=tbl_out.loc[tbl_out.country==country].copy()
            pltinit(suptitle='G weights, country=' + country,ncols=2,nrows=3,sharey=True)
            for until in untils:
                try: 
                    pltinitsub('next',title='until ' + str(until))
                except: break
                tbl_period=tbl.loc[tbl.until==until]
                for G in [14,21,28,35]:        
                    tblL=tbl_period.loc[tbl_period.G==G]
                    ser=serfromtbl(tblL,'weight','dayafter')
                    FvPlot(ser,'original',label='G=' + str(G),annot='max first last')
                pltshowsub(xlabel='days after last known data',commenttopright_out=str(len(tbl_period)) + ' record')
            pltshow()
# fn_G_weights_new(plot=False,csv=True)     # utolsó G_all
# exit()

G_weights_byperiods, G_weights_untils = None,None
def sub_get_G_weights(day,flatten=True):        # fn_lognorm_test2 hívja
    '''
    day:  float or datetime   (date of prediction)
    '''
    global G_weights_byperiods,G_weights_untils
    if G_weights_byperiods is None:
        tbl_weights = Read_csv(Datapath('G_weights lognorm FINAL','csv'))
        tbl_country=tbl_weights.loc[tbl_weights.country=='all']     # common for all country
        # reading until values
        G_weights_untils = sorted(tbl_country.until.unique())         # including "all" or "last"

        G_weights_byperiods = []     # three dimension:   G_weights_byperiods[i_until][iG][i_day]      len(G)+1 columns,   iG=len(aG) column contains the sum of the weights factors
        for until in G_weights_untils:      
            tbl_period=tbl_country.loc[tbl_country.until==until]
            weights = []
            # weights:  two dimension    weights[iG][i_day]      len(G)+1 oszlop, az iG=len(aG) oszlop a súlyfaktorok összegét tartalmazza
            for G in [14,21,28,35]:
                weights.append(tbl_period.loc[tbl_period.G==G].sort_values('dayafter')['weight'].values)
            weights.append(tbl_period.loc[tbl_period.G==0].sort_values('dayafter')['weight'].values)   # súly-összegek
            G_weights_byperiods.append(weights)            

    day=datestr(floatdate(day),'c10-')        # 'yyyy-MM-dd' format

    if G_weights_untils[-1]=='all':    # increasing periods
        for i_until,until in enumerate(G_weights_untils):
            if day>until: return G_weights_byperiods[i_until]
            # - az until értékek rendezettek;  a legkorábbi olyan találat kell, amelyre day>until

        # Ha nem volt találat, tehát a days korábbi mindegyik until értéknél, akkor az 'all' érvényesül
        return G_weights_byperiods[-1]       # until='all' áll az utolsó helyen

    elif G_weights_untils[-1]=='last':      # diszjunkt periódusok
        bVersionL=True
        if bVersionL:        # kettővel korábbi periódusból vegye (fél éves periódusok, éves szezonalitás)
            for i_until,until in enumerate(G_weights_untils):
                if day<until: 
                    if i_until>=2: i_until -= 2     # kettővel korábbi időszak
                    else: i_until += 2       # ha nincs, akkor kettővel későbbi időszak
                    return G_weights_byperiods[i_until]
                # - az until értékek rendezettek;  a legkorábbi olyan találat kell, amelyre day>until

            # Elvileg nem kerülhet ide, mert a "last"-nál minden dátumstring kisebb. Ha mégis ...
            return G_weights_byperiods[0]       
        else:
            for i_until,until in enumerate(G_weights_untils):
                if day>until: return G_weights_byperiods[i_until]
                # - until values are ordered;   the first hit, where day>until

            # if there was no hit,  so the days is earlier as all until value, then then second period
            #   (the essence, that it would be in future)
            return G_weights_byperiods[1]       




# ============================================================================ #
# LEFTWIDTH BY COUNTRIES AND TIMESCALES
# ============================================================================ #

def fn_leftwidth_stats(flatten=False):
    '''
    the average of free-leftwidth by countries,   and the tuned leftwidth-mean
    G értékenként a free-leftwidth átlag, és a hangolt leftwidth átlag
    '''
    tbl_free = Read_csv(Datapath('decompose_surges lognorm G_all leftw_free' + stringif(' flatten',flatten),'csv'))
    tbl_tune = Read_csv(Datapath('decompose_surges lognorm G_all leftw_tune' + stringif(' flatten',flatten),'csv'))

    print('\n' + 'BY COUNTRIES')
    for country in fn_countries('Europe+'):
        tbl = tbl_free.loc[tbl_free.country==country]
        mean_free = tbl.leftwidth.mean()       # elemi lognormokra  (mindegyik G-re)
        tbl = tbl_tune.loc[tbl_tune.country==country]
        mean_tune = tbl.leftwidth.mean()       # G értékekre átlagol
        print(country + '  mean_free=' + strnum(mean_free) + '  mean_tune=' + strnum(mean_tune))

    print('\n' + 'BY G')
    for G in [14,21,28,35]:
        tbl = tbl_free.loc[tbl_free.G==G]
        mean_free = tbl.leftwidth.mean()       # elemi lognormokra  (mindegyik országra)
        tbl = tbl_tune.loc[tbl_tune.G==G]
        mean_tune = tbl.leftwidth.mean()       # országokra átlagol
        print(str(G) + '  mean_free=' + strnum(mean_free) + '  mean_tune=' + strnum(mean_tune))
    
# fn_leftwidth_stats()
# exit()

def fn_plot_leftwidth_bytime(country,G,flatten=True):
    '''
    country:  'all'  or a special country
    G:  'all'  14, 21, 28 35
    '''


    # Háttérinformációként az átlagos esetszám idősora
    ser_mean=fn_ser_cases_mean()

    pltinit(suptitle='Change of the average leftwidth during the pandemic',
            title='Country=' + country + '   G=' + str(G),
            left=0.09,right=0.88,bottom=0.08,top=0.85)
    FvPlot(ser_mean,'original',label='cases_normalized',annot='max last',normalize=140,area=True)

    tbl=Readcsv(Datapath('decompose_surges lognorm G_all leftw_free' + stringif(' flatten',flatten),'csv'),format='hu')

    if G!='all': tbl=tbl.loc[tbl.G==G]

    # tbl=tbl.loc[tbl.start_y_gauss>0.01]
    tblL=tbl.copy()
    if country!='all': tblL=tblL.loc[tbl.country==country]
    ser = serfromtbl(tblL,'leftwidth',indexcol='grad2_start',aggfunc='mean',orderby='index')
    ser = ser.loc[(ser<140) & (ser>5)]      # outlier szűrés

    # A mozgóátlagolás count paraméteréhez kell a pontsűrűség a dátumtengelyen
    days = int(ser.index.max()) - int(ser.index.min())
    len_per_day = (len(ser)-1) / days

    # ser=floatdate(ser)
    FvPlot(ser,'scatter regauss',G=180*len_per_day,label='GaussMA_180_centered',annot='gaussabs max last',
        annotcaption=ddef(gaussabs='y',max='GaussMA'),color='gray')


    # ser = FvLinearResample(ser,X_out=Range(int(ser.index.min()),int(ser.index.max())))

    # Gauss mozgóátlag hibája
    # ser_resample = FvLinearResample(ser,4)
    # for day in Range(datefloat('2020-09-01'),datefloat('2022-12-31'),add=30):
    #     ser_cut = ser_resample.loc[ser_resample.index<day]
    #     ser_cut = FvGaussAvg(ser_cut,gausswidth=G*4)
    #     FvPlot(ser_cut,'original',annot='')

    # 360 napos jobbra igazított mozgóátlag
    # ser = FvLinearResample(ser,count=400)
    ser_right = fn_moving_avg(ser,int(360*len_per_day),center=False)
    ser_right = FvLinearResample(ser_right,X_out=Range(int(ser.index.min()),int(ser.index.max())))
    
    # Halványabban jelenjen meg az induló szakasz
    FvPlot(ser_right.loc[ser_right.index<datefloat('2020-06-01')],'original',label='',annot='',alpha=0.3)
    FvPlot(ser_right.loc[ser_right.index>=datefloat('2020-06-01')],'original',label='SMA_360_right',
            annot='max last',annotcaption=ddef(max='SMA'),color='last')

    pltshow(ymin=0,xtickerstep='year',ylabel='Leftwidth')
# fn_plot_leftwidth_bytime(country='all',G='all')
# fn_plot_leftwidth_bytime(country='Germany',G=35)
# exit()

def fn_save_leftwidth_bytime(flatten=True):         # átlagos leftwidth időbeli változása országonként és G értékenként
    tbl=Readcsv(Datapath('decompose_surges lognorm G_all leftw_free' + stringif(' flatten',flatten),'csv'),format='hu')

    tbl_out=pd.DataFrame()
    for country in fn_countries('Europe+'):
        for G in [14,21,28,35]:
            tblL=tbl.loc[(tbl.country==country) & (tbl.G==G)]

            # csak azok az lognormok kellenek, amelyek a járvány aktív szakaszaiban fordulnak elő
            # tblL=tbl.loc[tbl.start_y_gauss>0.05]     # nagyjából a rekordok fele marad
            # - a predikciók során is ez a szűrő érvényesül

            ser = serfromtbl(tblL,'leftwidth',indexcol='grad2_peak',aggfunc='mean',orderby='index')
            ser = ser.loc[(ser<140) & (ser>5)]      # outlier szűrés
            
            # ser = FvLinearResample(ser,X_out=Range(int(ser.index.min()),int(ser.index.max())))

            # A mozgóátlagolás count paraméteréhez kell a pontsűrűség a dátumtengelyen
            days = int(ser.index.max()) - int(ser.index.min())
            len_per_day = (len(ser)-1) / days

            # ser = fn_moving_avg(ser,360,center=False)   # ne legyen benne semmilyen jövőbeli információ
            ser = fn_moving_avg(ser,int(360*len_per_day),center=False)
            # resample minden napra 
            ser = FvLinearResample(ser,X_out=Range(int(ser.index.min()),int(ser.index.max())))

            # ser = FvGaussAvg(ser,gausswidth=180)

            # A (country,G)[day] görbék összeillesztése (az indexben a dátum van)
            tbl_out[country + '_G' + str(G)]=ser
    Tocsv(tbl_out,'leftwidth_by_country_and_time lognorm' + stringif(' flatten',flatten),format='hu')
# fn_save_leftwidth_bytime(flatten=True)
# fn_save_leftwidth_bytime(flatten=False)
# exit()


tbl_leftwidth_M = None
def sub_leftwidth(country,day,G,flatten=True):
    global tbl_leftwidth_M
    # leftwidth beolvasása  (adott országra, adott napon)
    if tbl_leftwidth_M is None:
        tbl_leftwidth_M = Readcsv(Datapath('leftwidth_by_country_and_time lognorm' + stringif(' flatten',flatten),'csv'),
                                  index_col='id')

    try:
        leftwidth = tbl_leftwidth_M[country + '_G' + str(G)][day]
        # leftwidth=ser_leftwidth[day] * tbl_leftwidth_factors.loc[G,'factor']    # az indexben a grad2_peak van, etért nem kell visszamenni
        # leftwidth=ser_leftwidth[day-30] * tbl_leftwidth_factors.loc[G,'factor']   # 30 nappal korábbi
        # leftwidth = ser_leftwidth.loc[ser_leftwidth.index<day-20].mean() * tbl_leftwidth_factors.loc[G,'factor']    
                # 20 napnál korábban indulók átlaga
    except:
        leftwidth=np.nan
    if np.isnan(leftwidth):     # csak South Korea esetén fordul elő, két 2023-as dátumnál
        # print('MISSING LEFTWIDTH: ' + country + '  ' + datestr(floatdate(day)))
        # leftwidth=sers_leftwidth.mean() * tbl_leftwidth_factors.loc[G,'factor']
        leftwidth=tbl_leftwidth_M[country + '_G' + str(G)].mean()

    # leftwidth=leftwidth*1.05     # tapasztalati adat: az optimális leftwidth átlaga 1.1-szerese a globális leftwidth átlagnak

    return leftwidth







# ============================================================================ #
# # DECOMPOSE AND PREDICT  HELPER FUNCTIONS
# ============================================================================ #

def fn_gauss_and_grad2(ser,G,resample=5,gauss_and_grad_all=None):     # serG,serGrad1,serGrad2,X_start,X_peak előállítása
  
    day_first=ser.index[0]
    day_last=ser.index[-1]    

    def sub_gauss_and_grad2(ser,G,resample):
        serGauss=FvGaussAvg(ser,gausswidth=G)
        
        if resample>1: serGaussR = FvLinearResample(serGauss,density=resample)
        else: serGaussR=serGauss

        serGrad1=FvGradient(serGaussR)
        serGrad2=FvGaussAvg(FvGradient(serGrad1),gausswidth=14*resample)
        points_min,points_max=serlocalminmax(serGrad2,endpoints=False)
        X_max=[]
        if len(points_max)>0: X_max,_=unzip(points_max)
        X_min=[]
        if len(points_min)>0: X_min,_=unzip(points_min)
        X_start=list(X_max)
        X_peak=list(X_min)
        # Ha peak az első szélsőérték, akkor kell egy képzetes start a bal oldalra
        if len(X_peak)>0 and (len(X_start)==0 or X_peak[0]<X_start[0]):
            X_start.insert(0,ser.index[0])

        return serGauss,serGrad1,serGrad2,X_start,X_peak

    # Gyorsított számolás, ha meg vannak a teljes időszakra vonatkozó adatok (részlegesen felhasználható)
    if gauss_and_grad_all is not None and day_last >= day_first + 4*G:
        serGauss_all,serGrad1_all,serGrad2_all,X_start_all,X_peak_all = gauss_and_grad_all
        X_start_all = array(X_start_all)
        X_peak_all = array(X_peak_all) 

        serGauss_ok = serGauss_all.loc[serGauss_all.index < day_last - 2*G]
        serGrad1_ok = serGrad1_all.loc[serGrad1_all.index < day_last - 2*G]
        serGrad2_ok = serGrad2_all.loc[serGrad2_all.index < day_last - 2*G]
        X_start_ok = X_start_all[X_start_all < day_last - 2*G]
        X_peak_ok = X_peak_all[X_peak_all < day_last - 2*G]
        
        day_first=day_last - 4*G    # 2G + 2G, A +2G azért kell, hogy a Gauss simítás szél-effektusa 
                                    #   ne okozzon torzulást   
        ser_right = ser.loc[ser.index > day_last - 4*G]
        serGauss_right,serGrad1_right,serGrad2_right,X_start_right,X_peak_right = \
            sub_gauss_and_grad2(ser_right,G,resample=resample)
        X_start_right = array(X_start_right)
        X_peak_right = array(X_peak_right)

        # Az átfedés levágása
        serGauss = Concat(serGauss_ok,serGauss_right[serGauss_right.index >= day_last - 2*G])
        serGrad1 = Concat(serGrad1_ok,serGrad1_right[serGrad1_right.index >= day_last - 2*G])
        serGrad2 = Concat(serGrad2_ok,serGrad2_right[serGrad2_right.index >= day_last - 2*G])
        X_start = Concat(X_start_ok,X_start_right[X_start_right >= day_last - 2*G])
        X_peak = Concat(X_peak_ok,X_peak_right[X_peak_right >= day_last - 2*G])

        # Extrém esetben előfordulhat, hogy kiesik egy X_peak. Ilyenkor nincs más megoldás, mint a 
        #    teljes görbére végrehajtani a grad képzést és a szélsőérték-helyek keresését
        if len(X_peak)<len(X_start)-1:
            serGauss,serGrad1,serGrad2,X_start,X_peak = \
                sub_gauss_and_grad2(ser,G,resample=resample)
    else:
        serGauss,serGrad1,serGrad2,X_start,X_peak = \
            sub_gauss_and_grad2(ser,G,resample=resample)

    return serGauss,serGrad1,serGrad2,X_start,X_peak

def sub_fit_back(ser,serGauss,X_start,X_peak,G,leftwidth,kitevo,bFixLeftwidth=False,params_all=[],
                     country=None,flatten=True,corefunction='lognormal'):
    '''
    serGauss: the complete known curve, with Gaussian smoothing (no future knowledge in it)
    X_start, X_peak: the grad2 extreme points of the entire known curve (no future knowledge is included)

    G,leftwidth: fixed parameters
         - If bFixLeftwidth=True, the leftwidth is fixed
         - If bFixLeftwidth=False, the leftwidth is only an initial value, it can be adjusted freely
         - in the case of "mean", read the surges before the current startpos from the previous leftw_free decomposition
             average leftwidth value (360-day moving average, adjusted to the right edge)
    params_all: post-calculated lognorm parameters for the entire curve (may also contain future knowledge)
         - specified in case of forward matching (predict)
         - if specified, the "old" lognorms do not need to be recalculated
         - recalculates only the lognorms on the right edge for which day_right > serGauss.index[-1] - 2*G
     corefunction: 'lognormal', 'gauss' ('erlang' is not yet integrated into the function)
    '''
    p_len=4
    if corefunction=='gauss': p_len=3

    nSurges_all = len(params_all)//p_len
    day_last=serGauss.index[-1]

    leftwidth_in=leftwidth

    params = []
    day_left=None           # az utolsó illesztés zárónapja
    params_left=[]    # az utolsó illesztés multilognorm paraméterei
    for i_startpos in range(0,len(X_start)-1):    # az utolsó előtti X_start-ig (csak a lezárt felfutások érdekelnek)
        if leftwidth_in=='mean':
            leftwidth = sub_leftwidth(country,X_start[i_startpos],G,flatten)

        day_left=0
        if i_startpos>0: day_left= int(X_peak[i_startpos-1])
        # korábban az előző day_right-tól indult az illesztés. Nem látszik különbség a kettő között 
        

        day_right = int((X_peak[i_startpos] + X_start[i_startpos+1])/2)    
        # - to the halfway point of the descending section. In fact, it is not necessarily a descending
        #   A new upswing can also start at ascending section 
        #  (the actual peak day of the previous section may be later than X_peak)
        # - the maxpos is not fixed here, so the fitting may be uncertain before peak
        # - day_left is the day_right of the previous closed section

        # If we are far enough from the end of serGauss and the parameters of the current lognorm have already been calculated
        if nSurges_all>i_startpos and day_right < day_last - 2*G:
            popt=params_all[i_startpos*p_len:(i_startpos+1)*p_len]
            # if corefunction=='lognormal':
            #     popt=params_all[i_startpos*4:(i_startpos+1)*4]
            # elif corefunction=='gauss':
            #     popt=params_all[i_startpos*3:(i_startpos+1)*3]

        # Else fitting (one lognorm)
        else:
            # fitting parameters
            params0,lower_bounds,upper_bounds,x_scale = [],[],[],[]

            boostwidth=X_peak[i_startpos] - X_start[i_startpos]                # felfutó szakasz látható hossza

            maxpos_0 = X_peak[i_startpos]
            maxpos_min=X_start[i_startpos]   # +  (X_peak[i_startpos] - X_start[i_startpos])*0.2   # maradjon valamennyi illesztési hossz
            if maxpos_min>maxpos_0: maxpos_min=maxpos_0     # it can occur at the very beginning of the ser
            maxpos_max=min(X_start[i_startpos] + leftwidth,X_peak[i_startpos]+G)
            # - if it were farther than leftwidth, then the start position of lognorm would be later than the start point according to grad2
            # - the X_peak + G is needed because without it the secondary grad2-max places can result in completely wrong surges
            #   We assume from the outset that the maxpos is close to the peakpos according to grad2

            # maxpos_max=X_start[i_startpos+1]
            # maxpos_max=maxpos_0 + G
            # maxpos_max=min(X_start[i_startpos+1],X_start[i_startpos] + leftwidth)  # nem jó. Átfedő felfutások is megengedettek.
            # maxpos_max=X_start[i_startpos] + leftwidth_0    
            if maxpos_max<maxpos_0: maxpos_0=maxpos_max

            # bFixLeftwidth=False
            if bFixLeftwidth:
                leftwidth_min=leftwidth*0.999
                leftwidth_max=leftwidth*1.001
            else:
                leftwidth_min = X_peak[i_startpos] - X_start[i_startpos]
                if leftwidth_min>leftwidth: leftwidth_min=leftwidth
                leftwidth_max=leftwidth*5

           
            y_peak=servalue(serGauss,X_peak[i_startpos])   # serGauss.loc[X_peak[i_startpos]]
            maxvalue_0=y_peak*0.9      # small reduction because it may overlap with the previous lognorm
            maxvalue_min=0             # in the case of a subordinate ramp-up, it can be much smaller than Y_peak[i].
            maxvalue_max=y_peak*1.2     # in principle, it cannot go above y_peak, but X_peak may differ slightly from the peak location

            params0.append(maxpos_0)
            params0.append(maxvalue_0)
            params0.append(leftwidth)
            lower_bounds.append(maxpos_min)
            lower_bounds.append(maxvalue_min)
            lower_bounds.append(leftwidth_min)
            upper_bounds.append(maxpos_max)
            upper_bounds.append(maxvalue_max)
            upper_bounds.append(leftwidth_max)
            x_scale.append(10000)           # maxpos
            x_scale.append(1000)            # maxvalue
            x_scale.append(10)              # leftwidth

            if corefunction=='lognormal':
                modus=0.95          # hozzávetőleg a power=9-nek felel meg az Erlangnál
                modus_min=0.94     
                modus_max=0.96
                # - delimitation is definitely necessary, because very low modes can occur in short fits 
                #    and the extremely slow decay of these elementary surges can lead to overestimations in the following stages
                # - similarly to Erlang, a common value is required for each epidemic curve, and the common value 
                #    must be optimized for the smallest loss (with Erlang, this was done and gave a power exponent of 9)

                params0.append(modus)
                lower_bounds.append(modus_min)
                upper_bounds.append(modus_max)
                x_scale.append(1)       
            elif corefunction=='erlang':
                params0.append(9)
                lower_bounds.append(8.5)
                upper_bounds.append(9.5)
                x_scale.append(10)       

            # ser_fit:  
            if False:       # sokkal rosszabb az illesztés, ha az eredeti adatokra illesztek
                if i_startpos==0: ser_fit=ser[:day_right]
                else:
                    ser_right = ser[day_left:day_right]
                    X=list(ser_right.index)
                    ser_fit=ser_right - fn_lognormal_multi(X,*params_left)
            # Fitting to Gauss
            if i_startpos==0: ser_fit=serGauss[:day_right]
            else:
                serGauss_right=serGauss[day_left:day_right]
                X=list(serGauss_right.index)
                # ser_fit=serGauss_right - fn_multi(X,corefunction,*params_left)
                if corefunction=='lognormal':
                    ser_fit=serGauss_right - fn_lognormal_multi(X,*params_left)
                elif corefunction=='gauss':
                    ser_fit=serGauss_right - fn_gauss_multi(X,*params_left)

            X=list(ser_fit.index)
            Y=ser_fit.values
            # popt=fn_fit(X,Y,corefunction,params0,lower_bounds,upper_bounds,x_scale)
            if corefunction=='lognormal':
                popt=fn_lognormal_fit(X,Y,params0,lower_bounds,upper_bounds,x_scale)
            elif corefunction=='gauss':
                popt=fn_gauss_fit(X,Y,params0,lower_bounds,upper_bounds,x_scale)

        params = Concat(params_left,popt)
        # day_left=day_right
        params_left=params

    return params


def sub_fit_forward(ser_known,serGauss,serGrad1,serGrad2,X_start,X_peak,lognormparams,leftwidth,G,model,gauss_and_grad_all):
    '''
    return lognormparams,ser_fit,ser_fitted,d1,d2,until_peak,grad,phase,maxpos_band
    '''
    i_start = len(X_start)-1        # az utolsó felfutásra kell illeszteni
    day_last = serGauss.index[-1]
    grad=serGauss[day_last] / serGauss[day_last-int(leftwidth/4)]
    
    # Lognorm fázis (érdemben nincs használva)
    if serGrad1[day_last-7]>0:
        if serGrad2[day_last-7]>0: phase=1
        else: phase=2
    else:
        if serGrad2[day_last-7]<0: phase=3
        else: phase=4

    params0,lower_bounds,upper_bounds,x_scale = [],[],[],[]

    maxpos = fn_maxpos_from_model(model,i_start,serGrad2,X_start,X_peak,lognormparams,serGauss,leftwidth,G)

    until_peak = maxpos - day_last


    # MAXPOS CORRECTION
    # anomaly filtering at the start of the second run
    # decrease maxpos if we are too close to the start
    start_distance_limit=leftwidth*0.1
    start_distance = day_last - X_start[-1]
    if start_distance < start_distance_limit:  
        maxpos = maxpos + (start_distance - start_distance_limit)   
        # - in the case of start_distance < 2, don't comes here (fn_predict has a day_min<2 filter)
        # - reason: due to gauss averaging, the final stage of grad2 is not yet final, it will go down gradually. 
        #    This will cause the maxpos to shift to the left
    
    # Maxpos dynamics on the ascending sections: due to the gradual shift of the startpos to the right,
    #   the maxpos also migrates to the right, with a steady deceleration (approximation)
    # STRONG IMPROVEMENT   Moreover, just in the sharpest rising sections (around 0.3%)
    elif until_peak>0:
        # the maxpos forecasted 7 days earlier should be requested
        ser_known_7 = ser_known.loc[ser_known.index<=day_last-7]
        serGauss_7,serGrad1_7,serGrad2_7,X_start_7,X_peak_7 = \
                fn_gauss_and_grad2(ser_known_7,G,resample=5,gauss_and_grad_all=gauss_and_grad_all)     # serG,serGrad1,serGrad2,X_start,X_peak előállítása
        if len(X_start_7)==len(X_start) and len(X_peak_7)==len(X_peak):
            maxpos_7 = fn_maxpos_from_model(model,i_start,serGrad2_7,X_start_7,X_peak_7,lognormparams,serGauss_7,leftwidth,G)
            maxpos_diff=maxpos-maxpos_7
            if maxpos_diff>0:
                # Average change / day:
                diff_per_day = maxpos_diff/7   
                # An upper limit is needed, because overestimations can occur in certain situations
                #  Example:  Italy, 2021-12-27   diff_per_day = 2.4
                if diff_per_day>1: diff_per_day=1       # the effect is minimal for the final loss
                shift = until_peak * diff_per_day
                maxpos = maxpos + shift



    # ser_fit
    if i_start==0: ser_fit=ser_known
    else:
        fit_start = X_peak[i_start-1]       
        fit_end = day_last
        # serL=FvGaussAvg(ser_known,gausswidth=14)
        serL=ser_known
        ser_right = serL.loc[(serL.index>=fit_start) & (serL.index<=fit_end)]

        X=list(ser_right.index)
        ser_fit=ser_right - fn_lognormal_multi(X,*lognormparams)      # subtracting the earlier surges

    leftwidth_min=leftwidth*0.999
    leftwidth_max=leftwidth*1.001       # in the descending section it will be loosen (see under)
    if bVersion:
        modus=0.945
        modus_min=0.915
        modus_max=0.96
        # modus,modus_min,modus_max = dgets(version_params,'modus,modus_min,modus_max')  
        # if until_peak>leftwidth*0.2:  modus_min=max(modus_min,0.93)  # az inflexiós pont előtt ne menjen 0.93 alá

    else:
        modus=0.95
        modus_min=0.94
        modus_max=0.96


    # MAXPOS_MIN, MAXPOS_MAX
    maxpos_min=maxpos          # by default, do not allow it to shift to the left (it would result in significant deterioration)
    maxpos_max=maxpos + 0.001
    # Anomaly filtering, primarily for outliers at higher G (at secondary peaks)
    # - lognorm/base improves by 2.6%, but a strong deterioration also occurs in some countries (e.g. Sweden, Latvia)
    boostwidth = maxpos - X_start[-1]
    boostwidth_from=leftwidth/2.1       
    if boostwidth > boostwidth_from:    
        maxpos_min = maxpos - (boostwidth - boostwidth_from)/(boostwidth_from) * 15
        maxpos_max=maxpos
        # - the boostwidth model sometimes gives an overly distant maxpos value. In such cases, it is better to trust curve_fit to find the correct maxpos value on its own
        # - violent upswings are not affected, because they usually predict a short boostwidth (in these cases, bringing maxpos_min forward would make the lognormal too flat)
        # - it occurs that the matching does not exactly hit the max location even in this case (e.g. Hungary, G35, before '2022-04-20'). Nevertheless, the result improves somewhat in this case as well.

    if until_peak<=0:       # after peak (the entire epidemic curve may still be in an ascending phase)
        # - huge improvment     1,5%  
        maxpos_max = day_last + 8       # tuned
        maxpos_min = maxpos - (-until_peak)     # it can also be opened to the left in descending phases
        # Allow a flatter curve (until the next starting point is found)
        # - this is necessary so that the maxpos can extend at least to the current point
        leftwidth_max =  leftwidth + 2*(-until_peak)
        # - not sensitive for the special factor
    elif until_peak>0:   # in ascending phase
        tol_peak=11      # in the vicinity of peak  it can encrease the maxpos
        tol_start=1     # minimal tolerance at the start of the surge, because this is the only way to force a sufficiently strong growth
        tol_width=leftwidth*0.22    # can be gradually released around the inflection point
        if until_peak>tol_width: maxpos_tol = tol_start     # far form peak (in early phase)
        else: maxpos_tol = tol_start + (tol_width - until_peak)/tol_width * tol_peak     # 11 about peak (can be shifted to right, but not to left)

        maxpos_max=maxpos + maxpos_tol


    # MAXVALUE
    # approximation for maxvalue:   near linear change between  x_start  and   x_peak
    x_start = X_start[-1]
    y_start = servalue(serGauss,x_start)
    x_now = serGauss.index[-1]
    y_now = serGauss.values[-1]
    m = (y_now-y_start)/(x_now - x_start)
    maxvalue = y_start + m * (maxpos - x_start)
    if maxvalue<0: maxvalue = y_now / 2

    maxvalue_min = 0
    maxvalue_max = 4*maxvalue

    params0.append(maxpos)
    params0.append(maxvalue)
    params0.append(leftwidth)
    params0.append(modus)
    lower_bounds.append(maxpos_min)
    lower_bounds.append(maxvalue_min)
    lower_bounds.append(leftwidth_min)
    lower_bounds.append(modus_min)
    upper_bounds.append(maxpos_max)
    upper_bounds.append(maxvalue_max)
    upper_bounds.append(leftwidth_max)
    upper_bounds.append(modus_max)
    x_scale.append(10000)           # maxpos
    x_scale.append(1000)            # maxvalue
    x_scale.append(10)              # leftwidth
    x_scale.append(1)              # modus

    X=list(ser_fit.index)
    Y=ser_fit.values

    popt = fn_lognormal_fit(X,Y,params0,lower_bounds,upper_bounds,x_scale)
    lognormparams = Concat(lognormparams,popt)

    Y_fitted=fn_lognormal_multi(X,*popt)
    ser_fitted = pd.Series(Y_fitted,X)

    # indicators for the quality of fit (d1,d2)
    # d1:  average of absolute differences, for the right-side 10 days of the fitting
    ser_diff = pd.Series(abs(Y - Y_fitted),X)
    d1 = ser_diff.loc[ser_diff.index>day_last-10].mean() / serGauss.max()
    
    # the difference of angles (relative to 180 degree)
    grad1_max=abs(serGrad1).max()     # the maximal value of gradient  on the fit-curve
    d2 = abs(math.atan((Y[-1]-Y[-2])/grad1_max) - math.atan((Y_fitted[-1]-Y_fitted[-2])/grad1_max))  
    d2=d2/math.pi   # relative to 180 degree

    return lognormparams,ser_fit,ser_fitted,d1,d2,until_peak,grad,phase,(maxpos_min,maxpos_max)

def fn_plot_multi_G(country):       # fn_gauss_and_grad2 testing
    tbl=Fn_read_coviddata()
    ser=Fn_coviddata_country(tbl,country)
    ser=datefloat(ser)

    pltinit()    
    for G in [14,28,56,112]:
        resample=5
        serGauss,_,_,X_start,_ =   \
                fn_gauss_and_grad2(ser,G,resample=resample)
        serStarts=pd.Series(serGauss[X_start],X_start)
        FvPlot(floatdate(serGauss),'original',label='G=' + str(G),annot='max')
        FvPlot(serStarts,'scatter',annot='')
    pltshow()
# fn_plot_multi_G('Hungary')
# exit()




# ============================================================================ #
# DECOMPOSE
# ============================================================================ #

def fn_regressions_back(country,G=14,kitevo=9,leftwidth=None,bFixLeftwidth=False,       # decompose for one country
                    verbose=1,bPlot=False,tbl_covid=None,ser=None,scatter=False,flatten=True,plot_grad2=True,
                    corefunction='lognormal'):
    '''
    Lognorm decomposition for a country
    
    leftwidth: If bFixLeftwidth=True, this will be the leftwidth value of all lognormals
        In the case of bFixLeftwidth=False, it is only an initial value, the leftwidth value of surges may differ from this
         - an average value depending on G can be specified
         - if not specified, an approximate value depending on G is entered
    bFixLeftwidth: If false, leftwidth is also a free tuning parameter (per surge), leftwidth
             argument only defines the initial value

        
    bExcel: If true, it writes the data of the log norms to a file
         'return': returns the generated table
    out type:
         'tbl': detailed data of surges (for bExcel, this is written to a file)
         'loss':
         'lognormparams':
    '''

    p_len=4
    if corefunction=='gauss': p_len=3

    if leftwidth=='mean': leftwidth_0 = 10 * math.sqrt(G) * 0.7     # approximating calculation for Erlang; the 0.7 factor must be due to the transition to lognormal 
    else: leftwidth_0 = leftwidth

    if not leftwidth: leftwidth = leftwidth_0
    #  - empirical data, the leftwidth-tuned decompose gives similar averages

    if verbose>0: progress('Adatok beolvasása')
    if tbl_covid is None:  tbl_covid=Fn_read_coviddata()

    if ser is None:  ser=Fn_coviddata_country(tbl_covid,country,flatten=flatten)

    ser=datefloat(ser)
    lastday=ser.index[-1]

       
    if verbose>0: progress('Lognorm illesztések')

    serGauss,serGrad1,serGrad2,X_start,X_peak =   \
            fn_gauss_and_grad2(ser,G,resample=5)
    ser_max=serGauss.max()

    params = sub_fit_back(ser,serGauss,X_start,X_peak,G,leftwidth,kitevo,bFixLeftwidth=bFixLeftwidth,
                                    country=country,flatten=flatten,corefunction=corefunction)
    

    # LOSS SZÁMÍTÁS
    ser_true=floatdate(serGauss[:X_start[-2]])      # az utolsó lezárt felfutásig
    # Predictions at the very beginning of epidemic curves may show extremely high MAPE values, which would distort the loss value
    ser_true=ser_true.loc[ser_true > serGauss.max()/1000]
    Y_true=ser_true.values
    # Y_fitted=fn_lognormal_multi(list(ser_true.index),*params)
    if corefunction=='lognormal': Y_fitted=fn_lognormal_multi(list(ser_true.index),*params)
    elif corefunction=='gauss': Y_fitted=fn_gauss_multi(list(ser_true.index),*params)

    mape_max = (abs(Y_fitted-Y_true) / ser_true.max()).mean()
    mape = (abs(Y_fitted-Y_true)/Y_true).mean()

    stats_out= ('LOSS' + '\n' + 
                'MAPEmax=' + strnum(mape_max,'2%') + '\n' +
                'MAPE=' + strnum(mape,'%'))
    if verbose>0: print(stats_out)

    # INPUT FOR THE BOOSTWIDTH-TRAIN
    if verbose>0: progress('Calculating boostwidth input')
    records=[]
    for i_start in range(len(X_start)-1):
        inputs = fn_boostwidth_input(i_start,serGrad2,X_start,X_peak,params,serGauss,leftwidth_0,corefunction)

        records.append([country,G,leftwidth_0,floatdate(Round(inputs['maxpos']))]   # általános adatok
                        + list(dgets(inputs,'all')))         # kimeneti és input adatok

    columns=('country,G,leftwidth_0,day_maxpos,' +
                'grad2_start,grad2_peak,maxpos,maxvalue,modus,power,leftwidth,boostwidth,boost_ratio,' +
                'maxpos_last_x,descent_ratio,formfaktor,maxvalue_last,' +
                'start_y_gauss,start_y,start_last_y,peak_last_y,start_last_x,peak_last_x')
    tbl_out=pd.DataFrame(records,columns=columns.split(','))
    
    # PLOT
    if bPlot:
        # Fn_country_lognorms_test(country,days='back',tbl=tbl_covid,ser=ser,verbose=2,bPlot_loss_after=True,
        #                         **params)   # xmax='2022-07-30'
        if plot_grad2: 
            nrows=2
            height_ratios=[3,1]
        else:
            nrows=1
            height_ratios=None

        pltinit(suptitle='Lognormal decomposition, ' + country,
                nrows=nrows,ncols=1,height_ratios=height_ratios,sharey=False)
        
        params_out=('PARAMS' + '\n' +
                    'G=' + strint(G)
                    # 'exponent=' + strint(kitevo)   # + '\n' +
                    # 'leftwidth=' + strint(boostwidth*2.1)
                    )

        pltinitsub('next',title='Epidemic curve, elementary lognormal-surges and their superposition',)

        # plot Y_true
        ser_true=serGauss
        Y_true=ser_true.values
        FvPlot(ser_true,plttype='original',label='gauss',annot='max')         

        # plot Y_fitted
        # normfaktor=config.normfaktorlastG
        # Y_fitted=fn_multi(list(ser_true.index),corefunction,*params)
        if corefunction=='lognormal': Y_fitted=fn_lognormal_multi(list(ser_true.index),*params)
        elif corefunction=='gauss': Y_fitted=fn_gauss_multi(list(ser_true.index),*params)
        FvPlot(pd.Series(Y_fitted,ser_true.index),plttype='original',label='fitted',
                         annot='max')

        # Original case counts
        if scatter:
            FvPlot(ser,plttype='scatter',label='original',annotate='max')


        # Forecasted peaks
        plot_peaks=False
        if plot_peaks:
            X=tbl_out[['maxpos_last_x','descent_ratio','formfaktor','grad2_start_y']].to_numpy()
            boost_ratios = f_keras(X,fn_get_maxpos_model(day=None))     

            maxposs=[]
            for i_lognorm in range(len(boost_ratios)):
                grad2_start=X_start[i_lognorm]       # az X_start 1-gyel hosszabb a boost_ratios (lezárt felfutások)
                boost_ratio=boost_ratios[i_lognorm]
                if np.isnan(boost_ratio) or boost_ratio<0: boost_ratio=0.5
                elif boost_ratio>2: boost_ratio=2
                maxpos=Round(datefloat(grad2_start) + leftwidth_0*boost_ratio)
                if maxpos>datefloat(lastday): continue
                maxposs.append(maxpos)
            servalues
            serPeak=ser_true[maxposs]
            FvPlot(serPeak,plttype='scatter',annotate='',colors=ddef(color='green'))
            


        # Elementary surges
        leftwidths=[]
        count=len(params)//p_len
        for i_lognorm in range(count-1):     # the last one is not needed (only the completed)
            maxpos=params[i_lognorm*p_len]
            maxvalue=params[i_lognorm*p_len+1]
            leftwidth=params[i_lognorm*p_len+2]

            # boostwidth/leftwidth ratio
            if False:
                annot=''
                label=None
                ratio_labels=False
                if ratio_labels:
                    boostwidth=maxpos-X_start[i_lognorm]
                    label=strnum(boostwidth/leftwidth,'2%')
                    annot='localmax'

            # plot surge  (area plot)
            startpos=maxpos - leftwidth
            X=Range(startpos,count=int(3*leftwidth))
            Y=fn_multi(X,corefunction,*(params[i_lognorm*p_len:(i_lognorm+1)*p_len]))
            # if corefunction=='lognormal': Y=fn_lognormal_multi(X,maxpos,maxvalue,leftwidth,modus)
            # elif corefunction=='gauss': Y=fn_gauss_multi(X,maxpos,maxvalue,leftwidth)
            FvPlot(pd.Series(Y,floatdate(X)),'original',label='',        #'Lognorm_' + str(i_lognorm),
                    colors=ddef(color='gray',alpha=0.3),annot='',area=True)

            leftwidths.append([maxpos,leftwidth])

        # leftwidth
        plot_leftwidth=False
        if plot_leftwidth:
            FvPlot(SerFromRecords(leftwidths),'regauss',label='leftwidth',normalize=ser_true.max()*0.5)

        pltshowsub( xy_texts=[ddef(x=0.03,y=0.97,caption=stats_out,fontsize=7,ha='left',va='top',alpha=0.6,transform='gca'),
                              ddef(x=0.97,y=0.97,caption=params_out,fontsize=7,ha='right',va='top',alpha=0.6,transform='gca')],
                  )

        if plot_grad2:
            pltinitsub('next',title='Second derivative')

            serGrad2_out=floatdate(serGrad2)
            FvPlot(serGrad2_out,plttype='original',G=G,annot='last',label='grad2',annotcaption={'gaussabs':'{x}','max':'{label}'})

            color=pltlastcolor()

            # Local extrema of grad2
            points_min,points_max = serlocalminmax(serGrad2,endpoints=False)        # a szélső pontokat ne tekintse szélsőérték helynek
            serMin=SerFromRecords(points_min)
            color=color_darken(color)
            FvPlot(serMin,plttype='scatter',annotate='lower5',annotcaption='{x}',colors=ddef(color='navy'))
            serMax=SerFromRecords(points_max)
            FvPlot(serMax,plttype='scatter',annotate='upper5',annotcaption='{x}',colors=ddef(color='orange'))

        pltshow(annot_fontsize=7,xtickerstep='year')

    return params, mape_max, mape, tbl_out
       
def fn_decompose(countries,G=20,tbl_covid=None,plot=False,to_csv=True,scatter=False,plot_grad2=True,
                 leftwidth_type='free',division=None,ser_flatten=False,corefunction='lognormal'):
    '''
    countries: a list or a single country. fn_countries('Europe') fn_countries('Europe+')
    G: several G can be entered (list)
    plot: If true (G, country), one subplot per value pair (no more than 10)
         The grad2 plot is only displayed if there is a single subplot (single G and country)
    to_csv: only interesting in the case of several countries (in the case of one country, there is absolutely no writing to a file)
    scatter: the original points should also appear
    leftwidth_type:
         'fix': kitevo=9, leftwidth: average of previous matches (common for all countries, depends only on G)
             - required for boostwidth_train
         'mean': kitevo=9, leftwidth: 
         'tune': unique leftwidth per G value and per country, with tuning
             - the leftwidth value of the ramps is fixed for a given (G, country) pair
         'free':
             - each run-up can have a different leftwidth and modus value
             - starting value: 10*sqrt(G)*0.7 (experience approximation of tunings)
     division: if it is specified, it matches two periods: before division and after division
         - example: examination of the difference between the period before and after the vaccinations
     ser_resample: If true, the input ser is averaged as 7 day points (the weekly average is written for each day)

    '''

    p_len=4
    if corefunction=='gauss': p_len=3


    if tbl_covid is None:  tbl_covid=Fn_read_coviddata()

    if type(countries)==str: countries=countries.split(',')
    else: countries=countries

    if type(G)==int: 
        aG=[G]
        in_fname = 'G' + str(G)
    else:
        aG=G
        in_fname = 'G_all'

    in_fname += ' leftw_' + leftwidth_type
    if ser_flatten: in_fname += ' flatten'

    periods=1
    if division: periods=2

    bFixLeftwidth =  (leftwidth_type!='free')


    if plot and (len(countries)>1 or len(aG)>1):
        pltinit(suptitle='Lognormal decomposition',ncols=1,nrows=len(aG)*len(countries))


    records=[]
    tbl_out_all=pd.DataFrame()
    for G in aG:
        for country in countries:
            if leftwidth_type=='free': print()
            
            for period in range(periods):
                ser=Fn_coviddata_country(tbl_covid,country,flatten=ser_flatten)
                if division: 
                    if period==0: ser=ser.loc[ser.index<division]
                    elif period==1: ser=ser.loc[ser.index>=division]
                    period_out=str(period+1)
                else: 
                    period_out='all'
                progress('TUNE ' + country + '   G=' + str(G) + '   period=' + period_out)


                leftwidth_0 = 10 * math.sqrt(G)      # Erlang experience data (approximate value for leftwdith due to tuning)

                # Transfer to lognormal  (experience multiplier, too)
                if corefunction=='lognormal': leftwidth_0 = leftwidth_0 / 0.68

                if leftwidth_type=='mean': leftwidth_0='mean'   # sub_fit_back fogja beolvasni

                params=ddef(G=G,kitevo=9,leftwidth=leftwidth_0)    

                if leftwidth_type=='tune':
                    def f_loss(**params):
                        _,loss,_,_ = fn_regressions_back(country,bFixLeftwidth=True,bPlot=False,
                                                   tbl_covid=tbl_covid,verbose=0,ser=ser,**params)
                        return loss

                    loss=None

                    params,loss,_ = fn_tune_bygroping(f_loss,params,'leftwidth',add=2,addsub=0.5,xmin=5,xmax=100,lossstart=loss)
                    # params,loss,_ = fn_tune_bygroping(f_loss,params,'kitevo',add=0.5,xmin=4,xmax=20,lossstart=loss)


                if len(countries)==1 and len(aG)==1:
                    if not bFixLeftwidth:
                        print('RESULT' +
                            '   kitevo=' + str(params['kitevo']) + 
                            '   leftwidth=' +  strnum(params['leftwidth'])) 
                            # '   loss=' + strnum(loss,'3%'))
                    fn_regressions_back(country,bFixLeftwidth=bFixLeftwidth,bPlot=True,tbl_covid=tbl_covid,
                                        ser=ser,scatter=scatter,plot_grad2=plot_grad2,**params)
                
                else:
                    surge_params,loss,mape,tbl_out = \
                        fn_regressions_back(country,bFixLeftwidth=bFixLeftwidth,corefunction=corefunction,
                                        verbose=0,bPlot=False,tbl_covid=tbl_covid,ser=ser,**params)

                    print('TUNE ' + country + 
                          '   G=' + str(G) + 
                          '   period=' + period_out +
                          '   MAPEmax=' + strnum(loss,'2%') +
                          '   MAPE=' + strnum(mape,'%'))

                    if plot:
                        pltinitsub('next',title=country + '  gausswidth=' + str(G))

                        kitevo=params['kitevo']
                        leftwidth=params['leftwidth']

                        # plot gauss
                        serGauss=FvGaussAvg(ser,gausswidth=G)
                        FvPlot(serGauss,plttype='original',label='gauss',annot='max')         

                        # plot fitted
                        # normfaktor=config.normfaktorlastG
                        Y_fitted = fn_multi(list(serGauss.index),corefunction,*surge_params)
                        FvPlot(pd.Series(Y_fitted,serGauss.index),plttype='original',label='fitted',
                                        annot='max')

                        # plot surges
                        count=len(surge_params)//p_len
                        for i_lognorm in range(count-1):     # az utolsó nem kell (csak a befejezettek kellenek)
                            maxpos=surge_params[i_lognorm*p_len]
                            maxvalue=surge_params[i_lognorm*p_len+1]
                            leftwidth=surge_params[i_lognorm*p_len+2]
                            modus,power=None,None
                            if corefunction=='lognormal': modus=surge_params[i_lognorm*p_len+3]
                            elif corefunction=='erlang': power=surge_params[i_lognorm*p_len+3]

                            # area plot
                            startpos=maxpos - leftwidth
                            X=Range(startpos,count=int(3*leftwidth))
                            if corefunction=='lognormal': Y=fn_lognormal_multi(X,maxpos,maxvalue,leftwidth,modus)
                            elif corefunction=='erlang': Y=fn_erlang_multi(X,maxpos,maxvalue,leftwidth,power)
                            elif corefunction=='gauss': Y=fn_gauss_multi(X,maxpos,maxvalue,leftwidth)
                            FvPlot(pd.Series(Y,floatdate(X)),'original',label='',    
                                    colors=ddef(color='gray',alpha=0.3),annot='',area=True)


                        stats_out=( 'surge_count=' + str(count) + '\n' +
                                    'MAPEmax=' + strnum(loss,'2%')
                                    ) 
                        params_out=('PARAMS' + '\n' +
                                    'G=' + str(G)
                                    # 'exponent=' + strint(kitevo)   # + '\n' +
                                    )
                        xy_texts= \
                            [ddef(x=0.02,y=0.94,caption=stats_out,fontsize=7,ha='left',va='top',alpha=0.6,transform='gca'),
                            ddef(x=0.98,y=0.94,caption=params_out,fontsize=7,ha='right',va='top',alpha=0.6,transform='gca')]

                        pltshowsub(xy_texts=xy_texts,xtickerstep='year')

                    surge_count=len(tbl_out)
                    leftwidth=array(surge_params[2::4]).mean()     # paraméter-négyesek, a leftwidth a 3. helyen
                    modus=array(surge_params[3::4]).mean()    # paraméter-négyesek, a modus a 4. helyen

                    records.append([country,G,period_out,surge_count,leftwidth,modus,loss,mape,len(ser),surge_params])

                    tbl_out_all = Concat(tbl_out_all,tbl_out)

    if to_csv:
        columns='country,G,period,surge_count,leftwidth,modus,loss,mape,count_days,lognormparams'
        tbl_tune_back=pd.DataFrame(records,columns=columns.split(','))
        Tocsv(tbl_tune_back,'decompose_countries ' + corefunction + ' ' + in_fname,format='hu')        # korábban "tune back"

        Tocsv(tbl_out_all,'decompose_surges ' + corefunction + ' ' + in_fname,format='hu',indexcol=True)

    if plot and (len(countries)>1 or len(aG)>1):
        pltshow()
# fn_decompose(fn_countries('Europe+'),G=[14,21,28,35],to_csv=True,leftwidth_type='tune',ser_flatten=True)     # ,division='2021.06.30')
# fn_decompose(fn_countries('Europe+'),G=[14,21,28,35],to_csv=True,leftwidth_type='free',ser_flatten=True)     # ,division='2021.06.30')
# fn_decompose(fn_countries('Europe+'),G=[14,21,28,35],to_csv=True,leftwidth_type='free',ser_flatten=False)     # ,division='2021.06.30')

# fn_decompose(fn_countries('Europe+'),G=[14,21,28,35],to_csv=True,leftwidth_type='free',corefunction='gauss')     # ,division='2021.06.30')


# fn_decompose('Denmark',G=14,plot=True,scatter=False,plot_grad2=False,to_csv=False,leftwidth_type='free',ser_flatten=False)
# fn_decompose('Hungary',G=14,plot=True,scatter=False,plot_grad2=True,to_csv=False,leftwidth_type='free',ser_flatten=False)
# fn_decompose('Germany',G=[14,21,28,35],plot=True,scatter=False,plot_grad2=False,to_csv=False,leftwidth_type='free',ser_flatten=False)

# exit()


def fn_decompose_histograms(leftwidth_type='free'):
    tbl_lognorms = Read_csv(Datapath('decompose_surges lognorm G_all leftw_' + leftwidth_type + ' flatten','csv'))

    hist_plot(tbl_lognorms,cols='leftwidth,maxvalue,boostwidth,boost_ratio',G=0.1,groups='G:14,21,28,35')
# fn_decompose_histograms()
# exit()

def fn_decompose_loss():    # free-tune és flatten veriációk globális loss értéke
    print('DECOMPOSE LOSS')
    tbl = Read_csv(Datapath('decompose_countries erlang G_all leftw_free','csv'))
    print('Erlang free   ' + strnum(tbl.loss.mean(),'3%'))
    tbl = Read_csv(Datapath('decompose_countries erlang G_all leftw_free flatten','csv'))
    print('Erlang free flatten   ' + strnum(tbl.loss.mean(),'3%'))
    tbl = Read_csv(Datapath('decompose_countries erlang G_all leftw_tune','csv'))
    print('Erlang tune   ' + strnum(tbl.loss.mean(),'3%'))
    tbl = Read_csv(Datapath('decompose_countries erlang G_all leftw_tune flatten','csv'))
    print('Erlang tune flatten   ' + strnum(tbl.loss.mean(),'3%'))

    print('Lognormal')
    tbl = Read_csv(Datapath('decompose_countries lognorm G_all leftw_free','csv'))
    print('Lognormal free   ' + strnum(tbl.loss.mean(),'3%'))
    tbl = Read_csv(Datapath('decompose_countries lognorm G_all leftw_free flatten','csv'))
    print('Lognormal free flatten   ' + strnum(tbl.loss.mean(),'3%'))
    tbl = Read_csv(Datapath('decompose_countries lognorm G_all leftw_tune flatten','csv'))
    print('Lognormal tune flatten   ' + strnum(tbl.loss.mean(),'3%'))

    print('Gauss')
    tbl = Read_csv(Datapath('decompose_countries gauss G_all leftw_free','csv'))
    print('Gauss free   ' + strnum(tbl.loss.mean(),'3%'))
    # tbl = Read_csv(Datapath('decompose_countries gauss G_all leftw_free flatten','csv'))
    # print('Lognormal free flatten   ' + strnum(tbl.loss.mean(),'3%'))
    # tbl = Read_csv(Datapath('decompose_countries gauss G_all leftw_tune flatten','csv'))
    # print('Lognormal tune flatten   ' + strnum(tbl.loss.mean(),'3%'))

# fn_decompose_loss()
# exit()


def fn_boostwidth_ratio_MAE():

    for leftwidth_type in ['free','tune','fix']:
        print('LEFTWIDTH_TYPE=' + str(leftwidth_type))
        tbl_lognorms = Read_csv(Datapath('decompose_surges lognorm G_all leftw_' + leftwidth_type + ' flatten','csv'))
        for G in [14,21,28,35]:
            tbl = tbl_lognorms.loc[tbl_lognorms.G==G]
            mean = tbl.boost_ratio.mean()
            MAE = (abs(tbl.boost_ratio-mean)).mean()
            print('G=' + str(G) + '   MAE=' + strnum(MAE,'%'))
# fn_boostwidth_ratio_MAE()
# exit()





# ============================================================================ #
# # BASE FUNCTIONS
# ============================================================================ #

# PREDICTION
def fn_predict(ser_known,G,kitevo,leftwidth,gauss_and_grad_all,params_all,model):
    '''
    Lognormal prediction

    return: lognormparams,ser_fit,ser_fitted,d1,d2,until_peak,grad,phase
    
    ser_known:  case values per day,   0 instead of nan,  cumulative reports distributed over the previous missing-data days
    gauss_and_grad_all: gaussian moving average calculated for the entire curve, grad1, grad2, grad2 extrema
          (5 data per day due to resample)
         - the data must be recalculated for the 2G day before the last known day
     params_all: parameters calculated with the backward algorithm for the entire curve
         - the data must be recalculated for the 2G day before the last known day
     model:  keras model for forecasting maxpos

    '''
    day_first=ser_known.index[0]
    day_last=ser_known.index[-1]        

    # serGauss,serGrad1,serGrad2,X_start_X_peak számítása
    # if we have the data for the entire curve, only need to recalculate the Gaussian moving average 
    #   and extrema for the right edge of the curve (acceleration)
    serGauss,serGrad1,serGrad2,X_start,X_peak = \
        fn_gauss_and_grad2(ser_known,G,resample=5,gauss_and_grad_all=gauss_and_grad_all)     # serG,serGrad1,serGrad2,X_start,X_peak előállítása


    # If the startpoint is too near, don't start new surge   (anomaly filtering)
    # setting days_min
    # About peak (on leftwidth/4-es time scale)
    if serGauss[day_last-int(leftwidth/4)]<serGauss[day_last]: days_min=2
    else: days_min=1  # within one day startday don't start new surge
    if day_last - X_start[-1] <= days_min:  X_start=X_start[:-1]


    # backward fit
    lognormparams = sub_fit_back(ser_known,serGauss,X_start,X_peak,G,leftwidth,kitevo,
                                    bFixLeftwidth=False,params_all=params_all)

    # forward fit (last open surge)
    lognormparams,ser_fit,ser_fitted,d1,d2,until_peak,grad,phase,maxpos_band = \
        sub_fit_forward(ser_known,serGauss,serGrad1,serGrad2,X_start,X_peak,lognormparams,leftwidth,G,model,gauss_and_grad_all)

    return lognormparams,ser_fit,ser_fitted,d1,d2,until_peak,grad,phase,maxpos_band

# RETROSPECTIVE TESTs AND PREDICTIONs FOR ONE COUNTRY
def Fn_country_test(country,G=14,xmax=None,verbose=2,tbl=None,ser_orig=None,flatten=False,
                             plot=False,plot3='back',interval=7,above=0.05,reduced=False,bPred=False,
                             dayfirst=None,daylast=None,dayofweek='optimal',countryToDayofweek=None,
                             outtype='loss',count_back=7,count_avg=1,model_knn=None,tbl_mape_upper=None,
                             scatter=True):  
    '''
    Lognormal predictions for uniformly distributed dates or a specific date

    country: for which country (loose syntax)
    G: a single number or list. In the case of a list, weighted average for lognormal predictions, 
            different for each prediction day with weight factors
     xmax: single prediction  displaying detailed fitting information
         - predictions for previous days can also be requested
         - a date string or a number can also be entered
     interval:  test days frequency
     ser_orig:  recommended to give for repeated calls. Epidemic curve calculated without backfill
     plot:   If true, plot appears. By specifying fname_fix, it can be requested to be written to the plot file
            'sub':  only pltinitsub and pltshowsub calls (the pltinit-pltshow frame must be taken care of separately)
     above:  calm state filtering. The ratio compared to the country-level peak value.
     bPred:  live prediction. It also displays data for the last seven days
     datefirst: (date string or float) if not specified, then from the 40th day of the data series of the given country
         - not interested if xmax is specified
     dayofweek:   which weekday is the testing day [0:7)
         It is only relevant if interval = 7,14,...
         'optimal': the day with the smallest loss per country (form pretraind dataset)
         'opt_by_day': per country and test day, the day with the smallest loss (based on the previous 8 weeks)

    '''

    global bDebug

    G_in=str(G)
    if type(G)==int: aG=[G]
    else: aG=G

    leftw_type = 'free'     # in case of lognormal,  "free" is better
    tbl_lognormparams = Read_csv(Datapath('decompose_countries lognorm G_all leftw_' + leftw_type + stringif(' flatten',flatten),'csv'),
                                index_col=['country','G'])

    params_all,kitevo = [],[]
    for G in aG:
        kitevo.append(9)        # in case of lognormal, power is not relevant (Erlang parameter)

        rec = tbl_lognormparams.loc[(country,G)]
        params_all.append(json.loads(rec.lognormparams))

    if tbl is None: tbl=Fn_read_coviddata()
    if ser_orig is None: ser_orig=Fn_coviddata_country(tbl,country,backfill=False,flatten=flatten)

    ser_orig=datefloat(ser_orig)
    lastday=ser_orig.index[-1]      # utolsó adatközlés napja

    ser=ser_orig
    if not flatten: ser = fillna_back(ser_orig)
    # gauss average for the above criterium   (the relevant gauss evarege will be later, with resample)
    ser_gauss,ser_max = [],[]
    for G in aG:
        ser_gaussL=FvGaussAvg(ser,gausswidth=G)
        ser_gauss.append(ser_gaussL)
        ser_max.append(ser_gaussL.max())

    # gauss average for the G14 comparison
    ser_gauss14 = FvGaussAvg(ser,gausswidth=14) 
    ser_max14=ser_gauss14.max()

    if dayofweek=='optimal':    # globálisan optimal weekday  (see also: 'opt_by_day')
        if interval==7:
            if countryToDayofweek is None: countryToDayofweek = fn_dayofweek_countries()
            dayofweek = countryToDayofweek[country]
        else: dayofweek=None


    # TEST DAYS  
    if xmax: 
        xmax=datefloat(xmax)
        if bPred:
            count_back=7
            days=Range(xmax,xmax-(count_back-1),add=-1)
        else:
            days=Range(xmax-(count_back-1),xmax,add=1)
    else:
        if dayfirst is None: 
            dayfirst= max(datefloat('2020-07-01'),ser_orig.index[0] + 40)
            if type(dayofweek)==int and dayofweek>=0 and dayofweek<7:
                dayfirst = datefloat(fn_monday(floatdate(dayfirst))) + dayofweek   # 0=hétfő, 6=vasárnap
        if daylast is None:
            daylast = lastday + 1       # a rendelkezésre álló adatok utáni nap
        days=[]
        day=datefloat(dayfirst)
        while True:
            day += interval
            if day>=daylast: break
            if ser_gauss14[day] < ser_max14*above: continue

            if dayofweek=='opt_by_day':     # shift in the next 7 das  to the optimal day
                optday=fn_get_optimal_dayofweek(country,day)
                shift = optday - floatdate(day).weekday()
                if shift<0: shift += 7
                day += shift

            days.append(day)

      
    days_loss=[]


    # Basic data for the whole curve (serGauss,serGrad1,serGrad2,X_start,X_peak)
    #  - partly applicable in predictions (accelerator)
    gauss_and_grad_all = []
    grad1_phase,grad2_phase = None,None
    for iG,G in enumerate(aG):
        result= fn_gauss_and_grad2(ser,G,resample=5)
        gauss_and_grad_all.append(result)

        if G==28: _,grad1_phase,grad2_phase,_,_ = result
    
    # if G28 is missing form aG, it must be requested  to detetermine the phase
    if grad1_phase is None: _,grad1_phase,grad2_phase,_,_ = fn_gauss_and_grad2(ser,28,resample=5)

    if plot:
        # suptitle and init
        if plot!='sub':         # in case of "sub"  the caller will settle the subtitle
            title = 'Epidemic curve,  ' + country
            if xmax:    # don't needed the lower plot
                title += ' ' + datestr(xmax)
                if bPred: 
                    pltinit(suptitle='Lognormal-prediction test')
                else:
                    pltinit(suptitle='Lognormal-prediction test',nrows=2,ncols=1,height_ratios=(4,1),sharey=False,
                            left=0.076,right=0.93,height=0.8,top=0.85,bottom=0.085,hspace=0.26)
            elif plot=='compare':
                pltinit(suptitle='Retrospective tests, ' + country,ncols=1,nrows=1)
                title='Lognormal predictions every 8 days in the active periods'

            else:
                pltinit(suptitle='Lognormal-prediction test',nrows=3,ncols=1,height_ratios=(4,1,1),sharey=False,
                            left=0.076,right=0.93,height=0.8,top=0.85,bottom=0.085,hspace=0.26)

        # title and initsub
        if plot=='sub': title = country + ' ' + datestr(xmax)
        pltinitsub(axindex='next',title=title)

        serGauss,serGrad1,serGrad2,X_start,X_peak = gauss_and_grad_all[0]

        # Teljes Gauss
        if not xmax:
            # scatter plot also needed with original data
            if scatter:
                FvPlot(ser,'scatter',label='original',annot='localmax2',scatter_size=1,scatter_alpha=0.5)
            FvPlot(serGauss,'original',label='gauss' + str(aG[0]),annotate='max last',colors=ddef(color='gray'))

        # in case of xmax,  gauss until xmax is also to plot (last G/2 period)
        else:
            color_gauss='orange'
            if bPred: color_gauss='navy'

            # gauss_fact   (in two part)
            serGauss_fact = serGauss.loc[(serGauss.index>=xmax-50) & (serGauss.index<=xmax-int(aG[0]/2))]
            FvPlot(serGauss_fact,'original',label='gauss_known',annotate='left',color=color_gauss)
            serGauss_fact_right = serGauss.loc[(serGauss.index>=xmax-int(aG[0]/2)) & (serGauss.index<=xmax)]
            FvPlot(floatdate(serGauss_fact_right),'original',annot='',colors=ddef(color=color_gauss,alpha=0.4))
                # - floatdate for mindestens one plot  (date-captions for the x-axis)
            # gauss_future
            if xmax<serGauss.index[-1]:
                serGauss_future = serGauss.loc[(serGauss.index>=xmax) & (serGauss.index<=xmax+50)]
                # If the end of the whole Gauss is shown, then 7-day appears dimmed
                if serGauss_future.index[-1]>serGauss.index[-7]:
                    serGauss_future_ok = serGauss_future.loc[serGauss_future.index<=serGauss.index[-7]]
                    if len(serGauss_future_ok)>7: 
                        FvPlot(serGauss_future_ok,'original',label='gauss_future',annot='last',
                            colors=ddef(color=color_gauss,alpha=0.3))
                        FvPlot(serGauss_future.loc[serGauss_future.index>=serGauss.index[-7]],'original',label='',annot='',
                            colors=ddef(color=color_gauss,alpha=0.15))
                    else: 
                        FvPlot(serGauss_future_ok,'original',label='',annot='',
                            colors=ddef(color=color_gauss,alpha=0.3))
                        FvPlot(serGauss_future.loc[serGauss_future.index>=serGauss.index[-7]],'original',label='gauss_future',annotate='right',
                            colors=ddef(color=color_gauss,alpha=0.15))
                else:
                    FvPlot(serGauss_future,'original',label='gauss_future',annotate='right',
                        colors=ddef(color=color_gauss,alpha=0.3))

            # gauss_known  (inclinig to horzintal direction)
            ser_known = ser.loc[ser.index<xmax]
            if not bPred:
                serGauss_known = FvGaussAvg(ser_known,gausswidth=aG[0])
                FvPlot(serGauss_known[-int(aG[0]/2):],'original',label='gauss_known',annot='last',colors=ddef(color=color_gauss))

            # scatter plot with the original data
            if scatter:
                if bPred: scatter_size=1.5
                else: scatter_size=1
                FvPlot(ser_known.loc[ser_known.index>xmax-50],'scatter',label='original',annot='upper1',
                                scatter_size=scatter_size,scatter_alpha=0.5)

                # only one point at the actual day, with annotation
                FvPlot(pd.Series([0],[xmax]),'scatter',scatter_size=1,annot='lower',label=datestr(xmax),colors=ddef(color='green'))

    day_half = datefloat('2021-06-30')

    # PREDICTIONS FOR THE TEST DAYS
    lognormparams=None
    first_values,pred_peaks,Y_pred_total,Y_true_total = [],[],[],[]
    APE,APE_peak,APE_peak_base,APE_peak_linear,Y_SMAPE = [],[],[],[],[]
    knn_train,loss_train,until_peaks,phases_known,phases = [],[],[],[],[]
    baseline=None
    maxpos_band=None
    for i_day,day in enumerate(days):
        bDebug= (i_day==len(days)-1)


        day_last_known=day-1  # the day not included (the data is known until the day before)
        day_last_knownG=day_last_known-int(aG[0]/2)  # the last known data in the moveing average

        # Known curve on the day of test
        ser_known = ser.loc[:day_last_known].copy()
       
        ser_max14_known = ser_gauss14.loc[ser_gauss14.index<=day_last_knownG].max()  

        # PREDIKCIÓ
        lognormparams = []
        until_peaksL=[]
        phase_known_out=None
        for iG,G in enumerate(aG):
            lognormparamsG,ser_fit,ser_fitted,d1,d2,until_peak,grad,phase_known,maxpos_band = \
                fn_predict(ser_known,G,kitevo[iG],sub_leftwidth(country,day,G,flatten),gauss_and_grad_all[iG],
                                   params_all[iG],fn_get_maxpos_model(day))
            lognormparams.append(lognormparamsG)
            until_peaksL.append(until_peak)
            if (phase_known_out is None) or (G==28): phase_known_out=phase_known

        until_peak_mean=array(until_peaksL).mean()
        until_peaks.append(until_peak_mean)
        phases_known.append(phase_known_out)        # based on ser_known, on day-7

        # Determining phase  (G=28 timescale, based on the complete epidemic curve)
        phase_out=None
        try:
            if grad1_phase[day]>0:
                if grad2_phase[day]>0: phase_out=1
                else: phase_out=2
            else:
                if grad2_phase[day]<0: phase_out=3
                else: phase_out=4
        except:
            pass
        phases.append(phase_out)            # in the knowledge of the entire epidemic curve


        X_pred=Range(day,count=50)

        # Calculation of Y_pred (on X_pred values)
        if len(aG)==1: Y_pred=fn_lognormal_multi(X_pred,*lognormparams[0])
        else:
            # in case of multi G, is must be weighted (based on loss and environmental data)
            weights = sub_get_G_weights(day)    

            Y_predG = []        
            for iG,G in enumerate(aG):
                Y_predG.append(fn_lognormal_multi(array(X_pred),*lognormparams[iG]))
            Y_pred=[]
            for i_dayafter in range(50):
                y_pred=0
                for iG in range(len(aG)):
                    y_pred += Y_predG[iG][i_dayafter] * weights[iG][i_dayafter]
                Y_pred.append(y_pred / weights[len(aG)][i_dayafter])
            Y_pred=array(Y_pred)


        # LOSS CALCULATIONS
        MAPE_0_13,MAPE_peak_0_13  = np.nan,np.nan
        if day>lastday: 
            Y_true = array([np.nan]*50)
            Y_raw = array([np.nan]*50)
        elif day>lastday-50: 
            X_loss = Range(day,count=int(lastday-day+1))
            Y_true = Concat(ser_gauss14.loc[X_loss].values,[np.nan]*(50-len(X_loss)))
            Y_raw = Concat(ser.loc[X_loss].values,[np.nan]*(50-len(X_loss)))
        else:
            X_loss = X_pred
            Y_true=ser_gauss14.loc[X_loss].values
            Y_raw=ser.loc[X_loss].values     
            # - if day=lastday-49, then the last 7 days are not completely reliable

        Y_diff=abs(Y_pred-Y_true)
        Y_mean=(abs(Y_pred) + abs(Y_true))/2

        APE.append(Y_diff/Y_true)    # two dimension array    APE[i_testday][i_dayafter]
        APE_peak.append(Y_diff/ser_max14)  # two dimension array

        Y_SMAPE.append(Y_diff/Y_mean)  # two dimension array    Y_SMAPE[i_testday][i_dayafter]

        MAPE_0_13,MAPE_peak_0_13 = np.nan,np.nan
        if Count_notna(Y_diff)>=14:
            MAPE_0_13 = (Y_diff/Y_true)[:14].mean()
            MAPE_peak_0_13 = (Y_diff/ser_max14)[:14].mean()
        knn_train.append([G,until_peak,grad,MAPE_0_13])    # obsolote

        # Error of the base model
        mean_7days = ser_known.iloc[-7:].mean()  # the mean of the last 7 days
        Y_pred_base = [mean_7days  for x in X_pred]
        Y_diff_base=abs(Y_pred_base - Y_true)
        Y_score_factor = Y_diff_base / Y_diff        
        APE_peak_base.append(Y_diff_base/ser_max14)  # two dimensional array

        mean_7days_2 = ser_known.iloc[-7:].mean()       # the mean of the last 7 days
        mean_7days_1 = ser_known.iloc[-14:-7].mean()    # the mean of the previous 7 days
        grad1=(mean_7days_2 - mean_7days_1) / 7
        Y_pred_linear=array([mean_7days_2 + grad1 * (4 + (x-day))  for x in X_pred])   # in range(int(G/2)+1,int(G/2)+1+nLossDays)]
            # - the fitting starts from then middle of the last 7 days
        Y_pred_linear[Y_pred_linear<0] = 0
            # - if the result would go below 0, then it should be 0 
        Y_diff_linear=abs(Y_pred_linear-Y_true)
        APE_peak_linear.append(Y_diff_linear/ser_max14)  # two dimensional array



        # loss_train
        if day<=lastday-50:     # it should not contain records containing NAN
            for i_dayafter in range(50):
                mape = APE[-1][i_dayafter]
                smape = Y_SMAPE[-1][i_dayafter]
                mape_peak = APE_peak[-1][i_dayafter]
                # score:   [-1,1] 
                factor = (Y_pred[i_dayafter]-Y_true[i_dayafter])**2 / (Y_pred_base[i_dayafter]-Y_true[i_dayafter])**2
                if factor<1: score = 1 - factor             # if better as base   
                else: score = -1 + (1/factor)             # if worse as base  
                score_factor = Y_score_factor[i_dayafter]
                y_pred = Y_pred[i_dayafter] / ser_max14   # ser_max14_known
                y_true = Y_true[i_dayafter] / ser_max14   # ser_max14_known
                y_raw = Y_raw[i_dayafter] / ser_max14
                y_base = mean_7days / ser_max14
                y_linear = Y_pred_linear[i_dayafter] / ser_max14
                if Y_pred[i_dayafter]>Y_true[i_dayafter]: sign=1
                else: sign=0
                loss_train.append([country,day,G_in,until_peak_mean,phase_known_out,phase_out,i_dayafter,
                                   y_pred,y_true,y_raw,y_base,y_linear,mape,mape_peak,smape,sign,score,score_factor])


        days_loss.append(day)


        # PRINTING THE FIT-PARAMS
        if verbose>=2: 
            if len(aG)==1:
                lognorm_count=len(lognormparams[0])//4    # 4 parameter / lognorm (maxpos,maxvalue,leftwidth,modus)
                print('Illesztés  ' + country + '  Day=' + datestr(day) + 
                    '  d1=' + strnum(d1,'%') + 
                    '  d2=' + strnum(d2,'%') + 
                    '  Count=' + strint(lognorm_count) +
                    '  Until_peak=' + strint(until_peak) + 
                    '  Grad=' + strnum(grad) + 
                    '  Loss_0_13=' + strnum(MAPE_peak_0_13,'%'))
            else:
                print('Illesztés  ' + country + 
                    '  Day=' + datestr(day) + 
                    '  Loss_0_13=' + strnum(MAPE_peak_0_13,'%') + 
                    '  Until_peak_mean=' + strnum(until_peak_mean) +
                    '  Phase=' + str(phase_out))


        # PLOT, PREDICTION CURVE
        if plot:
            # serFitted
            if len(aG)>1:
                serFitted=pd.Series(Y_pred,X_pred)       # calculated earlier
            else:       # in the case of a single G, it can also be drawn backwards with G/2
                ser_out=ser.loc[day_last_knownG:day_last_known + 20]
                X=list(ser_out.index)
                # the fitted curve should be at least G/2 + 30 days (if we are nearing the end of the ser, it can be shorter)
                if len(X)<int(aG[0]/2) + 30:
                    X=Concat(X,Range(X[-1]+1,add=1,count=int(aG[0]/2) + 30 - len(X)))

                Y_fitted=fn_lognormal_multi(X,*lognormparams[0])
                serFitted=pd.Series(Y_fitted,X)

            # serFitted plot
            if xmax:
                if bPred:
                    color_pred = 'orange'
                    # baseline setting for the lasz day
                    if (i_day ==0):
                        FvPlot(serFitted,'original',label='prediction',annot='right',colors=ddef(color=color_pred))
                        baseline = serFitted
                    elif i_day<7: 
                        label=strint(day-xmax)        # -1,-2,...  (obsolote)
                        # Area to the right of today
                        serPlot = serFitted.loc[serFitted.index>=baseline.index[0]]
                        FvPlot(serPlot,'original',label=label,area=True,baseline=baseline, annot='')
                        # Just a line before today
                        serPlot = serFitted.loc[serFitted.index<=baseline.index[0]]
                        FvPlot(serPlot,'original',area=False,annot='',colors=ddef(color='gray',alpha=0.2))
                else:
                    FvPlot(serFitted.loc[serFitted.index>=day_last_known],'original',label='prediction', annot='last')    # colors=ddef(alpha=0.3),
            else:
                FvPlot(serFitted,'original',area=True,annot='')    # colors=ddef(alpha=0.3),

            # Error bands plot
            if bPred and i_day==0:
                if (model_knn is not None and tbl_mape_upper is not None
                        and (day<=lastday) and ser_gauss14[day]>=ser_max14*above):
                        # - there is no sample for those below max*0.05 in the knn model, therefore too high upper limits would appear
                    X=Range(day,day+44,add=1)
                    arr_50_lower,arr_90_lower,arr_50_upper,arr_90_upper = [],[],[],[]
                    for dayafter in range(45):      # Displays 35 days, but 25% overrun to avoid gauss margin-error
                        # Lower limit based on MAPE values calculated from previous predictions (multiplicative)
                        y=Y_pred[dayafter]
                        arr = tbl_mape_upper.loc[tbl_mape_upper['dayafter']==dayafter,['mape']].values  # felülbecslések dayafter értékenként
                        mape_90 = np.quantile(arr,0.9)      # a hibák 90%-a kisebb ennél
                        arr_90_lower.append((1/(mape_90+1))*y)  # inverz számolás, mape to y_true/y_pred


                        # The upper limit with knn (mixed multiplicative and additive characters)
                        x = [dayafter,y/ser_max14]      # ser_max14_known
                        Y=f_knn_neighbors_Y(x,model_knn)    # y_diff value of closest predictions (normalized to 1)
                        y_diff_90 = np.quantile(Y,0.9)
                        arr_90_upper.append(y + y_diff_90*ser_max14)    # ser_max14_known
                    baselineL = pd.Series(Y_pred[:len(X)],X)
                    ser_lower=FvGaussTAvg(pd.Series(arr_90_lower,X),G=0.3,positive=True,leftright='left').iloc[:35]
                    FvPlot(ser_lower,'original',annot='',area='noline',baseline=baselineL)
                    ser_upper=FvGaussTAvg(pd.Series(arr_90_upper,X),G=0.3,positive=True,leftright='left').iloc[:35]
                    FvPlot(ser_upper,'original',label='90% band',annot='last',area='noline',baseline=baselineL)


            # Drawing the details of the fitting for the last day (what was fitted, what was the fitted curve)
            if xmax and not bPred:
                if len(aG)==1 and xmax and i_day==len(days)-1:
                    # The last closed surge
                    nSurges = len(lognormparams[0])//4
                    i_lognorm=0
                    if nSurges>1:
                        i_lognorm=nSurges-2
                        maxpos=lognormparams[0][i_lognorm*4]
                        maxvalue=lognormparams[0][i_lognorm*4+1]
                        leftwidthL=lognormparams[0][i_lognorm*4+2]
                        modusL=lognormparams[0][i_lognorm*4+3]
                        startpos=maxpos - leftwidthL
                        X=Range(startpos,count=int(2.5*leftwidthL))
                        Y=fn_lognormal_multi(X,maxpos,maxvalue,leftwidthL,modusL)
                        FvPlot(pd.Series(Y,floatdate(X)),'original',label='surge_last',        #'Lognorm_' + str(i_lognorm),
                                colors=ddef(color='gray',alpha=0.3),annot='max',area=True)

                    # Actual surge (with the backward algorithm, as a benchnark)
                    plot_lognorm_future=False
                    if plot_lognorm_future:
                        nSurges_all = len(params_all[0])//4
                        if nSurges_all>i_lognorm+1:
                            maxpos=params_all[0][(i_lognorm+1)*4]
                            maxvalue=params_all[0][(i_lognorm+1)*4+1]
                            leftwidthL=params_all[0][(i_lognorm+1)*4+2]
                            modusL=params_all[0][(i_lognorm+1)*4+3]
                            startpos=maxpos - leftwidthL
                            X=Range(startpos,count=int(2*leftwidthL))
                            Y=fn_lognormal_multi(X,maxpos,maxvalue,leftwidthL,modusL)
                            FvPlot(pd.Series(Y,floatdate(X)),'original',label='Lognorm_future',        #'Lognorm_' + str(i_lognorm),
                                    colors=ddef(color='gray',alpha=0.2),annot='max',area=True)

                    # Curve to fit:   (based on last 1 or 2 grad2-maximum)  
                    if len(aG)==1:
                        FvPlot(ser_fit,'original',annot='max',label='fit_right',colors={"color":"blue","alpha":0.5})
                        # Fitted curve
                        FvPlot(ser_fitted,'original',annot='last',label='fitted_right',colors={"color":"blue","alpha":0.3})

            first_values.append(serFitted.loc[day])

            if day_last_known + until_peak <= lastday:
                pred_peaks.append(day_last_known + until_peak)      # in case of mukti-G, irrelevant

    APE = array(APE)     # APE[i_testday][i_dayafter]
    APE_peak = array(APE_peak)
    APE_peak_base = array(APE_peak_base)
    APE_peak_linear = array(APE_peak_linear)
    # APE_peak_band = array(APE_peak_band)

    loss_rel,loss,loss_base,loss_linear,lognorm_base = np.nan,np.nan,np.nan,np.nan,np.nan
    loss_35,loss_rel_35,loss_base_35,lognorm_base_35 = np.nan,np.nan,np.nan,np.nan
    # If xmax is None, then average over all test points
    if xmax is None:
        if len(APE)>0:
            # A 14-day average should only be calculated for those test cases that contain values for at least 14 prediction days
            i_test_lastok = len(APE)
            while i_test_lastok>0 and Count_notna(APE[i_test_lastok-1])<14: i_test_lastok -= 1
            if i_test_lastok>=0:
                loss_rel = APE[:i_test_lastok,:14].mean()             # for all testdays, 0-13 day after the testda
                loss = APE_peak[:i_test_lastok,:14].mean()
                loss_base = APE_peak_base[:i_test_lastok,:14].mean()             # for all testday, 0-13 afterdays
                loss_linear = APE_peak_linear[:i_test_lastok,:14].mean()
                lognorm_base = loss/loss_base

            # A 35-day average should only be calculated for those test cases that contain values for at least 35 prediction days
            i_test_lastok = len(APE)
            while i_test_lastok>0 and Count_notna(APE[i_test_lastok-1])<35: i_test_lastok -= 1
            if i_test_lastok>=0:
                loss_35 = APE_peak[:i_test_lastok,:35].mean()
                loss_rel_35 = APE[:i_test_lastok,:35].mean()             # for all testday, 0-34 afterdays
                loss_base_35 = APE_peak_base[:i_test_lastok,:35].mean()             # for all testday, 0-34 afterdays
                lognorm_base_35 = loss_35/loss_base_35
        
    elif xmax:
        if len(APE)>0 and Count_notna(APE[0])>=14:
            loss_rel = APE[0,:14].mean()             # for all testday, 0-13 afterdays
            loss = APE_peak[0,:14].mean()
            loss_base = APE_peak_base[0,:14].mean()             # for all testday, 0-13 afterdays
            loss_linear = APE_peak_linear[0,:14].mean()         # for all testday, 0-13 afterdays 
            lognorm_base = loss/loss_base

        if len(APE)>0 and Count_notna(APE[0])>=35:
            loss_35 = APE_peak[0,:35].mean()
            loss_rel_35 = APE[0,:35].mean()             # for all testday, 0-34 afterdays
            loss_base_35 = APE_peak_base[0,:35].mean()             # for all testday, 0-34 afterdays
            lognorm_base_35 = loss_35/loss_base_35



    stats_out=('LOSS' + '\n' +
        'MAPEmax (1-14 days): ' + strnum(loss,'%') + '\n' +
        'Base/Lognorm   (1-14 days): ' + strnum(1/lognorm_base,'%')
        )
    if len(aG)==1:
        params_out=('PARAMS' + '\n' +
                    'G=' + strint(aG[0])
                    )
    else:
        params_out=('PARAMS' + '\n' +
                    'G=' + str(aG) 
                    )
    if bVersion:
        for key,value in version_params.items():
            params_out = params_out + '\n' + str(key) + '=' + strnum(value)


    if verbose==2:
        print(params_out)
        print(stats_out)
    elif verbose==1:
        if xmax:
            print_out=(country + '  ' + datestr(xmax) + 
                  '  MAPEmax (1-14 days)=' + strnum(loss,'%') +  
                  '  Base/Lognorm=' + strnum(1/lognorm_base,'%') +
                  '  Phase=' + str(phases[0]))      # phase belongong to the first testpoint
            if bPred:   # calculation of the optimal dayofweek  in case of bPred
                dayofweek = strdate(xmax).weekday()
                dayofweek_means = np.mean(APE_peak[:,:14],axis=1)   # APE_peak[i_testday][i_dayafter]
                dayofweek_means_base = np.mean(APE_peak_base[:,:14],axis=1)   # APE_peak[i_testday][i_dayafter]
                dayofweek_means = dayofweek_means / dayofweek_means_base
                i_optimal = np.argmin(dayofweek_means)
                date_optimal = floatdate(xmax - i_optimal)    # the predictions went backwards from day xmax
                weekday_optimal = date_optimal.weekday()
                day_names=['hétfő','kedd','szerda','csütörtök','péntek','szombat','vasárnap']
                subtract = dayofweek - weekday_optimal
                if subtract<0: subtract += 7
                date_optimal = xmax - subtract
                # print_out += '  Optimal day: ' + day_names[weekday_optimal] + '  ' + datestr(date_optimal)
            print(print_out)
        else:
            print(country + '  ' + 
                  '  MAPEmax (1-14 days)=' + strnum(loss,'%') +  
                  '  Base/Lognorm=' + strnum(1/lognorm_base,'%'))


    if plot:
        # Points on the first day of prediction curves (scatter)
        if not xmax:
            FvPlot(pd.Series(first_values,floatdate(days_loss)),'scatter',annotate='')

        # Forecasted peaks
        if len(aG)==1 and xmax:         # in the case of an overview, the max places are not needed
            FvPlot(pd.Series(servalueA(ser_gauss[0],pred_peaks),pred_peaks),'scatter',label='keras_peak',
                annot='max',colors=ddef(color='red'))

        v_lines=None
        if bPred: v_lines=[ddef(x=xmax,color='green',alpha=0.2)]

        # draw xmax_band for fitting figure
        v_bands=None
        if xmax and not bPred and maxpos_band:
            v_bands = [ddef(x1=maxpos_band[0],x2=maxpos_band[1],color='gray')]
            FvAnnotateAdd(x=(maxpos_band[0]+maxpos_band[1])/2,y=plt.ylim()[1],caption='maxpos_band')

        
        if plot=='sub': 
            # Display the Lognorm/base for the test day
            xy_texts=None
            subcount= Attr(plt.gcf(),'nrows',1) * Attr(plt.gcf(),'ncols',1)   # pltinit-ben lett eltéve
            if subcount==1 and len(APE)>0:
                loss_rel_14 = APE[0,:14].mean()
                caption='MAPE_14=' + strnum(loss_rel_14,'%')
                if subcount<=2:
                    loss_rel_30 = APE[0,:30].mean()
                    caption = caption + '\n' + 'MAPE_30=' + strnum(loss_rel_30,'%')
                xy_texts=[ddef(x=0.03,y=0.97,caption=caption,fontsize=7,ha='left',va='top',alpha=0.8,transform='gca')]
            xtickerstep='month'
        else:
            xy_texts=[ddef(x=0.03,y=0.97,caption=stats_out,fontsize=7,ha='left',va='top',alpha=0.6,transform='gca'),
                      ddef(x=0.97,y=0.97,caption=params_out,fontsize=7,ha='right',va='top',alpha=0.6,transform='gca')]
            xtickerstep=None

        xmin=None
        xmax_=None
        if xmax and not bPred: xmin=xmax-90
        elif plot=='compare':
            if country in ['Japan','South Korea','Australia']: interval=('2021-01-01','2023-06-30')
            elif country in ['South Africa']: interval=('2020-01-01','2022-06-30')
            else: interval=('2020-07-01','2022-12-31')
            xmin=datefloat(interval[0])
            xmax_=datefloat(interval[1])
            xtickerstep='year'

        pltshowsub(
            xmin=xmin,xmax=xmax_,
            xy_texts=xy_texts,
            annot_count=ddef(localmax=20,gaussabs=50,other=20),
            annot_fontsize=ddef(localmax=6,upper=6,other=7),
            xtickerstep=xtickerstep,
            v_lines=v_lines,v_bands=v_bands
            )


        # grad2 curve
        if not bPred and plot!='compare':
            pltinitsub(1,title='Second derivative')
            serGrad2=gauss_and_grad_all[0][2]
            if xmax: serGrad2 = serGrad2.loc[(serGrad2.index>xmax-50) & (serGrad2.index<xmax+25)]

            serGrad2=floatdate(serGrad2)
            FvPlot(serGrad2,plttype='original',annot='last',label='grad2_future')

            color=pltlastcolor()

            # grad2 extrema
            points_min,points_max = serlocalminmax(serGrad2,endpoints=False)        # a szélső pontokat ne tekintse szélsőérték helynek
            serMin=SerFromRecords(points_min)
            color=color_darken(color)
            FvPlot(serMin,plttype='scatter',annotate='lower5',annotcaption='{x}',colors=ddef(color='navy'))
            serMax=SerFromRecords(points_max)
            FvPlot(serMax,plttype='scatter',annotate='upper5',annotcaption='{x}',colors=ddef(color='orange'))
            pltshowsub()

            # grad2_known
            if xmax:
                gauss_and_grad_known = fn_gauss_and_grad2(ser_known,G,resample=5)
                serGrad2_known = gauss_and_grad_known[2]
                FvPlot(serGrad2_known.loc[serGrad2_known.index>xmax-50],
                    'original',label='grad2_known',annot='gaussabs last',
                    annotcaption={'gaussabs':'{x}','max':'{label}'})


        # Elementary surges and predicted peak times
        if not xmax and plot3 and plot!='compare':
            if plot3=='back':
                pltinitsub(2,title='Decomposition')

                # Elemi felfutások
                count=len(params_all[0])//4
                for i_lognorm in range(count-1):     # az utolsó nem kell
                    maxpos=params_all[0][i_lognorm*4]
                    maxvalue=params_all[0][i_lognorm*4+1]
                    leftwidthL=params_all[0][i_lognorm*4+2]
                    modusL=params_all[0][i_lognorm*4+3]
                    startpos=maxpos - leftwidthL
                    X=Range(startpos,count=int(3*leftwidthL))
                    Y=fn_lognormal_multi(X,maxpos,maxvalue,leftwidthL,modusL)
                    FvPlot(pd.Series(Y,floatdate(X)),'original',        #'Lognorm_' + str(i_lognorm),
                            colors=ddef(color='gray',alpha=0.3),annot='',area=True)
                pltshowsub(annot_fontsize=7)

            # Loss
            elif plot3=='loss':
                pltinitsub(2,title='Loss')
                Y = [APE_peak[i,:14].mean() for i in range(APE_peak.shape[0])]
                FvPlot(pd.Series(Y,days_loss[:APE_peak.shape[0]]),plttype='regauss',G=0.05,colors=ddef(alpha=1),label='diffpercent_lognorm7_13',annotate='last')
                Y = [APE_peak_base[i,:14].mean() for i in range(APE_peak_base.shape[0])]
                FvPlot(pd.Series(Y,days_loss[:APE_peak_base.shape[0]]),plttype='regauss',G=0.05,colors=ddef(alpha=1),label='diffpercent_base7_13',annotate='last')
                pltshowsub()

            # phase
            elif plot3=='phase':
                pltinitsub(2,title='phase')
                FvPlot(pd.Series(phases,days_loss),plttype='scatter original',colors=ddef(alpha=1),label='phase',annotate='last')
                pltshowsub()


        # pltshow
        if plot!='sub':
            to_file=None
            if type(plot)==str:  
                if plot=='compare': to_file='Lognorm test, ' + country
                else: to_file=plot
            pltshow(to_file=to_file)



    
    if outtype=='loss_all':
        loss_days=[]
        APE=array(APE)
        if len(APE)>0:
            for i_dayafter in range(50):
                loss_days.append(APE[:,i_dayafter].mean())     # átlagolás oszloponként

        return (loss,loss_rel,loss_base,loss_linear,loss_35,loss_rel_35,lognorm_base,lognorm_base_35,
                loss_days,knn_train,loss_train)

    elif outtype=='loss':
        return loss,loss_rel

    elif outtype=='lognorm_base':
        return lognorm_base

    elif outtype=='APE':
        return APE

    elif outtype=='APE_peak':
        return APE_peak

    elif outtype=='APE_all':
        return APE,APE_peak,APE_peak_base,APE_peak_linear
# Fn_country_test('Portugal',G=[14,21,28,35],plot='compare',interval=8,flatten=True,scatter=False)  #,xmax='2022-01-14') #,dayofweek=5) #,xmax='2022-01-21',count_back=7)  # '2022-03-25') #interval=10) #,xmax='2022-12-20') #,xmax='2022-04-17')  #,xmax='2022-07-07')  #,xmax='2022-04-14') #xmax='2021-11-28') #xmax='2021-11-23') #xmax='2022-01-23') #'2022-01-21')  #'2021-03-23')
# Fn_country_test('Latvia',G=[14,21,28,35],plot=True,xmax='2022-03-03',bPred=True)  # '2022-03-25') #interval=10) #,xmax='2022-12-20') #,xmax='2022-04-17')  #,xmax='2022-07-07')  #,xmax='2022-04-14') #xmax='2021-11-28') #xmax='2021-11-23') #xmax='2022-01-23') #'2022-01-21')  #'2021-03-23')
# Fn_country_test('United States',G=[14,21,28,35],plot=True)  

# Fn_country_test('Spain',G=35,plot=True,interval=8,flatten=False,xmax='2021-12-23',count_back=1)  
# Fn_country_test('Hungary',G=[14,21,28,35],plot=True,interval=8,flatten=True,scatter=False) #,xmax='2022-02-21',count_back=1)  
# exit()


# Összehasonlító ábrák országonként
def Fn_country_tests(countries,params={}):        # Saves into files then diagrams (png)
    global version_params,bVersion

    bVersion=True
    version_params=params

    if beginwith(countries,'group:'):
        countries_in=cutleft(countries,'group:')
        countries=fn_countries(countries_in)
    else:
        countries_in=countries
        countries=countries.split(',')

    for country in countries:
        progress(country)
        Fn_country_test(country,G=[14,21,28,35],verbose=0,plot='compare',interval=8,flatten=True,scatter=False)
# Fn_country_tests('group:Europe+')
# Fn_country_tests('Italy',ddef(modus=0.945,modus_min=0.915,modus_max=0.96))
# exit()


# PREDIKCIÓS PLOT
def fn_plot_predictions(country,testday_first,weeks=8,optimalday=True,flatten=False):
    '''
    sublots (count = weeks), each subplot shows the weekly days    ending with the given day
          (area plot, with last day's base-line)
    
    optimalday: If true, it adjusts to the optimal dayofweek calculated for the entire epidemic period
         starting date (the first day can be postponed by a maximum of 6 days)
         In the case of a live prediction, the gauss_future curve indicates that data after the closing day are also available
        
    The function displays which day of the week it is recommended to adjust the testday_first date.
         The plot gives the best display of development trends with this setting.
    If the data is only received weekly, you should use the testday_first date for a data release
         adapt to the day
    
    '''
    tbl_covid=Fn_read_coviddata()
    ser_orig=Fn_coviddata_country(tbl_covid,country,backfill=False,flatten=flatten)

    lastday=datefloat(ser_orig.index[-1])
    
    bLastday=False
    if testday_first=='lastday': 
        bLastday=True
        testday_first = datestr((lastday - (weeks-1)*7) + 1)        


    tbl_loss_train = Readcsv(Datapath('loss_train lognorm FINAL','csv'))

    tbl_mape_upper_all =  tbl_loss_train.loc[tbl_loss_train['sign']==1].copy()    # felülbecslések 

    tbl_knn_all = tbl_loss_train.loc[tbl_loss_train['sign']==0].copy()   # alulbecslsések
    tbl_knn_all['y_diff_n'] = tbl_knn_all['y_true_n'] - tbl_knn_all['y_pred_n']

    day_names=['Mónday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    countryToDayofweek = fn_dayofweek_countries()
    weekday_optimal_global=countryToDayofweek[country]
    print('Optimal day for the whole epidemic period: ' + day_names[weekday_optimal_global])

    dayofweek_firstday = strdate(testday_first).weekday()
    subtract = dayofweek_firstday - weekday_optimal_global
    if subtract<0: subtract += 7

    if subtract==0: print('The specified start date is the same as the optimal one.')
    else:
        if optimalday:
            testday_first = datestr(datefloat(testday_first) - subtract)
            if bLastday: print('The closing day was shifted forward by ' + str(subtract) + ' days.')
            else: print('The start day was shifted forwars by ' + str(subtract) + ' days.')
        else:
            if bLastday: print('The closing day is not optimal: ' + day_names[dayofweek_firstday] )
            else:  print('The specified start date is not optimal: ' + day_names[dayofweek_firstday])

    days=Range(testday_first,count=weeks,add=7)


    pltinit(suptitle='Lognormal prediction',subcount=len(days),rowcols_aspect=(1,1),sharey=True,sharex=False)
    dayofweek_means=[]
    for day in days:
        model_knn=None
        tbl_mape_upper = tbl_mape_upper_all.loc[(tbl_mape_upper_all['day']<datefloat(day)-42)]
        # if len(tbl_mape_upper)<1000:       # túl alacsony tételszám esetén ne mutassa az alsó határt (dayafter értékenként legalább 20 mape érték álljon rendelkezésre)
        #     tbl_mape_upper=None    
        tbl_knn = tbl_knn_all.loc[(tbl_knn_all['day']<datefloat(day)-42)]
        # - nem lehet benne jövőbeni információ (35 + G/2 nap)
        if len(tbl_knn)>3000:  # túl alacsony tételszám esetén ne mutassa a felső határt (dayafter értékenként legalább 100 diff-érték álljon rendelkezésre)
            count_per_dayafter = len(tbl_knn) / 50
            n_neighbors= int(min(100,count_per_dayafter/10))    # ne legyen nagyobb az egy dayafter-ben lévő minták tizedénél
            model_knn = f_knn_train(tbl_knn,'dayafter,y_pred_n','y_diff_n',scale='',n_neighbors=n_neighbors)

        # Calling the base function
        APE_peak = Fn_country_test(country,G=[14,21,28,35],plot='sub',xmax=day,bPred=True,verbose=1,
                tbl=tbl_covid,ser_orig=ser_orig,flatten=flatten,outtype='APE_peak',
                model_knn=model_knn,tbl_mape_upper=tbl_mape_upper)

    commenttopright='Epidemic curve, ' + country
    if weeks>1: commenttopright += ('  from ' + testday_first + 
                            '  to ' + datestr(datefloat(testday_first)+weeks*7) + 
                            ', ' + str(weeks) + ' week')
    pltshow(commenttopright=commenttopright,commentfontsize_add=2)

# fn_plot_predictions('Latvia','2022-01-10',weeks=8,optimalday=False)        # heti adatközlések napján
# fn_plot_predictions('Germany','2022-01-21',weeks=8,optimalday=False)    # heti adatközlések napján (ez látszik a legjobbnak)
# fn_plot_predictions('Germany','2022-09-01',weeks=12)    # elszállás ellenőrzés
# fn_plot_predictions('Austria','2022-01-24',weeks=8)       # publikálható
# fn_plot_predictions('France','2021-12-30',weeks=8,optimalday=False,flatten=True)  # publikálható (heti adatközlés)    
# fn_plot_predictions('France','2022-02-12',weeks=1)    
# fn_plot_predictions('Italy','2021-12-31',weeks=8)           # legnagyobb csúcs környéke, túlfutás
# fn_plot_predictions('Italy','2022-01-14',weeks=4,flatten=True)           # hibasávhoz, publikálható
# fn_plot_predictions('Hungary','2021-11-13',weeks=6)        # legnagyobb felfutás előtti csúcs, publikálható
# fn_plot_predictions('Hungary','2022-01-09',weeks=8,optimalday=False)      # legnagyobb felfutás, publikálható
# fn_plot_predictions('Hungary','2020-03-19',weeks=8)      # 1. hullám
# fn_plot_predictions('Hungary','2020-10-29',weeks=8,optimalday=False)        # 2. hullám    publikálható
# fn_plot_predictions('Hungary','2022-08-24',weeks=8)        
# fn_plot_predictions('Poland','2021-11-12',weeks=4)       
# fn_plot_predictions('Poland','2022-01-03',weeks=8)
# fn_plot_predictions('Slovakia','2022-01-21',weeks=8,flatten=True)
# fn_plot_predictions('Bulgaria','2022-01-01',weeks=8,flatten=True)
# fn_plot_predictions('United States','2020-09-25',weeks=1)    # összehasonlítható az egyik cikkel
# fn_plot_predictions('United States','2021-12-25',weeks=8)      # legnagyobb hullám, publikálható
# fn_plot_predictions('South Korea','2022-02-16',weeks=8,optimalday=False)      


def fn_plot_live_prediction(country,weeks=4,optimalday=True,flatten=False):   
    fn_plot_predictions(country,testday_first='lastday',weeks=weeks,optimalday=optimalday,flatten=flatten)
# fn_plot_live_prediction('Hungary',weeks=4)
# fn_plot_live_prediction('United Kingdom',weeks=4)
# fn_plot_live_prediction('Sweden',weeks=4)         
# exit()



# VERSION TESTING  (bVersion ramifications in the program code)
def fn_test_version2(countries,G,interval=9,reduced=False):
    global bVersion

    tbl_covid=Fn_read_coviddata()


    if type(countries)==str: countries=countries.split(',')

    if type(G)==int: aG=[G]
    else: aG=G

    verbose=0
    plot=False
    if len(countries)==1 and len(aG)==1: verbose=1

    losss,loss_olds=[],[]
    for G in aG:
        for country in countries:
            ser_orig=Fn_coviddata_country(tbl_covid,country,backfill=False)

            bVersion=False      # globális változó
            if (len(countries)==1) and len(aG)==1: plot=country + ' G' + str(G) + ' v0'      # fájlba írás
            loss_old,_=Fn_country_test(country,G,interval=interval,verbose=verbose,plot=plot,plot3='loss',
                                tbl=tbl_covid,ser_orig=ser_orig,reduced=reduced)



            bVersion=True
            if len(countries)==1 and len(aG)==1: plot=country + ' G' + str(G) + ' v1'      # fáljba írás
            loss,_=Fn_country_test(country,G,interval=interval,verbose=verbose,plot=plot,plot3='loss',
                                tbl=tbl_covid,ser_orig=ser_orig,reduced=reduced)

            losss.append(loss)
            loss_olds.append(loss_old)


            print_out=country + ' G' + str(G) + '    loss=' + strnum(loss,'2%') + '   loss_old=' + strnum(loss_old,'2%')
            if loss<loss_old: print_out += '   IMPROVED (' + strnum(loss_old-loss,'3%') + ')'
            elif loss>loss_old: print_out += '   worsened (' + strnum(loss-loss_old,'3%') + ')'
            print(print_out)

    if len(countries)>1 or len(aG)>1:
        print()

        loss_old=array(loss_olds).mean()
        loss=array(losss).mean()

        print_out='END RESULT    loss=' + strnum(loss,'2%') + '   loss_old=' + strnum(loss_old,'2%')
        if loss<loss_old: print_out += '   IMPROVEMENT (' + strnum(loss_old-loss,'3%') + ')'
        elif loss>loss_old: print_out += '   worsened (' + strnum(loss-loss_old,'3%') + ')'
        print(print_out)
# fn_test_version2('Hungary,Austria,Germany,Italy,France,Poland',G=[[14,21,28,35]],interval=10,reduced=True)
# fn_test_version2('Japan,Croatia,Ireland,United Kingdom,United States',G=[[14,21,28,35]],interval=10,reduced=True)
# fn_test_version2('Poland,United Kingdom,Sweden')
# fn_test_version2('Spain',G=[[14,21,28,35]],interval=10,reduced=False)
# exit()



# SYSTEMATIC TESTS FOR COUNTRIES
def fn_test_all(countries,G,interval,plot=False,flatten=True,dayofweek='optimal',dayfirst=None,daylast=None,above=0.05):
    ''' 
    Systematic testing for multiple countries and multiple G
    Countries: comma-separated list, or "group:Europe+"
    G: number   or   list    or    "weighted"
        - "weighted" calculate each G, with general weighting according to loss
    plot: If true, generates plot files per pair of (country,G) values
        - in the case of several countries, a lot of png files can be created
        - the files are placed in the Downloads folder
    dayfirst, daylast: a testing period can be specified

    return: creates two csv
         country,G,loss,loss_rel,loss_base,loss_linear,loss_days (multiple columns, 1-35)    
    
    '''
    tbl_covid=Fn_read_coviddata()

    countryToDayofweek = fn_dayofweek_countries()


    if beginwith(countries,'group:'):
        countries_in=cutleft(countries,'group:')
        countries=fn_countries(countries_in)
    else:
        countries_in=countries
        countries=countries.split(',')

    weighted=False
    if type(G)==int: aG=[G]
    elif type(G)==str and G=='weighted': 
        aG = [14,21,28,35]
        weighted=True
    else: aG=G

    recs=[]
    loss_train_all=[]
    for country in countries:
        ser_orig=Fn_coviddata_country(tbl_covid,country,backfill=False,flatten=flatten)

        for G in aG:
            if plot and not weighted: plot=country + ' G' + str(G)       # fájlba írás

            (loss,loss_rel,loss_base,loss_linear,loss_35,loss_rel_35,lognorm_base,lognorm_base_35,
             loss_days,knn_train,loss_train) = \
                Fn_country_test(country,G,interval=interval,flatten=flatten,verbose=0,plot=plot,outtype='loss_all',
                                tbl=tbl_covid,ser_orig=ser_orig,
                                dayofweek=dayofweek,countryToDayofweek=countryToDayofweek,
                                dayfirst=dayfirst,daylast=daylast,above=above)
                                # tbl_knn=tbl_knn,tbl_mape_upper=tbl_mape_upper)

            print(country + '  G' + str(G) + 
                '  loss=' + strnum(loss,'2%') + 
                '  Lognorm/Base=' + strnum(lognorm_base,'2%') +
                '  loss_rel=' + strnum(loss_rel,'2%') 
                )

            recs.append(Concat([country,G,loss,loss_rel,loss_base,loss_linear,
                                loss_35,loss_rel_35,lognorm_base,lognorm_base_35],loss_days))
            loss_train_all=Concat(loss_train_all,loss_train)

    if not plot:
        columns =  ('country,G,loss,loss_rel,loss_base,loss_linear,loss_35,loss_rel_35,lognorm_base,lognorm_base_35'.split(',') + 
                    ['day' + str(i) for i in range(50)])
        tbl_out=pd.DataFrame(recs,columns=columns)
        Tocsv(tbl_out,'Lognorm test ' + countries_in,format='hu')

        tbl_loss=pd.DataFrame(loss_train_all,columns='country,day,G,until_peak,phase_known,phase_ok,dayafter,y_pred_n,y_true_n,y_raw_n,y_base_n,y_linear_n,mape,mape_peak,smape,sign,score,score_factor'.split(','))
        Tocsv(tbl_loss,'loss_train lognorm',format='hu')

    # Loss számolás és kiírás
    print('Summary')
    print('lognorm/base14=' + strnum(tbl_out['lognorm_base'].mean(),'%') + '     ' +
          'lognorm/base35=' + strnum(tbl_out['lognorm_base_35'].mean(),'%'))
    # print('loss_rel=' + strnum(tbl_out['loss_rel'].mean(),'%'))

    mape14_mean = tbl_loss.loc[tbl_loss.dayafter==13,'mape'].mean()         # -1 napon van az adatsor vége
    mape14_median = tbl_loss.loc[tbl_loss.dayafter==13,'mape'].median()
    print('mape14 median=' + strnum(mape14_median,'%') + '     ' +
          'mape14 mean=' + strnum(mape14_mean,'%'))

    mape_1_35_mean = tbl_loss.loc[tbl_loss.dayafter<35,'mape'].mean()
    print('mape_1_35 mean=' + strnum(mape_1_35_mean,'%'))

# fn_test_all('group:Europe+',G=[[14,21,28,35]],interval=8,flatten=True)   # dayfirst='2021-07-01')
# fn_test_all('group:Europe+',G=[14,21,28,35],interval=8,flatten=True)      # multiG változat G_weights tanításhoz
# fn_test_all('Lithuania',G=[[14,21,28,35]],interval=8,dayofweek=None,dayfirst='2021-07-01')
# fn_test_all('Bulgaria,Hungary,Sweden',G=[[14,21,28,35]],interval=8,flatten=True,plot=True) #,dayofweek=0)

# fn_test_all('group:Europe+',G=[[14,21,28,35]],interval=1,flatten=True,
#                     dayfirst=datefloat('2021-01-09')-14,daylast=datefloat('2021-01-31')-14,above=0)  
#         # LSTM való összehasonlításhoz, a tesztidőszak (01-09 - 01-31) a célnapra vonatkozik (14 napos eltolás)

# exit()



# Processing of systematic test results   (partly obsolote)
def fn_loss_statistics(fname,outs='loss_by_G'):
    '''
    outs:  vesszős felsorolás
        loss_by_G:  print(pivot)
        loss_by_country:  print(pivot)
        plot_loss
        plot_weights
    '''
    
    path=Datapath(fname,'csv')
    tbl=Read_csv(path)

    tbl['lognorm/base']=tbl['loss']/tbl['loss_base']


    # Átlag számítás G értékenként
    if vanbenne(outs,'by_G'):
        # Teljes átlag
        # columns='loss,loss_rel,loss_base,loss_linear,lognorm/base'
        columns='loss,loss_rel,loss_base,loss_linear,loss_35,loss_rel_35,lognorm_base,lognorm_base_35'

        ser_mean=tbl[columns.split(',')].agg('mean')
        print(ser_mean)

        tbl_pivot=Pivot(tbl,'G',columns.split(','))
        print(tbl_pivot)

    # Átlag számítás országonként
    if vanbenne(outs,'by_country'):
        tbl_pivot=Pivot(tbl,'country','loss,loss_rel,loss_base,loss_linear,lognorm/base'.split(','))
        tbl_pivot = tbl_pivot.sort_values(by='lognorm/base_mean',ascending=False)
        print(tbl_pivot)


    # Plot a naponkénti loss-ra  (G értékenként)
    if vanbenne(outs,'plot|save_weights'):
        columns=tbl.columns.values
        columns2=[]
        for i,col in enumerate(columns):
            if vanbenne(col,'day'): columns2.append(int(cutleft(col,'day')))
            else: columns2.append(col)
        tbl.columns=columns2

        columns=[i for i in range(50)]
        
        tbl_pivot=Pivot(tbl,'G',columns,cols_out=['G'] + [i for i in range(50)])
        tbl_pivot = tbl_pivot.set_index('G')
        tbl_pivot=tbl_pivot.transpose()
        tbl_pivot['mean']=tbl_pivot.agg('mean',axis=1)

        if vanbenne(outs,'plot_loss'):
            pltinit(suptitle='Loss',
                    title='Loss by days after detection')
            for G in [14,21,28,35]:
                ser= tbl_pivot[G]
                FvPlot(ser,'original',label='G=' + str(G))
            pltshow(xlabel='days after detection',
                    ylabel='MAPE Mean absolute percentage error',
                    ynumformat='0%')

        if vanbenne(outs,'plot_weights'):
            pltinit(suptitle='Weights',
                    title='Weights by days after detection')
            for G in [14,21,28,35]:
                ser= (tbl_pivot['mean'] / tbl_pivot[G])**3
                FvPlot(ser,'original',label='G=' + str(G))
            pltshow(xlabel='days after detection',
                    ylabel='Weights')

        if vanbenne(outs,'save_weights'):
            tbl_out=pd.DataFrame()
            aG=[14,21,28,35]
            for iG,G in enumerate(aG):
                tbl_out[iG] = (tbl_pivot['mean'] / tbl_pivot[G])**3
            tbl_out[len(aG)] = tbl_out.agg('sum',axis=1)
            Tocsv(tbl_out,'Lognorm weights',format='hu')


        # print(tbl_pivot)


    # for G in [14,21,28,35]:
    #     tblG = Query_tbl(tbl,G==G)
    #     columns=

    #     serG=pd.Series()

def sub_print_losses(tbl_losses,title=''):        # háromféle loss illetve score számolása és kiírása
    '''
    title:  a sor elején megjelenő bevezető felirat. 
        Példa:  "phase all"   "phase=1"
    '''
    tbl=tbl_losses
    count=len(tbl)/50       # 50 tesztpont predikciónként
    mape14_median = tbl.loc[tbl.dayafter==13,'mape'].median()
    tblL=tbl.loc[tbl.dayafter<14]
    score_median = tblL['score_factor'].median()
    lognorm_per_base = tblL['mape_peak'].mean() / tblL['mape_peak_base'].mean()
    print(title + ' ' +  
            strint(count) + ' pred   ' +
            'mape14_median=' + strnum(mape14_median,'%') + '   ' +  
            # 'score_median=' + strnum(score_median) + '   ' +  
            'lognorm/base=' + strnum(lognorm_per_base,'%')
            )
    return mape14_median,lognorm_per_base,count


def fn_plot_loss_hist(dayafter=14):   # signed SMAPE by phase and dayafter
    tbl=Read_csv(Datapath('loss_train MAPE SMAPE','csv'))

    ncols=4
    pltinit(suptitle='Loss distributions',
            nrows=6,ncols=ncols,sharey=False,indexway='col')

    for phase in range(1,5):
        tblL=tbl.loc[tbl.phase==phase]
        for dayafter in [0,7,14,21,28,35]:
            title = 'phase=' + str(phase) + '  dayafter=' + str(dayafter)
            print(title)
            tblLL = tblL.loc[tblL['dayafter']==dayafter]
            def f_apply(rec):
                if rec.sign==1: rec['smape_signed']=-rec.smape
                else: rec['smape_signed']=rec.smape
                return rec
            tblLL = tblLL.apply(f_apply,axis=1)

            ser = tblLL['smape_signed']
            arr = ser.values
            
            outpoints=int(Limit(500/ncols,min=100))         # ha túl kicsi, akkor egy auto-encoded változó csúcsainak szélessége ingadozni fog
            ser_hist=hist_KDE(arr,0.1,outpoints)

            pltinitsub('next',title=title)
            x_middle = np.quantile(arr,0.5)
            FvPlot(ser_hist,plttype='original',annot=str(x_middle),annotcaption='x',area=True)

            # középső 50%
            x1=np.quantile(arr,0.25)
            x2=np.quantile(arr,0.75)
            FvPlot(ser_hist[x1:x2],plttype='original',annot='',area=True)

            # középső 50%           Inkább elhagyom, túlzsúfolt és nehezen érthető lesz tőle az ábra
            # x1=np.quantile(arr,0.25)
            # x2=np.quantile(arr,0.75)
            # FvPlot(ser_hist[x1:x2],plttype='original',annot='',area=True)


            pltshowsub(xlabel='SMAPE signed',yticklabels=False)

            # subplot_hist_ser(ser,
            #         title=title,
            #         G=0.2,xlabel='SMAPE signed')

    pltshow(commenttopright='Within the charts, the darker band indicates' + '\n' +
            'the central 90% of prediction losses.',
            commentfontsize_add=1)

    # for dayafter in [0,7,14,21,28,35]:
    #     tblL = tbl.loc[tbl['dayafter']==14]
    #     hist_plot(tblL,'loss',G=0.1)

def fn_write_loss_boundaries(method='mape_and_mapepeak'):
    tbl=Read_csv(Datapath('loss_train lognrom FINAL','csv'))

    if method=='mape_lower':
        recs=[]
        for dayafter in range(50):
            title = 'dayafter=' + str(dayafter)
            print(title)
            tblLL = tbl.loc[tbl['dayafter']==dayafter]

            # Csak a felülbecslések kellenek
            tbl_upper = tblLL.loc[tblLL['sign']==1]
            ser = tbl_upper['mape']
            arr = ser.values
            mape_90 = np.quantile(arr,0.9)

            recs.append([dayafter,mape_90])

        columns='dayafter,mape_90'.split(',')
        tbl_out=pd.DataFrame(recs,columns=columns)
        Tocsv(tbl_out,'loss distribution',format='hu')
        return


    recs=[]
    for phase in range(1,5):
        tblL=tbl.loc[tbl.phase==phase]
        for dayafter in range(35):
            title = 'phase=' + str(phase) + '  dayafter=' + str(dayafter)
            print(title)
            tblLL = tblL.loc[tblL['dayafter']==dayafter]

            if method=='smape':
                def f_apply(rec):
                    if rec.sign==1: rec['smape_signed']=-rec.smape
                    else: rec['smape_signed']=rec.smape
                    return rec
                tblLL = tblLL.apply(f_apply,axis=1)

                ser = tblLL['smape_signed']
                arr = ser.values
                
                smape_05 = np.quantile(arr,0.05)
                smape_25 = np.quantile(arr,0.25)
                smape_40 = np.quantile(arr,0.4)
                smape_50 = np.quantile(arr,0.5)
                smape_60 = np.quantile(arr,0.6)
                smape_75 = np.quantile(arr,0.75)
                smape_95 = np.quantile(arr,0.95)

                recs.append([phase,dayafter,smape_05,smape_25,smape_40,smape_50,smape_60,smape_75,smape_95])

            elif method == 'mape_and_mapepeak':
                # Külön kell kezelni az alulbecsléseket és a felülbecsléseket
                # Alulbecslések esetén mape_peak a kiindulópont, felülbecslés esetén mape
                
                # Alulbecslések
                tbl_lower = tblLL.loc[tblLL['sign']==0]
                ser = tbl_lower['mape_peak']
                arr = ser.values

                mape_peak_20 = np.quantile(arr,0.2)
                mape_peak_50 = np.quantile(arr,0.5)
                mape_peak_90 = np.quantile(arr,0.9)

                # Felülbecslések
                tbl_upper = tblLL.loc[tblLL['sign']==1]
                ser = tbl_upper['mape']
                arr = ser.values

                mape_90 = np.quantile(arr,0.9)
                mape_50 = np.quantile(arr,0.5)
                mape_20 = np.quantile(arr,0.2)


                recs.append([phase,dayafter,mape_90,mape_50,mape_20,mape_peak_20,mape_peak_50,mape_peak_90])

    if method=='smape':
        columns='phase,dayafter,smape_05,smape_25,smape_40,smape_50,smape_60,smape_75,smape_95'.split(',')
    elif method=='mape_and_mapepeak':
        columns='phase,dayafter,mape_90,mape_50,mape_20,mape_peak_20,mape_peak_50,mape_peak_90'.split(',')

    tbl_out=pd.DataFrame(recs,columns=columns)
    Tocsv(tbl_out,'loss distribution',format='hu')










# ============================================================================ #
# # GRAD2-MAX DYNAMICS
# ============================================================================ #

def Fn_grad2max_dyinamics(country,day_first=None,day_last=None,freq=2,grad_level=2):  # starthely vándorlás
    tbl=Fn_read_coviddata()
    ser=Fn_coviddata_country(tbl,country)

    ser=datefloat(ser)

    day_ser_first = int(datefloat(ser.index[0]))
    day_ser_last = int(datefloat(ser.index[-1]))
    
    if not day_first: day_first=day_ser_first
    else: day_first = int(datefloat(Date(day_first)))

    if not day_last: day_last=day_ser_last
    else: day_last = int(datefloat(Date(day_last)))

    G=20
    Gfaktor=1.2

    # Végső maxhelyek
    serGauss=FvGaussAvg(ser,gausswidth=G)
    serGrad1=FvGaussAvg(FvGradient(serGauss),gausswidth=int(G*Gfaktor))
    serGrad2=FvGaussAvg(FvGradient(serGrad1),gausswidth=int(G*Gfaktor))
    serGrad3=FvGaussAvg(FvGradient(serGrad2),gausswidth=int(G*Gfaktor))

    if grad_level==3: serGrad_level=serGrad3
    elif grad_level==2: serGrad_level=serGrad2

    points_min,points_max=serlocalminmax(serGrad_level,endpoints=False)
    X_max_global,Y_max_global=unzip(points_max)
    X_max_global=array(X_max_global)
    X_max_global=X_max_global[X_max_global<=day_last]
    X_max_global=list(X_max_global)
    X_min_global,_=unzip(points_min)
    X_min_global=array(X_min_global)
    X_min_global=X_min_global[X_min_global<=day_last]
    X_min_global=list(X_min_global)
    if X_max_global[0]<X_min_global[0]: X_min_global.insert(0,0)

    y_max_global=array(Y_max_global).max()
   
    # maxpos pontosítás:
    for i in range(len(X_max_global)):
        X_max_global[i] = fn_maxpos_precise(serGrad_level,X_max_global[i],halfwidth=2)


    # 2020-08-01-től nézem 2021-03-31-ig
    # Két naponta vizsgálom az utolsó grad2-maxhely vándorlását
    maxposs=arrayoflists(len(X_max_global))     # végső érték és naponkénti értékek
    detect_days=[None]*len(X_max_global)
    for day in Range(day_first,day_last,add=freq):
        progress(datestr(floatdate(day)))
        i_ser = day - day_ser_first
        if i_ser<1: continue
        serL=ser.iloc[:i_ser+1]
        serGauss=FvGaussAvg(serL,gausswidth=G)
        serGrad1=FvGaussAvg(FvGradient(serGauss),gausswidth=int(G*Gfaktor))
        serGrad2=FvGaussAvg(FvGradient(serGrad1),gausswidth=int(G*Gfaktor))
        serGrad3=FvGaussAvg(FvGradient(serGrad2),gausswidth=int(G*Gfaktor))

        if grad_level==3: serGrad_level=serGrad3
        elif grad_level==2: serGrad_level=serGrad2


        points_min,points_max=serlocalminmax(serGrad_level,endpoints=False)

        if len(points_max)==0: continue

        X_max,Y_max=unzip(points_max)
        X_max=list(X_max)

        # A mindenkori utolsó maxhely érdekel (pontos hely kell)
        i_maxpos = len(X_max)-1
        # Maxhely pontosítása  (resample és gauss)
        maxpos=X_max[i_maxpos]
        maxpos=fn_maxpos_precise(serGrad_level,maxpos,halfwidth=2)   # pontosítás regauss-szal (resample + gauss)

        # Az első észlelés dátumának tárolása
        if len(maxposs[i_maxpos])==0: detect_days[i_maxpos]=day+0.5   # nap közepén


        maxposs[i_maxpos].append(maxpos)

    # Jobbra rendezéshez kell a max hossz
    len_max=0
    for i in range(len(maxposs)):
        len_max=max(len_max,len(maxposs[i]))

    pltinit(suptitle='Grad' + str(grad_level) + ' maxhelyek eltolódása',
            title='Maxhely eltolódása az első észlelés óta')
    for i in range(len(X_max_global)):
        if len(maxposs[i])==0: continue
        if Y_max_global[i]<y_max_global*0.01: continue

        maxpos_start=maxposs[i][0]
        x_first=detect_days[i]-maxpos_start
        serL=pd.Series(maxposs[i]-maxpos_start,Range(x_first,add=2,count=len(maxposs[i])))

        annotplus=None
        if i+1<len(X_min_global): annotplus={X_min_global[i+1]-maxpos_start:'peak'}

        FvPlot(serL,'gauss',G=5,label=datestr(X_max_global[i]),annot='last',gaussside='',annotplus=annotplus)
    pltshow(xlabel='Maxhely első észlelésétől eltelt idő',
            commenttopright='Country: ' + country)

    return pd.Series(maxposs,X_max_global)
    # print(ser_out)
# Fn_grad2max_dyinamics('Belgium',grad_level=2,day_first='2020-08-01',day_last='2021-03-31',freq=2)
# exit()

def fn_grad2max_dynamics2(country,G,day_first=None,day_last=None,freq=2):  # grad2 görbék közvetlen rajzolása
    tbl=Fn_read_coviddata()
    ser=Fn_coviddata_country(tbl,country)

    ser=datefloat(ser)

    day_ser_first = int(datefloat(ser.index[0]))
    day_ser_last = int(datefloat(ser.index[-1]))
    
    if not day_first: day_first=day_ser_first
    else: day_first = int(datefloat(Date(day_first)))

    if not day_last: day_last=day_ser_last
    else: day_last = int(datefloat(Date(day_last)))


    # Lognormparams és kitevo beolvasása
    tbl_tune = Read_csv(Datapath('decompose_countries lognorm G_all leftw_free','csv'),index_col=['country','G'])
    rec = tbl_tune.loc[(country,G)]
    lognormparams = json.loads(rec.lognormparams)
    kitevo = rec.kitevo
    leftwidth = rec.boostwidth * 2.1

    resample=5
    serGauss_all,_,serGrad2_all,X_start_all,X_peak_all =   \
            fn_gauss_and_grad2(ser,G,resample=resample)
    serGauss_all_cut=serGauss_all.loc[(serGauss_all.index>=day_first) & (serGauss_all.index<=day_last)]


    # Két subplot egymás alatt
    pltinit(nrows=2,ncols=1,sharey=False,
            suptitle='Grad2 dynamics')

    # Elsőként az alsót rajzolom, mert az előrehaladó grad2 görbék alapján számolt tetőzési időpontok a részlognormokon
    #   lesznek megjelenítve (a felső diagramon)
    pltinitsub(1,title='Second derivative and the wandering of the max-position')

    # A végső grad2 görbe rajzolása
    FvPlot(serGrad2_all.loc[(serGrad2_all.index>=day_first) & (serGrad2_all.index<=day_last)],
            'original',label='final curve',annot='last')
    
    # start-pontok a végső grad2 görbén
    X_start_all = array(X_start_all)
    starts_x = X_start_all[(X_start_all>=day_first) & (X_start_all<=day_last)]
    starts_y = serGrad2_all[starts_x]
    FvPlot(pd.Series(starts_y,starts_x),'scatter',label='maxpos',annot='upper3')


    # Előrehaladó grad2 görbék, x_start pontokkal és előrejelzett tetőzési pontokkal (az utóbbiak az x_start-tal egy magasságban jelennek meg)
    starts_x,starts_y,peaks_x,peaks_index = [],[],[],[]
    for day in Range(day_first,day_last,add=freq):
        progress(datestr(floatdate(day)))
        serL=ser.loc[ser.index<=day]
        serGauss,serGrad1,serGrad2,X_start,X_peak =   \
                fn_gauss_and_grad2(serL,G,resample=5)
        
        FvPlot(serGrad2.loc[serGrad2.index>day-1.2*G],'original',annot='',colors=ddef(color='gray',alpha=0.3))

        start_x=X_start[-1]
        # Hagyja figyelmen kívül azokat a maxhelyeket, amelyek G/4-nél közelebb vannak a görbe végéhez
        if day - start_x < G/4: continue       # elvileg ezek is figyelembe vehetők, de erősen ingadozó eredményt adnak    
        starts_x.append(start_x)
        starts_y.append(serGrad2[start_x])

        # Előrejelzett tetőzési időpont
        maxpos = fn_maxpos_from_model(fn_get_maxpos_model(day),len(X_start)-1,serGrad2,X_start,X_peak,lognormparams,serGauss,leftwidth,G)
        if day<=maxpos:     # csak a valódi előrejelzések érdekelnek (a visszamenőleges nem kell)
            peaks_x.append(maxpos)
            peaks_index.append(len(X_start)-1)
    # X_start pontok az előrehaladó grad2 görbéken
    FvPlot(pd.Series(starts_y,starts_x),'scatter',annot='')
    # Előrejelzett tetőzési pontok az x_start pontokkal egy magasságban
    # FvPlot(pd.Series(starts_y,peaks_x),'scatter',annot='')

    pltshowsub()


    # Felső diagram rajzolása
    pltinitsub(0,title='Lognorm decomposition   ' + country + ',  G=' + str(G) + '   ' + datestr(day_first) + ' - ' + datestr(day_last) )


    # serGauss_all
    FvPlot(serGauss_all_cut,plttype='original',label='original curve',annot='max')         

    # startpontok a végső görbén
    X_start_all = array(X_start_all)
    starts_x = X_start_all[(X_start_all>=day_first) & (X_start_all<=day_last)]
    starts_y = serGauss_all[starts_x]
    FvPlot(pd.Series(starts_y,starts_x),'scatter',annot='')
    
    # serFitted
    X=list(serGauss_all_cut.index)
    serFitted = pd.Series(fn_lognormal_multi(X,*lognormparams),X)
    FvPlot(serFitted,plttype='original',label='Fitted',annot='max')


    # Rész-Lognorm-ok, előrejelzett tetőzési időponttal
    maxposs,maxvalues = [],[]
    peaks_x_lognorm,peaks_y_lognorm,peaks_x_lognorm_7,peaks_y_lognorm_7 = [],[],[],[]
    count=len(lognormparams)//4
    for i_lognorm in range(count-1):     # az utolsó nem kell
        maxpos=lognormparams[i_lognorm*4]
        maxvalue=lognormparams[i_lognorm*4+1]
        leftwidthL=lognormparams[i_lognorm*4+2]
        modusL=lognormparams[i_lognorm*4+3]

        startpos=maxpos - leftwidthL
        endpos = startpos + 2*leftwidthL
        if endpos<day_first or startpos>day_last: continue

        X=Range(startpos,count=int(3*leftwidthL))    # a részlognormok hossza legyen 3*leftwidth
        Y=fn_lognormal_multi(X,maxpos,maxvalue,leftwidthL,modusL)
        serLognorm = pd.Series(Y,floatdate(X))
        FvPlot(serLognorm,'original',
                colors=ddef(color='gray',alpha=0.3),annot='',area=True)
    
        # Előrejelzett tetőzési időpont (végső grad2 alapján)
        maxpos = fn_maxpos_from_model(model,i_lognorm,serGrad2_all,X_start_all,X_peak_all,lognormparams,serGauss_all,leftwidth,G)
        maxvalue = servalue(serLognorm,maxpos,True,True)
        maxposs.append(maxpos)
        maxvalues.append(maxvalue)

        # Előrejelzett tetőzési időpontok az előrehaladó grad2 görbék alapján
        index_count = Count(array(peaks_index)==i_lognorm)
        if index_count is not None:
            point_index=0
            for k in range(len(peaks_x)):
                if peaks_index[k]!=i_lognorm: continue
                until_peak=index_count - point_index       # index_count: a tényleges tetőzési napig hány felfutási nap volt
                # Külön rajzolom a -7-es pontot, annotálással
                if until_peak==7: peaks_x_lognorm_7.append(peaks_x[k])
                else: peaks_x_lognorm.append(peaks_x[k])

                y_step = serGauss_all_cut.max()/100         # y_irányban mekkora távolságra legyenek a napi maxpos előrejelzések
                # Külön rajzolom a -7-es pontot, annotálással
                if until_peak==7: peaks_y_lognorm_7.append(maxvalue - until_peak*y_step)
                else: peaks_y_lognorm.append(maxvalue - until_peak*y_step)
                point_index +=1
                # peaks_y_lognorm.append(servalue(serLognorm,peaks_x[k],True,True))
    # Előrejelzett tetőzések az előrehaladó grad2 görbék alapján
    FvPlot(pd.Series(peaks_y_lognorm,peaks_x_lognorm),'scatter',annot='',
        colors=ddef(color='orange'),scatter_size=1)
    FvPlot(pd.Series(peaks_y_lognorm_7,peaks_x_lognorm_7),'scatter',label='-7day',annot='upper3',
        colors=ddef(color='red'),scatter_size=1)

    # Előrejelzett tetőzések a végső grad2 alapján
    FvPlot(pd.Series(maxvalues,maxposs),'scatter',label='pred peak',annot='upper3',colors=ddef(color='red'))  # egyetlen pont rajzolása


    pltshowsub()


    pltshow()
# fn_grad2max_dynamics2('Italy',G=28,day_first='2021-10-01',day_last='2022-03-31',freq=1)
# fn_grad2max_dynamics2('Poland',G=56,day_first='2020-08-01',day_last='2021-06-30',freq=1)
# exit()








# ARIMA TEST FOR EPIDEMIC CURVES
def fn_plot_arima(country):
    tbl_covid=Fn_read_coviddata()
    ser_in=Fn_coviddata_country(tbl_covid,country,backfill=True)
    ser_in = datefloat(ser_in)
    ser_gauss = FvGaussAvg(ser_in,gausswidth=14)

    pltinit()
    FvPlot(ser_in,'scatter',label='original')
    FvPlot(ser_gauss,'original',label='gauss')

    # arima=ARIMA(order=(40,2,0) ,seasonal_order=(0,0,0,0),
    #                 enforce_stationarity=False,enforce_invertibility=False)
    arima=ARIMA(order=(7,4,7) ,seasonal_order=(0,0,0,0),
                    enforce_stationarity=False,enforce_invertibility=False)

    for day in Range(18940,18960,add=3): # Range(18900,19100,add=50):
        print('day=' + str(day))
        ser=ser_gauss.loc[ser_gauss.index<day]  #datefloat('2022.01.01')]

        arima.fit(ser)
        ser_new=pd.Series(arima.predict(7))
        # ser_new=pd.Series(arima.predict(40))
        ser_new.index = ser_new.index + ser.index[0]
        
        ser_old=pd.Series(arima.predict_in_sample())

        FvPlot(ser_new,'original',label='pred')
        # FvPlot(ser_old,'original',label='arima')
    pltshow()
# fn_plot_arima('Hungary')
# exit()
















