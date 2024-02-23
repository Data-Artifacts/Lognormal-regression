# from dateutil.parser import parse 
# from argparse import ArgumentDefaultsHelpFormatter
# from httplib2 import RelativeURIError
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sb
import numpy as np
from numpy import array
import pandas as pd
from datetime import datetime,date,timedelta
import copy
# import ez
import colorsys
import math
import re
from scipy.interpolate import interp1d,UnivariateSpline
from scipy.integrate import trapz
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import statsmodels.tsa.stattools as stat

from sklearn import linear_model
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor,KernelDensity

from sklearn.preprocessing import StandardScaler            # nullára centrálás és skálázás a szórással ([-1,1]-be kerül a többség)
from sklearn.preprocessing import MinMaxScaler              # [0-1] tartományba transzformálja
from sklearn.preprocessing import MaxAbsScaler              # csak skálázás
from sklearn.preprocessing import RobustScaler              # a pontok fele kerüljön a [-1,1] tartományba

from pmdarima.arima  import auto_arima,ARIMA
import pmdarima as pmd


from ezhelper import *
from ezdate import *

from xlsxwriter import *

import config
import locale



# Matplotlib háttérbeállítások
def matplot_init():
    ''' FIGYELEM  a debug megfigyelt változói közé ne vedd fel a plt.gcf() vagy a plt.gca() hívást,
      mert a debug ilyen helyből meg fog jelenni egy diagram ablak
    '''

    locale.setlocale(locale.LC_TIME, "hungarian")

    mpl.style.use('seaborn')
    plt.rcParams.update({'figure.figsize': (10, 5), 'figure.dpi': 100, 'savefig.dpi':300,'figure.titlesize':'x-large', 
                         'figure.subplot.hspace':0.3, 'figure.subplot.wspace':0.22,
                         'lines.linewidth':1, 'lines.markersize':3, 'axes.prop_cycle':plt.cycler("color", plt.cm.tab10.colors),
                         'axes.grid':True, 'axes.labelsize':'small', 'xtick.labelsize':'small', 'ytick.labelsize':'small',
                         'axes.titlepad':8,
                         'font.sans-serif':'Century Gothic', 'axes.formatter.use_locale':True})

    # plt.ioff()      # interactive mode kikapcsolása


    # plt.rcParams.keys()     # lehetséges paraméterek listája


    pd.set_option('display.max_columns',14)         # efelett truncated nézet (default 6;  adatforrás felrétképezésekeor tovább növelhető)
    pd.set_option('display.max_colwidth',50)        # oszloponként max milyen szélesség (az oszlopnevek láthatósága miatt nem lehet túl kicsi, default 80)
    pd.set_option('display.width',400)              # ha ennél nagyobb hely kell, akkor blokkszintű tördelés
    pd.set_option('display.max_rows',200)           # default 60   Efölött truncated nézet (head és tail esetén is)
    pd.set_option('display.min_rows',14)            # truncated nézetben hány sor jelenjen meg (default 10)
    pd.set_option('mode.use_inf_as_na',True)        # a végtelent és az üres stringeket is kezelje NA-ként (isna)
    pd.set_option('display.float_format', '{:,.10g}'.format)    # 10 számjegy, ezres határolással (sajnos csak ',' lehet)

matplot_init()



# SERIES / TABLES ÁLTALÁNOS MŰVELETEK

def serdtype(tblorser,colname='values'):
    '''
    colname:  ser esetén "index" vagy "values" lehet;   tbl esetén "index" vagy oszlopnév lehet
    return:  "string"  "num"   "datetime"
    '''
    dtype=''
    if colname=='index':
        dtype=tblorser.index.dtype
    elif colname=='values':
        dtype=tblorser.dtype
    else:
        dtype=tblorser[colname].dtype

    if dtype=='object': dtype='string'          # vegyes is lehet (ha egyetlen None előfordul, akkor is ide kerül)
    elif dtype in ['float64','int64']: dtype='num'      # nan lehet benne
    elif dtype=='datetime64[ns]': dtype='datetime'      # nincs automatikus parse

    return dtype

def isser(var):
    return type(var) is pd.Series

def istbl(var):
    return type(var) is pd.DataFrame



# def serfromtbl(tbl,valuescol,indexcol='index'):   # ha külön oszlopban van az index
#     if indexcol=='index':
#         return tbl[valuescol]        
#     else:
#         return pd.Series(tbl[valuescol].values,index=tbl[indexcol].values)

def Select_from(tbl,cols,indexcol='index',groupby='',query='',orderby='',agg=None,xmin=None,xmax=None):
    '''
    cols:       lazy felsorolása is megengedett  (lásd Tblcols)
                - ha groupby is meg van adva egynél több csoportértékkel, akkor csak egyelemű lehet
                - megadható "count(*)"  továbbá mean(), sum(), first(), ... statisztikai függvények írhatók a mezőkhöz
                  Ha legalább egy szerepel ezek körül, akkor az ismétlődő indexértékek összevonása, a megadott aggregálásokkal
                    Ha egynél meg van adva, akkor mindegiyknél meg kell adni
                  Az azonos aggregáló függvényhez tartozó mezők összevonhatóak. 
                    Példa: "count(*), mean(population,area), first(comment)"
                  Ugyanaz a mező többféle statisztikai függvénnyel is szerepelhet 
                  A kimeneti mezőnév megegyezik az eredetivel, kivéve ha ugyanaz a mező többféle függvénnyel is szerepel (a második esettől _mean utótagot kap)
                - statisztikai függvények:
                  'count', 'size', 'nunique',             - a count csak a notna értékeket számolja, a size az összeset
                  'sum', 'prod', 'min', 'max', 'idxmax', 'idxmin'
                  'mean', 'median', 'std', 'sem', 'var', 'skew', 'quantile',    - a quantile 0.5-tel értendő (ezen a módon nem paramétrezhető)
                  'first', 'last', 'any', 'all'           - szöveges mezőkre is működik

    indexcol:   a kimeneti tbl index-értékeit tartalmazó oszlop     ("index" esetén az input tbl indexe)
                - a kimenet indexe mindenképpen egy-oszlopos lesz
                - nem kötelező az unicitás (a cols-ban megadott statisztikai függvényekkel kérhető az ismétlődések összevonása, illetve az oszlopok aggregálása)
    groupby:    példa:  "country:hungary,slovakia"
                - szűrőfeltétel jellegű, nem foglal magába aggregálást, viszont új oszlopokat hozhat létre. A csoportosító mező nem lehet azonos a kimeneti indexxel.
                - ha csak a csoportosító mező van megadva, akkor az adott mező összes distrinct értéke megjelenik kimeneti oszlopként
                - ha több csoportosító érték van, akkor a cols csak egy mezőt tartalmazhat és a kimenetben minden 
                    csoportosító értékhez létrejön egy-egy oszlop (a megnevezése a csoportosító érték lesz)
                - a csoportosító mező nevét pontosan kell megadni (általában egy szöveges vagy egy int mező)
                - a csoport-értékek lazy szintaxissal adhatók meg (bárhol a szövegben; case insensitive; több találat esetén az első)
                    A függvény előzetesen bekéri a lehetséges csoportosító értékeket.
                - idézőjelezéssel jelezhető, hogy csak pontos egyezés fogadható el (kisbetű-nagybetű eltérés ilyenkor is megengedett)
    query:      SQL jellegű feltétel egy vagy több mezőre (and or használható). Az esetleges aggregálás előtt hajtódik végre.
                'country=="Hungary"'
                'country==@country'         # változónevek lehetnek benne  (a változó akár list is lehet, pl. 'country in @countries')         
                'A in [1,2,None]'
                'date>="2020-01-01" and date<"2020-01-02"'                  # egy nap összes időpontja  
                'country.str.contains("united|unified", case=False)'        # bárhol,  több mintával (regex)
                'country.str.contains("\bincome\b", case=False)'            # whole word (regex)
                'country.str.contains("united.*", case=False)'              # ugyanaz, mint a % az SQL-ben
                'country'                                   # a mező nem üres    
                'country.isna().values'                     # jó még:  country.notna().values  country.isnull().values
                'index.str.contains("united.*")             # az indexre az "index" képzetes oszlopnévvel lehet hivatkozni
                'A not in B'                                # az A mező értéke nincs benne a B mezőben
    having:     a kimeneti táblázatra vonaktozó feltétel. Elsősorban akkor van jelentősége, ha aggregálás történt.
                - "count(*)" vagy a kimeneti oszlopnevek adhatók meg (töb oszlopos groupby esetén a csoportnevek)
    orderby:    vesszős mezőfelsorolás, opcionálisan "desc" jelzésekkel.  Példa: "continent, country desc"
                - csak kimeneti oszlopnevek adhatók meg (töb oszlopos groupby esetén a csoportnevek). "index" is megadható. Aggregálás esetén count(*) is megadható.
                - kisbetű-nagybetű érdektelen, de teljes egyezés az elvárás  
    xmin,xmax:  szűrés a kimeneti index-re   (határok is beleértve)

    return tbl_out
    '''
    
    groupfilters=[]
    if groupby:
        groupcol,groupfilter = splitfirst(groupby,':')
        groupcol=groupcol.strip()
        if groupfilter: groupfilter=groupfilter.strip()
        # A táblában előforduló összes lehetséges csoportosító érték bekérése
        groups=tbl[groupcol].unique()       # distinct
        groupfilters = filternames(groups,groupfilter)
        if len(groupfilters)==0: print('ERROR  serfromtbl  group  Érvénytelen csoportosító érték: "' + groupfilter + '"')
        else:
            for groupfilter in groupfilters:
                tbl=tbl.loc[tbl[groupcol]==groupfilter]


    if query:
        tbl=tbl.query(query)


    # cols feldolgozása   fields és funcs
    fields,funcs = [],[]
    tokens=tokenize(cols,',',['hide_quot','hide_arg'])
    for token in tokens:
        m = re.match(r'(\w+)\(([^)]*)\)',token)
        if m:
            func = m.groups()[0]
            args = m.groups()[1]
            args = tokenize(args,',',['hide_quot'])
            for arg in args:
                fields.append(arg.strip('" '))
                funcs.append(func) 
        else:
            fields.append(token.strip('" '))
            funcs.append('')

    # Ha van legalább egy func, akkor mindegyikre meg kellett adni
    

    nFuncDb = len([x for x in funcs  if x!=''])
    if nFuncDb>0 and nFuncDb<len(funcs):
        print('ERROR  tblbysql  Vagy mindegyik oszlophpz függvényt kell megadni vagy egyikhez sem')
        return

    # Ha aggregálásra van szükség  (az index ismétlődő értékeinek összevonása)
    if nFuncDb>0:
        # dictionary előállítása a fields és a funcs alapján
        #  { 'colname_out':(colname,func), ... }
        kwargs={}
        for i,field in fields:
            func=funcs[i]
            # Értse az "avg" függvénynevet is
            if func=='avg': func='mean'
            # Ha többször is előfordul az adott field, akkor a függvénynevet is mőgé kell írni
            if len([x  for x in fields  if x==field])>1: colname_out=field + '_' + func
            else: colname_out=field
            kwargs[colname_out]=(field,func)
        
        tbl=tbl.groupby(indexcol).agg(**kwargs)
    # Ha nincs szükség aggregálásra, akkor a pandas pivot függvényét lehet használni

def Query_tbl(tbl,where=None,orderby=None,bCase=False,**wheres):     # Leválogatás, rendezés (az oszlopok nem változnak)  NEM INPLACE
    '''
    where vagy wheres:   
        dict:  euqal, in, like, tartomány   feltétel egy vagy több mezőre (pl. szűrőérték egy unique mezőre vagy mezőcsoportra)
            - a részszűrők közvetlen argumentumként is megadhatók (**wheres)
                Elsősorban akkor javasolt a közvetlen argumentumként vagy ddef-ben való megadás, ha equal 
                feltételről van szó (más esetekben szokatlan az '=' jel a részfeltételekben)
            - lista is megadható, szűrőértékként   ( field in [...] )
                Ha a lista első eleme 'NOT',  akkor "not in"  feltétel
            - tuple esetén >= és < feltétel   (ha valamelyik határ nem kell, akkor None)
                Stringek esetén is működik, de számít a kisbetű-nagybetű eltérés 
                None, '', np.nan értékeket jól kezeli, de nem fordulhat elő vegyesen szám és string az oszlopban.
            - ha egy string-érték elejére és/vagy végére "*" vagy "%" jelet írunk, akkor like jellegű feltételként értelmezi
                (csak egyedi értékként működik, listában jelenleg nem érvényesül)
            - np.nan, None,   'np.nan',  'not None', 'notna'   is megadható szűrőértékként  (többféle változat)
                (listában is szerepelhet np.nan vagy None;  a lista esetén hívott tbl.isin() helyesen kezeli)
                Az np.nan és a None a keresések szempontjából egyformán viselkedik (a '' viszont önálló eset)
                Példa:   B = ['NOT',np.nan,None,'',0]     - az összes érdemi érték
        query_string:    FONTOS  nem lehet benne @-változó, csak konstans feltétellel működik
            - a hívóhelyen viszont alkalmazható a tbl.eval(query_string), ami @-hivatkozások esetén is előállítja a szükséges bool-series-t
        loc-filter:   pl.  (tbl.A==1) | (tbl.B>"cc")      Ebben az esetben egy bool-series a tényleges input
            - az oszlopnevek elé mindenképpen beírandó a tbl, viszont nincsenek korlátozások 
                (pl. OR, != feltételek,  tetszőleges függvények és zárójelezések, oszlopok közötti relációk, ...)
            - AND : &,  OR : |, NOT : ~   és zárójelezni kell a részfeltételeket 
        where=None vagy where='all' esetén az összes sorra vonatkozik a művelet
        
        - kulcskeresés esetén dictionary-t érdemes megadni, de az in és like feltételek kezelése is ezzel a legegyszerűbb
        - konstans feltételeket egyszerűen megadhatók query szintaxissal is (pl. nem kell minősíteni az oszlopneveket)
        - bonyolult feltételeket inkább loc-filter szintaxissal
   
    orderby:    vesszős mezőfelsorolás, opcionálisan "desc" jelzésekkel.  Példa: "continent, country desc"
                - "index" is megadható
                - kisbetű-nagybetű érdektelen, de teljes egyezés az elvárás 

    bCase:  számít-e a kisbetű-nagybetű eltérés a like kereséseknél 

    PÉLDA:
        Query_tbl(tbl, ddef(A=1,B='aa*',C=(10,None))  )         # A==1 AND B like "aa%" AND C>=10
        
    '''

    if where is None and wheres is not None:  where=wheres

    # WHERE
    where = sub_where_to_boolser(tbl,where,bCase=bCase)

    if where is not None:
        tbl = tbl.loc[where]


    # ORDERBY
    if type(orderby)==str:
        orderby=orderby.lower().strip()
        if orderby=='index': tbl=tbl.sort_index(orderby)
        elif orderby=='index desc': tbl=tbl.sort_index(orderby,ascending=False)
        # Áttérés list-re
        else: orderby=orderby.split(',')

    if type(orderby)==list:
        orderbycols=[]
        ascending=[]
        for orderbycol in orderby:
            orderbycol=orderbycol.strip().lower()
            if endwith(orderbycol,' desc'): 
                orderbycol=cutright(orderbycol,' desc')
                ascending.append(False)
            else: ascending.append(True)
            colnamesout=filternames(list(tbl),orderbycol)
            if not colnamesout:
                print('ERROR   orderby   Érvénytelen oszlopnév: "' + orderbycol + '"')
                break
            else: 
                if len(colnamesout)>1: print('WARNING  orderby  A(z) "' + orderbycol + '" érték többféleképpen is előfordul (' + str(colnamesout) + ')')
                orderbycols.append(colnamesout[0])
        if len(orderbycols)>0:
            tbl=tbl.sort_values(orderbycols,ascending=ascending)

    return tbl

def Update_tbl(tbl,set,where,return_count=False,bCase=False):      # SQL update szintaxis;  INPLACE
    '''
    where:  
        dict:  "==", in, like, tartomány    feltétel egy vagy több mezőre (pl. szűrőérték egy unique mezőre vagy mezőcsoportra)
        query_string:    FONTOS  nem lehet benne @-változó, csak konstans feltétellel működik
        loc-filter:   pl.  (tbl.A==1) | (tbl.B>"cc")      Ebben az esetben egy bool-series a tényleges input
        None vagy 'all' esetén az összes sorra vonatkozik a művelet
        - részletesen lásd a Query_tbl-nél
    set:   dict vagy [cols,values]   
        dict:  pl. ddef(col1=1,col2='aa')
        [cols,values]:  a cols list vagy vesszős colname felsorolás, a values list vagy egyetlen érték
    return_count:  vissszaadja-e a módosított sorok számát
    bCase:  a like feltételek legyenek-e case-sensitivek

    Példák:
       Update_tbl(tbl, ddef(B=3,C=2),       ddef(A=1))
       Update_tbl(tbl, ddef(Adatgazda='Závecz'),  tbl.eval('Adatgazda.str.startswith("ZRI")'))
       Update_tbl(tbl, ddef(Adatgazda='Závecz'),  tbl.Adatgazda.str.startswith("ZRI")' )
       Update_tbl(tbl, ddef(A=0,B='BB'),    tbl.eval('A in [None,1] or B==6 or C not in ["b","d"]') )
       Update_tbl(tbl, ddef(A=0,B='BB'),    tbl.eval('A.isna() or B>6'))


    '''
    # set feldolgozása
    if type(set)==dict:
        cols=list(set.keys())
        values=list(set.values())
    elif type(set)==list:
        cols,values = set
        if type(cols)==str: cols=cols.split(',')
        if type(values)==str: 
            if len(cols)==1: values=[values]
            else: values=values.split(',')
   

    # Szűrés
    where = sub_where_to_boolser(tbl,where,bCase=bCase)


    if return_count: 
        if where is not None:
            count = where.sum()         # series, bool adatokkal
            if count>0: tbl.loc[where,cols]  =  values    
        else:
            tbl[cols] = values
            count=len(tbl)
        return count
    else:
        if where is not None:
            tbl.loc[where,cols]  =  values
        else:
            tbl[cols] = values

def Update_tbl_count(tbl,set,where,bCase=False):        # rekordszámot is visszaad;   INPLACE
    return Update_tbl(tbl,set,where,return_count=True,bCase=bCase)

def Update_or_insert_into(tbl,set,where,othervalues=None,bCase=False):   # NEM INPLACE
    '''
    Ha van már ilyen rekord, akkor update, egyébként insert

    set:    dict, módosítandó mezők a beírandó értékkel
    where:  dict, kulcsfeltétel     példa: ddef(index=154)  ddef(country='Hungary',day='2022-02-02')
            FONTOS: csak equal feltétel adható meg egy vagy több mezőre (ideértve az "index" mezőt is)
            Ha a where több rekordot ad eredményül, akkor mindegyikre végrehajtódik a módosítás
    othervalues:    dict,   az insert-hez szükséges olyan további adatok, amelyek sem a set-ben, sem a where-ben 
            nem szerepelnek 
    return:  tbl  (nem inplace, az insert lehetősége miatt)
    '''
    count=Update_tbl_count(tbl,set=set,where=where,bCase=bCase)
    if count==0: 
        insert = d_add(set,where)
        if othervalues:  insert = d_add(insert,othervalues)
        tbl = Insert_into(tbl,insert)

    return tbl


def Insert_into(tbl,record):    # Rekord hozzáfüzése autosorszámozott tbl-höz. NEM INPLACE
    '''
    Az új rekord a tábla végére kerül (autosorszámmal).
    A rekord új mezőket is tartalmazhat (a korábbi rekordokba nan érték kerül az új mezőhöz (akkor is, ha az új rekordban egy string szerepel))

    NE HASZNÁLD CIKLUSBAN.   Körülményes a használata, ráadásul lassú is.
    FIGYELEM   nem inplace jellegű     tbl=tblappend(tbl,records)
    Csak kis táblákra alkalmazható, viszonylag lassú művelet.  Előtte: tbl=pd.DataFrame()
    Ha a tbl-nek explicit indexe van, akkor tbl[indexvalue]=record   (ahol a record egy dict)
    
    HELYETTE:   
        list.append(rec) sorozat (a rec lehet list, tuple vagy dict), majd a végén: 
            tbl=pd.DataFrame(records,columns=['col1','col2',...])       # columns nélkül is jó, 0,1,2,... nevű oszlopokat vesz fel
            Ha meglévő táblához kell hozzáfűzni:   tbl = tblappend(tbl,records)
        
    record:   dict,  ser  vagy [cols,values]   
        dict:  pl. ddef(col1=1,col2='aa')
        ser:   pd.Seried([1,'aa'],['col1','col2'])
        [cols,values]:  a cols list vagy vesszős colname felsorolás, a values list
        - a record-ban nem szereplő oszlopok üres értéket kapnak
    '''
    if type(record)==dict:
        cols=list(record.keys())
        values=list(record.values())
    elif type(record)==list:
        cols,values = record
        if type(cols)==str: cols=cols.split(',')
    elif type(record)==pd.Series:
        cols=list(record.index)
        values=list(record.values)

    return pd.concat([tbl,pd.DataFrame([values],columns=cols)],ignore_index=True)

def Delete_from(tbl,where=None,bCase=False,**wheres):     # NEM INPLACE
    '''
    A feltételnek megfelelő rekordok törlése (elhagyása) a táblából

    where:     részletesen lásd a Query_tbl-nél
        dict:  "==", in, like, tartomány    feltétel egy vagy több mezőre (pl. szűrőérték egy unique mezőre vagy mezőcsoportra)
            A részfeltételek közvetlen argumentumként is megadhatók (**wheres)
        query_string:    FONTOS  nem lehet benne @-változó, csak konstans feltétellel működik
        loc-filter:   pl.  (tbl.A==1) | (tbl.B>"cc")      Ebben az esetben egy bool-series a tényleges input
        None vagy 'all' esetén az összes sorra vonatkozik a művelet
            Ha where=None és **wheres argumentumok sincsenek megadva, akkor az összes rekord törődik (csak az oszlopadatok maradnak)

    return  a törlések utáni tbl
    '''
    if where is None and wheres is not None:  where=wheres

    bool_ser = sub_where_to_boolser(tbl,where,bCase=bCase)


    if bool_ser is None: 
        return tbl.iloc[0:0]       # üres táblázat, az oszlopok maradnak
        # - a None arra utal, hogy nincs szűrés, tehát a táblázat összes rekordja törlendő
    else: 
        bool_ser = (bool_ser==True)   # előfordulhat benne None értéke, ami negaáláskor hibát okoz (pl. Delete_from_tbl)
        return tbl.loc[~bool_ser]

def sub_where_to_boolser(tbl,where,bCase=False):    # dict / string / loc-filter  formátumú feltétel  TO  ser of bools
    '''
    Részletes leírást lásd a Query_tbl-nél.
    Felhasználható Query-ben és Update-ben is

    return:  None vagy  bool-series   (a kulcs megegyezik a táblázat kulcsával)
    '''
    if type(where)==dict: 
        where_out_all=None
        for colname,value in where.items():
            where_out=None 
            if type(value)==list:
                if len(value)==0: where_out=None
                elif len(value)>1 and value[0]=='NOT': 
                    where_out = (~tbl[colname].isin(value))
                else:
                    where_out = (tbl[colname].isin(value))
            elif type(value)==tuple and len(value)==2:
                if value[0]:
                    if value[1]: where_out = (tbl[colname]>=value[0]) & (tbl[colname]<value[1])
                    else:  where_out = (tbl[colname]>=value[0])
                elif value[1]:
                    where_out = (tbl[colname]<value[1])
            elif type(value)==str:
                valueL=value.lower()
                if valueL in ['notna','notnan','not nan','not na','not np.nan']:
                    where_out = tbl[colname].notna()
                elif valueL in ['nan','np.nan']:
                    where_out = tbl[colname].isna()
                elif valueL in ['not none','notnone']:
                    where_out = tbl[colname]!=None
                if len(value)>1:
                    # Ha a végén * vagy %  van
                    if value[-1] in ['*','%']:
                        if value[0] in ['*','%'] and len(value)>2:
                            where_out = tbl[colname].str.contains(value[1:-1], case=bCase)
                        else:
                            where_out = tbl[colname].str.contains('^' + value[:-1] + '.*', case=bCase)
                    elif value[0] in ['*','%']:
                        where_out = tbl[colname].str.contains('.*' + value[1:] + '$', case=bCase)
            elif value is None:
                where_out = tbl[colname].isin([None])     # az isin() jól kezeli
            elif isinstance(value,(float,int)) and np.isnan(value):
                where_out = tbl[colname].isna()

            # minden més esetben equal feltétel
            if where_out is None: 
                where_out =  (tbl[colname]==value)

            if where_out_all is None: where_out_all = where_out
            elif where_out is not None:  where_out_all = where_out_all & where_out
        where=where_out_all
    elif type(where)==str:
        if where=='all': where=None
        else: where=tbl.eval(where)      # Series az eredmény, bool értékekkel
            # - nem lehet benne @-jeles változónév, mert csak a hívóhelyen ismertek a lokális változók

    return where


def serfromtbl(tbl,valuescol,indexcol='index',group='',query='',orderby='',aggfunc=None, xmin=None,xmax=None):
    '''
    valuescol:  y-értékeket tartalmazó oszlop   
                Speciális:  "count(*)"     Az indexcol unique értékkészletét adja vissza, tételszámokkal (default rendezés: count(*) desc)
                Extra esetben "index" is lehet: a táblázat indexe kerüljön bele 
    indexcol:   x-értékeket tartalmazó oszlop     ("index" esetén a tábla-index)
    group:    példa:  "country:hungary"
                - a csoportosító mező nevét pontosan kell megadni (általában egy szöveges vagy egy int mező)
                - a csoport-érték lazy szintaxissal adható meg (bárhol a szövegben; case insensitive; több találat esetén az első)
                    A függvény előzetesen bekéri a lehetséges csoportosító értékeket.
                - idézőjelezéssel jelezhető, hogy csak pontos mezőnév megfelelő (kisbetű-nagybetű eltérés ilyenkor is megengedett)
    query:      SQL jellegű feltétel az eredeti tábla egy vagy több mezőjére (and or használható)
                'country=="Hungary"'
                'A == [1,2]'
                'date>="2020-01-01" and date<"2020-01-02"'                  # egy nap összes időpontja  
                'country.str.contains("united|unified", case=False)'        # bárhol,  több mintával (regex)
                'country.str.contains("\bincome\b", case=False)'            # whole word (regex)
                'country.str.contains("united.*", case=False)'              # ugyanaz, mint a % az SQL-ben
                'country'                                   # a mező nem üres    
                'country.isnull()'                          # jó még:  country.notna()  country.isna()
                'index.str.contains("united.*")             # az indexre az "index" képzetes oszlopnévvel lehet hivaktozni
                'A not in B'                                # az A mező értéke nincs benne a B mezőben
    orderby:    vesszős mezőfelsorolás, opcionálisan "desc" jelzésekkel.  Példa: "continent, country desc"
                - kisbetű-nagybetű érdektelen, de teljes egyezés az elvárás  
                - csak a kimeneti mezők, továbbá az "index" szerepelhet benne
    aggfunc     Akkor lehet szükséges, ha az indexcol kimeneti értékkészlete nem unique. 
                Aggregálás az indexcol ismétlődő értékeire (általában csökken a rekrodok száma a value pedig az aggregált érték lesz)
                  'count', 'size', 'nunique',             - a count csak a notna értékeket számolja, a size az összeset
                  'sum', 'prod', 'min', 'max', 'idxmax', 'idxmin'
                  'mean', 'median', 'std', 'sem', 'var', 'skew', 'quantile',    - a quantile 0.5-tel értendő (ezen a módon nem paramétrezhető)
                  'first', 'last', 'any', 'all'           - szöveges mezőkre is működik
                count(*) jellegű aggregálást nem itt lehet kérni, hanem valuescol="count(*)" aggregálással
    xmin,xmax:  szűrés az indexcol-ra   (határok is beleértve)
    '''     
        
    groupfilters=[]
    if group:
        groupcol,groupfilter = splitfirst(group,':')
        groupcol=groupcol.strip()
        if groupfilter: groupfilter=groupfilter.strip()
        # A táblában előforduló összes lehetséges csoportosító érték bekérése
        groups=tbl[groupcol].unique()       # distinct
        groupfilters = filternames(groups,groupfilter)
        if len(groupfilters)==0: print('ERROR  serfromtbl  group  Érvénytelen csoportosító érték: "' + groupfilter + '"')
        else:
            if len(groupfilters)>1: print('WARNING  group  A(z) "' + groupfilter + '" érték többféleképpen is előfordul (' + str(groupfilters) + ')')
            tbl=tbl.loc[tbl[groupcol]==groupfilters[0]]
    if query:
        tbl=tbl.query(query)
    

    if orderby==valuescol or orderby==valuescol + ' desc': orderby=orderby.replace(valuescol,'values')
    if indexcol!='index' and (orderby==indexcol or orderby==indexcol + ' desc'): orderby=orderby.replace(indexcol,'index')
    # tbl rendezés
    if orderby and orderby not in ['index','index desc','values','values desc']:
        if type(orderby)==str:
            orderby=orderby.lower().strip()
            if orderby=='index': tbl=tbl.sort_index()
            elif orderby=='index desc': tbl=tbl.sort_index(ascending=False)
            # Áttérés list-re
            else: orderby=orderby.split(',')

        if type(orderby)==list:
            orderbycols=[]
            ascending=[]
            for orderbycol in orderby:
                orderbycol=orderbycol.strip().lower()
                if endwith(orderbycol,' desc'): 
                    orderbycol=cutright(orderbycol,' desc')
                    ascending.append(False)
                else: ascending.append(True)
                if orderbycol[0]!='"': orderbycol='"' + orderbycol + '"'
                colnamesout=filternames(list(tbl),orderbycol )
                if not colnamesout:
                    print('ERROR   orderby   Érvénytelen oszlopnév: ' + orderbycol)
                    break
                else:   # elvileg nen fordulhat elő, mert időézjelezve van (legfeljebb kisbetű-nagybetű eltérés miatt)
                    if len(colnamesout)>1: print('WARNING  orderby  A(z) ' + orderbycol + ' érték többféleképpen is előfordul (' + str(colnamesout) + ')')
                    orderbycols.append(colnamesout[0])
            if len(orderbycols)>0:
                tbl=tbl.sort_values(orderbycols,ascending=ascending)
    
    
    if valuescol=='count(*)':
        if indexcol=='index': values=tbl.index
        else: values=tbl[indexcol].values
        ser=pd.Series(values,index=values)
        ser=ser.groupby(ser.index).count()
    else:  
        if indexcol=='index':
            ser=tbl[valuescol]        
        elif valuescol=='index':
            ser=pd.Series(tbl.index,index=tbl[indexcol].values)
        else:
            ser=pd.Series(tbl[valuescol].values,index=tbl[indexcol].values)

        if aggfunc:
            ser=Ser_setunique(ser,aggfunc)
    
    if len(groupfilters)==1: ser.name=groupfilters[0]
    else: ser.name=valuescol

    # ser rendezés
    if orderby and orderby in ['index','index desc','values','values desc']:
        if orderby=='index': ser=ser.sort_index()
        if orderby=='index desc': ser=ser.sort_index(ascending=False)
        if orderby=='values': ser=ser.sort_values()
        if orderby=='values desc': ser=ser.sort_values(ascending=False)

    
    if xmin: ser=ser[ser.index>=xmin]
    if xmax: ser=ser[ser.index<=xmax]
        
    return ser

def SerFromRecords(aRec:list,xname:str='id',yname:str='',sortindex=False):      # -> pd.Series
    # a visszaadott series indexe nem feltétlenül rendezett (ser.sort_index()-re lehet szükség). Az sem elvárás, hogy unique legyen
    if len(aRec)==0: return pd.Series()
    aX,aY=unzip(aRec)
    return SerFromArrays(aX,aY,xname,yname,sortindex)

def SerFromArrays(aX:list,aY:list,xname:str='id',yname:str='',sortindex=False):      # -> pd.Series
    # a visszaadott series indexe nem feltétlenül rendezett (ser.sort_index()-re lehet szükség). Az sem elvárás, hogy unique legyen
    # xname: 'date', 'datum' esetén  to_datetime
    ser=pd.Series(aY,aX)
    if xname: ser.index.name=xname
    if yname: ser.name=yname
    if xname in ['date','datum']: ser.index=pd.to_datetime(ser.index)
    if sortindex: ser.sort_index(inplace=True)
    return ser

def TblFromRecords(aRec:list,colnames:str,indexcols:str='',sortindex=False):    
    # colnames:  pl 'id,adat1,adat2'    vesszős felsorolás
    # indexcols:  pl. 'id'        nem kötelező, hogy legyen index (ha nincs, akkor autosorszámozott)
    #    - nem kell unique-nak lennie (ellenőrzés: index.is_unique()), és a rendezettség sem elvárás (plot előtt tbl.sort_index()-re lehet szükség)
    columns=colnames.split(',')
    if indexcols: index=indexcols.split(',') 
    else: index=None
    tbl=pd.DataFrame.from_records(aRec,columns=columns,index=index)
    if sortindex: tbl.sort_index(inplace=True)
    return tbl


def Array_num(arr_list):         # csak az int,float,bool  elemeket tartja meg
    '''
    return:  float tömb (bool értékek helyett is).  Kivéve: ha csak int értékek, akkor int-tömb
            - az nan értékek is bekerülnek
            - None értékek nem kerülnek be
    '''
    return array([x for x in arr_list  if isinstance(x,(int,float))])

def Array_str(arr_list):         # csak az str elemeket tartja meg
    '''
    return:  str tömb (dtype=object).
            - az '' értékek is bekerülnek
            - None értékek nem kerülnek be
    '''
    return array([x for x in arr_list  if isinstance(x,str)])

    
def Count(filter):      # tbl, ser, array megadott feltételnek megfelelő értékeinek száma
    '''
    Szűrőfeltételnek megfelelő elemek száma
    Ha maga a szűrés kell, akkor  ser[ser>2]  arr[arr>2]  ...

    filter: array, ser, tbl objektum elemire vonatkozó szűrőfeltétel 
        - list esetén előtte array konverzió kell
        - list esetén használható ehelyett a list.count(value) függgvény, de ez általános szűrésre nem alkalmas

    Példa:
        Count(tbl==0)       mindig működik (string-cellák nem okoznak hibát).  0-t tartalmazó cellák száma   (string-konstansra is jó)
        Count(tbl<0)        csak akkor működik, ha a tbl-ben nincs string-adat
        Count((ser<10) | (ser>20))          fontosak a zárójelek.  Csak a values oszlopot nézi (az indexet nem)   
        Count((arr>=10) & (arr<=20))        
        
        Count(~np.isin(arr,[0,False,'',None]))    összes elem, kivéve 0,False,'',None

        Count(array(list)!=None)    nem None elemek  (list csak array konverzióval működik)
                            - ugyanez nem működik az np.nan-nel (csak az np.isna() működik)
        

        Count(tbl.isna())
        Count(ser.notna())
        Count(ser.null())
        Count(~np.isnan(arr))

        Count((ser==0) | (ser=='') | (ser.isna()) | (ser is None))       Fontosak a zárójelek
    '''
    if type(filter)==pd.DataFrame: return (filter).sum().sum()          # bool-értékekből álló dataframe
    elif type(filter)==pd.Series: return (filter).sum()
    elif type(filter)==np.ndarray: return (filter).sum()

def Count_nan(arr_ser_tbl):         # nan értékek száma. Vegyes tömbökre és listákra is működik
    '''  Az nan értékek számának lekérdezéséhez nem jó az arr=np.nan szűrés
    arr_ser_tbl:   array, list, Series, Tbl  
        - ha str vagy None értékek is vannak benne, akkor először kiszűri csak a számokat (exception kezeléssel)
    '''
    if type(arr_ser_tbl)==list: arr_ser_tbl=array(arr_ser_tbl)

    if type(arr_ser_tbl)==np.ndarray: 
        try:
            return np.isnan(arr_ser_tbl).sum()
        except:     # ha vegyes adattípusú
            return np.isnan(Array_num(arr_ser_tbl)).sum()       # előtte csak a 
    elif type(arr_ser_tbl)==pd.DataFrame: return arr_ser_tbl.isna().sum().sum()        # bool-értékekből álló dataframe
    elif type(arr_ser_tbl)==pd.Series: return arr_ser_tbl.isna().sum()

def Count_notna(arr_ser_tbl):         # nem nan értékek száma
    '''  Megjegyzés: Az nan értékek számának lekérdezéséhez nem jó az arr=np.nan szűrés
    arr_ser_tbl:   array, list, Series, Tbl
        - list esetén csak a számként értelmezhető értékeket veszi figyelembe (None, str értékeket nem)
    '''
    if type(arr_ser_tbl)==list: arr_ser_tbl=array(arr_ser_tbl)

    if type(arr_ser_tbl)==np.ndarray: 
        try:
            return (~np.isnan(arr_ser_tbl)).sum()
        except:
            return (~np.isnan(Array_num(arr_ser_tbl))).sum()
    elif type(arr_ser_tbl)==pd.DataFrame: return (arr_ser_tbl.notna()).sum().sum()        # bool-értékekből álló dataframe
    elif type(arr_ser_tbl)==pd.Series: return (arr_ser_tbl.notna()).sum()


def Concat(list_arr1,list_arr2):      # egydimenziós listák vagy tömbök összefűzése
    '''
    list, tuple, array, DataFrame és Series objektumokra működik.
    A két list_arr objektumnak alapesetben azonos típusúnak kell lennie, de az egydimenziós list-tuple-array 
        párosítások megengedettek. DataFrame és Series esetén viszont elvárás a típusegyezés, kétdimenziós
        array esetén pedig a második index méretének is egyeznie kell.
        
    Azért hasznos, mert élesen eltér a konkatenáció szintaxisa listák illetve array objektumok esetén, ráadásul
        a "+"  művelet egészen mást jelent array-ek esetén   (keveredést okozhat, mert nem mindig egyértelmű, hogy
        történt-e list_array konverzió)
    list1 + list2   illetve   np.concatenate(arr1,arr2)

    Kétdimenziós tömbökre is működik, de elvárás, hogy az oszlopok száma megegyezzen (második index)
       A második tömb rekordjait hozzáfűzi az első tömb rekordjaihoz.
    Series esetén összefűzés. Nem figyeli az indexértékek unicitását.
    Dataframe-ek esetén eltérő oszlopok is megengedettek. Ha egy oszlop nem szerepel valamelyik tbl-ben,
        akkor nan értékek kerülnek bele
        Az indexértékeknek nem kötelező unique-nak lenniük (nem egyesíti az azonos indexű sorokat)
    '''
    if list_arr1 is None:  return list_arr2
    elif list_arr2 is None: return list_arr1
        
    if type(list_arr1)==list:
        return list_arr1 + list(list_arr2)
    elif type(list_arr1)==tuple:
        return list(list_arr1) + list(list_arr2)
    elif type(list_arr1)==np.ndarray:
        return np.concatenate([list_arr1,list_arr2])
    
    elif type(list_arr1)==pd.DataFrame and type(list_arr2)==pd.DataFrame:
        return pd.concat([list_arr1,list_arr2])

    elif type(list_arr1)==pd.Series and type(list_arr2)==pd.Series:
        return pd.concat([list_arr1,list_arr2])
    


def serisempty(ser):        # nincs benne üres értéken kívüli elem (nem None,'',False,0)
    ''' 
    True, ha None, nincs sora, vagy minden értéke üres (None, nan, '')
    - a ser.empty nem ugyanezt jelenti, mert egy nan értékeket tartalmazó ser-t nem tekint empty-nek
    - az üres string csak akkor minősül üres értéknek, ha be van állítva:  pd.set_option('mode.use_inf_as_na',True)
    '''
    return ser is None or len(ser)==0 or ser.notnull().sum()==0

def servalues(ser,bIndex=False):      # Értékeloszlás (leggyakoribb értékek legelöl)
    if bIndex: return ser.index.value_counts()
    else: return ser.value_counts()

def servalue(ser,x,assume_sorted=False,assume_unique=True):    # interpolációval
    '''
    ser interpolált értéke tetszőleges x pontban   (a ser egy mintavételes függvényként értelmezhető)
    - először megnézi, hogy van-e ilyen x érték a ser x-tömbjében. Ha van, akkor a hozzá tartozó ser-value-t adja vissza.
    - ha x nincs benne az [xmin,xmax] tartományban, akkor az utolsó ismert szakasz meredekségével számol
    - ha egyetlen pont van a ser-ben, akkor konstans függvénynek tekinti a ser-t (y=y0)

    ser:  az x-értékek dátumok is lehetnek
    x:   dátum is lehet
        
    assume_sorted:  True, ha rendezve volt a ser a hívás előtt (x értékekre)
    assume_unique:  True, ha a ser x-értékei garantáltan nem ismétlődnek  
        False esetén a függvény összevonja az ismétlődő x-értékeket (átlag az y értékekre)    
    '''

    try:
        y=ser[x]
        if type(y)==float: return y     # előfordulhat, hogy series-t ad vissza
    except:
        pass

    if len(ser)==1: return ser[0]
    
    if not assume_sorted: ser=ser.sort_index()
    if not assume_unique: ser=ser.groupby(level=0).mean()
    
    x=datefloat(x)
    aXSer=datefloatA(ser.index.array) 
    aYSer=ser.array

    i = np.searchsorted(aXSer,x)
    if i>0:
        # ha jobbra esik az aXser-től
        if i==len(aXSer): i=i-2
        # ha belül van az intervallumon   (ha balra van, tehát i==0, akkor a [0,1] indexű szakasz kell)
        else: i=i-1
    
    x1=aXSer[i]
    x2=aXSer[i+1]
    y1=aYSer[i]
    y2=aYSer[i+1]
    
    m=(y2-y1) / (x2-x1)
    y=y1 + m * (x-x1)

    return y

    
def servalue_old(ser,x,assume_sorted=False):    # interpolációval
    # Mintavételes függvény interpolált értéke tetszőleges x pontban
    # - az   [xmin,xmax] tartományon kívül az utolsó ismert szakasz meredekségével számol
    
    # Először megnézi, hogy van-e ilyen x érték a ser x-tömbjében. Ha van, akkor a hozzá tartozó ser-value-t adja vissza.
    # Ha nincs, akkor lineáris interpolációval kéri be az értéket (az x-nek a végpontokon belül kell lennie) 
    # - a végpontokon kívüli tartományokban megbízhatatlan lehet (előtte egy simításra lehet szükség, pl. gauss, spline)
    # - lényegesen időigényesebb  (sok pont esetén lásd servalueA)
    #     Megjegyzés: egy sokkal gyorsabb algoritmust is lehetne írni, a két szomszédos (xy) pár alapján

    try:
        y=ser[x]
        if type(y)==float: return y     # előfordulhat, hogy series-t ad vissza
    except:
        pass

    aXSer=datefloatA(ser.index.array) 
    aYSer=ser.array


    
    f=interp1d(aXSer,aYSer,assume_sorted)
    aY=f([datefloat(x)])
    return aY[0]

  

def servalueA(ser,aX,outtype='array'):     # interpolációval, több x-re
    # A ser folytonos függvényként való kiterjesztése (lineáris interpoláció a pontok között, és a végpontokon kívül is) 
    #     és a megadott x-értékekhez a függvényérték bekérése
    # A végpontokon kívüli tartományokban megbízhatatlan lehet (előtte egy simításra lehet szükség, pl. gauss, spline)
    # outtype:  'array' vagy 'dict' vagy 'records'

    aXSer=datefloatA(ser.index.array) 
    aYSer=ser.array

    f=interp1d(aXSer,aYSer,assume_sorted=False)
    if outtype=='array': return f(datefloatA(aX))
    elif outtype=='dict': return dict(zip(aX,f(datefloatA(aX))))
    elif outtype=='records': return zip(aX,f(datefloatA(aX)))


def serfirstvalue(ser):         # az első olyan xy pár, amiben az y nem NaN.  Ha nincs ilyen xy, akkor None,None
    values=ser.values
    ifirst=None
    for i in range(len(values)):
        if pd.isna(values[i]): continue
        ifirst=i
        break
    if ifirst==None: return (None,None)
    else: return (ser.index[ifirst],values[ifirst])

def serlastvalue(ser,maxstep=None,out_='tuple'):         # az utolsó olyan xy pár, amiben az y nem NaN.  Ha nincs ilyen xy, akkor None,None
    '''
    Megkeresi az utolsó olyan pontot, ahol a ser értéke nem nan és nem 0
    Elsősorban idősorok esetén alkalmazható
    Ha nincs utolsó érték, akkor return None
    
    maxstep:  max hány lépést (pl. napot) mehet vissza az idősor végétől
        megadható <1 float-ként is  (hányadrésze a teljes hossznak)
    out:
    '''
    
    values=ser.values

    nlen=len(values)
    rangelast=-1
    if maxstep: 
        if maxstep<1: maxstep = int(nlen*maxstep)
        else: maxstep=int(maxstep)
        rangelast=nlen-maxstep-1
        if rangelast<-1: rangelast=-1
    
    ilast=None
    for i in range(nlen-1,rangelast,-1):
        if pd.isna(values[i]) or values[i]==0: continue
        ilast=i
        break
    
    if out_=='tuple':
        if ilast==None: return (None,None)
        else: return (ser.index[ilast],values[ilast])
    elif out_=='ilast':
        return ilast

def serlocalmax(ser,maxmin='max',halfwidth=2,mindistance=0,sidecut=0,endpoints=False):    #  out:  aXy - lokális max-pontok (xy),  legnagyobb y legelöl
    '''
    A visszadatt tömb üres is lehet  (ha nincs egyetlen lokális maxpont sem)
    ser:  az x értékek float vagy datetime típusúak és növekvő sorrendben rendezettek, az y float  (nan is lehet benne)
        - ha a ser nem rendezett x-re, akkor előtte ser=ser.sort_index()
    maxmin:  'max'  vagy 'min'    lokális max-helyek vagy lokális min-helyek
    halfwidth:  
        - a lokális max-helyek elvárt (fél)szélessége   
        - lokális maximum, ha mindkét irányban legalább ennyi szomszédos pontra teljesül a max-elvárás (1-nél nem lehet kisebb)
    mindistance (0-1):  a túl sűrűn következő maxhelyek elhagyása; legalább ekkora távolságnak kell lennie két localmax között (a teljes x-szélességhez viszonyítva)
    sidecut (0-0.5):  a jobb és a bal szél figyelmen kívül hagyása  (a teljes x-szélesség ekkora hányadát kell figyelmen kívül hagyni mindkét szélen)
    endpoints: True esetén a szélső pontok is bekerülhetnek (csak befelé ellenőriz)
    '''

    if serisempty(ser): return []
    ser=ser.dropna()

    width=datefloat(ser.index.max()) - datefloat(ser.index.min())    # x-re elvárás a rendezettség, ezért ser.index[-1] és ser.index[0] is lehetne
    if width==0:  return []

    if sidecut>0:
        if sidecut>=0.5: return []
        aX=datefloatA(ser.index)
        indexfirst=len(aX)
        for i in range(len(aX)):
            if aX[i]-aX[0]>sidecut*width:
                indexfirst=i
                break
        indexlast=0
        for i in range(len(aX)-1,-1,-1):
            if aX[-1]-aX[i]>sidecut*width:
                indexlast=i
                break
        if indexlast<indexfirst: return []
        ser=ser[indexfirst:indexlast]

    nLen=len(ser)

    aY=ser.values
    if halfwidth==None: halfwidth=2
    if halfwidth<1: return []
    


    aIndex=[]   # lokális szélsőértékek pozíciója (max vagy min)
    # Minden pontra vizsgálandó, hogy lokális max-ról van-e szó
    if endpoints:  rangeL=range(0,len(aY))          # ha a széleken lévő pontok is bekerülhetnek az out-ba
    else: rangeL=range(halfwidth,len(aY)-halfwidth)       # alapesetben a széleken nem vizsgálandó halfwidth számú pont
 
    if maxmin=='max':
        for i in rangeL:    # ciklus az összes pontra (kivéve a széleken halfwidth)        
            bOk=True
            for j in range(1,halfwidth+1):
                if ((i-j)>=0 and aY[i]<=aY[i-j]) or ((i+j)<nLen and aY[i]<=aY[i+j]): 
                    bOk=False
                    break
            if bOk: aIndex.append(i)
    else:   # min
        for i in rangeL:        # ciklus az összes pontra (kivéve a széleken halfwidth) 
            bOk=True
            for j in range(1,halfwidth+1):
                if ((i-j)>=0 and aY[i]>=aY[i-j]) or ((i+j)<nLen and aY[i]>=aY[i+j]): 
                    bOk=False
                    break
            if bOk: aIndex.append(i)

    aPoints=[]
    for i in range(len(aIndex)):
        aPoints.append((ser.index[aIndex[i]],ser.iloc[aIndex[i]]))
    aPoints.sort(key=lambda x: x[1],reverse= (maxmin=='max'))       # y szerinti rendezés (max esetén reverse)

    if mindistance>0:
        mindistance=mindistance*width
        aPointsL=[]
        aXL=[]
        for point in aPoints:
            mindistanceL=width
            x=datefloat(point[0])
            for xL in aXL:
                distance=abs(x-xL)
                if distance<mindistanceL: mindistanceL=distance
            if mindistanceL>mindistance:
                aXL.append(x)
                aPointsL.append(point)
        aPoints=aPointsL

    return aPoints

        
def serlocalminmax(ser,halfwidth=2,mindistance_x=0,mindistance_y=0,endpoints=True):   # rendezett (váltakozó) localmin-localmax sorozatokat ad vissza
    '''
    Felváltva keres lokális max és minhelyeket. 
    Elvárás, hogy min után a következő max nagyobb legyen, illetve max után a következő min kisebb (nem elég az egyenlőség)
    A szélekhez megpróbál valamilyen szélsőértéket hozzárendelni  (kikapcsolható az endpoints=False beállítással)
        kivéve: konstans függvény, vagy a legszélső érték outlier jellegű
    Lokális maximum: >= a 2*halfwidth környezetében  (akár minden szomszédja egyenlő lehet vele)
    Lokális minimum: <= a 2*halfwidth környezetében  (akár minden szomszédja egyenlő lehet vele)

    Megjegyzés: érdemes lenne próbálkozni a scypy find_peaks() vagy argrelmax() függvényeivel, mert viszonylag
        lassú a függvény. Mindamellett a max és minhelyek szigorú váltakozása, továbbá az az elvárás, hogy a
        platókat is meg kell találnia, viszonylag nehézzé teszi az említett függvények felhasználását.
        (a find_peaks() elvileg tudja kezelni a platókat)
    Futásidő:  1500 pont, 100 minmaxhely  =  0.005 sec körül


    ser:  az x értékek float vagy datetime típusúak, az y float  (nan is lehet benne)
        - az x értékeknek rendezettnek kell lenniük  (szükség szerint előtte ser=ser.sort_index())
    halfwidth:  
        - a lokális szélsőérték-helyek elvárt (fél)szélessége   
        - lokális maximum, ha mindkét irányban legalább ennyi szomszédos pontra teljesül a max-elvárás (1-nél nem lehet kisebb)
    mindistance_x:  a szélsőérték helyek között legalább ekkor x-távolság (a két szélső pontra ez nem feltétlenül érvényesül)
        0 esetén nincs ilyen szűrés
    mindistance_y:  NEM KELL    A grad2 lüktetésre végül egy újabb gauss átlagolás jobb megoldást adott.
        törlendő minden olyan szélsőérték pár, amelynél az y-irányú eltérés kisebb ennél, kivéve ha a 
            maxhely másorendben is maxhely vagy a minhely másodrendben is minhely
        0 esetén nincs ilyen szűrés
        - lüktetés kiszűrésére használható (példa: gauss simítás esetén is lehet egy maradvány-lüktetés, ami a közel
            vízszintes platókon indokolatlan szélsőértékeket eredményezhet)
        - megjegyzés: elvileg felmerülhet többféle periódusidejű lüktetés is. Jelenleg csak a legalacsonyabb periódusú
            lüktetést szűri ki
    endpoints: True esetén a két szélső pont minmax helynek minősül (kivéve: konstans függvény, vagy a legszélső érték outlier jellegű)

    return points_min,points_max    
    '''

    if serisempty(ser): return [],[]
    ser=ser.dropna()

    width=datefloat(ser.index.max()) - datefloat(ser.index.min())    # x-re elvárás a rendezettség, ezért ser.index[-1] és ser.index[0] is lehetne
    if width==0:  return [],[]

    nLen=len(ser)

    aX=ser.index
    aY=ser.values
    if halfwidth==None: halfwidth=2
    if halfwidth<1: return [],[]    
    
    if nLen<halfwidth + 1: return [],[]

    # Potenciális szélsőérték helyek előzetes feltérképezése
    
    minmax_indexes=[]
    minmax_types=[]         # -1, 1,  vagy 0  (ha utólag törölni kell, akkor 0 kerül bele)

    # Az összes potenciális szélsőérték pont felderítése
    if endpoints:   i,i_last = 0,nLen-1
    else:           i,i_last = halfwidth,nLen-halfwidth-1
    
    while i<=i_last:
        # nNagyobbDb = np.sum(aY[max(0, i - halfwidth) : i] > aY[i]) + np.sum(aY[i+1 : min(nLen, i + halfwidth + 1)] > aY[i])
        # nKisebbDb = np.sum(aY[max(0, i - halfwidth) : i] < aY[i]) + np.sum(aY[i+1 : min(nLen, i + halfwidth + 1)] < aY[i])

        # Átugrom a monoton növekvő illetve csökkenő szakaszt
        #  - ha nem túl hektikusak a pontok, akkor ezzel felezhető a futásidő
        if i>0 and i<i_last:
            # monoton növekedés
            if aY[i-1]<aY[i] and aY[i]<aY[i+1]:
                i+=1
                while aY[i]<aY[i+1] and i<i_last: i+=1 
            # monoton csökkenés
            elif aY[i-1]>aY[i] and aY[i]>aY[i+1]:
                i+=1
                while aY[i]>aY[i+1] and i<i_last: i+=1 

        nNagyobbDb=0    # nagyobb szomszédok száma 
        nKisebbDb=0     # kisebb szomszédok száma
        for j in range(1,halfwidth+1):      # j=1,..,halfwidth
            if (i-j)>=0:
                if aY[i-j]>aY[i]: nNagyobbDb+=1
                elif aY[i-j]<aY[i]: nKisebbDb+=1
            if (i+j)<nLen:
                if aY[i+j]>aY[i]: nNagyobbDb+=1
                elif aY[i+j]<aY[i]: nKisebbDb+=1
        if nNagyobbDb>0 and nKisebbDb==0:
            minmax_indexes.append(i)
            minmax_types.append(-1)
        if nNagyobbDb==0 and nKisebbDb>0:
            minmax_indexes.append(i)
            minmax_types.append(1)
        i+=1
    if len(minmax_indexes)==0: return [],[]

    def nexttwo(k):
        # következő kettő nem törölt szélsőhely keresése (k,kk)
        while k<len(minmax_types) and minmax_types[k]==0: k+=1      # következő nem-törölt keresése
        if k>=len(minmax_types)-1: return None,None
        kk=k+1
        while kk<len(minmax_types) and minmax_types[kk]==0: kk+=1      # következő nem-törölt keresése
        if kk>=len(minmax_types): return k,None
        return k,kk
        
    
    # Törlendő minden olyan potenciális szélsőérték pont, amelyiknek a típusa megegyezik az előtte lévővel
    last=minmax_types[0]
    k=0
    while k<len(minmax_types):
        k,kk = nexttwo(k)
        if not kk: break
        if minmax_types[kk]==minmax_types[k]: 
            minmax_types[kk]=0
        else:
            k=kk


    # Törlendők azok az egymás utáni szélsőhely-párok, amelyekre nem teljesül hogy max>min (egyenlőség nem megengedett)
    k=0
    while k<len(minmax_types)-1:
        k,kk = nexttwo(k)
        if not kk: break

        if ((minmax_types[k]==1 and minmax_types[kk]==-1 and aY[minmax_indexes[k]]<=aY[minmax_indexes[kk]])
                or (minmax_types[k]==-1 and minmax_types[kk]==1 and aY[minmax_indexes[k]]>=aY[minmax_indexes[kk]])):
            minmax_types[k]=0
            minmax_types[kk]=0
            k=kk+1
        else:
            k=kk

    # Törlendők azok a szélsőérték párok, amelyek mindistance_x-en belül vannak
    if mindistance_x>0:
        k=0
        while k<len(minmax_types)-1:
            k,kk = nexttwo(k)
            if not kk: break

            if abs(aX[minmax_indexes[kk]]-aX[minmax_indexes[k]]) < mindistance_x:
                minmax_types[k]=0
                minmax_types[kk]=0
                k=kk+1
            else:
                k=kk

    
        

    # törlések véglegesítése a szélsőérték listákban
    def sub_commit():
        nonlocal minmax_types,minmax_indexes
        if minmax_types.count(0)==0: return False
        minmax_indexes_=[]
        minmax_types_=[]         
        for k in range(len(minmax_types)):
            if minmax_types[k]!=0:
                minmax_indexes_.append(minmax_indexes[k])
                minmax_types_.append(minmax_types[k])
        minmax_indexes=minmax_indexes_
        minmax_types=minmax_types_
        return True

    sub_commit()

        
    # Törlendők a lüktetésből adódó szélsőérték párok (jellemzően a vízszintes platókon fordulhatnak elő)
    #  FIGYELEM:  nem teljesen definit a végeremdény ...
    if mindistance_y>0:
        for i in range(2):    # max kétkörös lehet
            k=0
            while k<len(minmax_types)-1:
                k,kk = nexttwo(k)
                if not kk: break

                if abs(aY[minmax_indexes[kk]]-aY[minmax_indexes[k]]) < mindistance_y:
                    # Másorendű szélsőértékről van-e szó (ha igen, akkor nem szabd törölni)
                    def sub_secondorder(k):
                        nonlocal minmax_types
                        if k-2<0 or k+2>=len(minmax_indexes): return False
                        else:
                            return  (
                                (minmax_types[k]==-1      # minimumhely
                                    and aY[minmax_indexes[k]] <= aY[minmax_indexes[k-2]] 
                                    and aY[minmax_indexes[k]] <= aY[minmax_indexes[k+2]])
                                or
                                (minmax_types[k]==1       # maximumhely
                                    and aY[minmax_indexes[k]] >= aY[minmax_indexes[k-2]] 
                                    and aY[minmax_indexes[k]] >= aY[minmax_indexes[k+2]])
                                )
                    if not sub_secondorder(k) and not sub_secondorder(kk):  # mindkettőre vizsgálandó
                        minmax_types[k]=0
                        minmax_types[kk]=0
                        k=kk+1
                    else:
                        k=kk
                else:
                    k=kk
            if not sub_commit(): break
                    

                
    # ha egyetlen minmax hely sincs
    if len(minmax_indexes)==0:
        if aY[0]<aY[-1]:
            minmax_indexes=[0,nLen-1]
            minmax_types=[-1,1]
        elif aY[0]>aY[-1]:
            minmax_indexes=[0,nLen-1]
            minmax_types=[-1,1]

    elif endpoints:
        # A bal szélső pont legyen mindenképpen szélsőérték-hely
        if minmax_indexes[0]>0:
            # Ha az első minmaxhely halfwidth-en belül van, akkor ki kell hozni a szélére
            if minmax_indexes[0]<=halfwidth and minmax_indexes[0]<nLen-1:     # a jobb szélről ne hozza át
                minmax_indexes[0]=0
            # Ha az első minmaxhely halfwidth-en kívül van, akkor a bal szél a negáltja
            else:
                if minmax_types[0]==1 and aY[0]<aY[minmax_indexes[0]]: 
                    minmax_indexes.insert(0,0)
                    minmax_types.insert(0,-1)
                elif minmax_types[0]==-1 and aY[0]>aY[minmax_indexes[0]]:
                    minmax_indexes.insert(0,0)
                    minmax_types.insert(0,1)
        if minmax_indexes[-1]<nLen-1:
            # Ha az utolsó minmaxhely halfwidth-en belül van, akkor ki kell hozni a szélére
            if minmax_indexes[-1]>=nLen-1-halfwidth and minmax_indexes[-1]>0:     # a bal szélről ne hozza át
                minmax_indexes[-1]=nLen-1
            # Ha az utolsó minmaxhely halfwidth-en kívül van, akkor a jobb szél a negáltja
            else:
                if minmax_types[-1]==1 and aY[-1]<aY[minmax_indexes[-1]]: 
                    minmax_indexes.append(nLen-1)
                    minmax_types.append(-1)
                elif minmax_types[-1]==-1 and aY[-1]>aY[minmax_indexes[-1]]:
                    minmax_indexes.append(nLen-1)
                    minmax_types.append(1)
    
    points_min=[]
    points_max=[]
    for k in range(len(minmax_types)):
        if minmax_types[k]==-1: points_min.append((aX[minmax_indexes[k]],aY[minmax_indexes[k]]))
        elif minmax_types[k]==1: points_max.append((aX[minmax_indexes[k]],aY[minmax_indexes[k]]))
    return points_min,points_max

   

def serapply(ser,f_apply):          # apply series-re (x is elérhető a soronkénti segédfüggvényben)
    '''
    A ser.apply alapesetben csak az y értéket adja át az f_apply függvénynek. Ezt orvosolja a jelen függvény.
    A f_apply-ban a rec.name-mel lehet hivatkozni a kulcsértékre, és rec.y -nal a függő változóra
    Példa:
    def f_apply(rec):
        x=float(rec.name)
        if x>=x0:
            rec.y = rec.y * fn_gauss(x,x0,szigma)
        return rec
    Megjegyzés:  az x0 és a szigma itt környezeti változóként van kezelve. Megoldható lenne másodlagos argumentumként való átadása az f_apply-nak.
    '''
    return pd.DataFrame(ser.copy(),columns=['y']).apply(f_apply,axis=1)['y']


def serset(ser,keys,values,delimiter=','):        # xy párok hozzáadása a ser-hez (inplace)
    '''
    Elsősorban a dict-jellegű series-ek esetén hasznos  (az x értékek átlalában szövegesek, pl. mezőnév jellegűek)
    Ha van már ilyen kulcs a ser-ben, akkor felülírás
    
    Példa:  a pandas apply műveleteiben a rec argumentum egy ser,  colname to value  tartalommal
        - ha az f_apply-ban egy Sql lekérdezéssel kérünk be külős értékeket, akkor a kapott tuple-t ezzel a módszerrel lehet a
            legegyszerűbben beilleszteni a rec-be (a rec Series típusú)        

    keys:       list,tuple vagy vesszős felsorolás (a határolójel megadható külön argumentumban)
    values:     list, tuple vagy vesszős felsorolás
                - az elemszámának meg kell egyeznie a keys elemszámával  (összehangolt felsorolások)
    delimiter:  határolójel a felsorolás esetén  (pl. ','  ';'   '//'  )
    '''
    
    if keys is None or values is None: return

    if type(keys)==str: keys=keys.split(delimiter)
    if type(values)==str: values=keys.split(delimiter)

    if len(keys)!=len(values):
        print('ERROR  serset   A keys és a values elemszámának egyeznie kell')
        return

    for i,key in enumerate(keys):
        ser[key] = values[i]

    return ser          # inplace jellegű, de visszadja a ser-t returnként is
# lásd még: tblappend


def cols_to_front(tbl,colnames,to_pos=0):    # táblázat-oszlop(ok) előrehozatala (inplace, több oszlop is)
    '''
    INPLACE művelet  (mindamellett a return-be is bekerül a tábla)
    colnames:  több oszlopnév is megadható (listában vagy vesszős felsorolással). 
        - az elől álló a kimenetben is legelöl lesz
        - ha az oszlopnévben vessző van, akkor mindenképpen listaként kell megadni (egy esetén is)
    to_pos:  0 esetén a legelejére kerül,  1 esetén az első oszlop mögé, ...
    '''    
    if type(colnames)==str: colnames=colnames.split(',')
    for colname in colnames[::-1]:
        col = tbl.pop(colname)
        tbl.insert(to_pos, col.name, col)

    return tbl


def Encode_col(tbl,colname,utótag='_encoded',lics=True):  # string-oszlop kódolása (0,1,..N-1 kódok, ahol N a unique értékek száma)
    '''
    Ismétlődő értékeket tartalmazó string-oszlop kódolása (integer-oszloppá való átalakítása)
    Létrejön egy új oszlop és megmarad az eredeti is.
    A kódértékek az ismétlődő string-értékek ABC rendezésével adódnak.

    lics: True esetén lower és skipaccent.   Helyes lesz a rendezés, de a kimeneti codelist is lics feliratokat tartalmaz (ékezetek nélkül)
        (megjegyzés: elvileg lehetne egy korrekciós kör, ami kiválasztja lics-csoportonként a leggyakoribb vagy első feliratot)
    return:  tbl, colname_encoded, codelist
        - codelist:  kódlista a string-oszlop unique értékeivel
                    felirat = dget(codelist,index=kód)   ahol a kód egy 0-bázisú integer sorszám
            - felhasználható a FvPlot annotcaption argumentumaként
            - felhasználható a pltshow függvény xticklabel_map vagy xticklabel_map paramétereként 
    
    Elsősorban a gyors áttekintésre alkalmas, az ABC-rendezéses kódolás nem túlzottan cizellált megoldás
    '''
    if lics: ser=Lics(tbl[colname])
    else: ser=tbl[colname]
    encoded,codelist=pd.factorize(ser,sort=True)
    tbl[colname + '_encoded'] = encoded     # kódoszlop felvétele egy szöveges oszlophoz
    colname_encoded = colname + utótag
    codelist=list(codelist)
    return tbl,colname_encoded,codelist 

def Encode_ser(ser,bIndex=False,lics=True):  # string-series kódolása (0,1,..N-1 kódok, ahol N a unique értékek száma)
    '''
    
    Ismétlődő értékeket tartalmazó string-series kódolása (integer-oszloppá való átalakítása)
    A kódértékek az ismétlődő string-értékek ABC rendezésével adódnak.
    
    
    lics: True esetén lower és skipaccent.   Helyes lesz a rendezés, de a kimeneti codelist is lics feliratokat tartalmaz (ékezetek nélkül)
    return:  ser_encoded, codelist
        - codelist:  kódlista a string-oszlop unique értékeivel
                    felirat = dget(codelist,index=kód)   ahol a kód egy 0-bázisú integer sorszám
            - felhasználható a FvPlot annotcaption argumentumaként
            - felhasználható a pltshow függvény xticklabel_map vagy xticklabel_map paramétereként 
    
    Elsősorban a gyors áttekintésre alkalmas, az ABC-rendezéses kódolás nem túlzottan cizellált megoldás
    '''
    if bIndex:
        index = list(ser.index) 
        if lics: index=Lics(index)
        encoded,codelist=pd.factorize(index,sort=False)
        ser_encoded = pd.Series(ser.values,encoded)
        return ser_encoded,codelist
    else:
        if lics: ser_lics=Lics(ser)
        else: ser_lics=ser
        encoded,codelist=pd.factorize(ser_lics,sort=True)
        codelist=list(codelist)
        ser_encoded=pd.Series(encoded)
        return ser_encoded,codelist 

def Encode_ser_index(ser,bIndex=False,lics=True):  # series string-indexének kódolása (0,1,..N-1 kódok, ahol N a unique értékek száma)
    '''
    String-index dekódolása
    Mindössze annyi történik, hogy az index sorszámozott integereket kap, az index-értékek pedig bekerülnek egy
        összehangolt tömbbe
    
    return ser_encoded,codelist
    '''
    ser=ser.copy()
    codelist = list(ser.index)    
    ser.index = Range(0,len(ser)-1)
    return ser,codelist


# ser = pd.Series([12,23],['első','második'])
# ser,codelist = Encode_ser_index(ser)
# print(ser)
# print(codelist)
# exit()


def info(object,todo='info',**params):
    ''' dataframe esetén tblinfo(tbl,todo,toexcel,cols,groupby,plotparams) '''

    if 'DataFrame' in str(type(object)): return tblinfo(object,todo,params)
    # egyelőre csak dataframe objektumokra működik  (series objektumokra is kidolgozandó)
    else: return    

def tblinfo(tbl,todo='info',toexcel=False,cols='',query='',groupby='',orderby='',plotstyle='firstline',**plotparams):
    """A tábla oszlopainak gyors áttekintése (második körben: értékeloszlás egy-egy oszlopra, oszlop átnevezése, törlése, korrelációk). 
    tbl:  pl. egy csv-ből beolvasott tábla;  az esetleges rekordszűréseket előzetesen kell végrehajtani (pl. .query('location=="Hungary"')
       - nem kötelező, hogy unique legyen az index. A plot-hoz általában szükséges, de általában egy groupby oszloppal biztosítható az unicitás.
    todo: 
      "info": (default)  kimutatás az oszlopokról  (colindex, colname, dtype, Not_null, Unique, Distinct_values, Most_frequent) 
      "minmax":           colindex, colname, dtype, min, max, corr_plus, corr_minus
      "browse" ( vagy "rows"):  sorok listázása   (cols: oszlopszűrés,  query: sorszűrés)
      "plot":             összes dtype="num" és distinct_values>1 oszlop nyomtatása egy közös diagramba (mindegyik normalizálva, "scatter gauss")
                            - az x tengelyre az index_col értékei kerülnek
                            - kihagyja a számított oszlopokat (corr=1 vagy -1 egy előtte lévővel)
                            - oszlopindex-lista állhat mögötte (oszlop-sorszámok, lásd "info")
                            - groupby megadása esetén a distinct értékekre külön vonaldiagrammok (mezőnként és groupby értékekenként)
      "2", "oszlopnév"    második oszlop értékeloszlása  (series:  value, count,  leggyakoribb legelöl, első 20, toexcel esetén az összes)
      "hist":             értékeloszlások nyomtatása (columns-ban szűkíthető az oszlopok köre)
      "2 plot":           második oszlop nyomtatása (scatter gauss)     
      "2 corr":           második oszlop korrelációi a többi oszloppal  (plot)  Az oszlopsorszám a tbl összes oszlopára vonatkozik (lásd még "corr 2")
      "corr":             korrelációs táblázat megjelenítése (szöveges)
      "corr 2":           a korrelációs táblázat második oszlopának korrelációi a többi oszloppal (lásd még: "2 corr")
      "corrplot":         az összes korreláció megjelenítése diagrammon
      "corrmax":          legnagyobb pozitív korrelációk megjelenítése (szöveges)     "corrmax 2" - második oldal ...
      "corrmin":          legnagyobb negatív korrelációk megjelenítése (szöveges).    "corrmin 2" - második oldal ...

      "drop 1,2,5":       oszlopok törlése (utána az "info" ismételt megjelenítése)
      "rename 3 code2":   oszlop átnevezése (oszlopsorszám 0-bázisú, egy-szavas oszlopnév adható meg, ékezetes betűk megengedettek; utána újra "info")
 
    cols: oszlopnevek rövidített felsorolása. Ha üres, akkor az összes számoszlop (számított másodlagos oszlopok nélkül) 
            példa:  "total case,cumulative":  van benne "total" és "case", vagy van benne "cumulative"  (több oszlop lehet az eredmény)
            Oszloponként külön lineplot, alapesetben ugyanabban a diagramban (ha groupby is többes, akkor direktszorzat)
            Subplot határoló: //   Ha a groupby-nál is meg van adva, akkor a groupby-nak van prioritása (összehangolandó a kettő)
            Hatással van:  plot, rows, corr, corrplot, corrmax, corrmin 
    query:  alapszűrő (nem kötelező)  pl. 'Dátum>="2021.01.01"'    'Continent=="Europe"'    'Year>1900 and Year<=2020'   'Continent.str.startswith("Eu",case=False)'
    groupby:  külön browse vagy plot a groupby oszlop distinct értékeire (csoportosító szűrésről van szó, nem aggregálásról) 
            'Country'     külön leválogatás / diagram-vonal a 'Country' oszlop összes distinct értékére         
            'Country:hun,cze,united king,slo,rom'     a 'Country' oszlop megadott szűrőknek megfelelő értékeire (lásd: filternames())
            'Country:hun,cze,pol,rom//germ,austria,france'     két subplot (nrows,ncols paraméterekkel állítható be a subplopotok pozíciója)
    orderby:  oszlopnév vagy oszlopnevek felsorolása (lehet gyorsírásos is, lásd cols, NOT nélkül)). 
            Mindegyik oszlopnév után állhat "desc" 

    toexcel:  írja ki excel fájlba   ("info", "browse", "5", "corr", "corrmin", "corrmax")
    plotstyle:
            'area':         mindegyik vonal area-stílussal
            'line':         mindegyik vonal normál stílussal (eltérő színekkel)
            'firstarea'     az első area, a többi normál
            'firstline'     az első vonal, a többi area
    plotparams:  FvPlot, plotinit és plotshow összes lehetséges argumentuma (kivéve area, amit a pltstyle felülírhat)
            Legyakoribb:  suptitle,title,ylabel,plttype,label,annotate,normalize,color,  plttype='gauss' (ha nem kellenek a pontok is)
            Ha a groupby-ban subplot-ok vannak megadva, akkor // határolással subplot-szintű paraméterek adhatók meg 
                (ha nincs elég, akkor az utolsó érvényes a többi subplot-ra) 

    return: tbl (colname,dtype,not_null,unique,...)     Rendezés: filled (desc),  repeat (asc)
    """

    tblOut=None

    if isint(todo): todo=str(todo)

    words=todo.split()
    if len(words)==0: return
    firstword=words[0]

    if query:
        tbl=tbl.query(query)


    groupcol=''         # csoportosító oszlop
    subdiagrams=[['']]    # aldiagramonként egy-egy groupvalue felsorolás (ha nincs groupby, akkor egyetlen aldiagram)
    if groupby:
        subdiagrams=[]
        groupcol,groupfilter = splitfirst(groupby,':')
        groupcol=groupcol.strip()
        if groupfilter: groupfilter=groupfilter.strip()

        # A táblában előforduló összes lehetséges csoportosító érték bekérése
        groups=tbl[groupcol].unique()       # distinct
        if groupfilter:
            subdiagrams=groupfilter.split('//')             # Country:hun,pol//german,france       //-határolással külön subplot-ok
            for i in range(len(subdiagrams)):
                subdiagrams[i]=filternames(groups,subdiagrams[i])      # helper függvény
                # minden subdiagram egy lista a tbl-ben lévő oszlopnevekkel  (az oszlopnevek elején '^' állhat, ami kiemelést jelez)
        else:
            subdiagrams.append(groups)

    if cols: 
        # groupcol felvétele az oszloplistába (a szűréshez szükség lesz rá)
        if groupcol:
            partsL=cols.split(' NOT ')
            if not groupcol in partsL[0]: 
                cols=partsL[0].strip() + ',' + groupcol
                if len(partsL)>1: cols=cols + ' NOT ' + partsL[1]
        tbl=tblfiltercols(tbl,cols)

    if orderby:
        if type(orderby)==str:
            orderby=orderby.lower().strip()
            if orderby=='index': tbl=tbl.sort_index(orderby)
            elif orderby=='index desc': tbl=tbl.sort_index(orderby,ascending=False)
            # Áttérés list-re
            else: orderby=orderby.split(',')

        if type(orderby)==list:
            orderbycols=[]
            ascending=[]
            for orderbycol in orderby:
                orderbycol=orderbycol.strip().lower()
                if endwith(orderbycol,' desc'): 
                    orderbycol=cutright(orderbycol,' desc')
                    ascending.append(False)
                else: ascending.append(True)
                colnamesout=filternames(list(tbl),orderbycol)
                if not colnamesout:
                    print('ERROR   orderby   Érvénytelen oszlopnév: "' + orderbycol + '"')
                    break
                else: 
                    if len(colnamesout)>1: print('WARNING  orderby  A(z) "' + orderbycol + '" érték többféleképpen is előfordul (' + str(colnamesout) + ')')
                    orderbycols.append(colnamesout[0])
            if len(orderbycols)>0:
                tbl=tbl.sort_values(orderbycols,ascending=ascending)


    rowcount=len(tbl)
    colnames=list(tbl)
    colcount=len(colnames)
    dtypes=tbl.dtypes

    try:
        todo_colnameindex=colnames.index(todo)
    except:
        todo_colnameindex=None

    # Oszlopok kivonatos adatai
    if todo in ['info','minmax']:
        tblcorr=tbl.corr()
       
        out=('Rekordok száma: ' + str(rowcount) + '\n' +
            'Oszlopok száma: ' + str(colcount))
    
        colnames_all=['index'] + colnames
        
        aOut=[]
        nColindex=0
        for colname in colnames_all:
            #print(colname)
            if colname=='index': 
                sorszam='index'
                col=tbl.index
                dtype=str(tbl.index.dtype)
                colnameout=tbl.index.name       # nem mindig van neve
                if colnameout==None: colnameout=''
            else: 
                sorszam=str(nColindex)
                nColindex+=1
                col=tbl[colname]
                dtype=dtypes[colname].name
                colnameout=colname
            
            if dtype=='object': dtype='string'          # vegyes is lehet (ha egyetlen None előfordul, akkor is ide kerül)
            elif beginwith(dtype,'float'): dtype='float'
            elif beginwith(dtype,'int'): dtype='int'
            elif beginwith(dtype,'decimal'): dtype='dec'
            elif beginwith(dtype,'datetime'): dtype='datetime'      # nincs automatikus parse
        
            if rowcount==0: 
                if todo=='info': aOut.append((sorszam,colnameout,dtype,'','','','','',''))
                elif todo=='minmax': aOut.append((sorszam,colnameout,dtype,'','','',''))
                continue

            # üres értékek, feltöltöttség
            if todo=='info':
                if colname=='index': nulldb=0
                else: nulldb=col.isnull().sum()       # None, np.nan 
                nulladb=nulldb
                if dtype=='string': nulladb += (col=='').sum() + (col=='0').sum()
                elif dtype in ['float','int','dec']: nulladb += (col==0).sum()
                elif dtype=='bool': nulladb += (col==False).sum()

                if nulldb==0: notnull='100% (all)'
                else: notnull = '{:.2%}'.format((rowcount-nulldb) / rowcount)

                if nulladb==0: notnulla='100% (all)'
                else: notnulla = '{:.2%}'.format((rowcount-nulladb) / rowcount)


                # ismétlődő értékek száma
                repeat = len(duprows(tbl,colname,False))         # colname=="index" esetén is működik
                if repeat==0: unique='100% (unique)'
                else: unique='{:.2%}'.format((rowcount-repeat)/rowcount)
            
                # leggyakoribb érték
                distinct=0
                sample=''
                ser_valuecounts=col.value_counts()      # series     
                if len(ser_valuecounts)>0:
                    sample=ser_valuecounts.index[0]
                    samplecount=ser_valuecounts.iloc[0]
                    distinct=len(ser_valuecounts)
                    if distinct==rowcount: distinct=str(distinct) + ' (unique)'
                    if dtype in ['float','int','dec']: sample='{:,.6g}'.format(sample)
                    else: sample=sample=txtshorten(str(sample),25)
                    sample='"' + sample + '" (' + str(samplecount) + ')' 

                ser=col.dropna()               # None értékek kihagyása (enélkül hibaüzenet jelenhet meg)
                min=ser.min()                           # számok esetén az nan kimarad, de a 0 benne van
                if dtype not in ['float','int','dec']: min='"' + txtshorten(str(min),22) + '"'
                max=ser.max()
                if dtype not in ['float','int','dec']: max='"' + txtshorten(str(max),22) + '"'

                aOut.append((sorszam,colnameout,dtype,notnull,unique,distinct,min,max,sample))
            
            elif todo=='minmax':
                ser=col.dropna()               # None értékek kihagyása (enélkül hibaüzenet jelenhet meg)
                min=ser.min()                           # számok esetén az nan kimarad, de a 0 benne van
                if dtype not in ['float','int','dec']: min='"' + txtshorten(str(min),30) + '"'
                max=ser.max()
                if dtype not in ['float','int','dec']: max='"' + txtshorten(str(max),30) + '"'
                avg=''
                # if dtype in ['float','int','dec','bool']: avg=ser.mean()         # bool esetén is értelmezhető
        
                corr_plus=''
                corr_minus=''
                if dtype in ['float','int','dec','bool'] and colnameout:
                    ser=tblcorr[colname]             # azok a korrelációk, amelyekben az aktuális oszlop szerepel
                    ser=ser[ser.index!=colname]         # saját magával nem kell korreláció
                    ser_plus=ser[ser>0.1]
                    if ser_plus.count()>0: 
                        corr_plus=ser.idxmax() + ' (' + '{:.3g}'.format(ser.max()) + ')'
                    ser_minus=ser[ser<-0.1]
                    if ser_minus.count()>0: 
                        corr_minus=ser.idxmin() + ' (' + '{:.3g}'.format(ser.min()) + ')'

                aOut.append((sorszam,colnameout,dtype,min,max,corr_plus,corr_minus))
    
        if todo=='info':
            tblOut=TblFromRecords(aOut,'x,colname,dtype,Not_null,Uniqueness,Distinct_values,Min,Max,Most_frequent','x')
            # tblOut=TblFromRecords(aOut,'x,colname,dtype,Not_null,Not_null_0,Uniqueness,Distinct_values,Most_frequent','x')
        elif todo=='minmax':
            tblOut=TblFromRecords(aOut,'x,colname,dtype,min,max,corr_plus,corr_minus','x')
    
        print(out + '\n\n' + str(tblOut) + '\n')
    
    # Browse (listázás)
    elif firstword in ['browse','rows']:
        if toexcel: tblOut=tbl      # todo:  külön táblázatokba kellene beírni (azonos munkalapon vagy külön munkalapokon)
        else: 
            if len(subdiagrams)==0:    # ha nincs megadva group
                print(tbl)
            else:
                for groupvalues in subdiagrams:          # minden subdiagram egy lista (egyetlen '' is lehet benne, ha nincs groupby)
                    for groupvalue in groupvalues:         # vonalak csoportosító értékenként és oszloponként 
                        if groupvalue!='': tblL=tbl.loc[tbl[groupcol]==groupvalue]
                        else: tblL=tbl
                        print('\n\n' + groupvalue.upper())
                        print(tblL)


    # Plot
    elif firstword in ['plot','interpolate']:
       
        # megadható egy oszlopssorszám felsorolás   (pl.  "plot 1,3,5";  a cols argumentum-ban is megadhatóak az oszlopnevek)
        colindexes=None
        if len(words)>1: colindexes=words[1].split(',')

        # dtype="num" és distinct_values>1 oszlopok begyűjtése (a számított oszlopok nélkül)
        tblcorr=tbl.corr()
        colnamesL=[]
        colindexL=[]
        for i in range(len(colnames)):
            if colindexes and not (str(i) in colindexes): continue      # ha van explicit colindex felsorolás, akkor csak azon belül
            colname=colnames[i]
            dtype=dtypes[colname].name
            if not (dtype in ['float32','float64','int64','int32']): continue
            distinct=len(tbl[colname].value_counts())
            if distinct<2: continue
            # Számított oszlopok kiszűrése (korreláció=1)
            # bCalc=False
            # strL=''
            # corrindex=tblcorr.index.get_loc(colname)
            # for j in range(corrindex):          # előtte lévők
            #     c=tblcorr.iloc[j,corrindex]
            #     strL=strL + str(c) + ','
            #     if np.isclose(c,1): bCalc=True
            # if bCalc: continue
            colnamesL.append(colname)
            colindexL.append(i)
        
        # Hányszor lesz hívva a FvPlot
        nGroups=0
        for nSub in range(len(subdiagrams)):          # minden subroup egy lista (egyetlen '' is lehet benne, ha nincs groupby)
            groupvalues=subdiagrams[nSub]
            nGroups += len(groupvalues)
        nPlotlines=nGroups * len(colnamesL)


        # FvPlot paraméterek    
        if plotstyle in ['area','areas']: plotparams['area']=True
        elif plotstyle in ['line','lines']: plotparams['area']=False
        # - a 'firstline' és a 'firstarea' plot-onként lesz érvényesítve

        # normalize_default=1
        # if len(colnamesL)==1: normalize_default=None
        # dsetsoft(plotparams,normalize=normalize_default)
 
        annot=notnone(dget(plotparams,'annot'),dget(plotparams,'annotate'))
        if type(annot)!=dict:
            if annot==None: annot='localmax2 last'
            dset(plotparams,annotate=annot)
        # dict is lehet benne, amit majd görbénként kell egyedi értékre konvertálni

        G=plotparams.get('gausswidth')
        if not G: G=plotparams.get('G')
        resample=plotparams.get('resample')
        if type(resample)==bool: dset(plotparams,resample=None)     # lejjebb még figyelembe lesz véve a plttype_defailt beállításakor
        
        if G==0: 
            if resample: plttype_default='interpolated'
            else: plttype_default='original'
        else: 
            plttype_default='gauss'
            if resample: 
                plttype_default='regauss'

        points=plotparams.get('points')         # True esetén scatter, "kde" esetén pontsűrüség
        if points:
            if type(points)==bool and points: plttype_default='scatter ' + plttype_default
        elif points==None:
            if nPlotlines==1 and G!=0: plttype_default='scatter ' + plttype_default
        dsetsoft(plotparams,plttype=plttype_default)

        # pltinit paraméterek
        dsetsoft(plotparams,suptitle='Diagramok (tblinfo)',height=0.8,width=0.8,left=0.08,right=0.85,bottom=0.1)
        
        # A diagram (aldiagram) feletti címsor tartalma
        title_in=plotparams.get('title')     # lista is lehet  (több subplot esetén egyedi feliratok)
        if not title_in: plotparams.get('titles')     # titles atgumentum is megengedett
        if type(title_in)==str: title_in=[title_in]     # egységesen lista

        # y tengely címfelirata
        ytitles=notnone(plotparams.get('ylabel'),plotparams.get('ytitle'),plotparams.get('ytitles'))
        if type(ytitles)==str: ytitles=[ytitles]     # egységesen lista

        # x tengely címfelirata     (aldiagramok esetén mindegyikre közös)
        xtitle = notnone(plotparams.get('xlabel'),plotparams.get('xtitle'))
        if xtitle==None: xtitle = tbl.index.name


        # pltshow paraméterek
        # xmax,xmin: már a plot előtt érvényesítve lesz
        xmin=notnone(dget(plotparams,'xmin'),dget(plotparams,'x1'))
        xmax=notnone(dget(plotparams,'xmax'),dget(plotparams,'x2'))
        
        dsetsoft(plotparams,annot_fontsize=8)
        # if 'date' in str(type(tbl.index)): dsetsoft(plotparams,xtickerstep='date')

        ynumformat=dget(plotparams,'ynumformat')
        if ynumformat=='': dsetsoft(plotparams,yticklabels=False)       # ne jelenjenek meg
        elif ynumformat in ['date','month','year']:
            dset(plotparams,ynumformat=None,ytickerstep=ynumformat)
        
        xnumformat=dget(plotparams,'xnumformat')
        if xnumformat=='': dsetsoft(plotparams,xticklabels=False)       # ne jelenjenek meg
        elif xnumformat==None: 
            if beginwith(str(tbl.index.dtype),'date'): xnumformat='date'    # dátum típusú index esetén xtickerstep='date'
        if xnumformat in ['date','month','year']:
            dset(plotparams,xnumformat=None,xtickerstep=xnumformat)


        # Ha subplotok vannak, akkor mindenképpen kell nrows és ncols
        nSubDb=len(subdiagrams)
        dsetsoft(plotparams,subcount=nSubDb)

        nrows=plotparams.get('nrows')
        ncols=plotparams.get('ncols')
        if nSubDb>1:
            if (not nrows or not ncols):        # ha meg van adva a nrows vagy a ncols
                if not nrows: 
                    if ncols and ncols>0: nrows=int(math.ceil(nSubDb/ncols))
                    else: nrows=int(math.floor(math.sqrt(nSubDb)))
                if not ncols: ncols=int(math.ceil(nSubDb/nrows))
                dsetsoft(plotparams,nrows=nrows,ncols=ncols)

        colors_in=plotparams.get('colors')
        annotcolor_in=plotparams.get('annotcolor')

        # pltinit  (a rajzolás indítása)
        if firstword=='plot': pltinit(**plotparams)
        
        # Vonaldiagramok rajzolása
        labelToSeries={}
        aGausswidth=[]              # tájékoztató felirat lesz a jobb felső sarokban (plotonként eltérő lehet, ezért számítani kell)
        for nSub in range(len(subdiagrams)):          # minden subroup egy lista (egyetlen '' is lehet benne, ha nincs groupby)
            groupvalues=subdiagrams[nSub]
            
            # Title beállítása
            title=''
            if title_in:
                if nSub<len(title_in): title=title_in[nSub]
                else: title=title_in[-1]                        # ha nincs ilyen indexű title_in, akkor az utolsó 
            # default beállítás            
            if title=='': 
                title=notnone(', '.join(groupvalues),', '.join(colnames))
                if title: title = txtshorten(title,40)
            if title: plotparams['title']=title

            # ylabel beállítása
            ylabel=''
            if ytitles:
                if nSub<len(ytitles): ylabel=ytitles[nSub]
                else: ylabel=ytitles[-1]                        # ha nincs ilyen indexű title_in, akkor az utolsó 
            # default beállítás
            if ylabel=='' and len(colnames)==1: 
                ylabel=txtshorten(colnames[0],40)
            if ylabel==title: ylabel=''             # felesleges ismétlés lenne
            if ylabel: plotparams['ylabel']=ylabel
            
            plotparams['xlabel']=xtitle
            
                    
            # Több aldiagram esetén pltinitsub
            if len(subdiagrams)>1 and firstword=='plot': 
                pltinitsub(nSub,**plotparams)

            i_plot=0
            
            points=plotparams.get('points')     # kde miatt kell
            normalize=plotparams.get('normalize')   # kde miatt kell
            ser_kde=pd.Series()

            for groupvalue in groupvalues:         # vonalak csoportosító értékenként és oszloponként 
                if type(groupvalue)==str and len(groupvalue)>0 and groupvalue[0]=='^':
                    groupvalue=cutleft(groupvalue,'^')
                    plotparams['colors']={'color':'orange', 'linewidth':1.5, 'alpha':1}
                    plotparams['annotcolor']='kiemelt'
                else: 
                    plotparams['colors']=colors_in
                    plotparams['annotcolor']=annotcolor_in
                
                if groupvalue!='': tblL=tbl.loc[tbl[groupcol]==groupvalue]
                else: tblL=tbl

                # Ha van xmin,xmax, akkor azt itt kell érvényesíteni (a FvPlot előtt)
                if xmin is not None: tblL=tblL.loc[tblL.index>=xmin]
                if xmax is not None: tblL=tblL.loc[tblL.index<=xmax]
                    
                plttype=plotparams.get('plttype')
                if plttype and ('gauss' in plotparams.get('plttype')):
                    G=plotparams.get('gausswidth')
                    if not G: G=plotparams.get('G')
                    if not G: G=0.1
                    if type(G)!=str:        # string esetén felsorolás van benne. Ilyenkor ne jelenjen meg a G-felirat
                        if G>1: G=G/len(tblL)   # itt a teljes szélességhez viszonyított hányad érdekel
                        aGausswidth.append(G)   # nem garantált, hogy minden vonalra azonos (az átlag fog megjelenni kivonatos adatként)

                nColnames=len(colnamesL)
                for i in range(nColnames):
                    colname=colnamesL[i]
                    # label kitalálása
                    if (groupvalue=='') or len(groupvalues)==1: caption=colname
                    else:
                        if nColnames==1: caption=str(groupvalue)
                        else: caption=str(colindexL[i]) + ' ' + str(groupvalue) + '_' + colname
                    if firstword=='plot':
                        if plotstyle in ['firstarea','areafirst']: plotparams['area'] = (i_plot==0)
                        elif plotstyle in ['firstline','linefirst']: plotparams['area'] = (i_plot>0)

                        if type(annot)==dict:
                            annot_=dget(annot,caption)    # label azonosítással kéri be a görébhez tartozó annot paramétert
                            if annot_==None: annot_=dget(annot,'other')  # ha nincs ilyen label a dict-ben, akkor "other" label
                            if annot_==None: annot_='localmax2 last' 
                            dset(plotparams,annotate=annot_)
                        
                        if nPlotlines>5:
                            if len(subdiagrams)>1 and title!='':  progress('Subplot: ' + title)
                            elif caption!='': progress('Plot: ' + caption)
                        
                        # RAJZOLÁS
                        FvPlot(tblL[colname],label=caption,**plotparams)

                        # KDE plot-hoz: az aldiagram összes pontsorozatának begyűjtése
                        ser_kde = pd.concat([ser_kde,tblL[colname]])       # gauss simítás és átlagolás előtti pontok


                        i_plot+=1
                    elif firstword=='interpolate':
                        labelToSeries[caption]=FvPlot(tblL[colname],seronly=True,**plotparams)


            if len(subdiagrams)>1 and firstword=='plot': 
                
                if points=='kde':
                    plt.autoscale(enable=False)         # fontos, mert enélkül újraszámolja a az xlim, xlim határokat és a sáv nem fog a szélekig érni
                    FvPlot(ser_kde,'kde',normalize=normalize,kde_bandwidth=1.5)
                
                pltshowsub(**plotparams)

        
        # információk a jobb felső sarokban
        lines=[]
        # Ha a tbl-nek van source adata, akkor kiírja  (egyedi módon állítható be a beolvasáskor)
        try:
            if tbl.attrs['source']: lines.append('Forrás: ' + tbl.attrs['source'])
        except: pass
        try:
            if len(aGausswidth)>0: 
                G=np.average(aGausswidth)       # hányad
                G_abs=(datefloat(tblL.index.max())-datefloat(tblL.index.min())) * G
                lines.append('Gauss mozgóátlag szélessége: ' + strnum(G_abs,'3g') + ' (' + strnum(G,'0%') + ')' )  
        except: pass        # felsorolásos gausswidth esetén nem írja ki
        xy_texts=None
        if len(lines)>0: 
            # annottext={'x':0.95,'y':0.97,'caption':joinlines(lines),'ha':'right','va':'top','fontsize':7,'alpha':0.5}
            annottext=joinlines(lines)
            #plt.text(1,1.12,joinlines(lines),ha='right',va='top',fontsize=7,alpha=0.5,transform = plt.gca().transAxes)
            if plotparams.get('commenttopright'): plotparams['commenttopright'] += '\n' + annottext
            else: plotparams['commenttopright']=annottext
                             
        # Diagram megjelenítése
        if firstword=='plot': pltshow(**plotparams)

    elif firstword in ['hist','histogram']:
        # pltinit(suptitle='Értékeloszlások')
        tbl.hist(bins=int(len(tbl)/10))
        pltshow()
        return


    # Értékeloszlás:  info=5   info='5'   info='temperature'
    elif isint(todo) or todo_colnameindex:
        ncol=todo_colnameindex or int(todo)
        if ncol>=rowcount:
            print('Hiba: tblinfo, nincs ilyen indexű oszlop (' + str(ncol) + '), ezért nem jelenthető meg értékeloszlás\n' +
                  'Az elérhető oszlopindexeket a "tblinfo(tbl)" utasítással lehet lekérdezni.')
            return
        colname=colnames[ncol]
        firstword=colname    # excel-be íráskor lehet szükséges
        # tblOut=tbl[colname].value_counts().reset_index(drop=False).sort_values([colname,'index'],ascending=[False,True]).set_index('index')
        tblOut=tbl[colname].value_counts()
        # print(tblOut)
        print('"' + tbl.columns[ncol] + '" oszlop értékeloszlása:')
        print('  Előforduló értékek száma: ' + str(len(tblOut)))
        print(tblOut.head(20))
        if len(tblOut)>20: print('... (az összes előforduló érték toexcel=True hívással kérhető)')
    
    # info='corr'
    elif len(words)==1 and words[0]=='corr':
        tblOut=tbl.corr()
        print(tblOut)

    # info='corrplot'
    elif todo=='corrplot':
        tblcorr=tbl.corr()
        numcols=len(list(tblcorr))
        if numcols<=5: format='simple'
        else: format='normal'
        FvCorrPlot(tbl,format)
        return
            
    # info='5 corr'
    elif len(words)==2 and isint(words[0]) and words[1]=='corr' and int(words[0])<colcount:
        colname=colnames[int(words[0])]
        pltinit(suptitle='Korreláció',title=colname,left=0.4,right=0.9,width=0.5,height=0.8)
        tblcorr=tbl.corr()
        sb.heatmap(tblcorr[[colname]].sort_values(by=colname, ascending=False)[1:], vmin=-1, vmax=1, annot=True, cmap='BrBG')
        pltshow()

    # info='corrmax',  info='corrmin'
    elif firstword in ['corrmax','corrmin']:
        tblcorr=tbl.corr()
        ascending = (todo=='corrmax')
        nlen=len(tblcorr)
        aRec=[]
        for i in range(nlen):
            for j in range(i+1,nlen):
                aRec.append((tblcorr.index[i],tblcorr.columns[j], tblcorr.iloc[i,j]))
        tblOut=TblFromRecords(aRec,'adat1,adat2,corr')
        tblOut=tblOut.sort_values(by='corr',ascending=ascending)
        if len(words)==2 and isint(words[1]):
            nPage=int(words[1])
            if nPage<0:
                print(str(tblOut.tail(30)) + '\n... lapozás: tblinfo(tbl,"' + firstword + ' 2")')        
            else:
                i=(nPage-1)*30
                print(str(tblOut.iloc[i:(i+30)]) + '\n... lapozás: tblinfo(tbl,"' + firstword + ' ' + str(nPage+1) + '")')
        else:  
            print(str(tblOut.head(30)) + '\n... lapozás: tblinfo(tbl,"' + firstword + ' 2")')        

           
    # info='drop 4'
    elif len(words)>=2 and words[0]=='drop':
        dropped=0
        for i in range(1,len(words)):
            if isint(words[i]) and int(words[i])<colcount: 
                tbl.drop(tbl.columns[int(words[i])], axis=1,inplace=True)
                dropped+=1
            else: 
                print('Hiba: tblinfo, "drop" után oszlopsorszámo(ka)t kell megadni')
        if dropped>0:
            print('tblinfo, ' + str(dropped) + ' oszlop lett törölve a táblázatból')
            tblinfo(tbl)

    # info='rename 6'
    elif len(words)==3 and words[0]=='rename' and isint(words[1]) and int(words[1])<colcount:
        tbl.rename(columns={tbl.columns[int(words[1])]: words[2]},inplace=True)
        print('Oszlop átnevezve\n')
        tblinfo(tbl)

    else:
        print('Hiba:  tblinfo,  oszlopsorszám, "drop" "rename" "corr" adható meg első szóként')
    

    if toexcel and len(tblOut)>0:
        path=nextpath('tblinfo_' + firstword,'xlsx')
        try: 
            tblOut.to_excel(path)
            print('Excel fájlba írás: ' + path)
        except:
            print('A fájlba írás nem sikerült')

    if firstword=='interpolate':
        return labelToSeries    # A kirajzolt vonalakhoz tartozó series objektumok (felhasználható pl. interpolációra (servalueA))
    elif firstword=='plot':
        config.tblinfoplotsG = labelToSeries   

Info=tblinfo





# FÁJL BEOLVASÁS

def Readcsv(path,format='hu',index_col=None,parse_dates=False):
    '''
    Két-három nagságrenddel gyorsabb, mint az excel-beolvasás. 1 MByte felett mindenképpen a csv formátumban való tárolás javasolt
        Példa:   1.7 MByte   6.5 sec  -  0.01 sec
    Akkor is be tudja olvasni, ha meg van nyitva excelben   (a Readexcel ilyenkor hibát küldene)
    Hibaüzenet, ha nincs ilyen fájl   (except FileNotFoundError: ...)
            
    A dátumkonveriókat érdemes önállóan végrehajtani, a format megadásával (pd.to_datetime())
    Az indexbe vitelt is utólag érdemes végrehajtani (ritkán szükséges,  tbl.set_index())
      Kivétel: ha van egy "index" oszlop a beolvasandó táblázat elején (pl. Tocsv kiírással jött létre a fájl),
        akkor index_col='index' argumentummal érdemes hívni a függvényt

    path:  ha '\' jelek vannak benne (pl. a windows fájlkezelőből való bemásoláskor), akkor r"..." formátumot kell alkalmazni
    format:
         None,'en','':      sep=','     decimal='.'
        'hu':               sep=';'     decimal=','   encoding='cp1250'
        'hu_utf8':          sep=';'     decimal=','
        'utf16':            encoding='uft_16_le'      hőmérsékleti adatsorok beolvasásakor fordult elő
    index_col:   ez legyen a tbl indexe.  Több oszlopos index is megadható (list)
        - ha nincs megadva, akkor autosorszámozott index
        - nem kell unique-nak lennie   (de ha nem unique, akkor nem szerencsés index-be vinni, mert nem lesz
                azonosítója a soroknak)
        - alapesetben nem érdemes használni  (kivéve pl. kódjegyzékeknél)
        - több oszlopos index esetén rekord kérés:    tbl.loc[('Hungary',56)] 
    parse_dates:  próbálkozzon-e a dátumoszlopok feltérképezésével
        - nagy táblák esetén lassú lehet. Az egyedi pd.to_datetime hívások, formátum-információval, nagyságrendekkel gyorsabbak lehetnek
        
    return:  DataFrame
    '''
    if format is None or format=='' or format=='en':      sep,decimal,encoding = ',', '.', None
    elif format=='hu':                                  sep,decimal,encoding = ';', ',', 'cp1250'
    elif format=='hu_utf8':                             sep,decimal,encoding = ';', ',', 'utf8'
    elif format=='utf16':                               sep,decimal,encoding = ',', '.', 'utf_16_le'
    
    return pd.read_csv(path,sep=sep,decimal=decimal,encoding=encoding,index_col=index_col,parse_dates=parse_dates)

def Readexcel(path,sheet=0,index_col=None,header_row=0,parse_dates=False):
    '''
    Két-három nagyságrenddel lassabb, mint a csv beolvasás. 1 MByte felett mindenképpen a csv formátumban való tárolás javasolt
        Példa:   1.7 MByte   6.5 sec  -  0.01 sec
    Hibaüzenet, ha meg van nyitva excelben.
        Kezelése:       except PermissionError:
                            print('Nem lehet hozzáférni a fájlhoz (feltehetőleg meg van nyitva Excel-ben)')
    Hibaüzenet, ha nincs ilyen fájl   (except FileNotFound: ...)
    
    A dátumkonveriókat érdemes önállóan végrehajtani, a format megadásával (pd.to_datetime())
    Az indexbe vitelt is utólag érdemes végrehajtani (ritkán szükséges,  tbl.set_index())
      Kivétel: ha van egy "index" oszlop a beolvasandó táblázat elején (pl. Toexcel kiírással jött létre a fájl),
        akkor index_col='index' argumentummal érdemes hívni a függvényt

    path:  ha '\' jelek vannak benne (pl. a windows fájlkezelőből való bemásoláskor), akkor r"..." formátumot kell alkalmazni
    sheet:  sheetname vagy 0-bázisú sorszám.  Lista esetén dict-of-DataFrame lesz a return
    index_col:   ez legyen a tbl indexe
        - ha nincs megadva, akkor autosorszámozott index
        - nem kell unique-nak lennie   (de ha nem unique, akkor nem szerencsés index-be vinni, mert nem lesz
                azonosítója a soroknak)
        - lehet egy korábbi kiírás "index" nevű oszlopa is
    header_row:  None esetén nincs fejléc sor
        - 0-bázisú
    parse_dates:  próbálkozzon-e a dátumoszlopok feltérképezésével
        - nagy táblák esetén lassú lehet. Az egyedi pd.to_datetime hívások, formátum-információval, nagyságrendekkel gyorsabbak lehetnek
    '''
    try:
        return pd.read_excel(path,sheet_name=sheet,index_col=index_col,header=header_row,parse_dates=parse_dates)
    except FileNotFoundError:
        print('ERROR  Nincs ilyen fájl (' + path + ')')
    except PermissionError:
        print('ERROR  Meg van nyitva az excel fájl, ezért nem olvasható (' + path + ')')


Read_csv=Readcsv
Read_excel=Readexcel

def Read_excelcopy(header=None,colnames=None):       # magyar nyelvű excelből másolt tartomány beolvasása tbl-be
    '''
    Előfeltétel: tetszőleges sor és oszlopszámú, de egybefüggő kijelölés másolása Excel-ből 
       (szűrt táblázat esetén csak a látható sorok kerülnek be)
    Ha a kijelölésben nincs benne a header, és colnames nincs megadva, akkor 0-bázisú sorszámozott oszlopnevek
    Az index autosorszámozott

    colnames:  list of string   vagy felsorolásos string (vessző határolással több oszlopnév és megadható)
    '''
    if type(colnames)==str: colnames = colnames.split(',')

    sep,decimal = '\t', ','

    return pd.read_clipboard(sep=sep,decimal=decimal,header=header,names=colnames)

def Read_xlscol_copy():             # Egymás alatti Excel-cellák másolása vágólapra majd beolvasás listába (header-rel nem foglalkozik)
    return pd.read_clipboard(sep='\t',decimal=',',header=None,names=['col'])['col'].values




def Detect_encoding(path):
    import chardet
    with open(r"C:\Users\Zsolt\OneDrive\Python\Projects\sklearn\Adatsorok\Tigáz Kürt\tigáz PRED ts_ARIMA.csv", 'rb') as rawdata:
        result = chardet.detect(rawdata.read(100000))
    return result



# FÁJLBA ÍRÁS

def Toexcel(tbl_ser_dict,fname_fix='tbl browse',dir='downloads',columns=None,headers=None,
            indexcol='named_only',append=False,numbering=True,colsizing=True):    # Excel fájlba írás, a Letöltések mappába
    '''
    tbl_ser_dict:  a fájba írandó táblázat vagy ser.  Több munkalap illetve táblázat esetén dictionary adható meg: munkalap to tblorser
        - list of records input is jó 
    fname_fix:  a fájlnév fix része  (append esetén a kiterjesztés és dir nélküli fájlnév)
    dir:  'downloads' - a letöltések mappába,    '' - a py file mappájába,    relatív és abszolút path is megadható
    
    columns:  ha csak a megadott oszlopokat kell kiírni (ser esetén érdektelen). Az oszlopok sorrendjének beállítására is alkalmas
        - több munkalap esetén list of list formátumban adható meg  (None is megadható egy vagy több munkalapra)
        - nem kötelező mindegyik munkalapra megadni ([] az adott helyen, ha nem szükséges)
    headers:  Fejléc feliratok.    
        - több munkalap esetén list of list formátum.  
        - akkor is megadható, ha a columns nincs megadva. Ha azonban a columns meg van adva, akkor igazodni kell hozzá.
        - nem kötelező mindegyik munkalapra megadni (None az adott helyen, ha nem szükséges)


    indexcol:  True esetén a bal szélre kiírja az index-értékeket (fejléc: az index neve vagy "index")
        'named_only':  csak az elnevezett indexet írja ki  (kivéve series, amikoris mindeképpen kiírja)
    append: autosorszámozott index esetén alkalmazható
            True esetén keres egy ilyen nevű fájlt. Ha van, akkor beolvassa és hozzáfűzi az új táblázatot vagy ser-t, az autosorszámozott
            index továbbfolytatásával (új oszlopok is megengedettek).  Ebben az esetben a fájlnévben nincs autosorszámozás
    colsizing:  True esetén automatikus oszlopszélesség méretezés (az első 1000 sor alapján, max 50 karakter)
    
    További paraméterezési lehetőségek:  
            startrow, strartcol:        csak extra esetben lehet szükséges
            float_format, date_format, datetime_format:    egyelőre nincs használva, de hasznos lehet
            merge_cells:                Ture esetén a multiindexek csoportosítva jelennek meg (ritkán lehet hasznos)
    '''
    
    if type(tbl_ser_dict)==dict:
        sheets=list(tbl_ser_dict.keys())
        tblorsers=list(tbl_ser_dict.values())
    else:
        sheets=[fname_fix]
        tblorsers=[tbl_ser_dict]

    if columns is not None:
        if len(sheets)==1: columns=[columns]
    if headers is not None:
        if len(sheets)==1: headers=[headers]

    # if columns is not None and headers is None:  headers=columns

    # path beállítása
    if numbering and not append: 
        path=nextpath(fname_fix,'xlsx',dir)
    else: 
        path=fname_fix + '.xlsx'
        if dir.lower()=='downloads': dir=str(Path.home() / "Downloads")
        if dir!='': path=dir + "\\" + path
        
        
    bWriteOk=True
    with pd.ExcelWriter(path) as excelwriter:           # implicit close() a végén
        
        
        for i,sheet in enumerate(sheets):
            tblorser=tblorsers[i]
            if type(tblorser)==list: tblorser = pd.DataFrame(tblorser)    # list of records input
            
            sheet=sheet[:30]        # excel előírás

            columns_=None
            if columns is not None: 
                if columns[i] is not None and len(columns[i])>0: columns_=columns[i]
            header=True
            if headers is not None: 
                if headers[i] is not None and len(headers[i])>0: header=headers[i]


            if indexcol=='named_only':
                indexcol= isser(tblorser) or (tblorser.index.name is not None)

            index_label=None
            if indexcol and tblorser.index.name is None: index_label='index'

            if not append:
                try: 
                    tblorser.to_excel(excelwriter,sheet_name=sheet,columns=columns_,header=header,
                                        index=indexcol,index_label=index_label,freeze_panes=(1,0),merge_cells=False)
                except Exception as e:
                    print('Hiba a fájlba íráskor: ' + str(e))
                    bWriteOk=False
            
            elif append:
                try:
                    tbl_old = pd.read_excel(path)
                    # unnamed és "index" oszlopok törlése  (a korábbi to_excel "index" vagy üres fejlécű oszlopba írhatta az indexet)
                    colstodrop=[]
                    for colname in tbl_old.columns:
                        if colname=='index' or beginwith(colname,'Unnamed'): colstodrop.append(colname)
                    tbl_old.drop(columns=colstodrop,inplace=True)

                except:
                    tbl_old=pd.DataFrame()

                tbl_new = tbl_old.append(tblorser,True)
                
                try: 
                    tbl_new.to_excel(excelwriter,sheet_name=sheet,index=indexcol,index_label=index_label,freeze_panes=(1,0),merge_cells=False)
                except:
                    bWriteOk=False

            if not bWriteOk: break

            # Formázások

            worksheet = excelwriter.sheets[sheet]  

            # formátum beállítása az egész munkafüzetre
            # fmt = excelwriter.book.add_format({"font_name": "Century Gothic"})
            # worksheet.set_column('A:Z', None, fmt)
            # worksheet.set_row(0, None, fmt)            
            
            if colsizing:
                if isser(tblorser):
                    max_len = max((
                        tblorser.astype(str).map(len).max(),       # konvertálás stringre, len() függvény alkalmazása minden elemre (a map itt apply jellegű)
                        len(str(tblorser.name))                     # len of column name/header
                        )) + 1  # adding a little extra space
                    max_len = Limit(max_len,max=50)
                    i_col=1
                    if not indexcol: i_col=0
                    worksheet.set_column(i_col, i_col, max_len)  # set column width
                    
                else:
                    try:
                        columns_=columns[i]     # munkalaponként külön-külön adható meg
                    except:
                        columns_=None
                    if columns_ is None: columns_=tblorser.columns
                    
                    for i_col, colname in enumerate(columns_):  # loop through all columns
                        ser_col = tblorser[colname].iloc[:1000]     # max 1000 sort nézzen végig
                        max_len = max((
                            ser_col.astype(str).map(len).max(),      # konvertálás stringre, len() függvény alkalmazása minden elemre (a map itt apply jellegű)
                            len(str(ser_col.name))  # len of column name/header
                            )) + 1  # adding a little extra space
                        max_len = Limit(max_len,max=50)
                        if indexcol: i_col=i_col + tblorser.index.nlevels
                        worksheet.set_column(i_col, i_col, max_len)  # set column width


    if bWriteOk:
        print('Excel fájlba írás: ' + path)

    else:
        print('A fájlba írás nem sikerült  (' + path + ')')


# ser=pd.Series(['aaaaaa aaaaaa aaaaaaaa aaaaa','b','c'],[11,12,13])
# ser
# Toexcel(ser,'teszt',headers=['Values oszlop'*10])



def Tocsv(tblorser,fname_fix='tbl browse',dir='downloads',format=None,indexcol='named_only',append=False,numbering=True):    # CSV fájlba írás, a Letöltések mappába
    '''
    FIGYELEM:   itt format=None a default   (a beolvasásnál viszont format='hu')
    
    tblorser:  a fájba írandó táblázat vagy ser
    fname_fix:  a fájlnév fix része  (append esetén a kiterjesztés és dir nélküli fájlnév)
    dir:  'downloads' - a letöltések mappába,    '' - a py file mappájába,    relatív és abszolút path is megadható
    indexcol:  True esetén a bal szélre kiírja az index-értékeket (fejléc: az index neve vagy "index")
        'named_only':  csak az elnevezett indext írja ki
    format:
         None:              sep=','     decimal='.'
        'hu':               sep=';'     decimal=','   encoding='cp1250'
        'hu_utf8':          sep=';'     decimal=','
        'utf16':            encoding='uft_16_le'      hőmérsékleti adatsorok beolvasásakor fordult elő
    append: autosorszámozott index esetén alkalmazható.
            True esetén keres egy ilyen nevű fájlt. Ha van, akkor beolvassa és hozzáfűzi az új táblázatot vagy ser-t, az autosorszámozott
            index továbbfolytatásával (új oszlopok is megengedettek).  Ebben az esetben a fájlnévben nincs autosorszámozás
    numbering:  ütközés esetén sorszámozza a fájlnevet (kivéve append).  False esetén felülírás
    '''

    if format is None:          sep,decimal,encoding = ',', '.', None
    elif format=='hu':          sep,decimal,encoding = ';', ',', 'cp1250'
    elif format=='hu_utf8':     sep,decimal,encoding = ';', ',', 'utf8'
    elif format=='utf16':       sep,decimal,encoding = ',', '.', 'utf_16_le'

    if indexcol=='named_only':
        indexcol=  (tblorser.index.name is not None)

    index_label=None
    if indexcol and tblorser.index.name is None: index_label='index'

    if not append:
        if numbering: path=nextpath(fname_fix,'csv',dir)
        else: 
            path=fname_fix + '.csv'
            if dir.lower()=='downloads': dir=str(Path.home() / "Downloads")
            if dir!='': path=dir + "\\" + path
        try: 
            tblorser.to_csv(path,index=indexcol,index_label=index_label,sep=sep,decimal=decimal,encoding=encoding)
            print('CSV fájlba írás: ' + path)
        except:
            print('A fájlba írás nem sikerült')

    elif append:
        path=fname_fix + '.csv'
        if dir.lower()=='downloads': dir=str(Path.home() / "Downloads")
        if dir!='': path=dir + "\\" + path

        try:
            tbl_old = pd.read_csv(path,sep=sep,decimal=decimal,encoding=encoding)
            # unnamed és "index" oszlopok törlése  (a korábbi to_excel "index" vagy üres fejlécű oszlopba írhatta az indexet)
            colstodrop=[]
            for colname in tbl_old.columns:
                if colname=='index' or beginwith(colname,'Unnamed'): colstodrop.append(colname)
            tbl_old.drop(columns=colstodrop,inplace=True)
        except:
            tbl_old=pd.DataFrame()
               
        tbl_new = tbl_old.append(tblorser,True)
        
        try: 
            tbl_new.to_csv(path,index=indexcol,index_label=index_label,sep=sep,decimal=decimal,encoding=encoding)
            print('CSV fájlba írás: ' + path)
        except:
            print('A fájlba írás nem sikerült  (' + path + ')')




# SZŰRÉSEK

def indexlike(tbl_or_ser,like):     # keresés indexre, like mintával
    '''
    A megadott szövegrésszel kezdődő indexű rekordok leválogatása (case insensitive)
    Dataframe-re és Series-re is működik
    Csak stringindex-szel rendelkező táblázatokra működik (számindex esetén üres táblázat az eredmény)
    like: ha * van az elején, akkor bárhol előfordulhat az indexben
    '''
    
    if str(tbl_or_ser.index.dtype)!='object': 
        print('Warning: indexlike works properly if index is of type string')
        return tbl_or_ser.iloc[0:0]         # üres táblázatot ad vissza
    if len(like)==0: 
        return tbl_or_ser.iloc[0:0]         # üres táblázatot ad vissza

    if like[-1]=='*': like=like[:-1]    # a végén mindenképpen érvényesül a *
    if len(like)==0: return tbl_or_ser.iloc[0:0] 

    # Ha az elején is van csillag, akkor bárhol a szövegben
    if like[:1]=='*' and len(like)>1:
        return tbl_or_ser[tbl_or_ser.index.str.contains(like[1:],case=False)]
    # Ha csak a végén van csillag
    elif len(like)>0: 
        return tbl_or_ser[tbl_or_ser.index.str.contains('^' + like,case=False)]


def tbllike(tbl,col,like):
    '''
    A táblázat szűrése.  A megadott szövegrésszel kezdődő rekordok, vagy contains rekordok  (case érzéketlen)
    
    col:  stringoszlopnak kell lennie
    like:  keresőminta  (a végén lehet *, de nem kötelező)
        kisbetűvel érdemes megadni   (a keresés case-érzéketlen)
        ha *-gal kezdődik, akkor "bárhol előfordul" keresés
        a végén lévő csillag érdektelen  (mindenképpen úgy tekinti, hogy csillag van a végén)
    '''
    if like is None or len(like)==0:
        return tbl.iloc[0:0]         # üres táblázatot ad vissza
    
    if str(tbl[col].dtype)!='object': 
        print('ERROR   tbllike   cols:  csak string-oszlopra alkalmazható')
        return tbl.iloc[0:0]         # üres táblázatot ad vissza

    if like[-1]=='*': like=like[:-1]    # a végén mindenképpen érvényesül a *
    if len(like)==0: return tbl.iloc[0:0] 

    # Ha az elején is van csillag, akkor bárhol a szövegben
    if like[:1]=='*' and len(like)>1:
        return tbl[tbl[col].str.contains(like[1:],case=False)]
    # Ha csak a végén van (volt) csillag
    elif len(like)>0: 
        return tbl[tbl[col].str.contains('^' + like,case=False)]


def serlike(ser,like):
    '''
    Series szűrése.  A megadott szövegrésszel kezdődő rekordok, vagy contains rekordok  (case érzéketlen)
    
    like:  keresőminta  (a végén lehet *, de nem kötelező)
        kisbetűvel érdemes megadni   (a keresés case-érzéketlen)
        ha *-gal kezdődik, akkor "bárhol előfordul" keresés
        a végén lévő csillag érdektelen  (mindenképpen úgy tekinti, hogy csillag van a végén)
    '''
    if like is None or len(like)==0:
        return ser.iloc[0:0]         # üres táblázatot ad vissza
    
    if str(ser.dtype)!='object': 
        print('ERROR   serllike   cols:  csak string-oszlopra alkalmazható')
        return ser.iloc[0:0]         # üres táblázatot ad vissza

    if like[-1]=='*': like=like[:-1]    # a végén mindenképpen érvényesül a *
    if len(like)==0: return ser.iloc[0:0] 

    # Ha az elején is van csillag, akkor bárhol a szövegben
    if like[:1]=='*' and len(like)>1:
        return ser[ser.str.contains(like[1:],case=False)]
    # Ha csak a végén van (volt) csillag
    elif len(like)>0: 
        return ser[ser.str.contains('^' + like,case=False)]





def tblfiltercols(tbl,colnamefilter):       # oszlopnevek szűrése   Return: tbl_filtered vagy oszlopnevek
    '''
    Oszlopszűrő az oszlopnevekben előforduló szövegrészekre. 
    - keresés szavanként (ÉS kapcsolat, a szavak sorrendje érdektelen)
    - vesszővel elválasztva OR feltételek
    - NOT után kizáró feltétel
      példa:  "total case,cumulative NOT smoothed":  van benne "total" és "case", vagy van benne "cumulative" és nincs benne "smoothed"
    Return:  tbl_filtered   (nem inplace)
    '''
    if colnamefilter is None or colnamefilter=='': return tbl

    colnamesout=filternames(list(tbl),colnamefilter)
    if colnamesout:
        return tbl[colnamesout]


def Tblcol(tbl,col,msg=False):    # legközelebbi találat az anywhere találatok közül  (kisbetű-nagybetű érzéketlen)
    col_out=Find_in_list(col,list(tbl))
    if not col_out and msg: print('ERROR  Tblcol  Nincs ilyen oszlop: "' + col + '"')
    return col_out

def Tblcols(tbl,cols,basecol=None,corrtype='pearson',commalist=True):       # Szöveges szűrés vagy felsorolás     return:  list of colnames
    '''
    cols:  string vagy list   (ha üres, akkor az összes oszlop)
        egyetlen oszlop neve, vagy oszlopnevek vesszős felsorolása (teljes név, de case-insensitive)
        '**' az elején:  filtercols() függvénnyel kiválasztott oszlopok
            - kettős idézőjelben ilyenkor is megadhatók pontos oszlopnevek (teljes találat, de case-insensitive)
        'type=...' az elején  (float / int / text / date vesszős felsorolása)    példa:  "type=float,int"
            - utána állhat filternames szűrő  (szóközzel elválasztva; pozitív és negatív része is lehet, lásd filtercols) 
        'top', 'top5':  a basecol oszloppal legnagyobb korrelációjú oszlopok  (csak float vagy integer oszlopok)
            - utána állhat filternames szűrő  (szóközzel elválasztva; pozitív és negatív része is lehet, lásd filtercols) 
        list of colnames  is megadható  (return=cols;   nem ellenőrzi az oszlopnevek megfelelőségét)
        ha None, üres, vagy 'all' akkor az összes oszlop
    basecol:  csak cols='top...' esetén kell.   Ezzel az oszloppal kell képezni a korrelációkat
    corrtype:  csak cols='top...' esetén kell.   'pearson', 'kendall', 'spearman'
    commalist:  True esetén vesszős felsorolással is megadhatóak az oszlopnevek 
            (kikapcsolandó, ha lehet vessző az oszlopneveken belül is)

    '''
    if type(cols)==list:
        return cols
    elif cols is None or cols=='' or cols=='all':
        return list(tbl)
    elif beginwith(cols,'type='):
        cols = cutleft(cols,'type=')
        tipusok,filter = splitfirst(cols,' ')       # a típusfelsoroláson belül nem lehet szóköz
        b_float = vanbenne(tipusok,'float')
        b_int = vanbenne(tipusok,'int')
        b_text = vanbenne(tipusok,'text|txt')
        b_date = vanbenne(tipusok,'date')

        cols=[]
        colnames=list(tbl)
        for colname in colnames:
            dtype=tbl.dtypes[colname].name
            if b_float and vanbenne(dtype,'float'): cols.append(colname)
            elif b_int and vanbenne(dtype,'int'): cols.append(colname)
            elif b_text and vanbenne(dtype,'object'): cols.append(colname)
            elif b_date and vanbenne(dtype,'date'): cols.append(colname)

        if len(cols)>0 and filter:  cols=filternames(cols,filter)

    elif beginwith(cols,'top'):
        first,filter = splitfirst(cols,' ')
        count=None
        if first!='top': count=int(cutleft(first,'top'))
        cols=Topcorrs(tbl,basecol,method=corrtype)
        if filter: cols=filternames(cols,filter)
        if count: cols = cols[:count]
    else:
        if cols[:2]=='**': cols=filternames(tbl.columns,cols[2:])
        else:
            if commalist: cols_=cols.split(',')
            else: cols_=[cols]
            cols=[]
            for col in cols_:
                col_out=Tblcol(tbl,col)         # legközelebbi találat
                if col_out: cols.append(col_out)
                else: print('Tblcols  Ismeretlen oszlopnév: "' + col + '"')

    return cols



def filternames(names,filter):           # segédfüggvény az oszlopnév-szűrésekhez
    '''
    Szűrés a megnevezésekben előforduló szövegrészekre. A kimenet egy névlista.
    names:  választható nevek  (a kimenetbe ezek a nevek kerülhetnek)  
       list    vagy str (vesszős felsorolás)    vagy np.ndarray (np list)
       példa: ["cases", "total_cases", "cases_smoothed", "total_cases_smoothed"]
              "cases,total_cases,cases_smoothed"
       - a return mindenképpen list
    filter:    ha üres, akkor return names (nincs szűrés)
    - vesszős névfelsorolás is megadható (case-insensitive, bárhol a szövegben)
        Ha teljes találat kell (nem "bárhol előfordul"), akkor idézőjelezni kell az adott nevet (ilyenkor is case_insesitive)
    - általános esetben:
        - vesszővel elválasztva OR feltételek  (az idézőjelezés teljes OR-feltételrészekre alkalmatható, nem az AND-szavakra)
        - OR feltételeken belül:  keresés szavanként (ÉS kapcsolat, a szavak sorrendje érdektelen)
        - NOT után kizáró feltétel
      példa:  'total case,cumulative NOT smoothed':  van benne "total" és "case", vagy van benne "cumulative" és nincs benne "smoothed"
              'NOT smoothed':    az összes, kivéve a "smoothed" szövegrészt tartalmazók
    - a kimeneti nevek sorrendje a filter-hez igazodik
    - a kiemelés jelzések ("^") a kereséskor levágva, majd a kimeneti listába is bekerülnek az oszlopnevek elé  (pl. "^hun,ger,^slovakia,austria")
        Az "asc" és a "desc" zárószavakat is levágja, és beírja a kimeneti lista oszlopnevei után 
    Return:  List, a filternek megfelelő nevek listája
    '''
    if type(names)==np.ndarray: names=list(names)
    elif type(names)==str: names=names.split(',')
    
    filter=filter.strip()
    if filter=='':
        return names

    if filter.startswith('NOT '):
        parts=filter.split('NOT ')
    else:
        parts=filter.split(' NOT ')
    if len(parts)>2: parts=parts[:2]        # part[0]: pozitív feltétel,  part[1]: negatív feltétel
    
    aNamesout=[]
    bNot=False
    for nPart in range(len(parts)):     # part[0]: pozitív feltétel,  part[1]: negatív feltétel
        part=parts[nPart]
        if part=='':
            if nPart==0: aNamesout=names.copy()
            continue
        
        bKiemelt=False
        
        # vesszővel elválasztva VAGY feltételrészek
        orfilters=part.split(',')
        orfilters_kiemelt=[False]*len(orfilters)


        # ciklus a filterben szereplő mintákra  (a kimeneti lista sorrendje a filter-hez igazodik)
        for i,orfilter in enumerate(orfilters):    # orfilter:  ["hun"]   ["total","cases"]
            orfilter=orfilter.strip()
            if len(orfilter)==0: continue
            
            # Ha idézőjelezett az orfilter
            if len(orfilter)>2 and orfilter[0]=='"' and orfilter[-1]=='"':
                orfilter=orfilter[1:-1]
                name_out=Str_in_list(orfilter,names,case='clean',pos='equal')
                if name_out:    # ha van ilyen oszlopnév  (case-től eltekintve)
                    if nPart==0:   # pozitív esetén hozzáadás (unique ellenőrzéssel)
                        if not (name_out in aNamesout): aNamesout.append(name_out)
                    elif nPart==1:   # negatív esetén elhagyás
                        if name_out in aNamesout: aNamesout.remove(name_out)

                # if orfilter in names:
                #     if nPart==0: 
                #         if not (orfilter in aNamesout): aNamesout.append(orfilter)
                #     elif nPart==1: 
                #         if orfilter in aNamesout: aNamesout.remove(orfilter)
            
            # Ha az orfilter nem idézőjelezett
            else:
                orfilter=Lics(orfilter)
                bKiemelt=False
                if orfilter[0]=='^':                
                    orfilter=cutleft(orfilter,'^')
                    bKiemelt=True
                bDesc=False
                if endwith(orfilter,' asc'): orfilter=cutright(orfilter,' asc')
                elif endwith(orfilter,' desc'): 
                    orfilter=cutright(orfilter,' desc')
                    bDesc=True
                
                # szóközzel elválasztott keresőszavak, listává alakítás ("ÉS" kapcsolat, egyelemű is lehet)
                orfilter=orfilter.split()   # orfilter: ["total","cases"]
                
                # ciklus a választható nevekre
                for name in names:
                    if type(name)==str: nameL=Lics(name)
                    else: nameL=str(name)
                    bOkL=True
                    for sample in orfilter:         # ÉS minták    
                        if not (sample in nameL):   # bárhol előfordulhat a névben
                            bOkL=False              # ha van olyan ÉS minta, ami nem fordul elő, akkor nincs találat
                            break
                    # Ha a név megfelel az orfilter-nek
                    if bOkL:
                        if bKiemelt: name='^' + name        # a kimeneti listába is kerüljön be a '^' jel
                        if bDesc: name = name + ' desc'
                        if nPart==0: 
                            if not (name in aNamesout): aNamesout.append(name)
                        elif nPart==1: 
                            if name in aNamesout: aNamesout.remove(name)
            
    return aNamesout

def tblgroupby(tbl,groupcol,valuecol,indexcol=None,groupfilters=None,pivot=False):    # dict of ser a csoportosító mező értékeire (érték to ser)
    '''
    A rekordok szétdarabolása egy csoportosító mező értékei szerint. 
      Nem történik aggregálás, szűrésekről van szó
    Return:  dict of ser,    csoportosító értékek  to  series    (a series-be az indexcol és a valuecol értékei kerülnek)
    
    groupcol:  colname a tbl-ben  (általában véges sok kategorizáló értékkel)
    valuecol:  a kimeneti ser-ekbe ennek az oszlopnak az értékei kerülnek
    indexcol:  a kimeneti ser-ekbe kerülő index-értékek  (általában ez is egy különálló oszlop, pl. dátumok)
        - None esetén a tbl eredeti indexértékei
    groupfilters:  lásd filternames()    Ha üres, akkor a groupcol-ban előforduló összes érték. 
        keresőszavak felsorolása (köztük szóköz),  vesszővel elválasztva VAGY feltételek, 
           a végén NOT után olyan szórészletek, amelyek nem fordulhatnak elő a csoportosító értékben
    pivot:  
        False esetén a kimeneti ser-ek eltérő hosszúságúak lehetnek. Egyszerű szűrések az adott kategóriaértékekre
        True esetén mindegyik ser-be bekerül az összes index-érték (a ser-ek azonos hosszúságúak lesznek). 
            Ha egy indexértékhez nem volt adat az adott kategóriában, akkor nan
    '''
    if pivot:
        result={}
        tblL=tbl.pivot(index=indexcol,columns=groupcol,values=valuecol)
        for colname, colseries in tblL.items():
            result[colname]=colseries

    else:
        result={}
        groupvalues = tblgroupbyvalues(tbl,groupcol,groupfilters)
        for groupvalue in groupvalues:
            tblL=tbl.loc[tbl[groupcol]==groupvalue]
            if not indexcol: 
                result[groupvalue]=tblL[valuecol]
            else:
                result[groupvalue]=serfromtbl(tblL,valuecol,indexcol)
    return result

def tblgroupbyvalues(tbl,groupcol,groupfilters=None):       # groupby értékek listája, szűrési lehetőséggel
    ''' 
    GroupBy értékek listáját adja vissza. A lista ismeretében egy ciklus és a tábla szűrése a groupby értékekre
            for group in groups:
                tblL=tbl.loc[tbl[groupcol]==group]
    tbl:
    groupcol: colname a tbl-ben  (általában véges sok kategorizáló értékkel)
    groupfilters:  lásd filternames()    Ha üres, akkor a groupcol-ban előforduló összes érték. 
        keresőszavak felsorolása (köztük szóköz),  vesszővel elválasztva VAGY feltételek, 
           a végén NOT után olyan szórészletek, amelyek nem fordulhatnak elő a csoportosító értékben
    '''
    values=tbl[groupcol].unique()      # distinct

    # nan, None, '' ne kerüljön bele
    groupvalues=[]
    for value in values:
        if pd.isna(value) or value==None or value=='': continue
        groupvalues.append(value)

    if groupfilters:
        groupvalues=filternames(groupvalues,groupfilters)


    return groupvalues



# PIVOT

def Pivot(tbl,groupercol,cols,aggfunc='mean',toindex=False,cols_out=None):      # Excel-pivot
    '''
    groupercol:  általában egy oszlop, de több oszlop is megadható   (list,  vesszős felsorolás nem jó)
        - többnyire egy szöveges kategóriaoszlop vagy egy integer oszlop (integer kódok)
    cols:    egyetlen oszlopnév, vagy list of colnames    (vesszős felsorolás nem megy, mert az oszlopnevekben is lehet vessző)
    aggfunc:  több is megadható  (itt a vesszős felsorolás is megengedett   vagy list)
        - 'mean', 'count', 'sum', 'first', 'last', ....
        - az nan értékeket nem veszik figyelembe a függvények
        - string-oszlopra nem mindegyik alkalmazható  (pl. 'mean' csak számoszlopra)
        - ha több oszlop és több aggfunc van megadva, akkor direktszorzat lesz az eredmény
    cols_out: kimeneti oszlopok neve  (list)
        FIGYELEM: az oszlopnevek sorrendjével valami baj van ...
        - elemszám:  groupercols + cols * aggfunc   (a groupercol nevének is szerepelnie kell benne)
            Több cols és több aggfunc esetén a sorrend:  col1_agg1, col2_agg1, col1_agg2, col2_agg2
        - ha nincs megadva, akkor egyetlen cols esetén aggfunc, egyébként col + '_' + aggfunc
            Ha nem jó az elemszám, akkor hibaüzenet, és végrehajtás default elnevezésekkel
        - gyakran kell, mert a pivot tbl többnyire publikus megjelenítésre van szánva (pl. Toexcel)

    return:
        - DataFrame, autorszámozott index-szel  (rekordok száma:  groupercol unique értékeinak a száma)
        - a groupercol is külön oszlopba kerül  (több oszlopos group esetén sem csinál hierarchikus indexet)
        - az oszlopnevek colname + '_' + aggfunc mintát követik, kivéve ha cols_out-ban közvetlenül meg lettek adva
    
    '''
    
    if type(groupercol)==str: groupercol=[groupercol]
    if type(cols)==str: cols=[cols]
    if type(aggfunc)==str: aggfunc=aggfunc.split(',')

    if cols_out is not None and len(cols_out)!=len(groupercol) + len(cols)*len(aggfunc):
        print('ERROR: Pivot  cols_out  Érvénytelen elemszám, az argumentum figyelmen kívül hagyva' )
        cols_out=None
    
    tblP=tbl.pivot_table(index=groupercol,values=cols,aggfunc=aggfunc)

    # Az oszlopnevek egyszintűvé tétele
    columns2=[]
    columns=tblP.columns.values         #  pl. [('mean','col1'),('mean','col2'),('count','col1')]
    for i,col in enumerate(columns):      # a groupercol az index, ezért nem kerül ide
        agg,colname=col
        colname_new=None
        if cols_out is not None: colname_new = cols_out[i+len(groupercol)]    # az elején a groupercol neve áll
        if colname_new is None:
            if len(cols)==1: colname_new=agg
            else: colname_new=colname + '_' + agg
        columns2.append(colname_new)
    tblP.columns=columns2
   
    if not toindex:tblP=tblP.reset_index()      # a groupercol is önálló oszlopba kerül  (autosorszámozott index)
    
    if cols_out is not None:
        for i,colname in enumerate(groupercol):
            tblP=tblP.rename(columns={colname:cols_out[i]})

    return tblP

def Sumrow(tbl,pos='bottom',agg='sum',caption='Összesen',captioncol=0):     # összegzősor hozzáadása (sorszámozott indexű táblához)
    '''
    NEM INPLACE
    Elvárás:  a tábla autosorszámozott és az első oszlopa feliratokat tartalmaz (az első oszlopba kerül az "Összesen" felirat)
    Csak a számoszlopokra összegez (a többire nan)

    captionsol: melyik oszlopba kerüljön a felirat.  Oszlopsorszám vagy oszlopnév adható meg
    
        '''
    if type(captioncol)==str: captioncol=list(tbl).index(captioncol)

    if pos=='bottom':
        tbl=tbl.append(tbl.agg(agg,numeric_only=True),ignore_index=True)
        tbl.iloc[-1,captioncol]=caption     # utolsó sor
    elif pos=='top':
        tbltop=pd.DataFrame(columns=tbl.columns).append(tbl.agg(agg,numeric_only=True),ignore_index=True)
        tbl=tbltop.append(tbl,ignore_index=True)
        tbl.iloc[0,captioncol]=caption
    return tbl

def Sumcol(tbl,pos='right',agg='sum',caption='Összesen',cols='numeric'):
    '''
    pos:  'right', 'left'
    agg:  'sum', 'mean', ...
    cols:  melyik oszlopokra összegezzen
        'numeric':          összes számoszlop
        ['col1','col2']:    felsorolt oszlopnevek
    '''
    if cols=='numeric':  tbl_sum=Numcols(tbl)
    else: tbl_sum=tbl[cols]

    tbl[caption]=tbl_sum.agg(agg,axis=1)
    if pos=='left': cols_to_front(tbl,caption)
    return tbl

def Blankrow(tbl,count=1,pos='bottom'):        # üres sor hozzáadása (sorszámozott indexű táblához)
    '''
    Elsősorban kimeneti excel-táblák esetén használható
    '''
    return tbl.append([None]*count,ignore_index=True)

def Numcols(tbl,out_='tbl'):
    tbl_out=tbl.select_dtypes(include=np.number)
    if out_=='tbl': return tbl_out
    elif out_=='colnames': return tbl_out.columns.tolist()


# tbl=pd.DataFrame({'id':[1,2,3,2,3,4],'grouper':['A','A','A','B','B','B'],'value':[10,11,12,13,14,15]})
# tbl.pivot(index='id',columns='grouper',values='value')



# MERGE MŰVELETEK

def merge_ser(tbl,ser,colname_new,colnames_hiv):   # 1.hivatkozott kódjegyzék feliratának vagy parent objektum adatának átvétele, 2. kiegészítő oszlop bevétele
    '''
    nem inplace jellegű a művelet     tbl=tbl.merge_ser() 

    ser:  kódlista jellegű    
        A tbl colnames_hiv mezője (mezői) a ser indexére hivatkozik (nem elvárás a teljes körű hivatkozás)
        Megjegyzés: ha egy sorrendhelyes és azonos méretű array-t vagy listát kell felvenni a tbl egy 
            új oszlopába, akkor nem kell merge, egyszerű értékadással oldható meg   (Példa:  tbl[newcol] = arr)
    colnames_new:  a táblába felveendő új oszlop neve. Ha van már ilyen oszlop, akkor utótagot kap
    colnames_hiv:  egy-oszlopos hivatkozás esetén string, egyébként list
        - a ser indexére hivatkozik  (lehet benne nan is)
        - általában nem unique, ezért az adatbeolvasás redundanciát eredményez (sok rekordba íródhat ugyanaz az érték)
        - lehet unique is, pl egy kiegészítő adat átvételekor
        - ha egy hivatkozáshoz nincs találat, akkor nan kerül be   (a ser-nek nem feltétlenül kell teljesnek lennie)
    '''
    ser.name=colname_new
    return tbl.merge(ser,how='left',left_on=colnames_hiv,right_index=True)

def merge_from_parent(tbl, tblParent, on, parent_on, parent_cols):   # szülőobjektum adatainak bevétele
    '''
    Szülőtábla vagy mellérendelt tábla adatainak átvétele. Az alaptábla rekordjainak száma nem változik
        - oszlopnévütközés esetén az újonnan bekerül mező neve végére '_y' utótag kerül
        - a parent kapcsolómezői is bekerülnek (redundáns, de hasznos annak ellenőrzésére, hogy egy rekord parenthivatkozása helyes-e)
            Ha ez zavaró, akkor a parent-tábla indexébe érdemes tenni a kapcsolómező(ke)t (így nem kerülnek át az alaptáblába)
        - ha üres vagy téves a hivatkozás, akkor nan értékek kerülnek az új oszlopokba
    tblParent:  szűlőobjektum táblája vagy egy mellérendelt tábla.
        Hibaüzenet, ha a szülőtáblában nem unique a kapcsolómező
    on:  kapcsolómező neve az alaptáblában.  Több mező esetén felsorolás (lista vagy vesszős felsorolás)
        - 'index':  akkor fordulhat elő, ha mellérendelt tábláról van szó (ritka eset)
    parent_on:  kapcsolómező neve a parent táblában. Ha üres, akkor megegyezik az alaptábla hasonló adatával.  Több mező esetén felsrorolás (lista vagy vesszős felsorolás)
        - 'index':  ha a szülőtábla kapcsolómezőjét a tábla indexe tartalmazza (nem kötelező a unique azonosítót indexbe vinni)
    parent_cols:  egyetlen oszlop neve vagy oszlopnevek felsorolása (lista vagy vesszős felsorolás). 
        - a kapcsolómezőket nem kell megadni, azok mindenképpen bekerülnek.
    '''
    if not parent_on: parent_on=on

    left_index=False
    if on=='index': 
        left_index=True
        left_on=None
    else:
        if type(on)==str:
            if vanbenne(on,','): on=on.split(',')
            else: on=[on]
        left_on=tuple(on)
    
    right_index=False
    if parent_on=='index':
        right_index=True
        right_on=None
    else:
        if type(parent_on)==str: 
            if vanbenne(parent_on,','): parent_on=parent_on.split(',')
            else:parent_on=[parent_on]
        right_on=tuple(parent_on)

    if type(parent_cols)==str: 
        if vanbenne(parent_cols,','): parent_cols=parent_cols.split(',')
        else: parent_cols=[parent_cols]

    if parent_on!='index':
        for i,colname in enumerate(parent_on):
            if not (colname in parent_cols): parent_cols.insert(i,colname)


    return tbl.merge(tblParent[parent_cols],how='left',left_on=left_on,left_index=left_index,right_on=right_on,right_index=right_index,
                    validate='many_to_one',suffixes=(None,'_y'))

def merge_from_sub(tbl, tblSub, on, sub_on, aggs):    # Altábla mezőinek átvétele a szülőtáblába, aggregálással
    '''
    Altábla adatainak átvétele, aggregálással. Az alaptábla rekordjainak száma nem változik
    on: kapcsolómező(k) a szülőtáblában.  
        - 'index': a parent tábla indexe   (a kucsmezőt nem kötelező bevinni a tbl indexébe, néha jobb, ha "kint" marad)
    sub_on:  kapcsolómező(k) az alárendelt táblában  (hivatkozó mező;  normális esetben nem lehet az alárendelt tábla indexe, mert akkor nem kellene aggregálás)
        - többmezős hivatkozás esetén listát kell megadni
    aggs:  az aggregálandó mezők felsorolása   
            Példa:  [['sum','kWh'],['mean','m3']]     
        - ha csak egy mezőt kell átvenni, akkor elég egy mélységű lista   
            Példa: ['sum','kWh']
        - az első helyen az aggregáló művelet áll: 'count', 'sum', 'mean', 'median', 'min', 'max', 'first', 'last', ... 
        - második helyen: az aggregálandó mező az altáblában
            - a hivatkozó mező nem adható meg. 
            - string-mezőkre nem mindegyik művelet alkalmazható (pl. a mean csak számmezőkre működik) 
            - a 'count' művelet esetén is meg kell adni egy mezőt. Ügyelni kell arra, hogy csak azoknak az 
              alárendelt rekordoknak a száma kerül be, amelyekben az adott mezőnek van értéke (egy teljesen 
              feltöltött mezőt érdemes megadni, de ez nem lehet a hivatkozó mező)
              Megjegyzés:  nem sikerült megoldani a size() jellegű művelet integrálását
        - a harmadik adat opcionális:  az új mező neve a parent táblában (ha nincs megadva, akkor "sum_kWh", "mean_m3")
            Példa: [['sum','kWh','összesített kWh']] 
    '''

    # def add_groupcol_teszt():
    #     tblParent=pd.DataFrame({'id':[1,2,3,4],'adat':[10,11,12,13]})
    #     tblSub=pd.DataFrame({'id':[0,1,2,3],'parentid':[1,1,2,3],'kWh':[20,22,30,40],'m3':[30,32,40,50]})
    #     tbl=merge_sub(tblParent,tblSub,'id','parentid',[['count','id','count_sub'],['sum','kWh'],['mean','kWh'],['mean','m3']])
    #     print(tbl)



    if type(aggs[0])==str: aggs=[aggs]      # ha 1 mélységű lista lett megadva

    if type(sub_on)==str: 
        if vanbenne(sub_on,','): sub_on=sub_on.split(',')
        else: sub_on=[sub_on]   # egymezős esetben is legyen lista
    if type(on)==str: 
        if vanbenne(on,','): on=on.split(',')
        else: on=[on]   # egymezős esetben is legyen lista

    # Aggregálandó mezők listája
    cols_agg=[]
    f_aggs=[]
    rename={}
    colnames=[]
    for agg in aggs:
        f_agg,col_agg,col_new = unpack(agg,3)
        if not f_agg:
            print('merge sub, aggs argumentum, Nincs megadva az aggregáló művelet, "' + print(aggs) + '"')
            continue
        if not f_agg in f_aggs: f_aggs.append(f_agg)
        
        if not col_agg:
            print('HIBA: merge sub, aggs argumentum, Nincs megadva az aggregálandó oszlop, "' + str(aggs) + '"')
            continue
        elif col_agg in sub_on:
            print('HIBA: merge sub, aggs argumentum, Nem adható meg a hivatkozó mező aggregálandó mezőként, "' + str(aggs) + '"')
            continue
        if not col_agg in cols_agg: cols_agg.append(col_agg)

        colname=col_agg + '_' + f_agg   # a groupby().agg() 'kWh_sum' formátumú default oszlopneveket hoz lére (flatten után)
        if not col_new and tbl.equals(tblSub):     # merge_groupdata esetén fordul elő
            col_new='_'.join(sub_on) + '_' +  col_agg + '_' + f_agg   # kerüljön a név elejére a grouper mező is
        if col_new:     
            rename[colname]=col_new      
            colname=col_new
        colnames.append(colname)

    # if len(cols_agg)==0:
    #     tbl_grouped=tblSub[sub_on].groupby(sub_on).agg(f_aggs).rename(columns=rename)
        

    # tbl_grouped=tblSub[sub_on + cols_agg].groupby(sub_on).agg(f_aggs)[cols_agg].rename(columns=rename)
    tbl_grouped=tblSub[sub_on + cols_agg].groupby(sub_on).agg(f_aggs)
    # Az oszlopnevek egyszintűvé tétele
    tbl_grouped.columns = ['_'.join(col).strip() for col in tbl_grouped.columns.values]
    # Oszlop átnevezések
    if rename: tbl_grouped = tbl_grouped.rename(columns=rename)
    # A felesleges kombinációk elhagyása  (ha pl. sum és mean is kérés volt, akkor előfordulhat, hogy a mean nem mindegyik oszlopra kell)
    tbl_grouped = tbl_grouped[colnames]

    return tbl.merge(tbl_grouped,how='left',left_on=tuple(on),right_on=tuple(sub_on))

def merge_subrecords(tbl, tblSub, on, sub_on, sub_cols):    # altábla rekordjainak átvétele
    '''
    Átveszi az altábla adatait és többletrekordjait. Változik az alaptábla rekordszáma.
        - a rekordok elején ismétlődnek a főtáblás adatok
        - ha egy főtáblás rekordhoz nincs altáblás rekord, akkor az altáblás mezőkbe nan értékek kerülnek
    tbl: alaptábla  (főtábla)
    tbl_sub:  altábla.  Egy alaptábla rekordhoz több altábla rekord tartozhat (0 is megengedett)
    on:  alaptábla kapcsolómezője (elvárás: unique).  Több mezős kapcsolás esetén lista, vagy vesszős felsorolás.
        - 'index':  a kapcsolómezőt a tábla indexe tartalmazza (nem kötelező lehet külön mező is)
            NE HASZNÁLD   csak nagyjából működik, a kimeneti tábla indexe kissé furcsa lesz
    sub_on:  altábla kapcsolómezője (általában nem unique). Több mezős kapcsolás esetén lista, vagy vesszős felsorolás.
        - ha None vagy üres, akkor megegyezik az on-nal
        - nem jelent problémát, ha érvénytelen hivatkozások is vannak benne (figyelmen kívül maradnak)
    sub_cols:  az altáblából átveendő mezők felsorolása  (lista vagy vesszős felsorolás)
        - a kapcsolómezőnek nem kell benne lennie (mindenképpen átkerül az alaptáblába)
    '''
    if not sub_on: sub_on=on

    left_index=False
    if on=='index': 
        left_index=True
        left_on=None
    else:
        if type(on)==str:
            if vanbenne(on,','): on=on.split(',')
            else: on=[on]
        left_on=tuple(on)
    
    right_index=False
    if sub_on=='index':
        right_index=True
        right_on=None
    else:
        if type(sub_on)==str: 
            if vanbenne(sub_on,','): sub_on=sub_on.split(',')
            else:sub_on=[sub_on]
        right_on=tuple(sub_on)


    if type(sub_cols)==str: 
        if vanbenne(sub_cols,','): sub_cols=sub_cols.split(',')
        else: sub_cols=[sub_cols]

    if sub_on!='index':
        for i,colname in enumerate(sub_on):
            if not (colname in sub_cols): sub_cols.insert(i,colname)


    return tbl.merge(tblSub[sub_cols],how='left',left_on=left_on,left_index=left_index,right_on=right_on,right_index=right_index,
                    validate='one_to_many',suffixes=(None,'_y'))


def merge_groupdata(tbl, grouper, aggs):   # aggregált csoportadatok hozzáadása a táblához
    '''
    Aggregált csoportadatok hozzáadás a táblához. 
    A csoportokat a grouper mező értékei alapján képzi.
    Megjegyzés:  az alapeset az lenne, hogy a merge_from_sub függvénnyel az aggregált csoportadatok a grouper 
        által hivatkozott parenttáblába visszük. A pandas "flatten" logikája azonban általában azt ösztönzi, 
        hogy a legrészletezőbb táblában jelenjenek meg a magasabb szintű adatok is (redundánsan). Ráadásul
        előfordulhat olyan grouper is, amihez nincs parent tábla (pl. a grouper egy egyszerű kódlistás mező). Parent
        táblát ilyenkor is létre lehetne hozni, de általában hatékonyabb a résztáblában való implicit kezelés.
    grouper:  csoportosító mező (általában nem unique). Többmezős is lehet (lista vagy vesszős felsorolás)
    aggs:  az aggregálandó mezők felsorolása (a grouper-en kívüli egy vagy több saját mező)
            Példa:  [['sum','kWh'],['mean','m3']]     
        - ha csak egy mezőt kell aggregálni, akkor elég egy mélységű lista   
            Példa: ['sum','kWh']
        - az első helyen az aggregáló művelet áll: 'count', 'sum', 'mean', 'median', 'min', 'max', 'first', 'last', ... 
        - második helyen: az aggregálandó mező az altáblában
            - a grouper mező nem adható meg 
            - string-mezőkre nem mindegyik művelet alkalmazható (pl. a mean csak számmezőkre működik) 
            - a 'count' művelet esetén is meg kell adni egy mezőt. Ügyelni kell arra, hogy csak azoknak az 
              alárendelt rekordoknak a száma kerül be, amelyekben az adott mezőnek van értéke (egy teljesen 
              feltöltött mezőt érdemes megadni, de ez nem lehet a hivatkozó mező)
              Megjegyzés: a size() jellegű mező felvételéhez lásd: merge_groupsize
        - a harmadik adat opcionális:  az új mező neve a parent táblában 
            - ha nincs megadva, akkor "[grouper]_kWh_sum", "[grouper]_m3_mean")
            Példa: [['sum','kWh','összesített kWh']] 
    '''
    return merge_from_sub(tbl,tbl,grouper,grouper,aggs)

def merge_groupsize(tbl, grouper, colname='[grouper]_group_size'):   # grouper mezőhöz tartozó rekordok száma új oszlopba
    '''
    grouper mezőhöz tartozó rekordok számának felvétele új oszlopba
    grouper:  csoportosító mező (általában nem unique). Többmezős is lehet (lista vagy vesszős felsorolás)
    colname:  a felveendő oszlop neve
    '''
    if not colname or colname=='': 
        print('HIBA, merge_groupsize, A colname nem lehet üres' )
        return
    if not grouper or grouper=='': 
        print('HIBA, merge_groupsize, A grouper nem lehet üres' )
        return
    
    if type(grouper)==str:
        if vanbenne(grouper,','): grouper=grouper.split(',')
        else: grouper=[grouper]
    
    colname=colname.replace('[grouper]','_'.join(grouper))

    ser=tbl[grouper].groupby(grouper).size()
    ser.name=colname
    return tbl.merge(ser,how='left',left_on=tuple(grouper),right_on=tuple(grouper))






# ISMÉTLŐDŐ SOROK

def duprows(tblorser, col_or_cols='',keep=False,toexcel=False):   # ismétlődő rekordok
    '''
    Minden olyan rekord, amelyre az adott mező értéke ismétlődő (a result táblázat vagy ser, az inputtól függően)
    col_or_cols:  mire ellenőrizze az unicitást
       '' / 'all' /  'index'  /  oszlopnév   /   több oszlopnév listaként
         - 'index':  a kulcsértékek duplikációi   (a pandas-ban nem garantált a kulcsértékek unicitása) 
         - ser esetén csak 'index'  vagy  ''  adható meg
         - '':  táblázat esetén az 'all'-nak felel meg (összes oszlop)
    keep:  'first' / 'last'  / False
         - 'first':  az ismétlődők közül az elsőt tartsa meg (csak a többi kerül be a return-be)
         - False:  mindegyik sor bekerül a return-be
    TÖRÖLVE, NEM VOLT STABIL   sortby:  rendezés kiegészítése még egy oszloppal (csak tbl esetén)
         - a col_or_cols oszlopokra mindenképpen rendez, ehhez fűzi hozzá ezt a további oszlopot
         - elsősorban a dropduprows függvénynél lesz jelentősége, keep='first' vagy 'last' esetén
    '''
    
    if istbl(tblorser):
        tbl=tblorser
        if col_or_cols=='index': 
            tblout=tbl[tbl.index.duplicated(keep=keep)].sort_index()
        elif col_or_cols in ['','all']: 
            tblout=tbl[tbl.duplicated(keep=keep)]
        else: 
            # sortbyL=col_or_cols.copy()
            # if sortby: 
            #     if type(sortbyL)==list: sortbyL.append(sortby)
            #     elif type(sortbyL)==str: sortbyL+=',' + sortby
            # tbl.sort_values(by=sortbyL)
            tblout=tbl[tbl[col_or_cols].duplicated(keep=keep)].sort_values(by=col_or_cols)      
    
        if toexcel and len(tblout)>0: Toexcel(tblout,'duprows')
        return tblout
    
    elif isser(tblorser):
        ser=tblorser
        if col_or_cols=='index': serout=ser[ser.index.duplicated(keep=keep)].sort_index()
        else: serout=ser[ser.duplicated(keep=keep)].sort_values()

        if toexcel and len(serout)>0: Toexcel(serout,'duprows')
        return serout

    else:
        print('duprows csak pd.DataFrame-re vagy pd.Series-re alkalmazható')

def dropduprows(tbl_or_ser, col_or_cols='',keep='first'):    # ismétlődő sorok elhagyása
    '''
    Kidobja a másodlagos ismétlődő sorokat (minden csoportból az elsőt vagy az utolsó tartja meg,
        illetve keep=False argumentummal kérhető az összes elhagyása)
    Megjegyzés:  a DafaFrame.drop_duplicated és a Series.drop_duplicated nagyjából ugyanezt tudja,
        de az indexre nem alkalmazható  (viszont van inplace argumentuma)
    col_or_cols: mire ellenőrizze az unicitást
                '' / 'all' /  'index'  /  oszlopnév   /   több oszlopnév listaként
         - 'index':  a kulcsértékek duplikációi   (a pandas-ban nem garantált a kulcsértékek unicitása) 
         - ser esetén csak 'index'  vagy  ''  adható meg
         - '':  táblázat esetén az 'all'-nak felel meg (összes oszlop)
    keep:  'first' / 'last'  / False
         - 'first':  az ismétlődők közül az elsőt tartsa meg (csak a többit hagyja el)
         - False:  mindegyiket hagyja el
    TÖRÖLVE, NEM VOLT STABIL   sortby:  rendezés kiegészítése még egy oszloppal (csak tbl esetén)
         - a col_or_cols oszlopokra mindenképpen rendez, ehhez fűzi hozzá ezt a további oszlopot
         - keep='first' vagy 'last' esetén van jelentősége
    
    return:  NEM INPLACE    (a tbl vagy ser  drop_duplicated függvényének viszont van inplace argumentuma)

    Lásd még:   ser.groupby(ser.index).mean()       Átlagolás az ismétlődő azonosítójú rekordok számmezőire   (min max is lehet)
    '''

    if istbl(tbl_or_ser):
        tbl=tbl_or_ser
        if col_or_cols=='index':
            return tbl[~tbl.index.duplicated(keep=keep)]
        elif col_or_cols=='all':
            return tbl[~tbl.duplicated(keep=keep)]
        else:
            # if sortby:
            #     if type(col_or_cols)==list:
            #         sortbyL=col_or_cols.copy()
            #         sortbyL.append(sortby)
            #     elif type(col_or_cols)==str:
            #         sortbyL+=col_or_cols + ',' + sortby
            # else:
            #     sortbyL=col_or_cols
            # tbl.sort_values(by=sortbyL)
            return tbl[~tbl[col_or_cols].duplicated(keep=keep)]

    elif isser(tbl_or_ser):
        ser=tbl_or_ser
        if col_or_cols=='index':
            return ser[~ser.index.duplicated(keep=keep)]
        else:
            return ser[~ser.duplicated(keep=keep)]

    else:
        print('dropduprows csak pd.DataFrame-re vagy pd.Series-re alkalmazható')

def Ser_setunique(ser,aggfunc='mean'):       # ismétlődő indexek eltüntetése aggregálással
    '''   
    A számaggregációk csak float értékekre működik (az index lehet szöveges vagy date is)
    aggfunc:   mean, first, last, min, max, ...

    '''
    return ser.groupby(ser.index).agg(aggfunc)

   


# INDEX

def tblsetindex(tbl,column,inplace=True,drop=False,checkunique=True):
    dtype=serdtype(tbl,column)
    tbl.set_index(column,verify_integrity=checkunique,drop=drop,inplace=inplace)
    if not drop: tbl.index.name='index'          # fontos, mert egyébként névütközés lenne a megmaradó oszloppal
    if dtype=='datetime': tbl.index=pd.to_datetime(tbl.index)     # elvileg automatikus is lehetne

def fillna_back(ser,nullvalue_max=0):   # kumulatív adatszolgáltatások esetén a közölt értéket szétosztja az előtte lévő üres cellákra
    '''
    Kumulatív adatszolgáltatások esetén a közölt értéket szétosztja az előtte lévő üres cellákra 
        (a végösszeg nem változik, a szétosztott érték is csökken)
    A ser végén maradhatnak üres értékek.
    Megjegyzés:  a ser.fillna(method='bfill') függvény nem szétosztja, hanem átmásolja a később álló értéket
    
    nullvalue_max:  nullának tekintendő értékek felső határa 
        Példa:  a johns hopkins covid adatsorokban 1 értékek is előfordultak, lényegében 0 jelentéssel
    '''
    values=ser.array
    iLastNull=None

    if nullvalue_max>0:
        value_add=0
        for i,value in enumerate(values):
            if not value or (value==np.NaN):
                if not iLastNull: iLastNull=i
            elif value<=nullvalue_max:
                if not iLastNull: iLastNull=i
                value_add = value
            else:
                if iLastNull:
                    valueElosztott=(value + value_add) / (i - iLastNull + 1)
                    for j in range(iLastNull,i+1): values[j]=valueElosztott
                    iLastNull=None
                    value_add=0
                    
    else:                            
        for i,value in enumerate(values):
            if not value or (value==0) or (value==np.NaN):
                if not iLastNull: iLastNull=i
            else:
                if iLastNull:
                    valueElosztott=value / (i - iLastNull + 1)
                    for j in range(iLastNull,i+1): values[j]=valueElosztott
                    iLastNull=None
            
    return pd.Series(values,ser.index)



# SKLEARN
def Tbl_scale(tbl,type='StandardScaler',scaler=None,weights=None):
    '''
    tbl: csak float vagy int oszlopai lehetnek  (a szöveges oszlopokat előtte kvantálni kell)
    type:
        StandardScaler            Ez a leggyakoribb. Nullára centrálás és skálázás a szórással ([-1,1]-be kerül a többség)
        MinMaxScaler              # [0-1] tartományba transzformálja
        MaxAbsScaler              # csak skálázás
        RobustScaler              ez a legjobb. A kiugró értékeket is jól kezeli. 
                Medián illetve quantile jellegű a hangolás. A [-1,1] tartományt a domináns szóráshoz igazítja.
                Ezzel minden feature-re teljesül, hogy a pontok túlnyomó többsége a [-1,1] tartományba kerül, 
                0 átlagértékkel   (az outlierek esetleges aszimmetriája viszont akár még növekedhet is, 
                de ez nem jelent gondot)
    weights:  dictionary adható meg az oszlopnevekkel (nem kell teljes körűnek lennie)
        - szorzófaktorok:  a távolságméréseket is érdemben befolyásolja
        - negatív is lehet  (ha az adott feature skálájának eltérő az iránya)
            Az előjelváltás a távolságmérést nem befolyásolja, de az esetleges átlagérték képzést igen
    '''
    
    if not scaler: 
        if type=='StandardScaler': scaler=StandardScaler().fit(tbl)
        elif type=='MinMaxScaler': scaler=MinMaxScaler().fit(tbl)
        elif type=='MaxAbsScaler': scaler=MaxAbsScaler().fit(tbl)
        elif type=='RobustScaler': scaler=RobustScaler().fit(tbl)

    tbl_out=pd.DataFrame(scaler.transform(tbl),columns=tbl.columns,index=tbl.index)

    if weights:
        for column in tbl_out.columns:
            weight=dget(weights,column)
            if weight: tbl_out[column]=tbl_out[column] * weight

    return tbl_out,scaler



# PLOT FÜGGVÉNYEK

def FvPlot(ser,plttype='scatter gauss',label='',seronly=False,annotate='localmax last',normalize=None,normfaktor=None,
            area=False,baseline=None,colors=None,axindex=None,**params):
    ''' Series nyomtatása pont-diagramként / interpolálva / mozgóátlaggal / spline illesztéssel  (több típus is kérhető)
    A vonal(ak) annotálása is kérhető (a felirat label lehet egyedi,  pozíció:  last, first, max, min, konkrét x-érték idézőjelben)
    return (config.serplotlastG):  a transzformált és kinyomtatott series
    
    plttype:    megjelenítési típus;  több is felsorolható (szóköz határolás, nem kell vessző, sorrend érdektelen)
        original:   eredeti függvénygörbe  (NaN pontok megszakítják a görbét)
        scatter:    pontok megjelenítése (gauss simítás előtt)
        kde:        pontok sűrűségének megjelenítése (szürkeárnyalatos, átlátszó). A scatter-hez hasonló funkció, de sokkal látványosabb.
                    Két hangolható paraméter:  kde_alpha ([0:1], default: 0.25)  kde_bandwidth (1 körüli érték;  0.5-részletesebb,  2-elnagyoltabb) 
                        - ha több görbét kell nyomtatni kde-vel ugyanabba a diagramba, akkor az alpha csökkenthető (pl. 0.2)
                            Másik megoldás: egy közös ser létrehozása append műveletekkel, majd a végén egy külön FvPlot(ser_kde,'kde')
                        - az x,y limiteket általában egyedileg kell beállítani a végén (a kde kiterjeszti a határokat)
        interpolated:   nan értékek helyébe lineáris illesztés (eredeti görbe, de nincsenek megszakítások)
        gauss:      gaussAvg megjelenítése  (nan értékek is lehetnek a ser-ben)
        regauss:    resample+gauss; resample az eredeti pontok számának négyszeresével, gauss a resample pontok tizedével (felülírható)
                        Kiküszöböli a gauss átlagolásnak a mérési pontok inhomogenitásából fakadó fésületlenségeit.
                        gausswidth és resample közvetlenül is beállítható
        regausst:   resample+gauss+trend: a mozgóátlag számításakor a széleken túlfutva lineáris trenddel számol (illesztés: szélső G pontra)
                        Általában jobb értékeket ad a széleken, mert kiküszöböli a gauss vízszintesbe hajló trendjét.
                        Rosszabb eredményt adhat, ha a forrásdatok szórása rendkívül nagy (pl. dirac delták) és/vagy fix periodicitása van. 
        gausst:     gauss + trend:   =regausst, de resample nélkül. Elvárás:  az x-tengely mintavételi pozíció egyenletesen vannak elosztva
        gauss+spline:    egymás utáni alkalmazás. Bizonyos esetekben kisebb hullámjelenségek előfordulhatnak
        gaussgrad:  derivált megjelenítése  (előtte és utána is gauss simítás)
        linear:     egyenes illesztés (legkisebb négyzetek módszere)
        resample:   linear resample  (az eredeti pontok összekötése egyenesekkel, majd a megadott frekvenciának megfelelő mintavételek)
        spline:     resample + spline  (a spline előtt mindenképpen resample kell, enélkül szélsősége kilengések is előfordulnának) 
    label:        vonal annotáció (ha van legend, akkor ott is ez jelenik meg; lásd még annotcaption)
    seronly:      ne rajzoljon, csak az interpolált kimeneti series-t adja vissza (több plttype esetén az utolsónak megfelelő series)
    annotate (vagy "annot"):  'last'  'first'  'middle' 'right' 'left' 'max'  'min' 'maxabs' 'upper' 'lower'
                        'localmax' 'localmin' 'gaussabs' 'gaussmax', 'baseline', konkrét x-érték (pl.'2021-01-01', '12';  az x értéknek nem kell szerepelnie a ser x-tömbjében (interpoláció))    
                    - szóközzel elválasztva több is megadható, pl. "last max"  (a 'last', 'first'  xmin megadásakor kikerülhet a képtérből)
                    - megadhatók egyedi x értékek is, de a feliratozás pattern-je közös (annotcaption, illetve {label} vagy {plttype})
                        Teljesen egyedi feliratozást az annotplus-ban lehet megadni (ha nem elég az annotcaption-ben megadható közös pattern)
                    - last, first:  jobb szélső / bal szélső pont
                    - max, min:  a legnagyobb illetve a legkisebb pont annotálása (maxvalue, minvalue)
                    - middle,left,right:   a szélesség 0.5, 0.1, 0.9 pontján   pl. akkor hasznos, ha a 'first'/'last' x-limitálás miatt kicsúszhat a képtérből (interval megadása is segíthet)
                    - upper, lower:  a legnagyobb/legkisebb 10 pont annotálása ("upper20" esetén max 20 pont).  
                        Elsősorban nem-folytonos pontosorozatokra (pl. scatter). 
                    - localmax:  legalább 5 pont szélességű lokális maximum, ha több van, akkor a legnagyobb y értékűek
                    - localmax2:  ha egy integer áll a "localmax" után, akkor max ennyi localmax  (legalább xwidth*0.05 távolság közöttük)
                    - localmax2_butfirst:  max 2 db, de a lenagyobb nélkül (2. és 3.)
                    - localmin:  csak a pltshow-ban megadott baseline alattiak (default: y=0). A "min"-nél nincs ilyen szabály. Itt is alkalmazható: localmin2, localmin2_butfirst
                    - baseline:  localmax és localmin, baseline-hoz képest (ha nincs magadva a "baseline" argumentum, akkor egyszerűen 'localmax localmin')
                    - maxabs:   0 feletti maxhelyek és 0 alatti minhelyek  (maxabs4:  a 0-tól legtávolabbi 4 szélsőérték)
                    - gaussabs12:  mindkét irányú kilengés a kétszeres időskálájú trendvonalhoz képest (kétszeres G-vel számított baseline). A végén megadható a maximális darabszám.
                        legalább xwidth*0.05 távolság közöttük, legalább (xmax-xmin)*0.05 mértékű a kilengés (a legfelső és a legalsó mindenképpen megjelenik)
                    - gaussmax, gaussmin:  csak a felső illetve csak az alsó kilengések
                    - ha több vonalat is rajzol a függvény (pl. 'spline gauss'), akkor az utolsó görbére érvényesül
    normalize:    maxOut adható meg (általában 1; az abszolút értékre vonatkozik;  0 vagy None esetén nincs normalizálás)
    normfaktor:   ha meg van adva, akkor a normalize érdektelen. A kiszámolt normfaktor beíródik a config.normfaktorlastG-be (egy következő FvPlot-ban felhasználható)
                    - 'last': az előző plot norfaktor értéke
    area:         True esetén szürke kitöltés a "baseline"-tól vagy az x-tengelytől; scatter-nél érdektelen
                    - baseline megadása esetén két ser közötti sáv rajzolására is használható (leggyakoribb: átlagérték-től való eltérés rajzolása)
                    - 'noline': csak árnyékolás, szürke határolóvonal rajzolása nélkül
                    - 'colorline':  színes vonal szürke árnyékolással 
    baseline:     Konstans érték  VAGY  Y-tömb (összehangolva a ser-index-szel)  VAGY  pd.Series
                    - area=True esetén ettől a vonaltól rajzolandó a sáv (default: 0 azaz az x-tengely)
                    - az annot-ban megadott "baseline" kulcsszó arra utal, hogy a baseline-hoz képest kell keresni a lokális max és min-helyeket
                        Ebben az esetben általában a pltshow() "annot_baseline" argumentumában is ugyanezt kell megadni
                    - series esetén interpolációval állapítja meg az eredeti ser x-értékeihez tartozó értékeket
    colors:       {color,linewidth,alpha,faderight,fadeleft}   példa: {'color':'orange', 'linewidth':2, 'alpha':0.6, 'faderight':[100,122]}
                    - color,alpha,linewidth:  közvetlen argumentumként is megadható (ha meg van adva, akkor elsőbbsége van a colors-ban megadott értékkel szemben)
                    - ha nincs megadva, akkor default color (figyelembe véve a speccolors argumentumot is, amiben a label-hez tartozó speciális színek lehetnek)
                    - False: default color, a speccolors figyelembe vétele nélkül
                    - 'last':  az előző plot vonalszíne (pl. scatter-nél hasznos)
                    - fadeleft,faderight: indexek adhatók meg (ser.iloc). Alpha fokozatos csökkentése a megadott indexektől. Resample esetén jelenleg nincs automatikus indexkorrekció
    params:       
        gausswidth ("G"is jó) Az átlagolás Gauss-görbéjének talpponti vagy effektív szélessége (4*szórás). Dirac deltából ilyen szélességű gauss lesz. 
                    A zajszűrési periódusidő a Gauss szélesség fele (2*szórás). Az ilyen vagy ennél kisebb periódusú kétirányú kilengések lényegében eltűnnek (kb századrész). 
                    A széleken lévő megbízhatatlansági tartomány szintén a Gauss szélesség fele. 
                    A ténylegesen figyelembe vett pontok száma a gausswidth kétszerese (8*szigma)
                    Megadható a pontok száma (>1 integer) vagy az összes mérési ponthoz viszonyított hányad (0-1 közötti tört).
                        Egyenetlen eloszlású mérési pontok esetén regauss-t érdemes alkalmazni (előzetes resample)
                    Default: a teljes x-szélesség tizede.
                    Vesszős felsorolás is megadható (string).  Példa: '140,120,100,80,60'  (általában a legnagyobb legelöl; egyre halványabb megjelenítés) 
        gaussside   Milyen színnel rajzolja a megbízhatatlansági tartományban a görbét
                    'colorandcolor': a feléig az eredeti szín halványabban, utána még halványabb (default)
                    'colorhalf': csak a megbízhatatlansági tartomány feléig megy, az eredeti szín halványabb változatával
                    'gray':  szürke a teljes megbízhatatlansági tartományban   
                    'grayhalf': csak a megbízhatatlansági tartomány feléig megy
                    'toend' (vagy ''):  kimegy a végekig (eredeti színnel)
        trendpoint  gauss, spline:   (x,y) trendpont a jobb szélen (x legyen nagyobb minden meglévő x értéknél)
        positive    True esetén a széleken végrehajtott lineáris illesztés figyelembe veszi, hogy a vonalnak a pozitív tartományban kell maradnia  
        resample    spline,  pl '100,500'   az első szám az előzetes lineáris resample sűrűsége, a második a kimeneti görbéé
                    linear   pl '200'    kimeneti resample (a kimeneti egyenes hány pontra legyen kiszámolva, magát az illesztést nem érinti). 
                        Ne az eredeti x értékek kerüljenek be a kimeneti ser-be, hanem egy egyenletes mintavétel. Extend esetén kötelező.
                    gauss    ha meg van adva, akkor előzetes resample (kiszűri a mérési pontok inhomogenitási miatti torzulásokat)
                    regauss   akkor is resample, ha nincs megadva a resample paraméter (default: az eredeti pontok számának négyszerese) 
        trend       regausst és gausst esetén itt adható meg, hogy melyik oldalra (default: mindkettőre)
                        és milyen illesztési algoritmussal (default: "linear"; az "erlang" még nem működik)
                    példa: "right"  "right erlang" 
        extend      spline, linear, regausst:   a széleken lévő trend meghosszabbítása a megadott szélességben.
                    - spline és linear esetén a pontok számát kell megadni  (csak resample esetén érvényesül)
                        Két szám is megadható (tuple). Ebben az esetben a bal és a jobb oldalra eltérő. Példa: (0,30)
                    - regausst esetén float-ként is megadható: a G-hez viszonyított arányszám (példa: 0.5: G*0.5 szélesség)
        extend_to, extend_from:   kitejesztés jobbra/balra a megadott x-értékekig (ha nem terjed odáig). Az utolsó ismert adatot ismétli meg 
                        (konzervatív karakterű). Ha az x-tengelyen dátum van, akkor string-formátumban is megadható  
                    A diagram széle kitolódik a kiterjesztésig. Az esetleges resample ezután következik, tehát sok pontot eredményezhet.
        interval    tuple (a két határolóérték is jelenjen meg). Ha nincs megadva, akkor a ser.index határai a mérvadóak. Gauss mozgóátlag esetén
                        javasolt a megjelenítendőnél szélesebb ser-t megadni, ha az adatok a széleken túl is rendelkezésre állnak (ezzel 
                        eltűnhetnek a szürke szakaszok).  Példa: ("2022.01.01",None)  (120,200)
        splinediff  spline paraméter   másként: accuracy

        annotcaption     fix string / str.format pattern / függvény.    Példa pattern-re:  "{label} {plttype} ({y:.2%})" 
            - ha nincs megadva, akkor   label or plttype
            - megadható egy dictionary is, annot-típusonként eltérő felirat-sablonokkal
            - helyettesítőjelek a feliratsablonban:   {label}  {plttype}  {x}  {y}  {y_orig}     (az y_orig a skálázás előtti y érték)
            - a számadatoknál formázás is megadható:   {x:int}  {y:float}  {y:%}  {y_orig:%%}    (a "%" 1-re normált float-ra alkalmazható a "%%" 100-ra normált float-ra)
            - shorthands:  "label" =  "{label}"    "y" = "{y_orig:float}"   "y%" = "{y_orig:%}"     "x" = "x={x}"
                Példa: annotcaption = {"max":"{label} maxhely: {y:int}",   "min":"minimumhely",   "localmax":"y"}
            - függvény esetén  rec argumentum és caption return.   rec.type, rec.label,rec.plttype, rex.x, rec.y, rec.y_orig
                None return esetén  label or plttype   (üres stringgel sem nyomható el teljesen az annotálás)
        annotlength     az annotban megjelenő label max hossza (a feliratok teljes hossza nagyobb lehet ennél, mert a labelen kívül más is lehet benne) 
        annotplus       egyedi pont-feliratozások a plot-görbén, dictionary-ben vagy list-of-lists ben megadva:  {x1:felirat1,x2:felirat2,...}
                        - az x értékek lehetnek dátumok is (stringként megadva). Ha nincs ilyen dátum a ser-ben, akkor interpoláció.
                        - a feliratok fix szövegek (nem sablon)
                        - ha ismétlődő x értékek is előfordulhatnak, akkor list-of-list formátumban kell megadni [[x1,caption],...] 
                        - ha teljesen önállóan kell annotálni egy pontot vagy egy egyenest, akkor FvAnnotateAdd() hívásssal oldható meg
        annotcolor,annotcolorplus        az annotáció betűszíne  (pl. 'red';  ha nincs megadva, akkor a vonal színe ill. plus esetén 'darkolivegreen')
        annotpos                         annotációk közös tájolása (alapesetben típusonként eltérő)  pl. "right bottom" 
        annotposplus                     a kiegészítő annotációk tájolása  (pl. 'right bottom'  default: 'right top')
        annotnormfaktor csak akkor van hatása, ha az annotcaption-ben y_orig szerepel, és az adatsor már a jelen függvény hívása előtt normalizálva lett

        scatter_alpha    ha nincs megadva, akkor a pontok számához igazítja (5 / sqrt(count))
        scatter_size     ha nincs megadva, akkor a pontok számához igazítja (<100:3, <1000:2, egyébként 1)

        stdarea          True esetén szórás-sáv rajzolása (default False)
    '''

    len_ser=len(ser)
    if len_ser==0: return

    ax=None
    if not seronly:
        if axindex: ax=plt.gcf().axes[axindex]
        else: ax=plt.gca()

    if label in ['adat','data']: label=None         # a semmitmondó nevek ne fedjék el a plttype feliratokat (plttype teszteléskor fontos)

    annotcolor=None
    # Kiemelt label-ekhez rendelt vonalszínek (pl. label="Hungary")
    if (colors is None or colors==True) and config.linecolorsG:           # a config-ban adhatóak meg a kiemelt label-el a hozzájuk tartozó vonalszínnel
        colors=config.linecolorsG.get(label) or {}
        if colors: colors=colors.copy()                   # ha globális változóban van tárolva, akkor a globális változó értéke ne változzon
    if not colors: colors={}
    # else: annotcolor=colors.pop('annotcolor',None)        # a colors-ból törli az annotcolor-t

    color,alpha,linewidth = dgets(params,'color,alpha,linewidth')
    if color is not None:
        if colors is None:  colors=ddef(color=color)
        else: dset(colors,color=color)
    if alpha is not None:
        if colors is None:  colors=ddef(alpha=alpha)
        else: dset(colors,alpha=alpha)
    if linewidth is not None:
        if colors is None:  colors=ddef(linewidth=linewidth)
        else: dset(colors,linewidth=linewidth)
    

    if normfaktor=='last': normfaktor=config.normfaktorlastG
    elif normalize==0: normalize=None

    Y_baseline=0
    if baseline is not None:
        if isinstance(baseline,(int,float)): Y_baseline = baseline
        elif type(baseline)==list or type(baseline)==np.ndarray: Y_baseline=baseline
        elif type(baseline)==pd.Series: Y_baseline = servalueA(baseline,list(ser.index))

    interval = dget(params,'interval')

        
    def sub_normalize(ser):
        nonlocal normfaktor
        # Az alsó plttype-ra hajt végre normalizálást, a többi rajzolási módra érvényesíti a normfaktort
        if normfaktor: 
            return ser*normfaktor
        elif normalize: 
            ser,normfaktor=normalizeSer(ser,normalize,faktorout=True)
            return ser
        else: return ser
    
    def sub_interval(ser):
        if not interval: return ser
        if interval[0]: ser=ser.loc[ser.index>=interval[0]]
        if interval[1]: ser=ser.loc[ser.index<=interval[1]]
        return ser
        
    def sub_plotline(serplot,plttype,alpha=None,color=None):
        # alpha:  gauss mozgóátlag esetén a széleken lévő bizonytalan szakasz (halványabb)
        # color:  "last" esetén az utolsó rajzolás vonalszíne
        if serisempty(serplot):
            # print('HIBA: FvPlot  Nincs nyomtatható adat   label=' + label + ' plttype:' + plttype)
            pass
        else:
            labelout=label or plttype
            colorsL=colors.copy()
            if color=='last': color=pltlastcolor()
            if alpha: colorsL['alpha']=alpha
            if color: colorsL['color']=color

            if colorsL.get('color')=='last': colorsL['color']=pltlastcolor()

            # Kitöltés szürkeárnyalattal
            if area:
                try:
                    alphaL=0.08
                    # if alpha: alphaL=alpha*alphaL 
                    alphaLL=colorsL.get('alpha')
                    if alphaLL: alphaL=alphaLL*alphaL 
                    plt.fill_between(serplot.index,serplot.values,Y_baseline,alpha=alphaL,color='0.3')
                    #  Y_baseline default értéke 0, korábban lett beállítva az esetleges area_baseline argumentum alapján
                except Exception as e:
                    print('ERROR  FvPlot  Area plot  label="' + label + '"  ' + str(e))
                # Vonal-rajzolás paraméterei
                if colors=={}:      # ha nincs megadva explicit szín vagy alpha a FvPlot hívásban
                    line_color='0.5'
                    # ha gauss mozógátlag széléről van szó
                    if alpha: colorsL={'color':'0.5','alpha':0}   # középszürke, teljesen átlátszó 
                    else: 
                        colorsL={'alpha':0.2}     # kicsit sötétebb a háttérnél  
                        if area!='colorline': colorsL['color']='0.5'    # szürke árnyalat
                else: colorsL=colors

            # Vonal rajzolása halványítással
            if colorsL.get('faderight') or colorsL.get('fadeleft'):   # nincs még használva, de működik; az átfedések kezelése még nem tökéletes
                # A kettő közötti szakasz kitalálása
                index_middle_start=0
                index_middle_end=len(serplot)-1
                aIndexLeft=colorsL.get('fadeleft')
                if aIndexLeft: 
                    del colorsL['fadeleft']
                    aIndexLeft.sort(reverse=True)
                    index_middle_start=aIndexLeft[0]
                aIndexRight=colorsL.get('faderight')
                if aIndexRight: 
                    del colorsL['faderight']
                    aIndexRight.sort()
                    index_middle_end=aIndexRight[0]
                # A közbenső szakasz rajzolása
                ax.plot(serplot[index_middle_start:index_middle_end+1],label=labelout,**colorsL)
                # serplot[index_middle_start:index_middle_end+1].plot(label=labelout,ax=ax,**colorsL)
                if not colorsL.get('color'): colorsL['color']=pltlastcolor()    # a fade-szakaszok színe legyen ugyanez
                alpha=colorsL.get('alpha',1)
                colorsL['alpha']=alpha
                # Bal oldali szakaszok rajzolása
                if aIndexLeft: 
                    del aIndexLeft[0]
                    aIndexLeft.append(0)
                    colorsLL=colorsL.copy()
                    indexright=index_middle_start
                    # alpha_step=alpha/(len(aIndexLeft)+1)
                    alpha_faktor=20**(1/(len(aIndexLeft)+1))        # 20: ez határozza meg a lépésenkénti csökkentés erősségét (nagyobb érték, erősebb csökkenés)
                    for index in aIndexLeft:
                        colorsLL['alpha']=colorsLL['alpha']/alpha_faktor
                        # colorsLL['alpha']=colorsLL['alpha'] + alpha_step
                        ax.plot(serplot[index:indexright+1],**colorsL)
                        # serplot[index:indexright+1].plot(ax=ax,**colorsLL)
                        indexright=index
                # Jobb oldali szakaszok rajzolása
                if aIndexRight: 
                    del aIndexRight[0]
                    aIndexRight.append(len(serplot)-1)
                    colorsLL=colorsL.copy()
                    indexleft=index_middle_end
                    # alpha_step=alpha/(len(aIndexRight)+1)
                    alpha_faktor=20**(1/(len(aIndexRight)+1))    # 20: ez határozza meg a lépésenkénti csökkentés erősségét (nagyobb érték, erősebb csökkenés)
                    for index in aIndexRight:
                        # colorsLL['alpha']=colorsLL['alpha'] - alpha_step
                        colorsLL['alpha']=colorsLL['alpha']/alpha_faktor
                        if index+1>len(serplot): index=len(serplot)-1
                        if index<indexleft: break
                        # serplot[indexleft:index+1].plot(ax=ax,**colorsLL)
                        ax.plot(serplot[indexleft:index+1],**colorsLL)
                        indexleft=index

            # Vonal rajzolás
            elif area!='noline':
                ax.plot(serplot,label=labelout,**colorsL)
            # - az annotálás nem itt történik (csak az utolsó vonal kaphat annotálást)

    def sub_gauss(gausstype):
        ''' gausstype: "gauss", "regauss", "regausst", "gausst"  
        '''

        gausswidth=params.get('gausswidth')
        if not gausswidth: gausswidth=params.get('G')
        if not gausswidth: gausswidth=0.1

        # A gausswidth argumentumban felsorolás is megadható (egyre halványabb színnel rajzolva)
        if type(gausswidth)==int or type(gausswidth)==float: 
            if gausswidth>1: gausswidth=gausswidth / len(ser)   # átszámolom hányadra (resample esetén is a hányad a mérvadó)
            aGausswidth=[gausswidth]
        elif type(gausswidth)==str:       # veszővel elválasztva több is megadható
            aGausswidth=[]
            for number in gausswidth.split(','): 
                if isint(number): 
                    nInteger=int(number)
                    if nInteger<=1:
                        print('FvPlot, Érvénytelen gausswidth: "' + number + '"  in "' + gausswidth + '"')
                        aGausswidth.append(0.1)
                    else:
                        aGausswidth.append(nInteger/len(ser))
                else: aGausswidth.append(float(number))


        # Előzetes resample
        resample=params.get('resample')   # nem kötelező megadni  (ha nincs megadva, akkor "gauss" esetén nincs resample)
        if gausstype in ['regauss','regausst']  and not resample: resample=len(ser)*4    # alapesetben négyszeres sűrűség
        if resample: 
            serResample=FvLinearResample(ser,count=resample)
        else: 
            serResample=ser.copy()    # kell a copy, mert a "gaussr" hozzátehet pontokat a serResample-hez (plttype="interpolated gausst" esetén a széleken túlnyúló egyenesek jelentek meg)
            resample=len(ser)
        
        trendpoint=None
        if gausstype=='gauss': trendpoint=params.get('trendpoint')
        
        
        for i,gausswidth_sub in enumerate(aGausswidth):
            # Ha a gausswidth-ben a pontok száma lett megadva, akkor a resample nélküli pontokra utal
            if gausswidth_sub<1: gausswidth_sub=int(resample*gausswidth_sub)     # kötelezően ide kerül, mert át lett már számolva hányadra
            elif gausstype!='gauss': gausswidth_sub=int(gausswidth_sub*resample/len(ser))   # nem kerülhet ide


            # Gausstrend
            if gausstype in ['regausst','gausst']:
                trend=params.get('trend')
                leftright='leftright'
                if vanbenne(trend,'right',True): leftright='right'
                elif vanbenne(trend,'left',True): leftright='left'
                fit_method='linear'
                if vanbenne(trend,'erlang',True): fit_method='erlang'

                x_left_old=serResample.index.min()
                x_right_old=serResample.index.max()

                serGauss=FvGaussTAvg(serResample,G=gausswidth_sub,leftright=leftright,fit_method=fit_method,
                            extend=params.get('extend'),positive=params.get('positive'))

                # Ha ki lett terjesztve, akkor a halványításnak is igazodnia kell ehhez
                # if extend:
                #     x_left=serGauss.index.min()
                #     if x_left<x_left_old:

            
            else:
                serGauss=FvGaussAvg(serResample,gausswidth=gausswidth_sub,trendpoint=trendpoint)


            if serGauss.index.dtype in ['int64','int32']:       # FONTOS:  integer indexre serGauss.index[countGray] nem tömbindexre utalnak
                serGauss.index = serGauss.index.astype(float)

            serGaussOut=serGauss

            serGaussOut=sub_interval(serGaussOut)
            # if interval: serGaussOut=serGaussOut[interval[0]:interval[1]]  # integer index esetén problémás lehet


            if normfaktor or normalize: serGaussOut=sub_normalize(serGaussOut)
            # Rajzolás
            if not seronly: 
                # A széleken lévő alacsony megbízhatóságú szakasz külön rajzolandó, halványabb színnel
                # A szélessége G*0.5

                # KORÁBBI: a szélesség függjön a zajosságtól (nem kell, mindenképpen szükséges a G/2)
                # varianceG=serGauss.var(ddof=0,skipna=True)
                # varianceSer=serResample.var(ddof=0,skipna=True)   # itt elvileg nem fordulhat elő na
                # if varianceSer<=varianceG: faktor=0.25
                # elif varianceSer>=varianceG*4: faktor=0.5
                # else: faktor=math.sqrt(varianceSer/varianceG)*0.25

                faktor=0.5          # 0.5*G szélességben kell halványan színezni 
                
                if gausswidth_sub<1: countGray=int(gausswidth_sub*resample*faktor)
                else: countGray=int(gausswidth_sub*faktor)

                countGrayHalf=math.ceil(countGray/2)
                # countGrayHalf=int(countGray/2)
                label_plottype=gausstype + ' ' + str(gausswidth_sub)


                # sub_plotline(serLeft,label_plottype)
                # sub_plotline(serLeftLinear,label_plottype)


                # Normál színű vonal 
                gaussside=params.get('gaussside')
                if gaussside is None: gaussside='colorandcolor' 
                elif gaussside=='': gaussside='toend' 

                if gaussside in ['toend','']:
                    sub_plotline(serGaussOut,label_plottype)
                else:
                    sub_plotline(serGaussOut[serGauss.index[countGray]:serGauss.index[-countGray]],label_plottype)

                    if gaussside=='gray':
                        # Szürke vonal a bal szélen
                        sub_plotline(serGaussOut[serGauss.index[0]:serGauss.index[countGray+1]],label_plottype,alpha=0.2,color='0.5')
                        # szürke vonal a jobb szélen
                        sub_plotline(serGaussOut[serGauss.index[-countGray-1]:serGauss.index[-1]],label_plottype,alpha=0.2,color='0.5')

                    elif gaussside=='color':
                        # Halványabb színű vonal a bal szélen
                        sub_plotline(serGaussOut[serGauss.index[0]:serGauss.index[countGray+1]],label_plottype,color='last',alpha=0.3)
                        # Halványabb színű vonal a jobb szélen
                        sub_plotline(serGaussOut[serGauss.index[-countGray-1]:serGauss.index[-1]],label_plottype,color='last',alpha=0.3)

                    elif gaussside=='grayhalf':
                        # Szürke vonal a bal szélen
                        sub_plotline(serGaussOut[serGauss.index[countGrayHalf]:serGauss.index[countGray+1]],label_plottype,alpha=0.2,color='0.5')
                        # szürke vonal a jobb szélen
                        sub_plotline(serGaussOut[serGauss.index[-countGray-1]:serGauss.index[-countGrayHalf]],label_plottype,alpha=0.2,color='0.5')

                    elif gaussside in['colorhalf','colorandgray','colorandcolor']:
                        # Halványabb színű vonal a bal szélen (megbízhatatlansági tartomány feléig)
                        sub_plotline(serGaussOut[serGauss.index[countGrayHalf]:serGauss.index[countGray]],label_plottype,color='last',alpha=0.3)
                        # Halványabb színű vonal a jobb szélen (megbízhatatlansági tartomány feléig)
                        sub_plotline(serGaussOut[serGauss.index[-countGray]:serGauss.index[-countGrayHalf]],label_plottype,color='last',alpha=0.3)
                        if gaussside=='colorandgray':
                            # Szürke vonal a bal szélen
                            sub_plotline(serGaussOut[serGauss.index[0]:serGauss.index[countGrayHalf+1]],label_plottype,color='0.5',alpha=0.1)
                            # szürke vonal a jobb szélen
                            sub_plotline(serGaussOut[serGauss.index[-countGrayHalf-1]:serGauss.index[-1]],label_plottype,color='0.5',alpha=0.1)
                        elif gaussside=='colorandcolor':
                            # Szürke vonal a bal szélen
                            sub_plotline(serGaussOut[serGauss.index[0]:serGauss.index[countGrayHalf]],label_plottype,color='last',alpha=0.1)
                            # szürke vonal a jobb szélen
                            sub_plotline(serGaussOut[serGauss.index[-countGrayHalf]:serGauss.index[-1]],label_plottype,color='last',alpha=0.1)

                        # # Halványabb színű vonal a bal szélen (megbízhatatlansági tartomány feléig)
                        # sub_plotline(serGaussOut[serGauss.index[countGrayHalf]:serGauss.index[countGray+1]],label_plottype,color='last',alpha=0.3)
                        # # Halványabb színű vonal a jobb szélen (megbízhatatlansági tartomány feléig)
                        # sub_plotline(serGaussOut[serGauss.index[-countGray-1]:serGauss.index[-countGrayHalf]],label_plottype,color='last',alpha=0.3)
                        # if gaussside=='colorandgray':
                        #     # Szürke vonal a bal szélen
                        #     sub_plotline(serGaussOut[serGauss.index[0]:serGauss.index[countGrayHalf+1]],label_plottype,color='0.5',alpha=0.1)
                        #     # szürke vonal a jobb szélen
                        #     sub_plotline(serGaussOut[serGauss.index[-countGrayHalf-1]:serGauss.index[-1]],label_plottype,color='0.5',alpha=0.1)
                        # elif gaussside=='colorandcolor':
                        #     # Szürke vonal a bal szélen
                        #     sub_plotline(serGaussOut[serGauss.index[0]:serGauss.index[countGrayHalf+1]],label_plottype,color='last',alpha=0.1)
                        #     # szürke vonal a jobb szélen
                        #     sub_plotline(serGaussOut[serGauss.index[-countGrayHalf-1]:serGauss.index[-1]],label_plottype,color='last',alpha=0.1)

                
                
                if i==0: 
                    colors['color']=pltlastcolor()
                    colors['alpha']=1
                else: colors['alpha']=colors['alpha']*0.7       # egyre halványabb
        return serGaussOut

    extend_to=params.get('extend_to')
    if extend_to:
        ser=ser.copy()          # enélkül figyelmeztetések jelennek meg a futtatáskor
        if type(extend_to)==str: extend_to=pd.to_datetime(extend_to) 
        max_x=datefloat(max(ser.index))
        if max_x<datefloat(extend_to): ser[extend_to]=ser[max(ser.index)]
    extend_from=params.get('extend_from')
    if extend_from:
        if type(extend_from)==str: extend_from=pd.to_datetime(extend_from) 
        min_x=datefloat(min(ser.index))
        if min_x>datefloat(extend_from): ser[extend_from]=ser[min(ser.index)]   # a bal szélső értéket ismétli

    serOut=None
    words=plttype.split()
    plttypefirst=words[0]
    plttypelast=words[-1]

    if 'original' in words:         # az original és a scatter legyen elől, mert több plttype esetén a normfaktort ez határozza meg
        if normfaktor or normalize: serOriginal=sub_normalize(ser)
        else: serOriginal=ser
        serOriginal = sub_interval(serOriginal)
        if not seronly: sub_plotline(serOriginal,'original')
        if plttypelast=='original': 
            serOut=serOriginal
    if 'scatter' in words:
        if normfaktor or normalize: serScatter=sub_normalize(ser)
        else: serScatter=ser
        serScatter = sub_interval(serScatter)
        alpha = params.get('scatter_alpha')
        size = params.get('scatter_size')
        color = None
        if colors:
            color=colors.get('color')
            alpha=notnone(alpha,colors.get('alpha'))
            size=notnone(size,colors.get('size'))
        if not alpha: alpha = Limit( 10 / (len(ser)**0.5), 0.1,0.9)     # 100-ig alpha=0.9,  10 000-nél 0.1   milliónál 0.01
        if not size:  size = np.select([len_ser>1000,len_ser>100],[2,4], default=9)   # terület, points**2-ben 
        if not seronly: plt.scatter(serScatter.index,serScatter.values,label=(label or 'scatter'),
                                    c=color,s=size,alpha=alpha)
        if plttypelast=='scatter': 
            serOut=serScatter
    if 'kde' in words:
        if normfaktor or normalize: serKde=sub_normalize(ser)
        else: serKde=ser
        serKde = sub_interval(serKde)
        alpha=params.get('kde_alpha',0.25)
        bandwidth=params.get('kde_bandwidth',1)
        if not seronly: 
            cmap=sb.light_palette("gray", as_cmap=True)         # jó lenne háttérváltozóban kezelni
            sb.kdeplot(x=ser.index,y=ser.values,fill=True,cmap=cmap,alpha=alpha,bw_adjust=bandwidth)
        if plttypelast=='kde': 
            serOut=serKde

    if 'gauss' in words: 
        try:
            serGauss=sub_gauss('gauss')
            if plttypelast=='gauss': 
                serOut=serGauss 
        except Exception as e:
            print('ERROR  FvPlot  gauss  label="' + label + '"  ' + str(e))
            serOut=None
    if 'regauss' in words: 
        try:                # csupa nan esetén elszállhat a resample
            serGauss=sub_gauss('regauss')
            if plttypelast=='regauss': 
                serOut=serGauss 
        except Exception as e:
            print('ERROR  FvPlot  regauss  label="' + label + '"  ' + str(e))
            serOut=None
    if 'regausst' in words: 
        try:
            serGauss=sub_gauss('regausst')
            if plttypelast=='regausst': 
                serOut=serGauss 
        except Exception as e:
            print('ERROR  FvPlot  regausst  label="' + label + '"  ' + str(e))
            serOut=None
    if 'gausst' in words: 
        try:
            serGauss=sub_gauss('gausst')
            if plttypelast=='gausst':
                serOut=serGauss 
        except Exception as e:
            print('ERROR  FvPlot  gausst  label="' + label + '"  ' + str(e))
            serOut=None
    
    if 'gauss+spline' in words:
        gausswidth=params.get('gausswidth')
        trendpoint=params.get('trendpoint')
        serGauss=FvGaussAvg(ser,gausswidth=gausswidth,trendpoint=trendpoint)
        resample=params.get('resample')
        extend=params.get('extend')
        splinediff=params.get('splinediff')
        serSpline=FvSmoothSpline(serGauss,resample=resample,extend=extend,diffByMaxmin=splinediff)    # csak a gauss-ra érvényesül a trendpoint
        if normfaktor or normalize: serSpline=sub_normalize(serSpline)
        serSpline = sub_interval(serSpline)
        if not seronly: sub_plotline(serSpline,'gauss+spline')
        if plttypelast=='gauss+spline': 
            serOut=serSpline 
    if 'gaussgrad' in words:
        gausswidth=params.get('gausswidth') or int(len(ser)/10)
        serGauss=FvGaussAvg(ser,gausswidth=gausswidth)
        serGaussGrad=FvGradient(serGauss)
        serGaussGrad=FvGaussAvg(serGaussGrad,gausswidth=gausswidth*2)       
        if normfaktor or normalize: serGaussGrad=sub_normalize(serGaussGrad)
        serGaussGrad = sub_interval(serGaussGrad)
        if not seronly: sub_plotline(serGaussGrad,'gaussgrad')
        if plttypelast=='gaussgrad': 
            serOut=serGaussGrad 
    if 'spline' in words:
        resample=params.get('resample')
        extend=params.get('extend')
        splinediff=params.get('splinediff')
        trendpoint=params.get('trendpoint')
        serSpline=FvSmoothSpline(ser,resample=resample,extend=extend,diffByMaxmin=splinediff,trendpoint=trendpoint)
        if normfaktor or normalize: serSpline=sub_normalize(serSpline)
        serSpline = sub_interval(serSpline)
        if not seronly: sub_plotline(serSpline,'spline')
        if plttypelast=='spline': 
            serOut=serSpline 
    if 'linear' in words:
        resample=params.get('resample')
        extend=params.get('extend')
        serLinear=FvLinear(ser,resample=resample,extend=extend)
        if normfaktor or normalize: serLinear=sub_normalize(serLinear)
        serLinear = sub_interval(serLinear)
        if not seronly: sub_plotline(serLinear,'linear')
        if plttypelast=='linear': 
            serOut=serLinear 
    if 'resample' in words:
        serResample=FvLinearResample(ser)       # 4-szeres sűrűségű mintavétel
        if normfaktor or normalize: serResample=sub_normalize(serResample)
        serResample = sub_interval(serResample)
        if not seronly: sub_plotline(serResample,'resample')
        if plttypelast=='resample': 
            serOut=serResample 
    if 'interpolated' in words:
        serInterpolated=(ser.interpolate(limit_area='inside'))
        if normfaktor or normalize: serInterpolated=sub_normalize(serInterpolated)
        serInterpolated = sub_interval(serInterpolated)
        if not seronly: sub_plotline(serInterpolated,'interpolated')
        if plttypelast=='interpolated': 
            serOut=serInterpolated 

    if normalize!='last': config.normfaktorlastG=normfaktor       # Felhasználható egy következő FvPlot hívásban

    if seronly:
        if not serOut is None:
            config.serplotlastG=serOut.copy()       # a hívóhelyen felhasználható
        return serOut.copy()

    if not serisempty(serOut):
            #print('serOut:' + str(serOut))
            annotnormfaktor=params.get('annotnormfaktor')
            if not annotnormfaktor:
                if not normfaktor: annotnormfaktor=1
                else: annotnormfaktor=normfaktor
            
            stdarea=params.get('stdarea')           # szórássáv rajzolása
            if stdarea:
                aYOut=servalueA(serOut,ser.index.array)
                if normalize: aY=normalizeSer(ser,normalize).array
                else: aY=ser.array

                sum=0
                for i in range(len(aY)):
                    sum += (aYOut[i] - aY[i])**2
                std=math.sqrt(sum/len(aY))
                #print('std: ' + str(std))
                #print(list(zip(aY,aYOut)))

                plt.fill_between(serOut.index.array,(serOut+std).array,(serOut-std).array,color='silver',alpha=0.3)

            # Vonal-szintű annotációk
            annot=params.get('annot')   # alternatív argumentumnév
            if annot is not None: annotate=annot
            if annotate and plttypelast!='kde':     # önmagában álló kde-t ne annotáljon
                caption=label or plttypelast
                annotpos=params.get('annotpos')         # pl. 'right bottom'

                annotcaption=params.get('annotcaption')           # fix érték,   {}-jeles pattern,  függvény
                if annotcaption is None: annotcaption=params.get('annotatecaption')    # annotatecaption névváltozat is jó

                annotlength=params.get('annotlength')    
                if annotlength and type(annotlength)==int: label_in_annot=txtshorten(label,annotlength)
                else: label_in_annot=label

                                        
                def f_annotate_add(x,y,annottype,annottype_out=None,position=None):
                    caption=None
                    if annottype_out==None: annottype_out=annottype
                    
                    if callable(annotcaption): 
                        rec=pd.Series([annottype,label_in_annot,plttypelast,x,y,y/annotnormfaktor],['type','label','plttype','x','y','y_orig'])
                        caption = annotcaption(rec)
                    
                    else:
                        sablon=None
                        if type(annotcaption)==dict:         # annotációs típusonként is meg lehetett adni a sablont
                            sablon=annotcaption.get(annottype)
                        elif type(annotcaption)==str:
                            sablon=annotcaption
                        
                        # Ha nincs megadva sablon, akkor az annottype-tól függő default
                        if not sablon:
                            if annottype in ['maxabs','gaussmax','gaussabs']: sablon='y'
                            else: sablon='label'

                        # caption előállítása a sablon alapján
                        # shorthands
                        if sablon in ['label','y','y%']: sablon='{' + sablon + '}'
                        elif sablon=='x': sablon='x={x}'

                        def f_percent_replace(sablon):
                            # Százalékos formázás kérhető az x, y, y_orig jeleknél
                            # default érték a decimális digitek számára: 1
                            sablon=sablon.replace('%}','%1}')
                            # {y%2}  mintázat
                            sablon=re.sub(pattern=r'\{(\w+)%[.]?(\d*)}', repl=r'{\1:,.\2%}', string=sablon)     # keresés és csere
                            # {y%%2} mintázat
                            sablon=re.sub(pattern=r'\{(\w+)%%[.]?(\d*)}', repl=r'{\1:,.\2f}%', string=sablon)     # keresés és csere
                            return sablon
                            

                        # Kerekítés max 4 decimális digit-re
                        x_in=x                      
                        if isinstance(x,float): 
                            x_in=Round(x,4)
                            sablon=sablon.replace('{x}','{x:,.9g}')         # a tizedes ponttól balra 9 digitig meghet (utána e00 jelölés)
                            # sablon=f_percent_replace(sablon)
                        elif isdatetime(x):
                            x_in=datestr(x)        # yyyy.MM.dd

                        y_in=y
                        y_orig=y
                        if isinstance(y,float): 
                            y_in=Round(y,4)
                            y_orig=Round(y/annotnormfaktor,4)
                            sablon=sablon.replace('{y}','{y:,.9g}')
                            sablon=sablon.replace('{y_orig}','{y_orig:,.9g}')

                        sablon=f_percent_replace(sablon)

                        caption=sablon.format(label=label_in_annot,plttype=plttypelast,
                                    x=x_in,y=y_in,y_orig=y_orig)
                        # - a számadatok 4 digitre kerekítve (fent a "9g" arra vonatkozik, hogy mikortól váltson e+00 formátumra)

                    if not caption: 
                        caption=label_in_annot or plttypelast
                    
                    position=notnone(annotpos,position)

                    FvAnnotateAdd(x=x,y=y,caption=caption,annottype=annottype_out,position=position,color=annotcolor)
                

                words=annotate.split()
                # baseline: localmin és localmax    Ha van area_baseline, akkor a baseline-tól való eltérést nézi
                words_=[]
                for word in words:
                    if beginwith(word,'baseline'): 
                        darab=cutleft(word,'baseline')
                        if nincsbenne(annotate,'localmax'): words_.append('localmax' + darab)
                        if nincsbenne(annotate,'localmin'): words_.append('localmin' + darab)
                    else: words_.append(word)
                words=words_
                    
                annotcolor=params.get('annotcolor')        

                xfirst,yfirst=serfirstvalue(serOut)
                xlast,ylast=serlastvalue(serOut)

                halfwidth=2                 # localmax, localmin
                if len(serOut)<10: halfwidth=1


                # last, first, max, min, egyedi
                labels_x=[]         # 0-1 közötti label-pozíciók (ütközés-ellenőrzéshez kell a következő körben)
                for word in words:
                    if word=='last': 
                        if xlast is not None:
                            f_annotate_add(xlast,ylast,word)
                            labels_x.append(1)
                    elif word=='first': 
                        if xfirst is not None:
                            f_annotate_add(xfirst,yfirst,word)
                            labels_x.append(0)
                    elif word=='middle': 
                        if xfirst is not None and xlast is not None:
                            # x=floatdate((datefloat(xlast)+datefloat(xfirst))/2)
                            x=(xlast+xfirst)/2
                            y=servalue(serOut,x,True)
                            f_annotate_add(x,y,word)
                            labels_x.append((x-xfirst)/(xlast-xfirst))
                    elif word=='left': 
                        if xfirst is not None and xlast is not None:
                            x=xfirst + (xlast-xfirst)*0.1                            
                            y=servalue(serOut,x,True)
                            f_annotate_add(x,y,word)
                            labels_x.append((x-xfirst)/(xlast-xfirst))
                    elif word=='right': 
                        if xfirst is not None and xlast is not None:
                            x=xfirst + (xlast-xfirst)*0.9                            
                            y=servalue(serOut,x,True)
                            f_annotate_add(x,y,word)
                            labels_x.append((x-xfirst)/(xlast-xfirst))
                    elif word=='max': 
                        x=serOut.idxmax()
                        y=serOut.loc[x]
                        if type(y)==pd.Series: y=y.max()
                        f_annotate_add(x,y,word,position='left top')  # a max-min értékek alapesetben balra pozícionáltak (a localmax-localmin viszont jobbra)
                        if xfirst is not None and xlast is not None and xlast!=xfirst:
                            labels_x.append((x-xfirst)/(xlast-xfirst))
                    elif word=='min': 
                        x=serOut.idxmin()
                        y=serOut.loc[x]
                        if type(y)==pd.Series: y=y.min()
                        f_annotate_add(x,y,word,position='left bottom')  # a max-min értékek alapesetben balra pozícionáltak (a localmax-localmin viszont jobbra)
                        if xfirst is not None and xlast is not None:
                            labels_x.append((x-xfirst)/(xlast-xfirst))
                    elif word=='maxabs': 
                        if abs(serOut.max()) >= abs(serOut.min()):
                            x=serOut.idxmax()
                            position='left top'
                            type_='max'
                            y=serOut.loc[x]
                            if type(y)==pd.Series: y=y.max()
                        else: 
                            x=serOut.idxmin()
                            position='left bottom'
                            type_='min'
                            y=serOut.loc[x]
                            if type(y)==pd.Series: y=y.min()
                        f_annotate_add(x,y,'maxabs',type_,position=position)  # a max-min értékek alapesetben balra pozícionáltak (a localmax-localmin viszont jobbra)
                        if xfirst is not None and xlast is not None:
                            labels_x.append((x-xfirst)/(xlast-xfirst))
                    else:       # konkrét x-érték (csak egy-szavas lehet)
                        try:
                            x=datefloat(word)
                            y=servalue(serOut,x,True)
                            f_annotate_add(x,y,'xpos') 
                            if xfirst is not None and xlast is not None:  
                                labels_x.append((x-xfirst)/(xlast-xfirst))
                        except Exception as e:
                            if not beginwith(str(e),"Unknown string format"):       # a localmax ...  eleve idekerül
                                print('ERROR  FvPlot  annotate_add  label="' + label + '"  ' + str(e))
                
                        
                # localmax, localmin, gaussabs
                for word in words:
                    if beginwith(word,'upper'):       # felső n pont
                        maxdb=10
                        try: maxdb=int(cutleft(word,'upper'))
                        except: pass
                        serL=serOut.sort_values(ascending=False).iloc[:maxdb]
                        for x,y in serL.items():
                            f_annotate_add(x,y,'upper',position='left top') 
                    elif beginwith(word,'lower'):       # alsó n pont
                        maxdb=10
                        try: maxdb=int(cutleft(word,'upper'))
                        except: pass
                        serL=serOut.sort_values().iloc[:maxdb]
                        for x,y in serL.items():
                            f_annotate_add(x,y,'lower',position='right bottom') 
                    elif beginwith(word,'localmax'):
                        bButfirst =  endwith(word,'_butfirst')
                        if bButfirst: word=cutright(word,'_butfirst')
                        maxdb=0
                        try: maxdb=int(cutleft(word,'localmax'))  # baseline is ilyen hosszú
                        except: pass
                        word=word[:8]
                        ser_from_base=serOut - Y_baseline      # default: 0
                        mindistance = 10 / axesunits()[0]     # 2023.10.11 legalább 10 point legyen az annot pontok között (kb 2 karakter) Korábban 0.05 volt beállítva.
                        points=serlocalmax(ser_from_base,'max',halfwidth,mindistance=mindistance,endpoints=True)
                        if bButfirst and len(points)>0: points=points[1:]
                        if len(points)>0:
                            if maxdb>0: points=points[:maxdb]
                            for x,y in points:
                                # nem lehet túl közel a fix annotációkhoz
                                bOk=True
                                for label_x in labels_x:
                                    if abs((x-xfirst)/(xlast-xfirst) - label_x) < 0.02:
                                        bOk=False
                                        break
                                if bOk: 
                                    f_annotate_add(x,serOut[x],word[:8],position='left top') 
                                    # f_annotate_add(x,y,word[:8],position='left top') 
                    elif beginwith(word,'localmin'):
                        bButfirst =  (endwith(word,'_butfirst')!='')
                        if bButfirst: word=cutright(word,'_butfirst')
                        maxdb=0
                        try: maxdb=int(cutleft(word,'localmin'))
                        except: pass
                        word=word[:8]
                        ser_from_base=serOut - Y_baseline
                        mindistance = 10 / axesunits()[0]     # 2023.10.11 legalább 10 point legyen az annot pontok között (kb 2 karakter) Korábban 0.05 volt beállítva.
                        points=serlocalmax(ser_from_base,'min',halfwidth,mindistance=mindistance,endpoints=True)   # width*0.05 távolság egymástól
                        if bButfirst and len(points)>0: points=points[1:]
                        if len(points)>0:
                            if maxdb>0: points=points[:maxdb]
                            for x,y in points:
                                # nem lehet túl közel a fix annotációkhoz
                                bOk=True
                                for label_x in labels_x:
                                    if abs((x-xfirst)/(xlast-xfirst) - label_x) < 0.02:
                                        bOk=False
                                        break
                                if bOk: 
                                    f_annotate_add(x,serOut[x],word,position='left bottom') 
                                    # f_annotate_add(x,y,word,position='left bottom') 
                    elif beginwith(word,'gaussmax|gaussmin|gaussabs'):
                        maxdb=None
                        try: maxdb=int(cutleft(word,'gaussabs'))        # gaussmax, gaussmin is ugyanilyen hosszú
                        except: pass
                        word=word[:len('gaussabs')]
                        
                        discrete_x=ser.index.dtype in ['int32','int64']

                        subcount=Attr(plt.gcf(),'subcount',1)
                        if not maxdb:
                            if subcount==1: maxdb=100
                            else:
                                if discrete_x:
                                    maxdb=np.select([subcount<=4,subcount<=8,subcount<=12],[16,12,8],default=4)
                                else:
                                    maxdb=np.select([subcount<=4,subcount<=8,subcount<=12],[4,3,2],default=1)
                        

                        # x-irányú sűrűségre és a kilengés minimális mértékére vonatkozó feltétel
                        if subcount==1:         # több hely van, ezért több annot jelenhet meg
                            neighbor_percent = 0.005     # x irányban a teljes szélesség 0.5%-a
                            diff_percent = 0.01          # y irányban 1% a simított görbétől
                        elif discrete_x:
                            neighbor_percent = 5 / axesunits()[0]     # legalább 5 point legyen az annot pontok között (kb egy karakter szélessége)
                            diff_percent = 10 / axesunits()[1]   # legalább 10 point eltérés a gauss-simított görbe y-értékétől
                        else:       # float index esetén kevesebb szélsőérték-pont kell
                            neighbor_percent = 10 / axesunits()[0]     # legalább 10 point legyen az annot pontok között
                            diff_percent = 20 / axesunits()[1]   # legalább 20 point eltérés a gauss-simított görbe y-értékétől

                        
                        

                        pointsmax=[]
                        if word in ['gaussabs','gaussmax']:
                            pointsmax=serlocalmax(serOut,'max',halfwidth,mindistance=neighbor_percent,endpoints=False)   
                        pointsmin=[]
                        if word in ['gaussabs','gaussmin']:
                            pointsmin=serlocalmax(serOut,'min',halfwidth,mindistance=neighbor_percent,endpoints=False)   # width*0.03 távolság egymástól
                        points=pointsmax + pointsmin

                        if len(points)>0:
                            # kétszeres G-vel számolt görbéhez képest számolja az eltérést (ha nem volt gauss simítás, akkor G=0.1)
                            if plttypelast in ['gauss','regauss','gausst','regausst','gauss+spline','gaussgrad']:
                                gausswidth=params.get('G') or params.get('gausswidth')
                                if gausswidth>1: gausswidth=gausswidth/len_ser
                                gausswidth = max(min(gausswidth*2,(gausswidth+1)/2),gausswidth**0.7)
                            else: gausswidth=0.1
                            serMean=FvGaussAvg(serOut,gausswidth)   # mindenképpen a serOut kell, mert regauss esetén resample volt
                            
                            abs_distance=[]
                            for x,y in points:
                                abs_distance.append(abs(serMean[x].mean()-y))
                            sortarrays(abs_distance,points,reverse=True)

                            arr_x,arr_y = unzip(points)
                            y_max=array(arr_y).max()
                            y_min=array(arr_y).min()
                            y_diff=y_max-y_min
                            count_middle=0
                            for i,point in enumerate(points):
                                # Mindenképpen megjelenik a legfelső és a legalsó, továbbá a megadott keretszámot figyelembe véve azok a 
                                #   kilengések, amelyek legalább a max-min 5%-ával térnek el a baseline-tól
                                x,y=point
                                if (    (word in ['gaussabs','gaussmax'] and y==y_max) or
                                        (word in ['gaussabs','gaussmin'] and y==y_min) or
                                        (abs_distance[i]>y_diff*diff_percent and count_middle<maxdb-2)  ):
                                    if y!=y_max and y!=y_min: count_middle+=1
                                    if y<serMean[x].mean(): position='left bottom'
                                    else: position='left top'
                                    f_annotate_add(x,y,'gaussabs',position=position) 


            # Egyedi annotációk
            annotplus=params.get('annotplus')           # dictionary  {'2020.03.10':'Járvány kezdete','2022.02.22':'Járvány vége'}
            if annotplus is not None:
                annotcolorplus=params.get('annotcolorplus')        
                if annotcolorplus is None:
                    if annotcolor is None: annotcolorplus='darkolivegreen'
                    else: annotcolorplus=annotcolor
                annotposplus=params.get('annotposplus')
                if annotposplus is None: annotposplus='right top'
                if type(annotplus)==dict:
                    for x,caption in annotplus.items():
                        color=None
                        if beginwith(caption,'!'):
                            caption=cutleft(caption,'!')
                            color='red'
                        y=servalue(serOut,x,True)
                        FvAnnotateAdd(x=x,y=y,caption=caption,position=annotposplus,annottype='egyedi',color=color)
                elif type(annotplus)==list:
                    for rec in annotplus:
                        x,caption = rec
                        color=None
                        if beginwith(caption,'!'):
                            caption=cutleft(caption,'!')
                            color='red'
                        y=servalue(serOut,x,True)
                        FvAnnotateAdd(x=x,y=y,caption=caption,position=annotposplus,annottype='egyedi',color=color)
                    
        
    if not serOut is None:
        config.serplotlastG=serOut.copy()       # a hívóhelyen felhasználható

    return 

def plotnow(Y,X=None,labels=None,plttype='scatter regausst',**plotparams):   # tömbök azonnali nyomtatása (pl. függvényábrák)
    '''
    Y: értékek tömbje;  
        - több Y-tömb is megadható, egyező elemszámmal (két dimenziós input)
            példa:  Y=[aY1,aY2]         Az aY2 lehet pl egy illesztett egyenes (minden X-re kiszámítandó)
        - series is megadható  (ilyenkor az X érdektelen)
        - list of series is megadható  (X érdektelen)
    X: x értékek tömbje:   az elemszáma egyezzen az Y tömbével;  ha nincs megadva, akkor automatikus sorszámozás
    labels: a diagram-vonalak megnevezése (annotation-ként jelenik meg). Ha üres, akkor sorszámozott label
        - megadható string-felsorolásként vagy listaként is
    plotparams:  FvPlot, plotinit és plotshow összes lehetséges argumentuma (kivéve area, amit a pltstyle felülírhat)
            Legyakoribb:  suptitle,title,ylabel,plttype,label,annotate,normalize,color,  plttype='gauss' (ha nem kellenek a pontok is)

    '''
    if labels and type(labels)==str: labels=labels.split(',')
    else: labels=[]

    # Ha több series van megadva, akkor külön plot-ok (nem kerül be közös tbl-be)
    if type(Y)==list and type(Y[0])==pd.Series:
        if plttype=='scatter regausst': plttype='original'
        dsetsoft(plotparams,plttype=plttype)
        pltinit(**plotparams)
        for i,ser in enumerate(Y): 
            if i<len(labels): label=labels[i]
            else: label=str(i)
            plotparams['label']=label
            FvPlot(ser,**plotparams)
        pltshow(**plotparams)
        return

            
    if vanbenne(str(type(Y)),'Series'):
        X=Y.index
        Y=Y.values


    dCols={}
    if np.asarray(Y).ndim==2:
        for i,colvalues in enumerate(Y): 
            if i<len(labels): colname=labels[i]
            else: colname=str(i)
            dCols[colname]=colvalues
    else:
        if len(labels)>0: colname=labels[0]
        else: colname='0'
        dCols[colname]=Y
    
    
    dset(plotparams,plttype=plttype)
    if len(labels)==1:
        dsetsoft(plotparams,suptitle=labels[0])
    else:
        dsetsoft(plotparams,suptitle='Diagram')

    tbl=pd.DataFrame(dCols,X)
    tblinfo(tbl,'plot',plotstyle='line',**plotparams)



    

def pltinit(suptitle=None,tight=None,width=None,height=None,left=None,right=None,top=None,bottom=None,
            subcount=None,rowscols_aspect=(2,3),
            nrows=None,ncols=None,indexway='sor',height_ratios=None,width_ratios=None,
            hspace=None,wspace=None,sharex=True,sharey=True,
            title=None,xlabel=None,ylabel=None,**other):        
    '''
    Diagram rajzolási folyamat inicializálása
    Ha nincs megadva valamelyik adat, akkor marad az alapértelmezett
      
     suptitle:  16-os betűmérettel megjelenő felirat felül  (extra méretezés esetén maradjon üres, a plt.suptitle függvénnyel önálló megadás)
           - ha csak egy subplot van, akkor a title, xlabel és ylabel  is megadható (a pltshow-ban is megadható)
           - a comment-ek a pltshow vagy a pltshowsub hívásokban adhatók meg
     width, height:  ablakszélesség a képernyőmérethez képest,  (0-1) közötti szám
           - ha nincs megadva, akkor a default érték subplot-ok esetén igazodik a nrows,nheight adathoz is
     left,right,top,bottom:  rajzolási terület határai az ablakon belül,  (0-1) közötti szám, minden esetben a bal-alsó saroktól számítva)  
           - a felhasználói felület slidereivel kereshető meg az optimális érték
           - ha nincs megadva, akkor a default érték subplot-ok esetén igazodik a nrows,nheight adathoz is
     
     nrows,ncols:  subplot sorok és oszlopok száma.  Ha csak egy subplot van, akkor nem kell megadni.
            - ha nincs megadva, de van subcount, akkor a függvény számítja a subcount és a rowcols_aspect alapján
     subcount:  subplot-ok száma. Nem kötelező megadni
        - ha nrows,ncols nincs megadva, akkor az értéküket a függvény állítja be a subcount alapján
        - nrows és ncols megadása esetén is érdemes megadni, mert a felesleges subplot cellákat ez alapján üríti a függvény
     rowscols_aspect:   sorok és oszlopok aránya. Csak akkor érdekes, ha subcount>1 és nrows,ncols nincs megadva
        - speciális eset: ha ncols=0, akkor függőleges növelés nrows sorig, túlcsorduláskor új oszlop (tetszőleges számú oszlop lehet)    
        - speciális eset: ha nrows=0, akkor vízszintes növelés ncols oszlopig, túlcsorduláskor új sor (tetszőleges számú sor lehet)    
     indexway:  'row' vagy 'sor' vagy 'horz' - vízszintesen halad (default).    'col' vagy 'vert':  függőlegesen halad 
     height_ratios:  subplotok esetén a sorok magasságarányai  pl [1,3,4] - az első sor 1 egység, a második 3 stb.
     width_ratios:  subplotok esetén az oszlopok szélességarányai   pl.  [5,1]
     hspace,wspace:  height-space és width-space a subplot-ok között (0-1, az átlagos subplot mérethez viszonyítva)
     sharex,sharey:  subplot tengelyek összhangolása (a függőleges tick-label-ek csak a bal szélen jelennek meg) 
     tight:  subplot-ok esetén érdemes megadni (nrows,ncols meg van adva).
            True  esetén tightlayout default beállításokkal. A left,right,top,bottom ilyenkor érdektelen.
            Float számérték is megadható:  a subplot-ok közötti vertikális távolságot jellemzi
            1: egy sor maradjon a jelölőknek    2: két sor maradjon a jelölőknek    (default: 1.08)
     
     title: a diagram felett megjelenő 10-es betűméretű felirat.  
        - ha több subplot van, akkor nem veszi figyelembe a függvény  (a pltinitsub hívásokban kell megadni)
     xlabel,ylabel:   tengely-felirat
        - ha több subplot van, akkor nem veszi figyelembe a függvény  (a pltinitsub vagy a pltshowsub hívásokban kell megadni)

     **other: nincs használva   (ismeretlen argumentumokra nincs hibaüzenet, a tblinfo() miatt kell)

    '''

    # if suptitle: print('pltinit, suptitle:' + suptitle)

    subplots= (nrows or ncols or subcount)
    if subplots:
        if not nrows and not ncols:
            nrows0,ncols0 = rowscols_aspect 
            nrows0=notnone(nrows0,2)
            ncols0=notnone(ncols0,3)

            ratio0=None
            if nrows0>0 and ncols0>0: ratio0=ncols0/nrows0

            # nrows, ncols optimális kiválasztása
            nrows,ncols = f_optimize_grid(subcount,nrows0,ncols0)

            # Ablakméretezés, ha akár a width, akár a height nincs megadva
            if not width or not height:            
                # Aránytartó méretezés
                if ratio0:
                    ratio=ncols/nrows
                    if ratio/ratio0 > 1.1:    # szélesség növelés, magasság csökkentés
                        if not width: width=0.95
                        if not height: height = Limit( width / (ratio/ratio0), 0.6,0.95)
                    elif ratio/ratio0 < 0.9:   # magasság növelés, szélesség csökkentés
                        if not height:
                            if nrows>1: height=0.95   
                            else: height=0.9        # egysoros esetén ne növelje 0.9 fölé
                        if not width: width = Limit( height * (ratio/ratio0), 0.6,0.95)
                    # ha sok subplot van, akkor arányos növelés a képernyő minél jobb kitöltése érdekében
                    if subcount>20:
                        if not height: height=0.92
                        if not width: width=0.95
                        if height<0.92 and width<0.95:
                            if 0.92-height<0.95-width:
                                width=0.92*width/height
                                height=0.92
                            else:
                                height=0.95*height/width
                                width=0.95

                    # if ratio/ratio0 > 1.1:    # magasság növelés, szélesség csökkentés
                    #     if not height:
                    #         if nrows>1: height=0.95   
                    #         else: height=0.9        # egysoros esetén ne növelje 0.9 fölé
                    #     if not width: width = Limit( height / (ratio/ratio0), 0.6,0.95)
                    # elif ratio/ratio0 < 0.9:   # szélesség növelés, magasság cökkentés, 
                    #     if not width: width=0.95
                    #     if not height: height = Limit( width * (ratio/ratio0), 0.6,0.95)
                elif nrows0>0 and ncols0==0:     # Függőleges haladás, túlcsorduláskor új oszlop (tetszőleges számú oszlop lehet)
                    if not height:
                        height=0.5 + 0.45 * nrows/nrows0
                    if not width:
                        width=dget([0.6,0.9,0.95],ncols)
                elif nrows0==0 and ncols0>0:     # Vízszintes haladás, túlcsorduláskor új sor (tetszőleges számú sor lehet)
                    if not width:
                        width=0.5 + 0.45 * ncols/ncols0
                    if not height:
                        height=dget([0.6,0.9,0.95],nrows)

        if not nrows: nrows=1
        if not ncols: ncols=1
        if not subcount: subcount=nrows*ncols

        # sorok számától függő beállítások
        if not top:
            top=np.select([nrows>7,nrows>5,nrows>3,nrows>2,nrows>1],
                          [0.89,   0.88,   0.87,   0.86,   0.84],      0.81)
            # tops=[0.81,0.83,0.85,0.87,0.88]
            # if nrows<len(tops): top=tops[nrows]
            # else: top=tops[-1]
        # if top is None: top=0.85            # 0.88 helyett

        if not bottom:
            bottom=np.select([nrows>10,nrows>7,nrows>4,nrows>3,nrows>2,nrows>1],
                             [0.05,    0.06,   0.07,   0.08,   0.09,   0.1],      0.11)
            # bottoms=[0.1,0.09,0.08]
            # if nrows<len(bottoms): bottom=bottoms[nrows]
            # else: bottom=bottoms[-1]
        # if bottom is None: bottom=0.08          # 0.11 helyett

        if not hspace:
            hspace=np.select([nrows>9,nrows>7,nrows>5,nrows==5,nrows==4,nrows==3],
                            [0.92,   0.85,    0.8,    0.7,     0.55,    0.4],      0.35)
            # hspaces=[0.35,0.35,0.45,0.5,0.5]
            # if nrows<len(hspaces): hspace=hspaces[nrows]
            # else: hspace=hspaces[-1]

        if not wspace:
            wspace=np.select([ncols>9,ncols>7,ncols>5,ncols>3,ncols>2],
                            [0.31,   0.29,   0.27,   0.25,   0.23],      0.22)

        if not right: 
            right=np.select([ncols>7,ncols>5,ncols>3,ncols>2,ncols>1],
                            [0.97,   0.96,   0.95,   0.94,   0.92],      0.9)

        if not left:
            left=np.select([ncols>7,ncols>5,ncols>3,ncols>2,ncols>1],
                            [0.04,  0.06,   0.08,   0.09,   0.1],      0.12)


        
        gridspec_kw={}
        if width_ratios: gridspec_kw['width_ratios']=width_ratios
        if height_ratios: gridspec_kw['height_ratios']=height_ratios
        if len(gridspec_kw)==0: grdspec_kw=None

        fig, axes = plt.subplots(nrows=(nrows or 1), ncols=(ncols or 1),
                                 gridspec_kw=gridspec_kw,
                                 sharex=sharex,sharey=sharey)
        # plt.sca(axes)
    else:
        nrows,ncols,subcount=1,1,1

    # elteszem későbbi felhasználásra (egyedi mező az osztályban).   A plt.gcf() példányban érhető el
    fig=plt.gcf()
    fig.subcount=subcount
    fig.nrows=nrows
    fig.ncols=ncols
    fig.axindex=-1      # pltinitsub használja
    fig.indexway=indexway


    if width or height: 
        dpi=plt.rcParams['figure.dpi']
        window = plt.get_current_fig_manager().window
        if width:
            screen_x = window.winfo_screenwidth() / dpi
            width=screen_x*width
        else: width=plt.rcParams['figure.figsize'][0]
        if height:
            screen_y = window.winfo_screenheight() / dpi      # inch-ben
            height=screen_y*height
        else: height=plt.rcParams['figure.figsize'][1]
        plt.gcf().set_size_inches(w=width, h=height)

    if suptitle: 
        if tight: 
            x=0.5           # plt.gcf().get_size_inches()[0]/2
            #print('x:' + str(x))
            horizontalalignment='center'
        else:
            x=left or plt.rcParams['figure.subplot.left']
            horizontalalignment='left'
        plt.suptitle(suptitle,x=x, horizontalalignment=horizontalalignment, fontsize=16)


    plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom,hspace=hspace,wspace=wspace)

    if subcount==1:
        if title!=None: plt.title(title, fontsize=10, pad=8,    # point-ban
                                x=0, horizontalalignment='left')
        if xlabel!=None: plt.xlabel(xlabel,alpha=0.6,fontsize=8,labelpad=4.)        
        if ylabel!=None: plt.ylabel(ylabel,alpha=0.6,fontsize=8,labelpad=4.)       


    config.aAnnotG=[]
    config.aScatterG=[]


    if tight:
        h_pad=None
        if type(tight) in [float,int]: h_pad=tight
        plt.tight_layout(h_pad=h_pad)
    

    # A felesleges subplotok eltüntetése
    if subplots:
        for axindex in range(subcount,ncols*nrows):    # maxrows*maxcols
            if plt.gcf().indexway in ['col','oszlop','vert']:
                nrows=plt.gcf().nrows   # pltinit-ben lett eltéve
                ncols=plt.gcf().ncols   # pltinit-ben lett eltéve
                nRow=axindex//nrows         
                nCol=axindex % nrows
                axindex = nCol*ncols + nRow         # nCol és nRow felcserélve
            plt.sca(plt.gcf().axes[axindex])
            plt.gca().axis('off')
        plt.sca(plt.gcf().axes[0])          # legyen a nulladik az aktuális subplot


    # Title megjelenítése több subplot esetén  (ha csak egy subplot van, akkor közvetlenül a diagram-terület felett jelenik meg)
    if title and subcount>1:
            x=left or plt.rcParams['figure.subplot.left']
            y = plt.rcParams['figure.subplot.top'] + 0.04
            plt.text(x=x,y=y,s=title,ha='left',va='top',fontsize=10,alpha=0.7,transform=plt.gcf().transFigure)
        


def f_optimize_grid(count,nrows,ncols):       # optimalizált grid
    '''
    A bal felső sarokból haladva kell kitölteni count rácspontot
    Háromféle paraméterezés:
        nrows>0, ncols=0:       Függőleges haladás, túlcsorduláskor új oszlop (tetszőleges számú oszlop lehet)
        nrows=0, ncols>0:       Vízszintes haladás, túlcsorduláskor új sor (tetszőleges számú sor lehet)
        nrows>0, ncols>0:       Aránytartás  (közelítőleg)   A sorok és oszlopok aránya legyen minél közelebb ehhez és illeszkedjen a count-hoz

    count:  összesen hány rácspont szükséges 
    '''
    

    
    # Függőleges haladás, túlcsorduláskor új oszlop (tetszőleges számú oszlop lehet)
    if ncols==0: 
        ncols=int((count-1)/nrows) + 1
        nrows=min(count,nrows)

    # Vízszintes haladás, túlcsorduláskor új sor (tetszőleges számú sor lehet)
    elif nrows==0:
        nrows=int((count-1)/ncols) + 1
        ncols=min(count,ncols)
    
    # Aránytartás  (közelítőleg) 
    else:
        ratio0 = ncols/nrows
        recs=[]
        for col in range(1,10):           # 9 feletti felosztás kizárható
            for row in range(1,10):
                if row*col<count or row*col>=2*count: continue      # count és 2*count közötti párosítások merülhetnek fel
                loss_count = abs((row*col)/count - 1)
                # ratio = max(col/row, row/col)
                loss_ratio = abs((col/row) / ratio0 - 1)
                loss = loss_count + 1*loss_ratio
                recs.append((col,row,loss,loss_count,loss_ratio)) 
        tbl=pd.DataFrame.from_records(recs)
        tbl=tbl.sort_values(by=2)           # generált oszlopnevek: 0-bázisú sorszám
        ncols=tbl[0].iloc[0]
        nrows=tbl[1].iloc[0]
    
    return nrows,ncols



def pltinitsub(axindex=None,title=None,xlabel=None,ylabel=None,annotinit=True,**other):
    '''
    Subplot rajzolási folyamat inicializálása (ha csak egy subplot van, akkor a pltinit is elég)
     Előfeltétel:  nrows és ncols megadása a pltinit-ben, 
      
    axindex:  ha nincs subplot, akkor None. Subplot estén a pltinit-ben megadott nrows*ncols subplot közül melyikre vonatkozik (0-bázisú)
        'next':  következő sublot  (pltinit nullázta, a jelen függvény beállítja az aktuálisat a végén)
    title:  10-es betűmérettel megjelenő felirat a subplot felett
    xlabel,ylabel:  koordináta tengelyek feliratai
    **other: nincs használva   (ismeretlen argumentumokra nincs hibaüzenet, a tblinfo() miatt kell)
    '''


    nrows=Attr(plt.gcf(),'nrows',1)   # pltinit-ben lett eltéve
    ncols=Attr(plt.gcf(),'ncols',1)   # pltinit-ben lett eltéve
    subcount=nrows*ncols

    # A subplot aktuálissá tétele  (a plt.gca() erre a subplot-ra mutasson)
    if axindex!=None: 
        if type(axindex)==str and axindex=='next': axindex=plt.gcf().axindex + 1
        plt.gcf().axindex=axindex
        # ha nem a default horizontális haladás kell, akkor axindex transzponálandó
        if plt.gcf().indexway in ['col','oszlop','vert']:
            # nrows=plt.gcf().nrows   # pltinit-ben lett eltéve
            # ncols=plt.gcf().ncols   # pltinit-ben lett eltéve
            nRow=axindex//nrows         
            nCol=axindex % nrows
            axindex = nCol*ncols + nRow         # nCol és nRow felcserélve
        plt.sca(plt.gcf().axes[axindex])



    if title!=None: plt.title(title, fontsize=np.select([subcount>30,subcount>10],[8,9],10),
                              pad=np.select([subcount>30,subcount>16,subcount>8,subcount>4,subcount>2],[2,3,4,5,6],8),    # point-ban
                              x=0, horizontalalignment='left')
    if xlabel!=None: plt.xlabel(xlabel,alpha=0.6,fontsize=np.select([subcount>10],[7],8),
                                labelpad=np.select([subcount>30,subcount>10],[2.,3.],4.))        # point-ban
    if ylabel!=None: plt.ylabel(ylabel,alpha=0.6,fontsize=np.select([subcount>10],[7],8),        
                                labelpad=np.select([subcount>30,subcount>10],[2.,3.],4.))        # point-ban
    

    if annotinit:
        config.aAnnotG=[]
        config.aScatterG=[]

def pltshow(annot=True,legend=False,subplot=False,to_file=None,**params):
    '''
     A rajzolás befejezése (záró formázások) és a diagram megjelenítése (subplot=False esetén)
     Kötelező hívni a rajzolási folyamat legvégén (de a formázások nem kötelezőek)
     A subplot-ok végén a pltshowsub függvény hívása opcionális (akkor kell, ha subplot szintű formázásokra van szükség)
    
     to_file:  ha meg van adva, akkor fájlba írás. A fájlnév fix részét kell megadni. A fájl a "Letöltések" vagy a pngdir-ben megadott
         mappába kerül és a kiterjesztése "png". Ha van már ilyen fájl, akkor fájlnév végére sorszám kerül.

     params:
      x1,x2,y1,y2:   koordináta határok   (x esetén dátumstring is megadható;   féloldalasan is megadható;  xmin,xmax,ymin,ymax is jó
      xlabel,ylabel:  tengelyfeliratok;  pltinit-ben is megadható
      area_under      a megadott y-érték alatti rész legyen sötétebb árnyalatú és jelenjen meg egy halvány vízszintes vonal is
    
      commenttopright: 
            - pltshow:     a diagram felett, a jobb oldalon megjelenő felirat (több soros is lehet, három sor kényelmesen elfér, a negyedik szűkösen)
            - pltshowsub:  a koordináta-területen belül jelenik meg
      commenttopright_in:   a koordináta-terület jobb felső sarka 
      commenttopright_out:
            - pltshowsub esetén alkalmazható:  a koordináta-terület felett illetve alatt jelenik meg (az alsó átfedhet a tick-feliratokkal)
      commentbottomright_out:
            - pltshow esetén alkalmazható, a teljes képtér jobb alsó sarkában jelenik meg (a bottom adat beállításával elkerülhető az átfedés)
      commenttopleft,commentbottomleft,commentbottomright,  commentright,commentleft,commenttop,commentbottom: 
            - a diagramon belül megjelenő feliratok (pltshow és pltshowsub esetén is belül)
            - szürke, 7-es méret, többsoros is lehet;   a legend-del átfedhet (a legend pozícióját a matplot automatikusan állítja be)
      commentfontsize_add:  alapesetben a comment-ek 7-es méretűek

      xtickerstep:   milyen lépésközzel kövessék egymást az x tengely tick-jei  
            - dátum tengelyre megadható "date": optimalizált feliratok (month és year is lehet a szélességtől függően)
                    "month": centrált hónapnevek      "year": centrált évszámok
      xticklabels,yticklabels:  False esetén nem jelennek meg feliratok (a rácsvonalak és a tick-ek megmaradnak)
            - a helyfenntartás is megszűnik, az esetleges xlabel, ylabel közelebb kerül a tengelyhez
      xticklabel_map,yticklabel_map:   ticklabel számok meppelése feliratokra. Lista esetén indexként értelmezi az eredeti
            számokat, dict vagy function esetén számToCaption
      ticksize:   fontméret (mindkét tengely jelölőire).  Default: 8    
      tickpad:    milyen messze legyenek a jelölőfeliratok a tengelytől (point-ban).  Default: 2

      xnumformat,ynumformat:  koordinátatengelyek jelölőinek számformátuma
          gyorsformátumok:   '%', '0%'  '%%'  'int'  ','   '3f'      lásd FvNumFormat
            ','   ='{x:,}'        integer, ezres határolással (szóköz határolás sajnos nincs, csak vessző vagy underline) 
            '3f'  ='{x:,.3f}'    fix decimal digits (0-k lehetnek a végén, e+00 kitevő semmiképpen, ezres határoló)
            '5g'  ='{x:,.5g}'    az összes digit száma van megadva (lehet kevesebb is, záró nullák nincsenek). Az e+00 akkor jelenik meg, ha a szám >= e+05 
            '4e'  ='{x:,.4e}'    mindenképpen e+00 formátum. A decimális digitek száma van megadva (0-k lehetnek a végén)
            '%'  ='{x:,.1%}'    1.0 bázisú százalék (szorzás százzal).  A decimális digitek száma van megadva (0-k lehetnek a végén)
            '%%' ='{x:,.1f}%'   nincs szorzás (közvetlen százalékos adatok). A decimális digitek száma van megadva (0-k lehetnek a végén)
      annot_fontsize:    default 9
      annot_xoffset,annot_yoffset:    csak pozitív érték adható meg  (a FvAnnotateAdd, határozza meg az irányt)
      annot_count:   globálisan vagy típusonként megadható, hogy hány annotáció jelenjen meg (az alapvonaltól legtávolabbi jelölők jelennek meg)
         - szám, dictionary vagy kódlista-string.    Default:  'first:20//middle:20//last:20//localmax:10//localmin:10//other:20'   
         - default alapvonal: x tengely  (lásd annot_baseline)
      annot_ymax,annot_ymin:  a címkék ne kerüljenek megadott y fölé illetve alá (egyik sem kötelező). Alsó szél: 0. Felső szél:1. (alá és fölé is mehet)
      annot_ystack:  azonos y értékhez tartozó címkék lépcsőzetes eltolása
      annot_baseline:  a localmin és a localmax annotációk közül csak a baseline alatti illetve feletti annotációk maradnak meg (default:0)
            - megadható egyetlen szám (konstans függvény) vagy egy ser

      annotcallback:  egy callback függvény, amivel konvertálhatóak a label feliratok (pl. x,y értéket megjelenítő annotáció átskálázása, 
            kategória-adat esetén kódToFelirat konverzió, ...).  

      v_bands (vagy y_bands):  Függőleges sávok rajzolása    
            list of dict(x1,x2,color='green',alpha=0.07,caption='',align='bottom',fontsize=7)
            Példa:   y_bands = [{'x1':120,'x2':130}]       (egy sáv esetén elmaradható a szögletes zárójel)
            - x1,x2: mettől meddig
            - x1 lehet 'start', 'begin',   x2 lehet 'end'.   Dátumtengely esetén stringként is megadható pl "2022-11-09"
            - align:  a felirat pozícionálása
                    'top' 'bottom'      függőleges tájolás, alulra vagy felülre igazítva
                    'topcenter':        széles sávok esetén;  felül középen, vízszintes tájolással
            - annotálás is megoldható (caption helyett) a FvAnnotateAdd önnálló hívásával
      h_bands (vagy x_bands):  Vízszintes sávok rajzolása        
            list of dict(y1,y2,color='green',alpha=0.07,caption='',align='left',fontsize=7)
            align:  a felirat pozícionálása     'left' / 'right'   (a tájolás mindenképpen vízszintes, y irányban a felső részén)

      v_lines (vagy y_lines): Függőleges vonalak rajzolása 
            list of dict(x,width=1,color='green',alpha=0.2,caption='',align='left',fontsize=7)
            - annotálás is megoldható (caption helyett) a FvAnnotateAdd önnálló hívásával
      h_lines (vagy x_lines): Vízszintes vonalak rajzolása 
            list of dict(y,width=1,color='green',alpha=0.2,caption='',align='bottom',fontsize=7)
            
      xy_circles:  Annotációs körök rajzolása a diagramvonalakon kívül        
            list of dict(x,y,size,color,caption)       
            - ha csak annotáció kell adott (x,y) helyre, akkor a FvAnnotateAdd függvényt érdemes hívni
      xy_texts:  a diagramvonalaktól független magyarázó feliratok,           
            list of dict(x,y,caption,ha='left',va='top',fontsize=fontsize0-1,alpha=0.5,color=default,transform='gcf'/'gca')
    '''

    nrows=Attr(plt.gcf(),'nrows',1)   # pltinit-ben lett eltéve
    ncols=Attr(plt.gcf(),'ncols',1)   # pltinit-ben lett eltéve
    subcount=nrows*ncols


    paramsL=params.get('params')    # másodlagos hívás esetén előfordulhat, hogy be van ágyazva
    if paramsL: params=paramsL

    x1,x2,y1,y2,xmin,xmax,ymin,ymax,xtickerstep,ytickerstep,xticklabels,xticklabel_map,yticklabels,yticklabel_map,ticksize,tickpad, \
                xnumformat,ynumformat,area_under, \
                annot_fontsize,annot_xoffset,annot_yoffset,annot_count,annot_baseline,annot_ymax,annot_ymin,annot_ystack, \
                y_bands,x_bands,x_lines,y_lines,v_bands,h_bands,v_lines,h_lines, \
                xy_circles,xy_texts,annotcallback,bottom,xlabel,ylabel,pngdir, \
                commenttopright,commenttopright_in,commenttopright_out,commentbottomright_out,commenttopleft,commenttopleft_in,commentbottomleft,commentbottomright, \
                commentright,commentleft,commenttop,commentbottom,commentfontsize_add, \
                  = dgets(params,
       'x1,x2,y1,y2,xmin,xmax,ymin,ymax,xtickerstep,ytickerstep,xticklabels,xticklabel_map,yticklabels,yticklabel_map,ticksize,tickpad,' +
               'xnumformat,ynumformat,area_under,' +
               'annot_fontsize,annot_xoffset,annot_yoffset,annot_count,annot_baseline,annot_ymax,annot_ymin,annot_ystack,' +
               'y_bands,x_bands,x_lines,y_lines,v_bands,h_bands,v_lines,h_lines,' +
               'xy_circles,xy_texts,annotcallback,bottom,xlabel,ylabel,pngdir,' +
               'commenttopright,commenttopright_in,commenttopright_out,commentbottomright_out,commenttopleft,commenttopleft_in,commentbottomleft,commentbottomright,' +
               'commentright,commentleft,commenttop,commentbottom,commentfontsize_add')

    
    if bottom:      # Azért szükséges, mert resample esetén az induláskor beállított bottom elveszhet. 
                    # Jellemzően a tblinfo-tól érkező hívás esetén fordulhat elő, hogy van "bottom" adat a params-ban 
        plt.subplots_adjust(bottom=bottom)

    commentfontsize=7
    if commentfontsize_add: commentfontsize += commentfontsize_add

    if commenttopleft_in is not None and commenttopleft is None: commenttopleft=commenttopleft_in
    
        
    if not xy_texts: xy_texts=[]
    if commenttopright:     # bal alsó sarok az origó (gcf esetén a teljes képtér, gca esetén koordináta-tér)
        xy_texts.append({'x':0.95,'y':0.97,'caption':commenttopright,'ha':'right','va':'top','fontsize':commentfontsize,'alpha':0.5})
    if commenttopright_in:     # bal alsó sarok az origó (gcf esetén a teljes képtér, gca esetén koordináta-tér)
        xy_texts.append({'x':0.95,'y':0.97,'caption':commenttopright_in,'ha':'right','va':'top','fontsize':commentfontsize,'alpha':0.5,'transform':'gca'})
    if commenttopright_out:     
        w_,h_ = axesunits('point')      # a koord-terület felett, 14 point-tal (átszámítom koord-faktorra)
        xy_texts.append({'x':1,'y':1 + 14/h_,'caption':commenttopright_out,'ha':'right','va':'top','fontsize':commentfontsize,'alpha':0.5,'transform':'gca'})
    if commentbottomright_out:     # bal alsó sarok az origó (gcf esetén a teljes képtér, gca esetén koordináta-tér)
        xy_texts.append({'x':0.95,'y':0.03,'caption':commentbottomright_out,'ha':'right','va':'bottom','fontsize':commentfontsize,'alpha':0.5})
    if commenttopleft:
        xy_texts.append({'x':0.05,'y':0.97,'caption':commenttopleft,'ha':'left','va':'top','fontsize':commentfontsize,'alpha':0.5,'transform':'gca'})
    if commentbottomleft:
        xy_texts.append({'x':0.05,'y':0.03,'caption':commentbottomleft,'ha':'left','va':'bottom','fontsize':commentfontsize,'alpha':0.5,'transform':'gca'})
    if commentbottomright:
        xy_texts.append({'x':0.95,'y':0.03,'caption':commentbottomright,'ha':'right','va':'bottom','fontsize':commentfontsize,'alpha':0.5,'transform':'gca'})
    if commentleft:
        xy_texts.append({'x':0.05,'y':0.5,'caption':commentleft,'ha':'left','va':'center','fontsize':commentfontsize,'alpha':0.5,'transform':'gca'})
    if commentright:
        xy_texts.append({'x':0.95,'y':0.5,'caption':commentright,'ha':'right','va':'center','fontsize':commentfontsize,'alpha':0.5,'transform':'gca'})
    if commenttop:
        xy_texts.append({'x':0.5,'y':0.97,'caption':commenttop,'ha':'center','va':'top','fontsize':commentfontsize,'alpha':0.5,'transform':'gca'})
    if commentbottom:
        xy_texts.append({'x':0.5,'y':0.03,'caption':commentbottom,'ha':'center','va':'bottom','fontsize':commentfontsize,'alpha':0.5,'transform':'gca'})


    if x1==None: x1=xmin
    if x2==None: x2=xmax
    if y1==None: y1=ymin
    if y2==None: y2=ymax

    if x1 is not None or x2 is not None:     # megadható csak az egyik is
        if type(x1)==str: x1=pd.to_datetime(x1)
        if type(x2)==str: x2=pd.to_datetime(x2)
        
        x1_,x2_=plt.xlim()
        x1_=x1_ - (x2_ - x1_)*0.02      # az autoscale miatt ténylegesen ez volt a határ
        x2_=x2_ + (x2_ - x1_)*0.02      # az autoscale miatt ténylegesen ez volt a határ

        # y1_,y2_=plt.ylim()
        # y1_=y1_ - (y2_ - y1_)*0.05      # az autoscale miatt ténylegesen ez volt a határ
        # y2_=y2_ + (y2_ - y1_)*0.05      # az autoscale miatt ténylegesen ez volt a határ

        if x1==None: x1=x1_
        if x2==None: x2=x2_
        # if y1==None: y1=y1_
        # if y2==None: y2=y2_

        plt.xlim(x1,x2)         # az xlim beállítása kikapcsolja az autoscale-t és vele a 0.05-ös margin-t 
        # plt.ylim(y1,y2)         # az ylim beállítása kikapcsolja az autoscale-t és vele a 0.05-ös margin-t

    if y1 is not None or y2 is not None:   # megadható csak az egyik is
        y1_,y2_=plt.ylim()
        y1_=y1_ - (y2_ - y1_)*0.02      # az autoscale miatt ténylegesen ez volt a határ
        y2_=y2_ + (y2_ - y1_)*0.02      # az autoscale miatt ténylegesen ez volt a határ

        if y1==None: y1=y1_
        if y2==None: y2=y2_
        plt.ylim(y1,y2)         # az ylim beállítása kikapcsolja az autoscale-t és vele a 0.05-ös margin-t

    # if not xtickerstep:     # dátum-tengely esetén alapértelmezett formázás
    #     if 'date' in str(type(x1)):     # NEM JÓ   mindenképpen float

    fontsize0=9
    tickpad0=None
    labelpad0=None
    if subplot: 
        fontsize0=np.select([subcount>30,subcount>10],[6,7],8)
        tickpad0=np.select([subcount>10],[1],1.5)               # point-ban
        labelpad0=np.select([subcount>10,subcount>5],[1,2],3)      # point-ban

        
            
    if xtickerstep:
        if type(xtickerstep)==str:
            # if xtickerstep=='date':
            #     x1_,x2_=plt.xlim()
            #     daydiff= (x2_ - x1_)
            #     if daydiff>10*365: xtickerstep=''   # általános formázás
            #     elif daydiff>2.5*365: xtickerstep='year'   # centrált évszámok
            #     elif daydiff>190: xtickerstep='month'   # centrált hónapnevek
            #     else: xtickerstep='date'

            if xtickerstep=='date': 
                locator = mpl.dates.AutoDateLocator()
                formatter=mpl.dates.ConciseDateFormatter(locator)
                formatter.formats = ['%y',  # ticks are mostly years
                                    '%b %#d',   # ticks are mostly months    A "#" karakter a 0 padding elhagyását kéri
                                    '%d',       # ticks are mostly days
                                    '%H:%M',    # hrs
                                    '%H:%M',    # min
                                    '%S.%f', ]  # secs
                plt.gca().xaxis.set_major_formatter(formatter)
            
            elif xtickerstep=='month': 
                plt.gca().xaxis.set_major_locator(mpl.dates.MonthLocator())
                # 16 is a slight approximation since months differ in number of days.
                plt.gca().xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=16))

                plt.gca().xaxis.set_major_formatter(mpl.ticker.NullFormatter())    # a major tick-sorozat ne látszódjon
                plt.gca().xaxis.set_minor_formatter(mpl.dates.DateFormatter('%b\n%Y'))   # rövidített hónapnevek
                    # - az évszám itt még mindenhova bekerül, de később az első/utolsó illetve a "jan." 
                    #   kivételével törölve lesz (lásd lejjebb)

                # Remove the minor tick lines 
                plt.gca().tick_params(axis='x', which='minor', tick1On=False, tick2On=False)

                # Align the minor tick label
                for label in plt.gca().get_xticklabels(minor=True): label.set_horizontalalignment('center')
                
                # Évszám csak az első, utolsó és a jan. hónapoknál maradjon meg
                plt.gcf().canvas.draw()     # enélkül a get_xticklabels üres feliratokat adna vissza
                labels=plt.gca().get_xticklabels(minor=True)
                labelsmod=labels
                for i,label in enumerate(labels): 
                    if not (i==0 or i==len(plt.gca().get_xticklabels(minor=True))-1 
                            or beginwith(label.get_text(),'jan')): 
                        labelsmod[i].set_text(cutright(label.get_text(),'\nyyyy'))
                plt.gca().set_xticklabels(labelsmod,minor=True,fontdict={'fontsize':fontsize0})

            elif xtickerstep=='year': 
                plt.gca().xaxis.set_major_locator(mpl.dates.YearLocator())      # csak vonalak, feliratok nélkül
                plt.gca().xaxis.set_minor_locator(mpl.dates.YearLocator(month=7,day=1))   # csak feliratok, vonalak nélkül

                plt.gca().xaxis.set_major_formatter(mpl.ticker.NullFormatter())    # a major tick-sorozat ne látszódjon
                plt.gca().xaxis.set_minor_formatter(mpl.dates.DateFormatter('%Y'))    

                # Remove the minor tick lines 
                plt.gca().tick_params(axis='x', which='minor', tick1On=False, tick2On=False)

                # Align the minor tick label
                for label in plt.gca().get_xticklabels(minor=True): label.set_horizontalalignment('center')
               

        else:
            plt.gca().xaxis.set_major_locator(mpl.ticker.MultipleLocator(xtickerstep))

    if ytickerstep:
        plt.gca().yaxis.set_major_locator(mpl.ticker.MultipleLocator(ytickerstep))


    if xlabel or xlabel=='': plt.xlabel(xlabel,alpha=0.6,fontsize=fontsize0,labelpad=labelpad0)     # ez is a resample miatt romolhatott el
    if ylabel or ylabel=='': plt.ylabel(ylabel,alpha=0.6,fontsize=fontsize0,labelpad=labelpad0)     # ez is a resample miatt romolhatott el


    if ticksize!=None:
        plt.gca().tick_params(axis='both', which='major', labelsize=ticksize)
    elif subplot: 
        plt.gca().tick_params(axis='both', which='major', labelsize=fontsize0)         # default: 8
        fontsize_offsettext=fontsize0
        if fontsize_offsettext>7: fontsize_offsettext-=1
        plt.gca().xaxis.offsetText.set_fontsize(fontsize_offsettext)      # pl. '1e+6' 
        plt.gca().yaxis.offsetText.set_fontsize(fontsize_offsettext)


    if tickpad:
        plt.gca().tick_params(axis='both', which='major', pad=tickpad)
    elif subplot: plt.gca().tick_params(axis='both', which='major', pad=tickpad0)         # default: 1.5
   
    if xticklabels==False:
        plt.gca().xaxis.set_ticklabels([])
    else:
        if xticklabel_map is not None:   
            # - a dict ser function formátumokkal nem foglalkozom (nem garantált a találat az integerek-re)
            def f_formatter_x(x,pos):
                if abs(x-round(x))>0.01: return strnum(x,'2g')  # return ''         # csak egész számokra jelenít meg értéket
                i=int(round(x))
                ticklabel=dget(xticklabel_map,key=i,index=i)
                if ticklabel is None or ticklabel=='': return strnum(x,'2g')  # return ''
                # rövidítés
                fontsize=notnone(ticksize,fontsize0)
                charwidth=fontsize * 5 / 8       # átlagos karakterszélesség, közelítő érték
                w,h = axesunits('point')
                max_chars = Limit(int(1.2 *  w / 10 / charwidth),min=5)         # max 10 tick jelenik meg (az 1.2 egy konyhaszabály - ennyi még belefér)
                return txtshorten(ticklabel,max_chars)

            plt.gca().xaxis.set_major_formatter(f_formatter_x)

    if yticklabels==False:
        plt.gca().yaxis.set_ticklabels([])
    else:
        if yticklabel_map is not None:   
            def f_formatter_y(y,pos):
                if abs(y-round(y))>0.01: return strnum(y,'2g')  # return ''         # csak egész számokra jelenít meg értéket
                i=int(round(y))
                ticklabel=dget(yticklabel_map,key=i,index=i)
                if ticklabel is None or ticklabel=='': return strnum(y,'2g')   # return ''
                # rövidítés
                return txtshorten(ticklabel,8)

            plt.gca().yaxis.set_major_formatter(f_formatter_y)
    
    if xnumformat: plt.gca().xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter(FvNumformat(xnumformat,'x')))
    if ynumformat: plt.gca().yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter(FvNumformat(ynumformat,'x')))

    # Sávok rajzolása (y_bands, x_bands)
    def plot_yband_label(caption,x1,x2,fontsize,color,align):
        if not fontsize: fontsize=7
        if not align: align='left'
        if align=='topcenter':
            rotation='horizontal'
            va='top'
            ha='center'
            try: text_x=(x1 + x2)/2
            except: text_x=floatdate((datefloat(x1) + datefloat(x2))/2)
            text_y=plt.ylim()[1] - (plt.ylim()[1]-plt.ylim()[0])/50
        else:
            text_x=datefloat(x1) + 2* (plt.xlim()[1]-plt.xlim()[0]) / axesunits()[0]  
                #  2 point eltolás jobbra, mert alapesetben balra kilógnának a karakterek
            rotation='vertical'
            if align=='left': align='bottom'
            elif align=='right': align='top'
            va=align    # "top" vagy "bottom"
            ha='left'
            if align=='top': text_y=plt.ylim()[1] - (plt.ylim()[1]-plt.ylim()[0])/50
            elif align=='center': text_y=(plt.ylim()[1]-plt.ylim()[0])/2
            else: text_y=plt.ylim()[0] +(plt.ylim()[1]-plt.ylim()[0])/50
        plt.text(text_x,text_y,caption,fontsize=fontsize,color=color,rotation=rotation,va=va,ha=ha)
            
    def plot_xband_label(caption,y,fontsize,color,align):
        if not fontsize: fontsize=7
        if not align: align='left'
        if align=='right': text_x=plt.xlim()[1] - (plt.ylim()[1]-plt.ylim()[0])/50
        elif align=='center': text_x=(plt.xlim()[1] - plt.xlim()[0])/2
        else: text_x=plt.xlim()[0] + (plt.ylim()[1]-plt.ylim()[0])/50
        plt.text(text_x,y,caption,fontsize=fontsize,color=color,va='bottom',ha=align)
            
    if not x_lines: x_lines=h_lines
    if not y_lines: y_lines=v_lines
    if not x_bands: x_bands=h_bands
    if not y_bands: y_bands=v_bands
            
    if isinstance(area_under,(float,int)):
        x_band = [ddef(y1='start',y2=area_under,color='gray')]
        if x_bands is not None:  x_bands=x_bands + x_band
        else: x_bands = x_band
        x_line = [ddef(y=area_under,color='black')]
        if x_lines is not None:  x_lines=x_lines + x_line
        else: x_lines = x_line
        
    if y_bands:
        if type(y_bands)==dict: y_bands=[y_bands]
        for band in y_bands:
            x1,x2,color,alpha,caption,align,fontsize = dgets(band,'x1,x2,color,alpha,caption,align,fontsize')
            if not x1: continue
            if not color: color='green'
            if not alpha: alpha=0.07
            if x1 in ['start','begin']: x1=plt.xlim()[0]
            elif x2=='end': x2=plt.xlim()[1]
            # plt.fill_betweenx(plt.ylim(),datefloat(x1),datefloat(x2),color=color,alpha=alpha)
            # plt.autoscale(enable=False)         # fontos, mert enélkül újraszámolja a az xlim, xlim határokat és a sáv nem fog a szélekig érni
            #  - az első sávrajzolás után kell befagyasztani
            plt.axvspan(datefloat(x1),datefloat(x2),color=color,alpha=alpha)
            if caption: plot_yband_label(caption,x1,x2,fontsize,color,align)
    if y_lines:
        if type(y_lines)==dict: y_lines=[y_lines]
        for line in y_lines:
            x,linewidth,color,alpha,caption,align,fontsize = dgets(line,'x,width,color,alpha,caption,align,fontsize')
            if x is None: continue
            if not linewidth: linewidth=1
            if not color: color='green'
            if not alpha: alpha=0.2
            plt.axvline(datefloat(x),linewidth=linewidth,color=color,alpha=alpha)
            if caption: plot_yband_label(caption,x,x,fontsize,color,align)
        
    if x_bands:
        if type(x_bands)==dict: x_bands=[x_bands]
        for band in x_bands:
            y1,y2,color,alpha,caption,align,fontsize = dgets(band,'y1,y2,color,alpha,caption,align,fontsize')
            if not y1: continue
            if not color: color='green'
            if not alpha: alpha=0.07
            if y1 in ['start','begin']: y1=plt.ylim()[0]
            elif y2=='end': y2=plt.ylim()[1]
            # plt.fill_between(plt.xlim(),y1,y2,color=color,alpha=alpha)
            # plt.autoscale(enable=False)         # fontos, mert enélkül újraszámolja a az xlim, ylim határokat és a sáv nem fog a szélekig érni
            plt.axhspan(y1,y2,color=color,alpha=alpha)
            if caption: plot_xband_label(caption,x1,fontsize,color,align)
    if x_lines:
        if type(x_lines)==dict: x_lines=[x_lines]
        for line in x_lines:
            y,linewidth,color,alpha,caption,align,fontsize = dgets(line,'y,width,color,alpha,caption,align,fontsize')
            if not linewidth: linewidth=1
            if not color: color='green'
            if not alpha: alpha=0.2
            plt.axhline(y,linewidth=linewidth,color=color,alpha=alpha)
            if caption: plot_xband_label(caption,y,fontsize,color,align)

    # Annotációs körök rajzolása a diagramvonalakon kívül
        # array of dict(x,y,size,color,caption)
        # - x,y:  a pont koordinátái (forrásadatokhoz igazodó)
        # - size: point-ban  (betűmérethez viszonyítható)
        # - color:  a kör színe és átlátszósága  (default: green)
        # - caption: (opcionális)  a körhöz tartozó annotáció
    if xy_circles:
        if type(xy_circles)==dict: xy_circles=[xy_circles]
        for annotcircle in xy_circles:
            x,y,size,color,caption = dgets(annotcircle,'x,y,size,color,caption')
            FvScatterAdd(datefloat(x),datefloat(y),color,size,caption)
        FvScatterPlot(maxsize=100,defaultsize=10,colorbar=None)

    # Diagramvonalaktól független magyarázó feliratok
        # array of dict(x,y,caption,ha,va,fontsize,alpha,colo,transform)
        # x,y:   relatív koordináták a bal alsó sarokhoz viszonyítva
        #   - subplot esetén a rajzterület origójához képest (x=-0.2: a rajzterülettől balra)
        #   - normál plot esetén a teljes képterület bal alsó sarkához képest
        # ha: horizontal alignment, 'left' 'center' 'right'
        # va: vertical alignment, 'top' 'center' 'bottom'
        # color:  'green'   '0.3':szürke árnyalat
        # transform: 'gcf'=teljes képterületen belül   'gca'=rajzterület origójához képest
    if xy_texts:
        if type(xy_texts)==dict: xy_texts=[xy_texts]
        for annottext in xy_texts:
            x,y,caption,ha,va,fontsize,alpha,color,transformL = dgets(annottext,'x,y,caption,ha,va,fontsize,alpha,color,transform')
            if not x or not y: continue
            if not ha: ha='left'
            if not va: va='top'
            if not fontsize: fontsize=fontsize0-1
            if not alpha: alpha=0.5
            if not transformL:
                if subplot: transform=plt.gca().transAxes
                else: transform=plt.gcf().transFigure
            elif transformL=='gcf': transform=plt.gcf().transFigure
            elif transformL=='gca': transform=plt.gca().transAxes
            plt.text(x,y,caption,ha=ha,va=va,fontsize=fontsize,alpha=alpha,transform = transform)
            

    # Annotáció rajzolása (diagram-vonalak illetve adat-pontok annotációja)
    if annot: 
        # print('annot')
        if not annot_fontsize: annot_fontsize=fontsize0
        FvAnnotatePlot(xoffset=annot_xoffset,yoffset=annot_yoffset,fontsize=annot_fontsize,annot_count=annot_count,
                    annotcallback=annotcallback,serbaseline=annot_baseline,ymax=annot_ymax,ymin=annot_ymin,ystack=annot_ystack)
    if legend: plt.legend(fontsize=fontsize0)


    if not subplot: 
        if to_file: 
            path=nextpath(to_file,'png', dir=pngdir)
            try: 
                plt.savefig(path)
                plt.close()
                print('Diagram fájlba írása: ' + path)
            except:
                print('A diagram fájlba írás nem sikerült')

        else: plt.show()

def pltshowsub(annot=True,legend=False,**params):
    '''
    Subplot rajzolásának befejezése (záró formázások, annotációk és legend megjelenítése).  
    A rajzolási folyamat legvégén kötelező hívni a pltshow függvényt
    Ha csak egy subplot van, akkor nem kell külön subplot szintű hívás
    
    params:
        x1,x2,y1,y2:   koordináta határok   (x esetén dátumstring is megadható;   féloldalasan is megadható)
        xtickerstep:   milyen lépésközzel kövessék egymást az x tengely tick-jei  (dátum tengelyre 'date', 'month')
        ticksize:   fontméret (mindkét tengely jelölőire).  Default: 8
        tickpad:    milyen messze legyenek a jelölőfeliratok a tengelytől (point-ban).  Default: 2
        xnumformat,ynumformat:  koordinátatengelyek jelölőinek számformátuma
         '{:,}        integer, ezres határolással (szóköz határolás sajnos nincs, csak vessző vagy underline) 
         '{:,.3f}'    fix decimal digits (0-k lehetnek a végéne, e+00 kitevő semmiképpen, ezres határoló)
         '{:,.5g}'    az összes digit száma van megadva (lehet kevesebb is, záró nullák nincsenek). Az e+00 akkor jelenik meg, ha a szám >= e+05 
         '{:,.4e}'    mindenképpen e+00 formátum. A decimális digitek száma van megadva (0-k lehetnek a végén)
         '{:,.2%}'    1.0 bázisú százalék (szorzás százzal).  A decimális digitek száma van megadva (0-k lehetnek a végén)
         '{:,.2f}%'   nincs szorzás (közvetlen százalékos adatok). A decimális digitek száma van megadva (0-k lehetnek a végén)

    További pltshow paraméterek:  (fontosabbak)
        commenttopright,commenttopleft,commentbottomleft,commentbottomright,  commentright,commentleft,commenttop,commentbottom: 
            - a subplot koordináta-területén belül megjelenő feliratok
            - szürke, 7-es méret, többsoros is lehet;   a legend-del átfedhet (a legend pozícióját a matplot automatikusan állítja be)
        commenttopright_out, commentbottomright_out:
            - a koordináta-terület felett illetve alatt jelenik meg (az alsó átfedhet a tick-feliratokkal)
    '''

    dsetsoft(params,annot_ymax=1.05)        # subplotok esetén kellemetlen az átfedés a title-lel, ezért jobb korlátozni

    pltshow(subplot=True,annot=annot,legend=legend,params=params)


def pltpercent(axis='y'):   # NINCS HASZÁLVA
    # "."     tizedes határoló.  Ha vessző is van előtte, akkor a vessző az ezres határoló (space nem adható meg, legfeljebb "_")
    # "g"     general   Az előtte lévő szám a digitek maximális száma (ha szükséges). A szélessége nem fix, felesleges 0-kat nem ír a végére és az elejére
    # hiba:  a diagram manuális átméretezésekor elromlik 
    #plt.gca().set_yticklabels(['{:.2g}%'.format(x*100) for x in plt.gca().get_yticks()])
    if axis=='y': plt.gca().yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0,decimals=0))
    elif axis=='x': plt.gca().xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0,decimals=0)) 

def FvPlotBkgcolorNegativ(color='0.7',alpha=0.1):   # NINCS HASZNÁLVA
    # Van jobb módszer is ...
    ylimL=plt.ylim()
    xlimL=plt.xlim()
    plt.autoscale(False)         # ne módosítsa az xlim ylim határokat
    plt.fill_between(xlimL,0,ylimL[0],color=color,alpha=alpha)




# HISTOGRAM PLOT,  SCALED PLOT
def hist_plot(tbl,cols='all',groups=None,G=0.02,style='sub',method='kde',            # NEW egy vagy több értékkészlet hisztogramja
            annotbyindex=False,outliers='show',labels=None,codelists=None,comment=None,
            suptitle=None,title=None,subparams=None,
            left=None,right=None,top=None,bottom=None,wspace=None,hspace=None,width=None,height=None,sharex=False):   
    '''
    Értékeloszlás megjelenítése egy vagy több adatra (kérhető a tbl-ben lévő összes számadatra is)
    
    tbl:   ha egyetlen arr-re kell plot, akkor előtte tbl=pd.DataFrame(arr,columns=[colname])
    cols:   mely oszlopokra jelenjen meg a hisztogram
        - string:  
            - 'all' vagy '':  összes oszlop
            - egyetlen oszlop neve  (vesszős felsorolás csak ** bevezetéssel megengedett)
            - ** kezdetű filtercols string   (lásd filtercols(),  lehet egyszerű vesszős felsorolás is, megadható NOT rész, ...)
            - "type=float,int"  kezdetű string (float,int,text,date adható meg, vesszős lista, tetszőleges sorrend). 
                A típusszűrő után megadható egy filtercols szűrő is (további szűrés a megadott típusú oszlopokon belül)
        - list of colnames
    groups:  csoportosító oszlop, opcionálisan a megjelenítendő csoport-értékek felsorolásával  (szöveges vagy int oszlop is lehet) 
            - ha a cols-ban több oszlop is meg van adva, akkor mindegyik subplot-ban több értékeloszlás lesz megjelenítve
        'country'     külön histogram-vonal a 'country' oszlop összes distinct értékére         
        'country:hun,cze,united king,slo,rom'    histogram-vonalak a country oszlop felsorolt értékeire  
    method:  'kde' (Kernel Density Estimation), 'bins' 
    style:  'sub' - subplotok
            'area' - közös diagram  (hasonló min-max értékű oszlopokra alkalmazható)
            'area_normalized' - közös diagram, az értékkészletek 1-re normalizálva
    annotbyindex:  localmax annotáció helyett a tbl indexében lévő feliratokat jeleníti meg (pl. országnevek)
            A két szélső értékhez tartozó felirat mindenképpen megjelenik, a közbensők mintavétel szerűen. 
            Nem túl nagy rekordszám esetén (pl. <50) a G csökkentésével elérhető, hogy egy tüskesorozat jelenjen meg.
            integer is megadható:  legfeljebb hány felirat jelenjen meg (True esetén automatikus hangolás)
            list-of_string is megadható:  a két szélsőn kívül csak a listában szereplő feliratok jelenjenek meg
            FvPlotBarH helyett is használható, ha túl magas a rekordok száma (pl. >20)
    outliers:  ha None, akkor nem jelennek meg az outlier jelölők
            'show':  kék pontok jelzik az outliereket a x irányban,      qauntile, de max 1% 
            'show_and_annot':  kék pontok és "outliers" annotáció (first last) 
            'drop':  x-irányban kidobja az outlier-eket,     y irányban pedig az átlagos sűrűség tízszerese az ymax
    labels:  több vonal esetén lista, függvény, dict is megadható (dget). Ha nincs megadva, akkor az oszlopnév.
    codelists:  Szöveges értékkészlet esetén. Több vonal esetén lista, függvény, dict is megadható (dget)
    comment:   dget     A teljes diagram-ablak jobb felső sarkában megjelenő kiemelt comment a "common" kulcsszóval adható meg 
    suptitle,title:  default "Értékeloszlás"     A title a közös feliratra utal a suptitle alatt
    subparpmas:  az aldiagramokhoz tartozó paraméterek, dict formátumban (colname to param)
        subparams = ddef(
            titles = ddef(                      # subdiagram felett megjelenő címfelirat
                colname1=<title col1>
                ...
            ),
            comments = ddef(                    # subdiagram felett jobbra megjelenő szürke comment
                ...
            )
        )
        További:  numformats,xmin,xmax,          konstans érték is megadható (közös mindegyikre)
                  mean,median       True esetén zöld színű mean és/vagy medián vonal razolása annotációval
                  iqr               True esetén a középső 50% sötétebb háttérszínnel, a határokhoz annotáció

    '''

    columns=Tblcols(tbl,cols)

    count=len(columns)
    if count==0:
        print('ERROR   plot_hist   cols: egyetlen oszlop sem lett megadva')
        return
        
    
    if style in ['sub','multi','subplots']: style='sub'


    if not suptitle: 
        suptitle='Értékeloszlás'
        if outliers=='drop': suptitle+=' (outlier-ek nélkül)'

    pltinit(suptitle=suptitle,title=title,subcount=count,rowscols_aspect=(5,3),indexway='vert',
            sharex=sharex,sharey=False,
            left=left,right=right,top=top,bottom=bottom,wspace=wspace,hspace=hspace,width=width,height=height)

    for i,colname in enumerate(columns):
        if i>=count: break

        label=dget(labels,colname,colname,i)
        codelist=dget(codelists,colname,None,i)

        if count>=5: progress('plot_hist:  ' + label)

        title = dget(dget(subparams,'titles'),colname,colname,i)
        subcomment = dget(dget(subparams,'comments'),colname,None,i)
        numformat = dget(dget(subparams,'numformats'),colname,None,i)
        xmax = dget(dget(subparams,'xmax'),colname,None,i)
        xmin = dget(dget(subparams,'xmin'),colname,None,i)
        mean = dget(dget(subparams,'mean'),colname,None,i)
        median = dget(dget(subparams,'median'),colname,None,i)
        iqr = dget(dget(subparams,'iqr'),colname,None,i)

        subplot_hist(tbl,[colname],title,G=G,groups=groups,annotbyindex=annotbyindex,
                    outliers=outliers,method=method,labels=label,codelist=codelist,
                    comment=subcomment,numformat=numformat,xmax=xmax,xmin=xmin,mean=mean,median=median,iqr=iqr)

    commentL=''
    if type(comment)==dict: commentL=comment.get('common')
    else: commentL=comment          
    
    pltshow(commenttopright=commentL)

def plot_hist_old(tbl,cols,G=0.04,style='sub',kde=False,outliers=0.01,drop_outliers=False,labels=None,comment=None):    #  OLD egy vagy több értékkészlet hisztogramja
    '''
    tbl:   ha egyetle arr-re kell plot, akkor előtte tbl=pd.DataFrame(arr,columns=[colname])
    cols:   mely oszlopokra jelenjen meg a hisztogram
        - string:  
            - egyetlen oszlop neve  (vesszős felsorolás csak ** bevezetéssel megengedett)
            - ** kezdetű filtercols string   (lásd filtercols(),  lehet egyszerű vesszős felsorolás is, megadható NOT rész, ...)
            - "type=float,int"  kezdetű string (float,int,text,date adható meg, vesszős lista, tetszőleges sorrend). 
                A típusszűrő után megadható egy filtercols szűrő is (további szűrés a megadott típusú oszlopokon belül)
        - list of colnames
    kde:  True esetén a KDE algoritmussal számol  (Kernel Density Estimation)
    style:  'sub' (vagy 'subplots')  'area'
        - ha az értékkészletek nincsenek normálva, akkor a style='sub' beállítás javasolt (független diagramok)
    outliers:       None esetén nincs jelzés
         0.01:       a leginkább kilógó 1%-t jelezze (method=isolation forest)
        'auto':     az isolation_forest algortimus dönti el, hogy hány pontot tart outlier-nek
        'all' vagy 'quantile':      quantile algoritmus  (itt sem kontrollált a pontok száma)
    label:  több vonal esetén lista, függvény, dict is megadható (dget)
    comment:   dget     A teljes diagram-ablak jobb felső sarkában megjelenő kiemelt comment a "common" kulcsszóval adható meg 
    '''

    columns=Tblcols(cols)

    count=len(columns)
    if count==0:
        print('ERROR   plot_hist   cols: egyetlen oszlop sem lett megadva')
        return
        
    
    if style in ['sub','multi','subplots']: style='sub'


    suptitle='Értékeloszlás'
    if drop_outliers: suptitle+=' (outlier-ek nélkül)'

    if style=='sub':
        pltinit(suptitle=suptitle,subcount=count,rowscols_aspect=(5,3),indexway='vert',
                sharex=False,sharey=False)
        annot='gaussmax'
    else:
        pltinit(suptitle=suptitle,title=label)
        annot='max last'
    
    ymax,ymin=None,None

    for i,colname in enumerate(columns):
        if i>=count: break

        label=dget(labels,colname,colname,i)

        if count>=5: progress('plot_hist:  ' + label)


        arr=tbl[colname].values

        # nan értékeket ki kell dobni (a histogram függvény nem szereti)
        arr=array(arr)
        arr=arr[~np.isnan(arr)]
        
        if len(arr)==0: continue       # ha csak nan értékeket tartalmaz, akkor nincs vele teendő

        # outliers
        outliers_=[]
        if type(outliers)==str: 
            if outliers=='all' or beginwith(outliers,'quant'): 
                if drop_outliers:  arr=fn_outliers(arr,quote='auto',method='quantile',drop=True)
                else: outliers_=fn_outliers(arr,quote='auto',method='quantile')
            elif outliers=='auto': 
                if drop_outliers:  arr=fn_outliers(arr,quote='auto',method='isolation forest',drop=True)
                else: outliers_=fn_outliers(arr,quote='auto',method='isolation forest')
        elif type(outliers)==float:
            if drop_outliers: arr=fn_outliers(arr,quote=outliers,method='isolation forest',drop=True)
            else: outliers_=fn_outliers(arr,quote=outliers,method='isolation forest')

        y_avg=5

        if kde:
            ser_hist=hist_KDE(arr,G,outpoints=400)
        else:
            ser_hist=hist_bybins(arr,y_avg)

        if style=='sub':
            pltinitsub(axindex='next',title=label)
            
            plttype='gauss'
            if kde: plttype='original'

            FvPlot(ser_hist,plttype=plttype,label='',annotate=annot,annotcaption='{x:.4g}',G=G,area=True,positive=True,gaussside='')

            if len(outliers_)>0: FvPlot(pd.Series([0]*len(outliers_),outliers_),plttype='scatter',annotate='',colors={'color':'navy','alpha':0.1})
            # y=0 értékek (a pontok az x-tengelyen)

            tickpad=None
            if count>2: tickpad=1.5

            commentL=''
            if type(comment)==dict: commentL=comment.get(label)          
            
            ymax,ymin=None,None
            if drop_outliers and ser_hist.values.max()>y_avg*10: 
                ymax=y_avg*10         # az átlagos sűrűség 10-szerese legyen a felső határ
                ymin=-2

            pltshowsub(tickpad=tickpad,commenttopright=commentL,ymin=ymin,ymax=ymax)
        else:
            area =  (style=='area') or (style=='firstarea' and i==0) or (style=='firstline' and i>0)
            FvPlot(ser_hist,plttype='gausst',label=label,G=G,area=area,positive=True,gaussside='',
                   annotate=annot,annotcaption='{x:.4g}')

            if len(outliers_)>0: FvPlot(pd.Series([0]*len(outliers_),outliers_),plttype='scatter',annotate='',colors={'color':'navy','alpha':0.1})
    
            if drop_outliers and ser_hist.values.max()>y_avg*10: 
                ymax=y_avg*10         # az átlagos sűrűség 10-szerese legyen a felső határ
                ymin=-2

    commentL=''
    if type(comment)==dict: commentL=comment.get('common')
    else: commentL=comment          
    
    if style=='sub': ymax,ymin=None,None        # a subplot-okban már be lett állítva

    pltshow(commenttopright=commentL,ymin=ymin,ymax=ymax)


def hist_bybins(arr,y_avg=5):         # bin-ek, numpy függvénnyel
    '''
    Series-t ad vissza, egyenletes eloszlású indexxel  (y értékek > sűrűség)
    
    arr:    list, np.array, ser.values, ser.index
    y_avg:      az eloszlásfüggvény átlaga  (egy bin-be átlagosan ennyi pont kerüljön)
        - ha nagyon alacsony a tömb elemszáma, akkor az eloszlásfüggvény átlaga kisebbnek adódhat  (legalább 10 bin-t mindenképpen felvesz)
    '''
    arr=array(arr)
    arr=arr[~np.isnan(arr)]
    
    if len(arr)==0: return       # ha csak nan értékeket tartalmaz, akkor nincs vele teendő


    bins=int(round(len(arr)/y_avg))
    if bins<10: bins=10

    values,bin_edges = np.histogram(arr,bins)
    # bin_edges eltolása fél bin-nel
    bin_width=bin_edges[1]-bin_edges[0]
    bin_centers = bin_edges + bin_width/2
    bin_centers=bin_centers[:-1]    # az utolsó határ nem kell (eggyel nagyobb az elemszáma a values-nál)

    return pd.Series(values,bin_centers)

def hist_KDE(arr,G=0.04,outpoints=500):        # KernelDensityEstimation, a listában előforduló értékek sűrűségfüggvénye (hisztogramja)
    '''
    Egy ser-t ad vissza, aminek az x változója a tömb legkisebb értékétől a legnagyobb értékéig terjed, egyenletes lépésközzel
    Az y változó az értékek előfordulási gyakoriságát jelzi  (hisztogram jellegű)

    Maga az algoritmus nem túl bonyolult:  az előforduló értékekhez rendelt elemi gauss görbék összeadásáról van szó.
    Az optimális sávszélességre vannak konyhaszabályok illetve hangolási mechanizmusok, de jobbnak tűnik külső változónak kezelni

    arr:  nem kell rendezettnek lennie. Ha van benne nan, akkor kihagyja a függvény.
    G:  <1 float,   az értékkészlet szélességének hány százaléka legyen a gaussok talpponti szélessége (4*szigma)
        - általában 0.01 körüli érték adható meg   (kis tételszám esetén lehet nagyobb is)
        - az ennél közelebbi pontok összeolvadnak
    '''
    arr=array(arr)
    arr=arr[~np.isnan(arr)]       # nan érték nem lehet benne

    if len(arr)==0: return       # ha csak nan értékeket tartalmaz, akkor nincs vele teendő

    # kostans függvény kezelése
    if arr.max()==arr.min():
        value=arr.max()
        if value>=1:
            arr_max=value*1.1
            arr_min=0
        elif value<=-1:
            arr_min=value*1.1
            arr_max=0
        else:
            arr_min=-1
            arr_max=1
        bandwidth=(arr_max-arr_min)*G / 4
        outrange=(arr_min,arr_max)
    else:
        bandwidth=(arr.max()-arr.min())*G / 4
        outrange=(arr.min(),arr.max())
    
    X=np.reshape(arr,(len(arr),1))
    kde = KernelDensity(bandwidth=bandwidth,kernel='gaussian').fit(X)        # gyors

    x=Range(outrange[0],outrange[1],count=outpoints)
    X_out=np.reshape(x,(outpoints,1))
    density=np.exp(kde.score_samples(X_out))            # viszonylag időigényes (ne legyen túl sok outpoint)

    return pd.Series(density,x)

def ser_xdensity(ser,G=0.04,outpoints=500):       # pontsűrűség az x irányban  (pl. korreláció-plot-okhoz hasznos)
    return hist_KDE(ser.index,G=G,outpoints=outpoints)
def ser_ydensity(ser,G=0.04,outpoints=500):       # érték-eloszlás
    return hist_KDE(ser.values,G=G,outpoints=outpoints)



def depend_plot(tbl,ycol,xcols,          # Hogy függ az ycol más változóktól (fx vagy ordered plot-ok)
            style='fx',G=0.1,minmax_annot='auto',
            normalize=False,corrtype='pearson',        # normalize: ha több ycol van, és lényegesen eltér az értékkészletük
            scatter=False,
            suptitle=None,
            xlabels=None,xnumformats=None,xcodelists=None,          # xcols méretű list vagy dict (ser,kódlista)
            ylabel=None,ynumformat=None,ylinelabels=None,ycodelist=None,       # ylinelabels: ycols méretű list (dict,kódlista,function,value)
            interval_y=None,
            
            rowscols_aspect=(2,3),indexway='vert',sharex=False,sharey=False,
            nrows=None,ncols=None,      # ha nincs megadva, akkor subcount alapú hangolás
            width=None,height=None,     # ha nincs megadva, akkor subcount alapú hangolás
            comments=None):             # dict    'common':topright   xcols:subloptonként
    '''
    Hogy függ az ycol változó az xcol változóktól?   (ycol is lehet több hasonló karakterű feature)
    Alapkoreográfia:  megegyezik az ycol, és vizsgálandó a többi változótól való függése
        - az ycol általában egy kimeneti (számított) jellemző, az x változók input karakterűek
        - az y változó általában folytonos karakterű, de az x változók lehetnek szövegesek is
        - kérhetőek a legnagyob korrelációt mutató x-változók
    Diagramtípus:
        - alapesetben fx plot, de kérhető ordered-plot, illetve a kettő vegyesen is

    tbl:        a tábla kulcsa érdektelen, mindegyik x_feature normál oszlopban van
    ycol:       a vizsgált változó (több is megadható listában, vagy **-felsorolásban).  
                Általában egy feature, de lehet több hasonló feature is. Szöveges oszlop is megadható.
                Mindegyik subploton ugyanez(ek) a diagram-vonal(ak) jelennek meg.
    xcols:      subplotok x változói (általában különböző, de lehet benne ismétlődés). Szöveges oszlopok is megadhatók.
                'top', 'top5':  a legnagyobb korrelációjú adatok  (csak float vagy integer oszlopok)
                'type=txt':  csak a szöveges oszlopok ('int', ...)
                '**' az elején:  több-találatos oszlopkiválasztás
    style:     'fx':   y(x) diagram,  x irányban outlier elhagyással, KDE árnyalással 
               'ordered':  az x-tengelyen xcol szerint rendezett sorszámok, az xcol is kirajzolva skálázással (area-plot)
               'both':   kétsoros megjelenítés, a felső sorban a 'ordered' plotok, az alsó sorban az 'fx' plotok
    
    xtitles,xlabels:   subplot title és xlabel.  Ha nincs megadva, akkor =xcols.  Megadható dict,list,fn   (fn esetén key,index argumentumok)
    ylabel,ynumformat:   mindegyik subplot-ra közös
    ylinelabels:    annotációk az y-vonalakhoz (mindegyik subplotban). Ha nincs megadva, akkor =ycols. Megadható dict,list,fn
    normalize:  ha több ycol van, és lényegesen eltér az értékkészletük, akkor normalize=1 javasolt
    comments:   dict vagy list     dict esetén 'common' key-vel adható meg a jobb-felső comment, key=xcol kulcsokkal pedig a subplot comment-ek
    '''

    if style in ['fx','f(x)','yx','y(x)','y','xy']: style='fx'


    ycols=Tblcols(tbl,ycol)
    # if type(ycol)==str:        # alapesetben csak egy oszlop, de megadható több hasonló jellegű (és értékkészletű) oszlop is
    #     if ycol[:2]=='**': ycols=filternames(tbl.columns,ycol[2:])
    #     else: ycols=[ycol]      # vesszős lista csak ** bevezetéssel megengedett, mert az oszlopnevekben is lehet vessző
    # elif type(ycol)==list:
    #     ycols=ycol
    # else:
    #     print('ERROR  depend_plot  ycol:  string vagy lista adható meg')
    #     return
    # ycol=None

    xcols=Tblcols(tbl,xcols,ycols[0],corrtype)

    if len(xcols)==0:
        print('ERROR  depend_plot:  nem lett megadva egyetlen xcol sem')
        return


    if not suptitle: 
        if not ylabel: ylabel=ycols[0]
        suptitle=ylabel + ' függése más változóktól'

    # def f_annot_x(x,codelist,default):
    #     if codelist is not None: return txtshorten(dget(codelist,key=int(x),index=int(x),default=default),20)
    #     else: return 'x=' + strnum(x,'4g')


    # szöveges ycol kódolása (ritkán szükséges, általában float)
    ycols_=ycols.copy()
    for i,ycol in enumerate(ycols):
        if ycodelist is None and serdtype(tbl,ycol)=='string':
            tbl,ycols_[i],ycodelist = Encode_col(tbl,ycol)        # _encoded utótagot kap az oszlop
    ycols=ycols_

    # plttype='gauss'
    # if scatter: plttype='scatter gauss'

    subcount=len(xcols)

    if style=='both':   # kétsoros elrendezés
        subcount=2*len(xcols)        
        if subcount>5: 
            print('FIGYELEM  depend_plot, style="both": max 5 xcol jeleníthető meg')
            subcount=10
            xcols=xcols[:5]
        rowscols_aspect=(2,0)
        indexway='vert'

    pltinit(subcount=subcount,rowscols_aspect=rowscols_aspect,nrows=nrows,ncols=ncols,indexway=indexway,
                sharey=sharey,sharex=sharex,
                suptitle=suptitle,
                width=width,height=height)
    
    for i,xcol in enumerate(xcols):
        if len(xcols)>3: progress('depend_plot:  ' + str(xcol))
        
        xlabel=dget(xlabels,xcol,xcol,i)

        if len(xcols)==1 and len(ycols)==1:
            if not ylabel: ylabel=ycols[0]
            title=ylabel + '  by  ' + xlabel
        else: title='by  ' + xlabel

        # xnumformat=dget(xnumformats,xcol,None,i)

        
        # xcodelist=dget(xcodelists,xcol,None,i)
        # if xcodelist is None and serdtype(tbl,xcol)=='string':
        #     tbl,xcol,xcodelist = Encode_col(tbl,xcol)        # _encoded utótagot kap az oszlop

        # tblL=tbl.dropna(subset=[xcol])          # ordered és depend plot esetén sem lehet mit kezdeni a hiányzó x értékekkel

        discrete_x = (tbl[xcol].dtype in ['int32','int64'])

        # xmin,xmax,interval=None,None,None
        # if not discrete_x:           # diszkrét görbe esetén nem kell outlier szűrés
        #     xmin,xmax = fn_outlier_limits(tblL[xcol].to_numpy())
        #     interval=(xmin,xmax)

        # Diszkrét esetén átlagolás kell az ismétlődő x-értékekre (a tbl rekordszáma nem változik, de bekerül egy új oszlop)
        # ycols_mean=[]
        # if discrete_x:
        #     for ycol in ycols:
        #         ser_mean = tblL.groupby(by=xcol).mean()[ycol]
        #         ycols_mean.append(ser_mean)
        #         tblL=merge_ser(tblL,ser_mean,colname_new=ycol + '_mean',colnames_hiv=xcol)   # tblL rekordszáma nem változik, de bekerül egy új oszlop



        style_=style
        if not style_:
            if discrete_x: style_='ordered'
            else: style_='fx'

        if style_ in ['ordered','both']:
            subplot_ordered(tbl,xcol,ycols,title=title,comment='corr',G=G,scatter=scatter,minmax_annot=minmax_annot,interval_y=interval_y)
            # subplot_ordered(tbl,xcol,ycols,title=title,comment='corr',G=G,scatter=scatter,xlabel=xlabel,ylabels=ylabel,
            #             xcodelist=xcodelist,ycodelist=ycodelist,minmax_annot=minmax_annot,interval_y=interval_y)
            
            
            # pltinitsub('next',title=title)
            # tblL=tblL.sort_values(by=xcol,ignore_index=True)      # teljes átrendezés xcol szerint
            
            # # ycols nyomtatása
            # normfaktors=[]
            # for j,ycol in enumerate(ycols):
            #     def f_annot2(rec):
            #         if rec.type in ['gaussabs']: 
            #             x=servalue(tblL[xcol],rec.x,True)       # az xcol görbe értéke ugyanebben a pontban
            #             return f_annot_x(x,xcodelist,ylabel)
            #         else: return ylabel

            #     # diszkrét
            #     if discrete_x:
            #         FvPlot(tblL[ycol + '_mean'],'gauss',G=G,
            #                annotate='gaussabs' + str(gaussabs) + ' right',annotcaption=f_annot2)
            #         if scatter:
            #             FvPlot(tblL[ycol],plttype='scatter',annotate='')    # nem az átlagértékeket nyomtatja, hanem az eredeti pontokat
            #     else:   # folytonos
            #         FvPlot(tblL[ycol],plttype=plttype,G=G,
            #                annotate='gaussabs' + str(gaussabs) + ' right',annotcaption=f_annot2)

            #     serG=config.serplotlastG
            #     normfaktors.append(max(abs(serG.min()),abs(serG.max())))
            
            # # xcol nyomtatása (area)            
            # normfaktor=max(normfaktors)
            # plttypeL='gauss'
            # if discrete_x: plttypeL='original'    # diszkrét görbe esetén ne legyen simítás
            # FvPlot(tblL[xcol],plttype=plttypeL,G=G,label=xlabel,annotate='right',area=True,normalize=normfaktor,gaussside='toend')


            # corr_='corr = ' + strnum(tblL[ycol].corr(tblL[xcol],method=corrtype),'2g')
            # pltshowsub(xlabel='rendezett sorszám (' + xlabel + ', ' + strint(len(tblL)) + ')',ylabel=ylabel,
            #             commenttopright_out=corr_,yticklabel_map=ycodelist,annot_fontsize=8,
            #             ynumformat=ynumformat,xnumformat=xnumformat)



        if style_ in ['fx','both']:
            subplot_fx(tbl,xcol,ycols,title=title,G=G,scatter=scatter,normalize=normalize,
                       interval_y=interval_y,
                       minmax_annot=minmax_annot)
            # subplot_fx(tblL,xcol,ycols,title=title,G=G,scatter=scatter,normalize=normalize,
            #            xlabel=xlabel,ylabels=ylinelabels,
            #            xnumformat=xnumformat,ynumformat=ynumformat,interval_y=interval_y,
            #            xcodelist=xcodelist,ycodelist=ycodelist,minmax_annot=minmax_annot)

            # pltinitsub('next',title=title)

            # ser_kde=pd.Series()
            # for j,ycol in enumerate(ycols):
            #     ylinelabel=dget(ylinelabels,ycol,ycol,j)

            #     ser=serfromtbl(tblL,ycol,xcol)
            #     ser=ser.sort_index()        # fontos    Enélkül az interval eleve elszáll a FvPlot-ban

            #     def f_annot(rec):
            #         if rec.type in ['gaussabs']:  return f_annot_x(rec.x,xcodelist,ylinelabel)
            #         else: return ylinelabel

            #     # diszkrét görbe esetén átlagolások kellenek
            #     if discrete_x:
            #         ser_means=serfromtbl(tblL,ycol + '_mean',xcol)
            #         ser_means=ser_means.sort_index()
            #         FvPlot(ser_means,plttype='gauss',G=G,label=ylinelabel,
            #                annotate='gaussabs' + str(gaussabs) + ' right',annotcaption=f_annot,
            #                normalize=normalize)
            #         if scatter:
            #             FvPlot(ser,plttype='scatter',label='',annotate='',normalize=normalize)    # nem az átlagértékeket nyomtatja, hanem az eredeti pontokat

            #         # Diszkrét x-értékek esetén az átlagértékek is kinyomtatandók  (túl sok pont esetén zavaró lenne)
            #         if len(ycols_mean[j])<20: FvPlot(ycols_mean[j],plttype='scatter',label='',annotate='',normalize=normalize)

            #     # folytonos görbe
            #     else:
            #         FvPlot(ser,plttype=plttype,G=G,label=ylinelabel,
            #                 annotate='gaussabs' + str(gaussabs) + ' right',annotcaption=f_annot,
            #                 interval=interval,normalize=normalize)
                    
            #     ser_kde = ser_kde.append(ser)       # gauss simítás és átlagolás előtti pontok

            # plt.autoscale(enable=False)         # fontos, mert enélkül újraszámolja a az xlim, xlim határokat és a sáv nem fog a szélekig érni
            # FvPlot(ser_kde,'kde',normalize=normalize,kde_bandwidth=1.5)
            
            # # commenttopright=dget(comments,xcol,'',i)
            # corr_='corr = ' + strnum(tblL[ycol].corr(tblL[xcol],method=corrtype),'2g')

            # pltshowsub(ynumformat=ynumformat,xnumformat=xnumformat,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,
            #         ylabel=ylabel,xlabel=xlabel,commenttopright_out=corr_,xticklabel_map=xcodelist)

    commenttopright=dget(comments,'common','',len(xcols))       # key='common', vagy a lista utolsó helyén 
    if style_=='fx': commenttopright=stringadd(commenttopright,
                        'A szürkeárnyalatos foltok a pontsűrűséget jelzik.')

    pltshow(commenttopright=commenttopright)            


def depend_plot_multi(tbl,ycol,xcol,xcol2,xcol2_segments=4,G=0.1):
    '''
    Kétváltozós függések vizsgálata  (kötelezően fx diagram, ordered itt áttekinthetetlen lenne)
    Egyetlen diagram, több vonallal    
            
    xcol:  általában az ycol-lal legerősebb korrelációt mutató változót érdemes megadni
    xcol2:  egy további változó, aminek az értékkészletét egyenlő szegmensekre kell osztani és mindegyik
                szegmensre külön vonalat kell rajzolni
    '''
    
    tbl=tbl.set_index(xcol)

    # Pontok felosztása xcol2 szerint
    tbl = tbl.sort_values(by=xcol2)
    segment_len = len(tbl)//xcol2_segments

    pltinit(suptitle='Depend plot',
            title='"' + ycol + '" by "' + xcol + '" and "' + xcol2 + '"')
    for i in range(xcol2_segments):
        if i==xcol2_segments-1: ser = tbl[ycol].iloc[i*segment_len:]
        else: ser = tbl[ycol].iloc[i*segment_len:(i+1)*segment_len]
        min=tbl[xcol2].iloc[i*segment_len]
        max=tbl[xcol2].iloc[(i+1)*segment_len-1]

        ser=ser.sort_index()
        FvPlot(ser,'scatter gauss',G=G,label=xcol2 + ' [' + strnum(min) + ':' + strnum(max) + ']')                 
    pltshow(xlabel=xcol,
            ylabel=ycol)
                

                
def effect_plot(tbl,sortcol,ycols=None,    # Milyen hatása van a változónak más változókra
                style='ordered',G=0.1,gaussabs=None,
                scatter=False, minmax_annot='auto',

                suptitle=None,
                xlabel=None,ylabels=None,
                xnumformat=None,ynumformat=None,
                xcodelist=None,ycodelists=None,

                rowscols_aspect=(2,3),indexway='vert',sharex=False,sharey=False,
                nrows=None,ncols=None,      # ha nincs megadva, akkor subcount alapú hangolás
                width=None,height=None,     # ha nincs megadva, akkor subcount alapú hangolás
                comments=None):
    '''
    Egy kiinduló feature szerint rendezve mutatja a tábla összes rekordját (az x tengely a rekordokat reprezentálja),
        egybevetve a többi feature gauss-simított változásaival (külön subplotok-ban) 
    - együttmozgások és lokális maximumok/minimumok felderítésére alkalmas (a maxmin helyek feliratozhatók)
    - sublplotok a ycols oszlopokra, a korreláció erősséges szerint rendezve
    - a sorter lehet szöveges vagy kategória-oszlop is (dtype=integer). 
        A kvantálás lehet gyenge is (relevánsak a lokális távolságok, de a globális távolság ad-hoc jellegű; pl. irsz, teáor)
        Példa: a teáor diagram nagyjából megfelel egy pivotnak, ágazatonkénti csoportosítással, mean, count  (count-ra top15)
            - eltérés amiatt lehet, hogy az effect_plot a szomszédos teáorokra is átlagol (pl. kisker és nagyker szomszédos, de bizonyos
                jellemzői élesen eltérhetnek)

    tbl:  DataFrame
        - az indexe sorszám jellegű  (a sortcol-nak külön oszlopban kell lennie; előtte tbl.reset_index(drop=False) hívásra lehet szükség)
        - szöveges oszlopok is lehetnek benne (a sortcol és az ycols is lehet szöveges)
    sortcol:   float, int és szöveges oszlop is lehet (nem lehet maga az index; ha szükséges külön oszlopba kell tenni)
        - olyan oszlopot érdemes választani, amitől más oszlopok függnek
        - dátumoszlop is választható, de csak style='fx' -szel alkalmazható
    ycols:  a másodlagos oszlopok felsorolása (opcionális).  A y-oszlopok külön subplotokba kerülnek
        - 'top', 'top5':  a sortcol-lal legnagyobb korrelációjú adatok  (csak float vagy integer oszlopok)
        - '**' az elején:  filtercols() függvénnyel kiválasztott oszlopok (szöveges vagy dátum-oszlop is lehet)
        - lista is megadható (szöveges vagy dátumoszlop is lehet)
    G: gauss simítás erőssége. Kisebb értékkel érdemes kezdeni (pl. 0.03), és akkorára állítani, hogy a trendek és a fontosabb 
          lokális kiugrások is látszódjanak (ha fontot a szórás érzékeltetése vagy a nagyobb részletgazdagság fontos, akkor maradjon alacsony értéken)
    style:  'ordered':  az x-tengelyen xcol szerint rendezett sorszámok, az xcol is kirajzolva skálázással (area-plot)   
            'fx':   f(x) diagram,  x irányban outlier elhagyással, KDE árnyalással  (ha a sortcol egy dátumoszlop, akkor csak style='fx' merülhet fel)
            'both':   kétsoros megjelenítés (max 5 ycol), a felső sorban a 'ordered' plotok, az alsó sorban az 'fx' plotok

    sharey:  True esetén mindegyik diagram [0:1] [-1:0] vagy [-1:1] osztású y-tengelyt kap
                A dependent görbére max-min annotáció (az érték jelenik meg)
             False esetén a dependent adatsor határaival jelenik meg az y tengely
             Javasolt a True, mert kevesebb az annot-ytick átfedés, nem okozhat félreértést, hogy az y-tengely melyik
                görbéhez tartozik, a dependent ill. sorter görbe eltérő előjele esetén is értelmezhető az 
                eredmény, jobban összehasonlíthatóak a dependent görbék egymással

    comment:   a jobb felső sarokban megjelenő kisbetűs comment (több soros is lehet)
        - style='sub' esetén  dictionary is megadható, amiben a label-ekhez, továbbá a 'common' kulcshoz adható meg comment. 
    
    
    '''

    if style in ['fx','f(x)','yx','y(x)','y','xy']: style='fx'

    sortcol=Tblcol(tbl,sortcol,True)
    if not sortcol: return

    ycols=Tblcols(tbl,ycols)
    if len(ycols)==0:
        print('ERROR  effect_plot  ycols:  nem lett megadva egyetlen oszlop sem')
        return

    xcodelist=notnone(xcodelist,dget(Attr(tbl,'codelists'),sortcol))


    # szöveges sortcol kódolása
    if xcodelist is None and serdtype(tbl,sortcol)=='string':
        tbl,sortcol,xcodelist = Encode_col(tbl,sortcol)        # _encoded utótagot kap az oszlop

    tbl=tbl.dropna(subset=[sortcol])        # a sortcol=nan rekordok nem lesznek rajta a diagramokon
    # tábla rendezése a sortcol szerint (a rekordok újrasorszámozásával)
    tbl=tbl.sort_values(by=sortcol,ignore_index=True)          

    discrete_x = (tbl[sortcol].dtype in ['int32','int64'])
    interval_x=None
    if not discrete_x:           # csak folytonos görbe esetén kell outlier szűrés (fx plot esetén)
        xmin,xmax = fn_outlier_limits(tbl[sortcol].to_numpy())
        interval_x=(xmin,xmax)


    xlabel=notnone(xlabel, dget(Attr(tbl,'labels'),sortcol), sortcol)
    if comments is None: comment={}



    subcount=len(ycols)

    if style=='both':   # kétsoros elrendezés
        subcount=2*len(ycols)        
        if subcount>5: 
            print('FIGYELEM  effect_plot, style="both": max 5 ycol jeleníthető meg')
            subcount=10
            xcols=xcols[:5]
        rowscols_aspect=(2,0)
        indexway='vert'


    if not suptitle: 
        suptitle=xlabel + ' hatása más változókra'



    pltinit(subcount=subcount,rowscols_aspect=rowscols_aspect,nrows=nrows,ncols=ncols,indexway=indexway,
                sharey=sharey,sharex=sharex,
                suptitle=suptitle,
                width=width,height=height)
    
    if not gaussabs:
        gaussabs=np.select([subcount<=4,subcount<=8,subcount<=12],[16,12,8],default=4)

    for i,ycol in enumerate(ycols):
        if len(ycols)>3: progress('effect_plot:  ' + str(ycol))
        
        ylabel=notnone(dget(ylabels,ycol,None,i), dget(Attr(tbl,'labels'),ycol), ycol)
        ycodelist = notnone(dget(ycodelists,ycol,None,i), dget(Attr(tbl,'codelists'),ycol))
        if ycodelist is None and serdtype(tbl,ycol)=='string':
            tbl,ycol,ycodelist = Encode_col(tbl,ycol)        # _encoded utótagot kap az oszlop

        tblL=tbl

        # Diszkrét x esetén átlagolás kell az ismétlődő x-értékekre (a rekordszám kisebb az eredetinél)
        if discrete_x:
            ser_mean = tblL.groupby(by=sortcol).mean()[ycol]
            tblL=merge_ser(tblL,ser_mean,colname_new=ycol + '_mean',colnames_hiv=sortcol)   # tblL rekordszáma nem változik, de bekerül egy új oszlop


        if style in ['ordered','both']:
            subplot_ordered(tbl,sortcol,ycol,title=ylabel,comment='corr',G=G,scatter=scatter,xlabel=xlabel,ylabels=ylabels,
                        xnumformat=xnumformat,ynumformat=ynumformat,xcodelist=xcodelist,ycodelist=ycodelist,interval_x=interval_x,
                        minmax_annot=minmax_annot)

            # pltinitsub('next',title=ylabel)
            
            # # ycol nyomtatása
            # def f_annot2(rec):
            #     if rec.type in ['gaussabs']: 
            #         x=servalue(tblL[sortcol],rec.x,True)       # az xcol görbe értéke ugyanebben a pontban
            #         return f_annot_x(x,xcodelist,ylabel)
            #     else: return ylabel

            # # diszkrét x esetén
            # if discrete_x:
            #     FvPlot(tblL[ycol + '_mean'],'gauss',G=G,
            #             annotate='gaussabs' + str(gaussabs) + ' right',annotcaption=f_annot2)
            #     if scatter:
            #         FvPlot(tblL[ycol],plttype='scatter',annotate='')    # nem az átlagértékeket nyomtatja, hanem az eredeti pontokat
            # else:   # folytonos
            #     FvPlot(tblL[ycol],plttype=plttype,G=G,
            #             annotate='gaussabs' + str(gaussabs) + ' right',annotcaption=f_annot2)

            # serG=config.serplotlastG
            # normfaktor=max(abs(serG.min()),abs(serG.max()))
            
            # # xcol nyomtatása (area)            
            # plttypeL='gauss'
            # if discrete_x: plttypeL='original'    # diszkrét görbe esetén ne legyen simítás
            # FvPlot(tblL[sortcol],plttype=plttypeL,G=G,label=xlabel,annotate='right',area=True,normalize=normfaktor,gaussside='toend')

            # corr_='corr = ' + strnum(tblL[ycol].corr(tblL[sortcol],method=corrtype),'2g')
            # if subcount<=2: xlabel_=xlabel + ' szerinti rendezés (' + strint(len(tblL)) + ' rekord)'
            # elif subcount<=9: xlabel_=xlabel + ' rendezés (' + strint(len(tblL)) + ')'
            # else: xlabel_=''

            # pltshowsub(xlabel = xlabel_,ylabel=ylabel,
            #             commenttopright_out=corr_,yticklabel_map=ycodelist,annot_fontsize=8,
            #             ynumformat=ynumformat,xnumformat=xnumformat)

        if style in ['fx','both']:
            subplot_fx(tbl,sortcol,ycol,title=ylabel,comment='corr',G=G,scatter=scatter,xlabel=xlabel,ylabels=ylabels,
                        xnumformat=xnumformat,ynumformat=ynumformat,xcodelist=xcodelist,ycodelist=ycodelist,interval_x=interval_x,
                        minmax_annot=minmax_annot)

            # pltinitsub('next',title=ylabel)

            # ylinelabel=dget(ylabels,ycol,ycol,i)

            # ser=serfromtbl(tblL,ycol,sortcol)
            # ser=ser.sort_index()        # fontos    Enélkül az interval eleve elszáll a FvPlot-ban

            # def f_annot(rec):
            #     if rec.type in ['gaussabs']:  return f_annot_x(rec.x,xcodelist,ylinelabel)
            #     else: return ylinelabel

            # # diszkrét görbe esetén átlagolások kellenek
            # if discrete_x:
            #     ser_means=serfromtbl(tblL,ycol + '_mean',sortcol)
            #     ser_means=ser_means.sort_index()
            #     FvPlot(ser_means,plttype='gauss',G=G,label=ylinelabel,
            #             annotate='gaussabs' + str(gaussabs) + ' right',annotcaption=f_annot)
            #     if scatter:
            #         FvPlot(ser,plttype='scatter',label='',annotate='')    # nem az átlagértékeket nyomtatja, hanem az eredeti pontokat

            #     # Diszkrét x-értékek esetén az átlagértékek is kinyomtatandók  (túl sok pont esetén zavaró lenne)
            #     if len(ser_mean)<20: FvPlot(ser_mean,plttype='scatter',label='',annotate='')

            # # folytonos görbe
            # else:
            #     FvPlot(ser,plttype=plttype,G=G,label=ylinelabel,
            #             annotate='gaussabs' + str(gaussabs) + ' right',annotcaption=f_annot,
            #             interval=interval)
                
            # plt.autoscale(enable=False)         # fontos, mert enélkül újraszámolja a az xlim, xlim határokat és a sáv nem fog a szélekig érni
            # FvPlot(ser,'kde',kde_bandwidth=1.5)
            
            # # commenttopright=dget(comments,xcol,'',i)
            # corr_='corr = ' + strnum(tblL[ycol].corr(tblL[sortcol],method=corrtype),'2g')

            # pltshowsub(ynumformat=ynumformat,xnumformat=xnumformat,xmin=xmin,xmax=xmax,
            #         ylabel=ylabel,xlabel=xlabel,commenttopright_out=corr_,
            #         yticklabel_map=ycodelist,xticklabel_map=xcodelist)


    commenttopright=dget(comments,'common','')       # key='common' 
    if style=='fx': commenttopright=stringadd(commenttopright,
                        'A szürkeárnyalatos foltok a pontsűrűséget jelzik.','\n')

    pltshow(commenttopright=commenttopright)            
   




def scaled_plot(tbl,sortcol,dependentcols=None,      #  NINCS KÉSZ ordered plot, skálázott y görbékkel
                G=0.05,style='sub',annotpos='middle',annot_maxpos=0,annotcallback=None,
               outliers=None,drop_outliers=False,scatter=False,comment=None,
               sharey=True):    # adatsorok gyors áttekintése
    '''
    Egy kiinduló feature szerint rendezve mutatja a tábla összes rekordját (az x tengely a rekordokat reprezentálja),
        egybevetve a többi feature gauss-simított változásaival 
    - együttmozgások és lokális maximumok/minimumok felderítésére alkalmas (a maxmin helyek feliratozhatók)
    - alapsesetben sublplotok a dependent feature-ökre, a korreláció erősséges szerint rendezve
    - a sorter lehet kategória-feature is, de előtte kvantálni kell. A kvantálás lehet gyenge is (relevánsak a lokális távolságok, 
            de a globális távolság ad-hoc jellegű; pl. irsz, teáor)
        Példa: a teáor diagram nagyjából megfelel egy pivotnak, ágazatonkénti csoportosítással, mean, count  (count-ra top15)
            - eltérés amiatt lehet, hogy az effect_plot a szomszédos teáorokra is átlagol (pl. kisker és nagyker szomszédos, de bizonyos
                jellemzői élesen eltérhetnek)

    tbl:  DataFrame
        - csak a float vagy int oszlopokra készül diagram  (további szűrés kérhető a tblfiltercols függvénnyel)
           (kategórikus oszlopok csak valamilyen kvantálással kerülhetnek be a kimutatásba)
    sortcol:  float vagy int típusú oszlop a tbl-ben  (kvantált kategória-feature is lehet)
        - ha None, akkor a legnagyobb korrelációt mutató számoszlopokra jelennek meg az együttmozgás diagrammok
        - olyan oszlopot érdemes választani, aminek az értékétől kisebb-nagyobb függést mutat egy vagy több másik számoszlop
        - ha egyedi label kell, akkor egyelemű dictionary   (oszlopnév to label)
    dependentcols:  a másodlagos oszlopok felsorolása (opcionális)
        - ha nincs megadva, akkor a tbl összes float vagy int oszlopa, a sortcol-lal való korreláció szerint rendezve
        - vesszős felsorolás, vagy lista
        - ha egyedi label-ek kellenek, akkor dictionary   (oszlopnév to label)
    G: gauss simítás erőssége.  Kissebb értékkel érdemes kezdeni (pl. 0.03), és akkorára állítani, hogy a trendek és a fontosabb 
          lokális kiugrások is látszódjanak (ha a szórás érzékeltetése is fontos, akkor maradjon alacsony értéken)
    style:  'sub' (vagy 'subplots')  'area'  'line'   'firstarea'   'firstline'
        - 2-3-nál több másodlagos oszlop esetén 'sub' javasolt
    annpotpos:  x irányban hol jelenjen meg a feature-neveket jelző annotációk   (0 - 1, vagy 'middle')    
        - 'sub' esetén csak a sortcol-ra jelenik meg
        - összevont razolás esetén mindegyik vonalra  (a skálázás előtti értékekkel)
    annot_maxpos:   jelenítsen meg annotációt a sortcol görbéjén, a dependentcol max-helyeinél. Számot kell magadni: hány maxhely
                Csak style='sub' esetén van hatása. Ha >0, akkor a featurename-annotációk nem jelennek meg.
    outliers:  (nem érdemes használni)       None esetén nincs jelzés   
         0.01:       a leginkább kilógó 1%-t jelezze (method=isolation forest)
        'auto':     az isolation_forest algoritmus dönti el, hogy hány pontot tart outlier-nek
        'all' vagy 'quantile':      quantile algoritmus  (itt sem kontrollált a pontok száma)
    drop_outliers:  True esetén a skálázást az outlier-ek figyelmen kívül hagyásával számolja (quantile)
    scatter:  pontokat is rajzoljon (alapértelmezésben csak Gauss)
    sharey:  True esetén mindegyik diagram [0:1] [-1:0] vagy [-1:1] osztású y-tengelyt kap
                A dependent görbére max-min annotáció (az érték jelenik meg)
             False esetén a dependent adatsor határaival jelenik meg az y tengely
             Javasolt a True, mert kevesebb az annot-ytick átfedés, nem okozhat félreértést, hogy az y-tengely melyik
                görbéhez tartozik, a dependent ill. sorter görbe eltérő előjele esetén is értelmezhető az 
                eredmény, jobban összehasonlíthatóak a dependent görbék egymással

    comment:   a jobb felső sarokban megjelenő kisbetűs comment (több soros is lehet)
        - style='sub' esetén  dictionary is megadható, amiben a label-ekhez, továbbá a 'common' kulcshoz adható meg comment. 
    
    
    '''

    def f_outliers(ser):        # mask-tömböt ad vissza
        if outliers is None:
            return [True]*len(ser)
        else:
            arr=ser.to_numpy()
            if type(outliers)==str: 
                if outliers=='all' or beginwith(outliers,'quant'): 
                    outliers_mask=fn_outliers(arr,quote='auto',method='quantile',out_='mask')
                elif outliers=='auto': 
                    outliers_mask=fn_outliers(arr,quote='auto',method='isolation forest',out_='mask')
            elif type(outliers)==float:
                outliers_mask=fn_outliers(arr,quote=outliers,method='isolation forest',out_='mask')

            return outliers_mask


    # if scatter: sharey=False        # enélkül az aoutlier-ek hazavágnák az összes ábrát


    # annotcaption='{label}'
    # annotnormfaktor=None
    annotate=''
    if annotpos is not None:
        if annotpos=='middle': annotpos=0.5
        annotate=str(tbl.index.min() + int((tbl.index.max()-tbl.index.min())*annotpos))  

    if type(sortcol)==dict:
        for colname,label in sortcol.items():
            sortcol_=colname
            sortcol_label=label
    else:
        sortcol_=sortcol
        sortcol_label=sortcol

    tbl=tbl.dropna(subset=[sortcol])        # a sortcol=nan rekordok nem lesznek rajta a diagramokon

    # tábla rendezése a sortcol szerint (a rekordok újrasorszámozásával)
    tbl=tbl.sort_values(by=sortcol_,ignore_index=True)          

    ymin,ymax = None,None
    if drop_outliers:
        if Count(tbl>0)>0: ymax=1.1
        if Count(tbl<0)>0: ymin=-1.1


    ser_sorter=tbl[sortcol_] 

    # ser_sortG=FvGaussAvg(ser_sorter,G)  
    #       - a sorter-re semmiképpen nem indokolt a Gauss simítás (eleve rendezetten jelenik meg, másrészt torzítaná
    #           a maximumhelyek leolvasását)

    ser_sorter_N,faktor_sorter=normalizeSer(ser_sorter,faktorout=True,robust=drop_outliers)

    if outliers: outliers_mask_sort=f_outliers(ser_sorter)    # az eredeti értékekre nézi (gauss simítás előtt)


    if comment is None: comment={}

    sers=[]
    labels=[]
    corrs=[]
    if dependentcols is None:
        colnames=list(tbl)
        for colname in colnames:
            if colname==sortcol_: continue
            dtype=tbl.dtypes[colname].name
            if not (dtype in ['float32','float64','int64','int32']): continue
            labels.append(colname)
            sers.append(tbl[colname])
            corr_=tbl[sortcol_].corr(tbl[colname])
            corrs.append(abs(corr_))
            comment[colname]=stringadd(comment.get(colname),'corr = ' + strnum(corr_,'2g'),'\n')
        sortarrays3(corrs,sers,labels,True)
    else:
        if type(dependentcols)==str: dependentcols=dependentcols.split(',')
        if type(dependentcols)==list:
            for colname in dependentcols:
                dtype=tbl.dtypes[colname].name
                if not (dtype in ['float32','float64','int64','int32']): continue
                labels.append(colname)
                sers.append(tbl[colname])
        elif type(dependentcols)==dict:
            for colname,label in dependentcols.items():
                dtype=tbl.dtypes[colname].name
                if not (dtype in ['float32','float64','int64','int32']): continue
                labels.append(label)
                sers.append(tbl[colname])

    # sortcol előre   (plotstyle=firstarea megjelenítéshez)
    cols_to_front(tbl,sortcol)

    
    if style in ['sub','multi','subplots']: style='sub'

    count=len(sers)

    suptitle='Sorter: ' + sortcol_label
    comment['common']=stringadd(comment.get('common'),
                    'Az x-tengely az összes rekord megjelenik, egyenletes eloszlással, "' + sortcol_label + '" szerint rendezve.\n' +
                    'A szürke diagram a(z) "' + sortcol_label + '" értékeit mutatja (skálázva).',
                    '\n')


    if style=='sub':
        if count>15: count=15       # max 15 subplot
        pltinit(suptitle=suptitle,subcount=len(sers),rowscols_aspect=(1,2),indexway='vert',sharex=True,sharey=sharey)
    else:
        if len(sers)==1: title=label[0]
        else: title=None
        
        pltinit(suptitle=suptitle,title=title)



    # sortcol rajzolása
    if style!='sub':
        ser_sorter_plot=ser_sorter_N
        area = style in ['area','firstarea']
        FvPlot(ser_sorter_plot,plttype='original',label='',annotate='',area=area,positive=True,gaussside='')
        if outliers: FvPlot(ser_sorter_plot.iloc[outliers_mask_sort[:len(ser_sorter_plot)]],plttype='scatter',annotate='',colors={'color':'gray','alpha':0.1})


    for i,ser_dependent in enumerate(sers):
        if i>=count: break

        if i>=len(labels): label_=labels[-1]
        else: label_=labels[i]

        if serisempty(ser_dependent): continue

        ser_dependent=ser_dependent.dropna()

        # ser_dependent=ser_dependent.interpolate()       # nem nagyon van jobb módszer (azt feltételezi, hogy hasonlóan viselkedik mint a rendezés szerinti szomszédai)
        if outliers: outliers_mask=f_outliers(ser_dependent)    # az eredeti értékekre nézi (gauss simítás előtt)

        positive = ser_dependent.all()>=0
        ser_dependent_G=FvGaussAvg(ser_dependent,G)

        if style=='sub':
            annotcaption='{y_orig:.4g}'
            annotplus={}
            if annot_maxpos>0:
                annotate=''         # kapcsolja ki a normál feliratozást (túlzsúfolt lenne)
                arr_localmax = serlocalmax(ser_dependent_G,'max',mindistance=0.05,endpoints=True)
                arr_localmax_x=[str(localmax[0]) for localmax in arr_localmax]
                arr_localmax_x=arr_localmax_x[:annot_maxpos]
                for i,x in enumerate(arr_localmax_x):
                    if annotcallback: caption=annotcallback(servalue(ser_sorter,x))
                    else: caption=strnum(servalue(ser_sorter,x),'4g')
                    annotplus[float(x)]=caption
                    # annotplus[float(x)]='(' + str(i+1) + '.) ' + caption

                arr_localmin = serlocalmax(ser_dependent_G,'min',mindistance=0.05,endpoints=True)
                arr_localmin_x=[str(localmin[0]) for localmin in arr_localmin]
                arr_localmin_x=arr_localmin_x[:annot_maxpos]
                for i,x in enumerate(arr_localmin_x):
                    if annotcallback: caption=annotcallback(servalue(ser_sorter,x))
                    else: caption=strnum(servalue(ser_sorter,x),'4g')
                    annotplus[float(x)]=caption
                    # annotplus[float(x)]='(-' + str(i+1) + '.) ' + caption

                

            if sharey:          # 1-re kell normálni a sorter-t és a dependent-et is
                ser_sorter_plot=ser_sorter_N    
                ser_dependent_plot,faktor_dependent=normalizeSer(ser_dependent_G,faktorout=True)
                ser_dependent_scatter=ser_dependent * faktor_dependent   # Gauss simítás nélküli görbéből kell kiindulni

            elif sharey==False:         # a sortert kell dependent max értékére normálni
                # Kell egy normfaktor a dependent-re. Ezzel kell normálni a sorter-t
                ser_dependent_N,faktor_dependent=normalizeSer(ser_dependent_G,faktorout=True,robust=drop_outliers)
                ser_dependent_plot=ser_dependent_G
                faktor_sorter=faktor_dependent
                ser_sorter_plot = normalizeSer(ser_sorter,1/faktor_sorter)
                ser_dependent_scatter=ser_dependent   # Gauss simítás nélküli görbéből kell kiindulni
            
            
            pltinitsub(axindex='next',title=label_)
            
            # sorter rajzolása
            if not scatter:
                FvPlot(ser_sorter_plot,plttype='original',label=sortcol_label,annotate=annotate,
                        annotnormfaktor=faktor_sorter,annotplus=annotplus,annotcolorplus='gray',annotposplus='right bottom',
                        area=True,positive=True,gaussside='')
            elif scatter:
                FvPlot(ser_sorter_plot,plttype='original',label=sortcol_label,colors={'color':'gray', 'alpha':0.5})
            
            if outliers: FvPlot(ser_sorter_plot.iloc[outliers_mask_sort[:len(ser_sorter_plot)]],
                                plttype='scatter',annotate='',colors={'color':'gray','alpha':0.1})
            
            # dependent rajzolása
            if scatter: FvPlot(ser_dependent_scatter,plttype='scatter',annotate='',colors={'color':'navy','alpha':0.2})
            FvPlot(ser_dependent_plot,plttype='original',label='',annotate='maxabs',annotcaption='{y_orig:.4g}',
                    annotnormfaktor=faktor_dependent,annotpos='left bottom',
                    area=False,positive=positive,gaussside='')
            if outliers: FvPlot(ser_dependent_plot.iloc[outliers_mask],plttype='scatter',annotate='',colors={'color':'navy','alpha':0.1})

            tickpad=None
            if count>2: tickpad=1.5

            commentL=''
            if type(comment)==dict: commentL=comment.get(label_)          
            
            if sharey:      # ne látszódjon az 1, -1 mert félrevezető   (csak a 0 látszódjon)
                plt.gca().set_yticks(ticks=[0])
            
            if sharey and scatter and not drop_outliers:      # scatter esetén ymin-ymax-ra lehet szükség, mert a scatter max-értéke nagyságrendekkel lehet a Guass simított max felett
                if ser_dependent_scatter.max()>1.5: ymax=1.5      # az 1.5 egyfajta kompromisszum (látszódjon még valami a túllógásokból)
                if ser_dependent_scatter.min()<-1.5: ymin=-1.5

            pltshowsub(tickpad=tickpad,commenttopright_out=commentL,ymin=ymin,ymax=ymax,annot_fontsize=8)

        else:
            ser_dependent_plot,faktor=normalizeSer(ser_dependent_G,robust=drop_outliers,faktorout=True)
            serScatter=ser_dependent_plot*faktor

            area =  (style=='area') or (style=='firstarea' and i==0) or (style=='firstline' and i>0)
            FvPlot(ser_dependent_plot,plttype='original',label=label_,area=area,positive=True)
            if scatter: FvPlot(serScatter,plttype='scatter',annotate='',colors={'color':'navy','alpha':0.2})
            if outliers: FvPlot(ser_dependent_plot.iloc[outliers_mask],plttype='scatter',annotate='',colors={'color':'navy','alpha':0.1})

    
    commentL=''
    if type(comment)==dict: commentL=comment.get('common')
    else: commentL=comment          
    
    pltshow(commenttopright=commentL,ymin=ymin,ymax=ymax)



    # if type(sortcol)==list:
    #     sortcol0=sortcol[0]
    # else: sortcol0=sortcol

    

    # tbl=tbl.sort_values(by=sortcol,ignore_index=True)
    # tblinfo(tbl,'plot',G=G,normalize=1,annotate=annotpos,annotcaption='{label} ({y_orig:.3g})',
    #     plttpye='gausst',plotstyle='firstarea',
    #     suptitle='Adatsorok összehasonlítása',
    #     title='Rendezés "' + str(sortcol) + '" szerint')



def feature_plot(tbl,xcol,ycol=None,G=0.1,symm=True,scatter=False,   # egy feature jellemzői, opcionálisan összehasonlítás egy (vagy több) másikkal
                 minmax_annot='auto'):    
    '''
    xcol: egyetlen oszlop (lazy name
        - integer oszlopok megengedettek (nem kell unique-nak lennie). Megadható 
        - szöveges oszlop esetén létrehoz egy ABC rendezett kódjegyzéket, és a sorszámozott kódértékekkel dolgozik
            Általában jobb megoldás az előzetes kvantálás, figyelembe véve az előforduló értékek szemantikáját
            Előzetes kvantálás esetén megadható az oszlophoz tartozó kódjegyzék
        - dátum oszlop esetén datefloat konverzió
    ycol: nem kötelező, Tblcols paraméterezés. Ha nincs megadva, akkor csak scatter-gauss és hisztogram
        - legfeljebb egy másodlagos oszlopot vesz figyelembe
        - itt is megengedettek az integer (nem unique) és szöveges oszlopok

    G:  gauss simítás erőssége (a diagramszélességhez képest értendő).  A hisztogram-ra nem vonatkozik.
    symm:  szimmetrikus megjelenítés. 
            symm=True vagy symm=1 esetén: további két plot, ordered és dependency  megfordítva is   
                    (a Gauss áltagolás miatt a megfordított diagramok távolról sem egyszerű tükrözések)
            symm=2 esetén:   original és histogram az ycol-ra is (kicsit zsúfolt)
    scatter: True esetén az ordered és a dependency plot-okon is megjelennek a pontok (nem csak a gauss). A szórás érzékeltetésére használható.

    Az label,numformat,codelist paramétereket a tbl attribútumaként lehet előzetesen megadni  
        (tbl.labels,tbl.numformats,tbl.codelists, mindegyik dictionary oszlopnevekkel)            
    '''
    
    if symm==False: symm=0
    elif symm==True: symm=1

    xcol=Tblcol(tbl,xcol,True)
    if not xcol: return

    
    ycols=Tblcols(tbl,ycol)
    
    if len(ycols)>0:
        ycol=ycols[0]        
    else: 
        ycol=None
        symm=0

    xlabel=notnone(dget(Attr(tbl,'labels'),xcol), xcol)
    ylabel=notnone(dget(Attr(tbl,'labels'),ycol), ycol)

    if symm==0:
        pltinit(suptitle='Feature: ' + xlabel,
                nrows=2,ncols=2,height=0.85,width=0.6,top=0.86,hspace=0.36,sharex=False,sharey=False,indexway='vert')
    elif symm==1:
        pltinit(suptitle='Feature: ' + xlabel,
                nrows=2,ncols=3,height=0.85,width=0.9,left=0.06,right=0.94,top=0.86,hspace=0.36,sharex=False,sharey=False,indexway='vert')
    elif symm==2:
        pltinit(suptitle='Feature: ' + xlabel + '  --  ' + ylabel,
                nrows=2,ncols=4,height=0.85,width=1,left=0.05,right=0.95,top=0.86,hspace=0.36,sharex=False,sharey=False,indexway='vert')


    # original plot   ("érkezési" sorrend vagy idő szerinti rendezés)
    subplot_scattergauss(tbl,xcol,ycols,'original',G=G)


    # histogram
    subplot_hist(tbl[xcol],'histogram',label=xlabel)
    

    if symm>1 and len(ycols)>0:
        subplot_scattergauss(tbl,ycol,xcol,'original',G=G)

        # histogram
        subplot_hist(tbl[ycol],'histogram',label=ylabel)
        

    # ordered_plot    
    subplot_ordered(tbl,xcol,ycol,title=ylabel,comment='by ' + xlabel,G=G,scatter=scatter,
                minmax_annot=minmax_annot)



    
    # depend_plot
    subplot_fx(tbl,xcol,ycol,title=ylabel,G=G,scatter=scatter,xlabel=xlabel,ylabels=ylabel,
                comment='by ' + xlabel,minmax_annot=minmax_annot)



    if symm>0:
        # ordered_plot    
        subplot_ordered(tbl,ycol,xcol,title=xlabel,comment='by ' + ylabel,G=G,scatter=scatter,
                    minmax_annot=minmax_annot)

    
        # depend_plot
        subplot_fx(tbl,ycol,xcol,title=xlabel,G=G,scatter=scatter,xlabel=ylabel,ylabels=xlabel,
                    comment='by ' + ylabel,minmax_annot=minmax_annot)


    # pltshow(commenttopright='by ' + ylabel + '    (corr = ' + strnum(tbl[xcol].corr(tbl[ycol]),'2g') + ')')
    pltshow(commenttopright='by ' + ylabel)



def subplot_scattergauss(tbl,col,othercols=None,title=None,G=0.1):      # scatter és Gauss, összes rekordra, eredeti sorrend
    '''
    Scatter-gauss plot, szöveges oszlopra is működik, további oszlopokra is megjeleníthető gauss

    col: pontos oszlopnév  (todo)
    othercols:  Tblcols  
    '''
    title = notnone(title, col)

    label=notnone(dget(Attr(tbl,'labels'),col), col)
    numformat=dget(Attr(tbl,'numformats'),col)
    codelist = dget(Attr(tbl,'codelists'),col)


    if codelist is None and serdtype(tbl,col)=='string':
        tbl,col,codelist = Encode_col(tbl,col)        # _encoded utótagot kap az oszlop

    # original plot   ("érkezési" sorrend vagy idő szerinti rendezés)
    pltinitsub('next',title='original')
    FvPlot(tbl[col],plttype='scatter gauss',G=G,label=label,annotate='max',colors={'color':'gray'})
    if othercols is not None:
        plt.autoscale(enable=False)         # az y határok ne módosuljanak
        othercols=Tblcols(tbl,othercols)
        for othercol in othercols:
            otherlabel=notnone(dget(Attr(tbl,'labels'),othercol), othercol)
            FvPlot(tbl[othercol],plttype='gauss',G=G,label=otherlabel,annotate='max')
    
    pltshowsub(xlabel='eredeti sorszám (' + strint(len(tbl)) + ')',ylabel=label,commenttopright_out=label,
            yticklabel_map=codelist,ynumformat=numformat,annot_fontsize=8)


def subplot_hist_ser(ser,title,G=0.02,method='kde',annotbyindex=False,outliers='show',
                label=None,xlabel=None,comment=None,codelist=None, normalize_x=None,numformat=None,
                xmax=None,xmin=None,mean=False,median=False,iqr=False):   # hisztogram subplot rajzolása
    '''
    Benne van a pltinitsub és a pltshowsub is. Előtte pltinit, utána pltshow
    - gaussabs annotációk ("x=...")
    - kódlistás változóra az annotációkban és a tengely-jelölőkön a feliratok jelennek meg a számok helyett (rövidítésekkel)
    - text változóra automatikus kvantálás (ABC sorrend szerint)
    - outlierek jelzése, elhagyási lehetőséggel
    - alapesetben egyetlen ser, de több ser értékeloszlásának közös plot-on való megjelenítése is lehetséges

    ser:  lehet ser-tömb is  (hasonló jellegű ser-ek; ha eltérőek, akkor normalize_x=1 argumentummal tehetők összehasonlíthatóvá
    method:     'kde'
                'bins'
    annotbyindex:  True esetén a ser indexében lévő feliratok jelennek meg annotálásként 
            False esetén gaussabs,last,first annotálás (a subdiagramok számától függően x=...  vagy "label (..)" formátumban)
            - annotbyindex esetén a két szélső mindenképpen megjelenik, a többit pedig véletlenszerűen választja ki
            - integer is megadható:  hány felirat jelenjen meg  (ha csak True van megadva, akkor a függvény hangolja)
    outliers:  ha None, akkor nem jelennek meg az outlier jelölők   (quantile, de max 1% (IsolationForest) )
            'show':  kék pontok jelzik az outliereket a x irányban,      qauntile, de max 1% 
            'show_and_annot':  kék pontok és "outliers" annot az elsőnél és utolsónál 
            'drop':  x-irányban kidobja az outlier-eket,     y irányban pedig az átlagos sűrűség tízszerese az ymax
    label:  vonal-felirat(ok)   ser_tömb esetén ez is egy tömb
    codelist:  csak egy kódlista adható meg  (vagy közös mindegyik ser-re, vagy None)
    normalize_x:   az értékkészlet normalizálása (általában 1-re). Több ser esetén lehet értelme
    '''
    
    if type(ser)==list:
        sers=ser
        labels=label
    else:
        sers=[ser]
        labels=[label]

    if numformat is None: numformat='4g'

    ymax,ymin=None,None

    pltinitsub('next',title=title)

    area=True
    if len(sers)>1: area='colorline'    # ha diagramonként több vonal is van, akkor a vonalak legyenek színesek

    v_lines=[]
        
    for i,ser in enumerate(sers):
        if ser.count()==0: continue

        label=labels[i]

        # string-series kvantálása
        if codelist is None and serdtype(ser)=='string':
            ser,codelist_=Encode_ser(ser)
        else: codelist_=codelist

        discrete = ser.dtype in ['int32','int64']

        # Kódlistás esetben kell az integer kódok értékeloszlása az annotációk pontosításához
        if discrete:
            code_counts=ser.value_counts()       # az integer kódok hány rekordban fordulnak elő
            code_width=code_counts.index.max() - code_counts.index.min()   # az integer kódtartomány szélessége
            



        arr=ser.values

        if normalize_x:             # több értékeloszlás egy diagramban való megjelenítéséhez lehet szükséges
            ser=normalizeSer(ser,1)

        # outliers
        outliers_percent=0.01               # 'auto'
        outliers_method='quantile'          # másik: "isolation_forest"  "lof"
        # outliers_method='lof'          # másik: "isolation_forest"  "lof"
        outliers_=[]
        if outliers=='drop': arr=fn_outliers(arr,quote=outliers_percent,method=outliers_method,drop=True)
        elif beginwith(outliers,'show'): 
            outliers_=fn_outliers(arr,quote=outliers_percent,method=outliers_method)
            outliers_=np.sort(outliers_)


        subcount=Attr(plt.gcf(),'subcount')
        ncols=Attr(plt.gcf(),'ncols')
        # nrows=Attr(plt.gcf(),'nrows')

        if method=='kde':
            outpoints=int(Limit(500/ncols,min=100))         # ha túl kicsi, akkor egy auto-encoded változó csúcsainak szélessége ingadozni fog
            ser_hist=hist_KDE(arr,G,outpoints)
            plttype='original'
        elif method=='bins':
            ser_hist=hist_bybins(arr)
            plttype='gauss'
        else:
            print('ERROR   subplot_hist   method: "kde" vagy "bins" adható meg')
            return
        

        # ylabel beállítása
        ylabel=''           # zavarja az annotációk olvashatóságát
        # ylabel='Előfordulási gyakoriság'
        # if nrows>5: ylabel=''
        # elif nrows>2: ylabel='Gyakoriság'

        # annotate beállítása
        annot_fontsize=None
        if subcount>40: 
            annotate=''             # túl sok subplot esetén egyáltalán ne legyen annotáció
        elif subcount>20: 
            annotate='first last'             # csak first és last annotáció
        else: 
            n_gaussmax=np.select([subcount==1,subcount==2,subcount<=4,subcount<=8,subcount<=12,subcount<=20],
                                [20           ,16        ,12          ,8          ,6           ,5         ],    4)
            annotate='gaussmax' + str(n_gaussmax) + ' first last'
            # annotate='gaussabs'
            ymax=1.4                # férjen ki az annotáció
            if subcount>1: annot_fontsize=8
        # if len(sers)>1:
        #     annotate = annotate + ' right'

        # Annotáció rövidítés
        shorten=np.select([ncols>=4,ncols==3,ncols==2],[10,15,20],30)

        def f_annot(rec):
            if beginwith(rec.type,'gauss|first|last'):  
                if discrete: 
                    # pontosítás a value_counts alapján (a gauss simítás miatt az egyszerű kerekítés ritkán találja el a tényleges csúcsot)
                    width=G/2 * code_width
                    if width==0 or rec.type in ['first','last']:    # ha egyetlen kód van
                        code=int(round(rec.x))
                        if codelist_ is not None:
                            annot=dget(codelist_,key=code,index=code,default=label)
                        else:
                            annot=strint(code)
                        try: count_maxpos = code_counts.loc[code]
                        except: count_maxpos=None
                    else:
                        neighbors=code_counts.loc[code_counts.index>=(rec.x - width/2)]
                        neighbors=neighbors.loc[neighbors.index<=(rec.x + width/2)]
                        code_maxpos = neighbors.idxmax()
                        count_maxpos = neighbors[code_maxpos]
                        if codelist_ is not None:
                            annot=dget(codelist_,key=code_maxpos,index=code_maxpos,default=label)
                        else:
                            annot=strint(code_maxpos)
                    if len(sers)>1: annot=label + ' ' + annot
                    annot=txtshorten(annot,shorten)
                    # Ha nincs túl sok subplot, akkor megjeleníthető a darabszám is
                    if subcount<=6 and count_maxpos: annot = annot + ' (' + strint(count_maxpos) + ')'
                    return annot
                else: 
                    if len(sers)>1: annot =txtshorten(label,shorten) + ' (' + strnum(rec.x,numformat) + ')' 
                    else: annot = 'x=' + strnum(rec.x,numformat)
                    return annot
            else: return label

        if annotbyindex:
            serL = ser.sort_values()
            captions = list(serL.index)
            values = serL.values
            annotplus = []
            if type(annotbyindex)==list:        # lista megadása esetén csak a felsorolt feliratok jelennek meg (a két szélsőn kívül)
                for i_caption,caption in enumerate(captions):
                    # A két szélső értékhez mindenképpen kell felirat, és az érték is megjelenik
                    if i_caption==0 or i_caption==len(captions)-1: 
                        caption = txtshorten(caption, shorten)
                        caption = '!' + caption +  ' (' + strnum(values[i_caption],numformat) + ')'
                        # - a '!' jellel kérhető a kiemelt megjelenés (piros betűszín)
                    else:
                        if not (caption in annotbyindex): continue
                        caption = txtshorten(caption, shorten)
                        caption = caption +  ' (' + strnum(values[i_caption],numformat) + ')'
                    annotplus.append([values[i_caption],caption])
                
            else:                     
                if type(annotbyindex)==int: nMaxDb=annotbyindex
                else:
                    nMaxDb=np.select([subcount==1,subcount==2,subcount<=4,subcount<=8,subcount<=12,subcount<=20],
                                    [20           ,16        ,12          ,8          ,6           ,5         ],    4)
                if len(captions)<nMaxDb:  indexes = Range(0,len(captions)-1)
                else:
                    indexes = [0,len(captions)-1] + Range(1,len(captions)-2,space='random_unique',count=nMaxDb-2)
                    # - a két szélső mindenképpen legyen benne
                for i_caption in indexes: 
                    caption = txtshorten(captions[i_caption], shorten)
                    # A két szélen az érték is jelenjen meg
                    if i_caption==0 or i_caption==len(captions)-1: 
                        caption = '!' + caption + ' (' + strnum(values[i_caption],numformat) + ')'
                        # - a '!' jellel kérhető a kiemelt megjelenés (piros betűszín)
                    annotplus.append([values[i_caption],caption])

            FvPlot(ser_hist,plttype=plttype,G=G,label=label,annotate='',annotplus=annotplus,area=area,normalize=1,gaussside='')
        
        else:
            FvPlot(ser_hist,plttype=plttype,G=G,label=label,annotate=annotate,annotcaption=f_annot,area=area,normalize=1,gaussside='')

        if len(outliers_)>0: 
            annot=''
            if outliers=='show_and_annot': annot='first last'
            FvPlot(pd.Series([0]*len(outliers_),outliers_),plttype='scatter',label='outliers',
                   annot=annot,annotpos='right top',color='navy',alpha=0.1)
        # y=0 értékek (a pontok az x-tengelyen)

        # drop_outliers esetben y irányban is korlátozás:  az átlag-sűrűség max tízszereséig mutatja
        if outliers=='drop' and ser.dtype not in ['int32','int64']:
            mean=ser_hist.values.mean()
            max=ser_hist.values.max()
            if mean>0 and max/mean > 10: 
                ymax=(10 * mean)/max     # az átlagos sűrűség 10-szerese legyen a felső határ
                ymin=(-0.1 * mean)/max        # ne kerüljön legalulra a nullavonal

        if mean:
            ser_mean=ser.mean()
            v_lines = v_lines + [ddef(x=ser_mean)]      # annotáció a vonal közepén
            FvAnnotateAdd(x=ser_mean,y=ymax*0.5,color='green',caption='mean=' + strnum(ser_mean,numformat))
        if median:
            ser_median=ser.median()
            v_lines = v_lines + [ddef(x=ser_median)]    # annotáció a vonal közepe táján
            FvAnnotateAdd(x=ser_median,y=ymax*0.4,color='green',caption='median=' + strnum(ser_median,numformat))
        if iqr:
            iqr1 = ser.quantile(0.25)
            iqr2 = ser.quantile(0.75)
            FvPlot(ser_hist,plttype=plttype,G=G,interval=(iqr1,iqr2),area='noline',annot='',normalize=1)
            FvAnnotateAdd(x=iqr1,y=ymax*0.2,caption='IQR1=' + strnum(iqr1,numformat))
            FvAnnotateAdd(x=iqr2,y=ymax*0.2,caption='IQR2=' + strnum(iqr2,numformat))

                
    # xlabel beállítása
    if not xlabel:
        xlabel=''
        if len(sers)==1:
            xlabel=labels[0]
            if subcount>20: xlabel=''


    # commenttopright_out beállítása
    if not comment:
        comment=''  
        if len(sers)==1:         # több ser esetén nincs comment
            if title!=labels[0]:
                comment=labels[0]
            elif ncols<3:
                comment=strint(sers[0].nunique()) + ' féle érték'
            elif ncols==4:
                comment=strint(sers[0].nunique()) + ' érték'
            else:           # ha túl sok az oszlop, akkor ne jelenjen meg
                comment=''  

              
    pltshowsub(xlabel=xlabel,
               ymax=ymax,ymin=ymin,ylabel=ylabel,commenttopright_out=comment,
               yticklabels=False,xticklabel_map=codelist_,annot_fontsize=annot_fontsize,
               xnumformat=numformat, annot_count=100, xmax=xmax,xmin=xmin, 
               v_lines=v_lines)       # annotcount: ne legyen korlát            

def subplot_hist(tbl,cols,title,G=0.02,groups=None,method='kde',annotbyindex=False,
                outliers='show',labels=None,codelist=None,normalize_x=None,
                comment=None,numformat=None,xmax=None,xmin=None,mean=False,median=False,iqr=False):   # hisztogram subplot rajzolása
    '''
    Benne van a pltinitsub és a pltshowsub is. Előtte pltinit, utána pltshow
    - gaussabs ("x=...") vagy byindex annotációk.  Az annotbyindex=True változat a FvPlotBarH-t helyettesítheti
        nagy rekordszám esetén (pl. 20 feletti rekordszám esetén a FvPlotBarH áttkeinthetetlenné válhat)
    - kódlistás változóra az annotációkban és a tengely-jelölőkön a feliratok jelennek meg a számok helyett (rövidítésekkel)
    - text változóra automatikus kvantálás (ABC sorrend szerint)
    - outlierek jelzése, elhagyási lehetőséggel
    - több hasonló oszlop értékeloszlása is megjeleníthető
    - csoport-szűrési lehetőség, külön értékeloszlás-vonalakkal
    - a labels, numformats, codelists adatok előzetesen, a tbl attribútumaként is megadhatók (dict, key=colname)

    tbl:  
    cols:   általában egyetlen oszlop, de megadható több hasonló értékkészletű oszlop 
                - normalize_x=1 hívással erősen eltérő értékkészletek is összehasonlíthatók
    method:     'kde'
                'bins'
    annotbyindex:  True esetén a ser indexében lévő feliratok jelennek meg annotálásként 
            False esetén gaussabs,last,first annotálás (a subdiagramok számától függően x=...  vagy "label (..)" formátumban)
            - annotbyindex esetén a két szélső mindenképpen megjelenik, a többit pedig véletlenszerűen választja ki
            - integer is megadható:  hány felirat jelenjen meg  (ha csak True van megadva, akkor a függvény hangolja)
    outliers:  ha None, akkor nem jelennek meg az outlier jelölők   (quantile, de max 1% (IsolationForest) )
            'show':  kék pontok jelzik az outliereket a x irányban,      qauntile, de max 1% 
            'drop':  x-irányban kidobja az outlier-eket,     y irányban pedig az átlagos sűrűség tízszerese az ymax
    groups:  csoportosító oszlop, opcionálisan a megjelenítendő csoport-értékek felsorolásával  (szöveges vagy int oszlop is lehet) 
            - csak akkor érvényesül, ha egyetlen oszlop van megadva (cols)
            - a csoport-értékek száma legfeljebb 5 körüli legyen, mert efelett áttekinthetetlenné válik a diagram
        'country'     külön histogram-vonal a 'country' oszlop összes distinct értékére         
        'country:hun,cze,united king,slo,rom'    histogram-vonalak a country oszlop felsorolt értékeire  
        ['country:hun,cze','year:2010,2020']    több oszlopos csoportképzés, listával  (todo, ez még nem működik)
    labels:  több oszlop esetén ez is egy tömb.  groups esetén érdektelen (a csoportosító értékek lesznek a címkék)
    codelist:  csak egy kódlista adható meg  (vagy közös mindegyik ser-re, vagy None)

    normalize_x:   az értékkészlet normalizálása (általában 1-re). Több ser esetén lehet értelme (ritkán kell)
    mean,median:    boolean   True esetén egy függőleges vonal jelenik meg és annotációval megjelenik az érték is
    iqr:  boolean   True esetén a középső 50 % sötétebb háttérszínnel látszik és a két határnál annotáció jelenik meg
    '''
    
    cols=Tblcols(tbl,cols)
    if len(cols)==0: 
        print('subplot_hist  cols:  nincs ilyen oszlop')
        return

    # tbl-ben tárolt attribútumok bekeérése (a tbl.dropna művelettel törlődne, ezért itt kell bekérni)
    labels,numformats,codelists = Attrs(tbl,'labels,numformats,codelists')
    # - nincs használva



    codelist=dget(codelists,cols[0])
        
    if len(cols)==1 and groups is not None:
        groupcol,groupfilter = splitfirst(groups,':')
        dict_of_ser = tblgroupby(tbl,groupcol,cols[0],groupfilters=groupfilter)

        comment_default='by ' + groupcol

        sers=[]
        labels_=[]
        for groupvalue,ser in dict_of_ser.items():
            sers.append(ser)
            labels_.append(str(groupvalue))
            if len(dict_of_ser)==1: comment_default += ':  ' + str(groupvalue)
        
        if not comment: comment=comment_default

        subplot_hist_ser(sers,title=title,G=G,method=method,annotbyindex=annotbyindex,outliers=outliers,
                        label=labels_,xlabel=cols[0],comment=comment,
                        codelist=codelist,normalize_x=normalize_x,numformat=numformat,xmax=xmax,xmin=xmin,
                        mean=mean,median=median,iqr=iqr)


    else:
        sers=[]
        labels_=[]
        for col in cols:
            sers.append(tbl[col])
            labels_.append(dget(labels,col,col))
        subplot_hist_ser(sers,title=title,G=G,method=method,annotbyindex=annotbyindex,outliers=outliers,
                        label=labels_,codelist=codelist,normalize_x=normalize_x,comment=comment,numformat=numformat,
                        xmax=xmax,xmin=xmin,mean=mean,median=median,iqr=iqr)





def subplot_fx(tbl,xcol,ycols,title=None,comment='corr',G=0.1,scatter=False,normalize=None,interval_x='quantile',interval_y=None,
               xlabel=None,ylabels=None,xnumformat=None,ynumformat=None,xcodelist=None,ycodelist=None,minmax_annot='auto'):
    '''
    Szolgáltatások:
        - gauss mozgóátlag  (G effektív szélesség,  (0:1)-közötti float, a teljes szélességhez viszonyított arányszám)
        - szöveges xcol esetén automatikus kvantálás (ABC rendezéssel). Az annotációk és a ticklabel-ek is szövegesek lesznek.
            - a gauss simítást az x-értékenként átlagolt y-értékekre alkalmazza (enélkül ad-hoc jellegű lenne az x-értékek közötti összekötés helye)
            - ha az x-értékek száma <20, akkor nem csak a gauss jelenik meg, hanem a tényleges átlagértékeket is pontok jelzik
        - KDE háttérárnyékolás (pontsűrűség jelzése). Több ycol esetén az eredő pontsűrűség
        - max-min helyek annotálása
        - korreláció megjelenítése a diagram jobb felső sarkában (xcol és ycol közötti korreláció; több ycol esetén az első ycol-lal)
        - kérhető outlier-szűrés x-irányban (quantile alapú szűrés)
        - kérhető scatter  (nem csak a gauss jelenik meg, hanem az eredeti függvénypontok is)
        - a labels, numformats, codelists adatok előzetesen, a tbl attribútumaként is megadhatók (mindegyik dict, key=colname)
    
    xcol:  egyetlen oszlop pontos neve
    ycols:  egy vagy több oszlop  (Tblcols() hívás;  lehet pontos oszlopnév, **-kezdetű string, lista, rendezés az xcol-lal való kooreláció szerint,...)
        - elvárás, hogy az oszlopok értékkészlete és szemantikai tartalma hasonló legyen
    xcodelist,ycodelist:  dget formátumok (dict, ser, list, function)
    minmax_annot:  'auto' esetén a subplotok számától, az x-irányú sűrűségtől és az elvárt minimális kilengéstől függő megjelenítés
        - integer esetén a maximális minmax_annotációk száma
        - több ycol esetén csak az első ycol-ra jellennek meg minmax annotációk
    '''

    xcol=Tblcol(tbl,xcol,True)
    if not xcol: return
    ycols=Tblcols(tbl,ycols)

    if type(minmax_annot)==int: 
        if int(minmax_annot)<=0: gaussabs=''
        else: gaussabs='gaussabs' + str(minmax_annot)
    else: gaussabs = 'gaussabs'

    ylabel_common=dget(ylabels,'common',ycols[0],-1)
    if not title: title=ylabel_common


    pltinitsub('next',title=title)
    subcount=Attr(plt.gcf(),'subcount',1)

    
    # tbl-ben tárolt attribútumok bekeérése (a tbl.dropna művelettel törlőde, ezért itt kell bekérni)
    labels,numformats,codelists = Attrs(tbl,'labels,numformats,codelists')

    tbl=tbl.dropna(subset=[xcol])           # lehet, hogy copy kellene
    
    xlabel=notnone(xlabel, dget(labels,xcol), xcol)
    xnumformat=notnone(xnumformat, dget(numformats,xcol))
    xcodelist = notnone(xcodelist, dget(codelists,xcol))


    if xcodelist is None and serdtype(tbl,xcol)=='string':
        tbl,xcol,xcodelist = Encode_col(tbl,xcol)        # _encoded utótagot kap az oszlop


    discrete_x = (tbl[xcol].dtype in ['int32','int64'])

    if not interval_x or interval_x=='':
        xmin,xmax,interval_x=None,None,None
    elif interval_x=='quantile':
        xmin,xmax,interval_x=None,None,None
        if not discrete_x:           # diszkrét görbe esetén nem kell outlier szűrés
            xmin,xmax = fn_outlier_limits(tbl[xcol].to_numpy())
            interval_x=(xmin,xmax)
    else:
        xmin,mmax=interval_x

    ymin,ymax=None,None
    if interval_y is not None:
        ymin,ymax=interval_y



    ser_kde=pd.Series()
    for j,ycol in enumerate(ycols):
        ylabel=notnone(dget(ylabels,ycol,None,j), dget(labels,ycol), ycol)
        ynumformat=notnone(ynumformat, dget(numformats,ycol))
        ycodelist = notnone(ycodelist, dget(codelists,ycol))

        if j>0: gaussabs=''     # minmax annotáció csak az első görbére

        if ycodelist is None and serdtype(tbl,ycol)=='string':
            tbl,ycol,ycodelist = Encode_col(tbl,ycol)        # _encoded utótagot kap az oszlop

        ser=serfromtbl(tbl,ycol,xcol)
        ser=ser.sort_index()        # fontos    Enélkül az interval_x eleve elszáll a FvPlot-ban

        def f_annot(rec):
            if rec.type in ['gaussabs']:  
                if xcodelist is not None: 
                    x=int(round(rec.x))     # itt is hasznos lenne több szomszédos kódra végignézni
                    annot=dget(xcodelist,key=x,index=x,default=ylabel)
                    return txtshorten(annot,20)
                else: return 'x=' + strnum(rec.x,'4g')
            else: return ylabel

        # diszkrét görbe esetén átlagolások kellenek
        if discrete_x:
            ser_mean = tbl.groupby(by=xcol).mean()[ycol]
            if not ycol + '_mean' in tbl.columns:       
                tbl=merge_ser(tbl,ser_mean,colname_new=ycol + '_mean',colnames_hiv=xcol)   # tblL rekordszáma nem változik, de bekerül egy új oszlop
            ser_means=serfromtbl(tbl,ycol + '_mean',xcol)
            ser_means=ser_means.sort_index()
            FvPlot(ser_means,plttype='gauss',G=G,label=ylabel,
                    annotate=gaussabs + ' right',annotcaption=f_annot,
                    normalize=normalize)
            if scatter:
                FvPlot(ser,plttype='scatter',label='',annotate='',normalize=normalize)    # nem az átlagértékeket nyomtatja, hanem az eredeti pontokat

            # Diszkrét x-értékek esetén az átlagértékek is kinyomtatandók
            if len(ser_mean)<20:    # csak akkor, ha nem túl sok diszkrét x-érték van
                FvPlot(ser_mean,plttype='scatter',label='',annotate='',normalize=normalize)

        # folytonos görbe
        else:
            plttype='gauss'
            if scatter: plttype='scatter gauss'

            FvPlot(ser,plttype=plttype,G=G,label=ylabel,
                    annotate=gaussabs + ' right',annotcaption=f_annot,
                    interval=interval_x,normalize=normalize)
            
        ser_kde = pd.concat([ser_kde,ser])       # gauss simítás és átlagolás előtti pontok

    plt.autoscale(enable=False)         # fontos, mert enélkül újraszámolja a az xlim, xlim határokat és a sáv nem fog a szélekig érni
    FvPlot(ser_kde,'kde',normalize=normalize,kde_bandwidth=1.5)
    
    # commenttopright=dget(comments,xcol,'',i)
    if not comment or comment=='corr':
        comment='corr = ' + strnum(tbl[ycols[0]].corr(tbl[xcol]),'2g')

    pltshowsub(ynumformat=ynumformat,xnumformat=xnumformat,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,
            ylabel=ylabel,xlabel=xlabel,commenttopright_out=comment,
            xticklabel_map=xcodelist,yticklabel_map=ycodelist)
    

def subplot_ordered(tbl,sortcol,ycols,title=None,comment='corr',G=0.1,scatter=False,normalize=None,interval_x=None,interval_y=None,
                    minmax_annot='auto'):
    '''
    Két oszlop (sortcol és ycol) együttmozgásának jelzése, sortcol szerinti rendezéssel.
    Az x tengelyen a tbl indexe látható (alapesetben egyszerű sorszámozás, sortcol szerint rendezve)
    A sortcol area plotként jelenik meg.
    A sortcol és az ycol is lehet szöveges, vagy diszkrét. Szöveges oszlopra automatikus ABC rendezéses kódolás.

    A labels, numformats, codelists adatok előzetesen, a tbl attribútumaként is megadhatók (mindegyik dict, key=colname)
        - ha meg van adva az xlabel,xnumformat,cxodelist, ...  argumentum, akkor az argumentumnak van prioritása
        - ha nincs megadva label, akkor az oszlopnév jelenik meg
    
    sortcol:  egyetlen oszlop (lazy name)
    ycols:  egy vagy több oszlop  (Tblcols() hívás;  lehet pontos oszlopnév, **-kezdetű string, lista, rendezés az xcol-lal való kooreláció szerint,...)
        - elvárás, hogy az oszlopok értékkészlete és szemantikai tartalma hasonló legyen
    minmax_annot:  'auto' esetén a subplotok számától, az x-irányú sűrűségtől és az elvárt minimális kilengéstől függő megjelenítés
        - integer esetén a maximális minmax_annotációk száma
        - több ycol esetén csak az első ycol-ra jellennek meg minmax annotációk
    '''

    sortcol=Tblcol(tbl,sortcol,True)
    if not sortcol: return

    ycols=Tblcols(tbl,ycols)
    if len(ycols)==0: return

    if type(minmax_annot)==int: 
        if int(minmax_annot)<=0: gaussabs=''
        else: gaussabs='gaussabs' + str(minmax_annot)
    else: gaussabs = 'gaussabs'

    if scatter: normalize=None          # scatter esetén nem értelmezhető a normalize

    # tbl-ben tárolt attribútumok bekeérése (a tbl.dropna művelettel törlőde, ezért itt kell bekérni)
    labels,numformats,codelists = Attrs(tbl,'labels,numformats,codelists')

    
    if not title: 
        title=notnone(dget(labels,ycols[0]), ycols[0])


    pltinitsub('next',title=title)
    subcount=Attr(plt.gcf(),'subcount',1)


    tbl=tbl.dropna(subset=[sortcol])           # lehet, hogy copy kellene
    tbl=tbl.sort_values(by=sortcol,ignore_index=True)      # teljes átrendezés sortcol szerint


    xlabel=notnone(dget(labels,sortcol), sortcol)
    xnumformat=dget(numformats,sortcol)
    xcodelist = dget(codelists,sortcol)
    
    if xcodelist is None and serdtype(tbl,sortcol)=='string':
        tbl,sortcol,xcodelist = Encode_col(tbl,sortcol)        # _encoded utótagot kap az oszlop
    
    discrete_x = (tbl[sortcol].dtype in ['int32','int64'])


    for j,ycol in enumerate(ycols):
        if ycol=='gázátadó':
            print('hahó')

        ylabel=notnone(dget(labels,ycol), ycol)
        ynumformat=dget(numformats,ycol)
        ycodelist = dget(codelists,ycol)


        if j>0: gaussabs=''     # minmax annotáció csak az első görbére

        if ycodelist is None and serdtype(tbl,ycol)=='string':
            tbl,ycol,ycodelist = Encode_col(tbl,ycol)        # _encoded utótagot kap az oszlop

        # ycol nyomtatása
        def f_annot(rec):
            if rec.type in ['gaussabs']: 
                x=servalue(tbl[sortcol],rec.x,True)       # az xcol görbe értéke ugyanebben a pontban
                if xcodelist is not None: 
                    annot=dget(xcodelist,key=x,index=x,default=ylabel)
                    return txtshorten(annot,20)
                else: return 'x=' + strnum(x,'4g')
            else: return ylabel

        # diszkrét x esetén
        if discrete_x:
            ser_mean = tbl.groupby(by=sortcol).mean()[ycol]
            if not ycol + '_mean' in tbl.columns:       
                tbl=merge_ser(tbl,ser_mean,colname_new=ycol + '_mean',colnames_hiv=sortcol)   # tblL rekordszáma nem változik, de bekerül egy új oszlop
                # középindexek kitalálása
                tbl['index_copy']=tbl.index
                arr_középindex = tbl.groupby(by=sortcol).median()['index_copy'].apply(int)
                ser_mean=pd.Series(ser_mean.values,arr_középindex)                
            FvPlot(tbl[ycol + '_mean'],'gauss',G=G,
                    annotate=gaussabs + ' right',annotcaption=f_annot,normalize=normalize)
            if scatter:
                FvPlot(tbl[ycol],plttype='scatter',annotate='')    # nem az átlagértékeket nyomtatja, hanem az eredeti pontokat

            # Diszkrét x-értékek esetén az átlagértékek is kinyomtatandók
            if len(ser_mean)<20:    # csak akkor, ha nem túl sok diszkrét x-érték van
                FvPlot(ser_mean,plttype='scatter',label='',annotate='',normalize=normalize)

        else:   # folytonos
            plttype='gauss'
            if scatter: plttype='scatter gauss'
            FvPlot(tbl[ycol],plttype=plttype,G=G,
                    annotate=gaussabs + ' right',annotcaption=f_annot)

        serG=config.serplotlastG
        normfaktor=max(abs(serG.min()),abs(serG.max()))
        
        # xcol nyomtatása (area)            
        plttype='gauss'
        if discrete_x: plttype='original'    # diszkrét görbe esetén ne legyen simítás
        FvPlot(tbl[sortcol],plttype=plttype,G=G,label=xlabel,annotate='right',area=True,normalize=normfaktor,gaussside='toend')

    if not comment or comment=='corr':
        comment='corr = ' + strnum(tbl[ycols[0]].corr(tbl[sortcol]),'2g')

    if subcount<=2: xlabel_=xlabel + ' szerinti rendezés (' + strint(len(tbl)) + ' rekord)'
    elif subcount<=9: xlabel_=xlabel + ' rendezés (' + strint(len(tbl)) + ')'
    else: xlabel_=''

    xmin,xmax=None,None
    if interval_x is not None:
        xmin,xmax=interval_x
    ymin,ymax=None,None
    if interval_y is not None:
        ymin,ymax=interval_y

    pltshowsub(xlabel = xlabel_,ylabel=title,
                commenttopright_out=comment,yticklabel_map=ycodelist,annot_fontsize=8,
                ynumformat=ynumformat,xnumformat=xnumformat,
                xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax)




# CORR PLOT, KORRELÁCIÓK
def FvCorrPlot(tbl,format='simple',order=False,**pltparams):
    '''
    format:
      'simple'      seaborn default formázás
      'triangle'    annotálásokkal, alul 10-ig lépcsőzetes annotálások, 10 felett ferde feliratok
    '''
    
    tblcorr=tbl.corr()

    if order:
        tblcorr['corrsum']=tblcorr.sum()
        tblcorr=tblcorr.sort_values(by='corrsum')
        tblcorr=tblcorr[tblcorr.index.array]        # oszlopok átrendezése a sorok sorrendje alapján ('corrsum' oszlop kimarad)

    if format=='simple':
        pltinit(pltparams)
        sb.heatmap(tblcorr, cmap="BrBG_r",vmin=-1, vmax=1)
        pltshow()
    
    elif format=='normal':
        dsetsoft(pltparams,suptitle='Korrelációk',height=0.8,left=0.25,bottom=0.3,right=0.97)
        pltinit(**pltparams)
        numcols=len(list(tblcorr))

        aCaption=tblcorr.index.array
        sb.heatmap(tblcorr, cmap="BrBG_r",vmin=-1, vmax=1)
        plt.xticks(rotation=-35,ha='left',fontsize=8)
        plt.tick_params(axis='x', which='major', pad=1)
        
        pltshow()      # annot_ystack: lépcsőzetes elrendezés 12-es csoportokban        


    elif format=='triangle':
        print('pltparams:' + str(pltparams))
        dsetsoft(pltparams,suptitle='Korrelációk',height=0.8,left=0.25,bottom=0.3,right=0.97)
        print('pltparams:' + str(pltparams))
        pltinit(**pltparams)
        numcols=len(list(tblcorr))

        aCaption=tblcorr.index.array
        mask = np.triu(np.ones_like(tblcorr))
        mask = mask[1:, :-1]
        tblcorr = tblcorr.iloc[1:,:-1].copy()
        sb.heatmap(tblcorr, cmap="BrBG_r",mask=mask,vmin=-1, vmax=1,xticklabels=(numcols>12))
        # anotációk az átlóban
        for i,caption in enumerate(aCaption):
            if i>0 and i<len(aCaption)-1: FvAnnotateTwo((0.5 + i,i),(i,i-0.5),caption,5,8)
        
        # xlabels
        y0=len(aCaption)-1    # a seaborn diagram bal felső sarka az origo
        if numcols<=12:
            # x tengely feliratozása
            for i,caption in enumerate(aCaption):
                FvAnnotateAdd(0.5 + i,y0,caption,'bottom right')    
        else:
            plt.xticks(rotation=-35,ha='left',fontsize=8)
            plt.tick_params(axis='x', which='major', pad=1)
        
        pltshow(annot_fontsize=8,annot_yoffset=10,annot_ystack=(y0,12))      # annot_ystack: lépcsőzetes elrendezés 12-es csoportokban        

def corr(x,y):
    # Az nan értékeket nem szereti
    # pvalue is kérhető (hipotézis elemzéshez használható)

    # A Seires-nek és a DataFrame-nek is van corr() függvénye. A pandas függvény annyiban többet tud, hogy eltérő hosszúságú vagy 
    #    részben eltérő index-állományú sorozatokat is képes kezelni  (előzetes összegigazítás)

    return pearsonr(x, y)[0]

def Topcorrs(tbl,col_with,count=None,method='pearson'):    # top[10] legerősebben korreláló számoszlop, egy adott oszloppal 
    '''
    A korreláció abszolút értékét nézi (negatív is lehet a korreláció)
    metod:  'pearson', 'kendall', 'spearman'

    return:  oszlopnevek listája  (top10 korreláló oszlop)
    '''
    
    topcols,topcorrs_=[],[]
    for colname in list(tbl):
        if colname==col_with: continue
        dtype=tbl.dtypes[colname].name
        if not (dtype in ['float32','float64','int64','int32']): continue
        corr_=abs(tbl[col_with].corr(tbl[colname],method=method))
        if np.isnan(corr_): continue
        topcols.append(colname)
        topcorrs_.append(corr_)
    sortarrays(topcorrs_,topcols,reverse=True)

    if count: topcols=topcols[:count]
    return topcols





def FvCrosscorr(tbl1,country1:str,tbl2,country2:str,par='',datumfirst='',omit=0):      # ->(lag,korr,ser,tblout)
    # tbl1,tbl2:  pl. tbl_cases, tbl_deaths
    # par: "grad":  a gradiensre nézze a keresztkorrelációt
    # datumfirst:   milyen dátumtól
    # omit:  hány napot hagyjon el a szélekről

    # ha lag negatív, akkor a country2 felfutásai általában megelőzik a country1 felfutásait

    smoothhalf=20
    smoothlen=2*smoothhalf

    if not datumfirst:
        datumfirst=tbl1.index.min()+pd.DateOffset(omit)
        # Az első nem NaN record keresése
        for i in range(len(tbl1.index)):
            if not np.isnan(tbl1[country1][i]) and not np.isnan(tbl2[country2][i]):
                if i>smoothhalf: 
                    datumfirst=tbl1.index[i]
                break

    datumlast=tbl1.index.max()-pd.DateOffset(omit)

    ser1=tbl1.loc[datumfirst:datumlast,country1]
    ser2=tbl2.loc[datumfirst:datumlast,country2]       # kipróbáltam negatív előjellel is, de a korreláció (az előjelet leszámítva) ugyanaz lett

    if par=="grad":
        ser1 = pd.Series(FvNormalize(np.gradient(ser1.to_numpy())),ser1.index)
        ser1 = ser1.rolling(smoothlen, min_periods=1, center=True, win_type='gaussian').mean(std=smoothlen/4)
        ser2 = pd.Series(FvNormalize(np.gradient(ser2.to_numpy())),ser2.index)
        ser2 = ser2.rolling(smoothlen, min_periods=1, center=True, win_type='gaussian').mean(std=smoothlen/4)
    
    tblout=pd.DataFrame({
        tbl1.name + '.' + country1:ser1,
        tbl2.name + '.' + country2:ser2
    })

    #print(ser1)
    #print(ser2)

    # Keresztkorreláció  (ccf: cross correlation function)
    forwards=stat.ccf(ser2,ser1,adjusted=False,fft=False)         # jó lenne, ha meg lehetne adni a lag-tartományt, de a ccf nem tud ilyesmit
    backwards=stat.ccf(ser1,ser2,adjusted=False,fft=False)[::-1]      # megfordítás
    backwards=backwards[:-2]        # lag=0 nem kell  (átfedés a forwards-szel)

    ser=pd.concat([pd.Series(backwards,index=range(-len(backwards),0)),pd.Series(forwards,index=range(0,len(forwards)))])
    if country1!=country2 or tbl1.name!=tbl2.name:
        ser=ser.loc[-100:100]        # 100 nap feletti korreláció nem érdekel (kivéve: autokorreláció)

    if par=='onlypositive': ser=ser.loc[0:]

    return (ser.idxmax(),ser.max(),ser,tblout)






# SCATTER PLOT

def FvScatterPlot(maxsize=1000,defaultsize=100,alpha=0.7,cmap='viridis',colorbar={"label":""},colordefault=None,colorminmax=None):
    '''   Buborék diagram rajzolása.  Előtte FvBubbleAdd sorozat
    A buborékoknak yx koordinátája, mérete, színe és felirata van.  
    A színt és a méretet egy-egy számadat határozza meg (FvBubbleAdd)
    A méretnek pozitívnak kell lennie. Nem kell előzetesen skálázni.
    A szín lehet negatív is, és itt sincs szükség előzetes skálázásra. Diszkrét értékekek is lehetnek.
    
    maxsize:  mekkora legyen a legnagyobb pont mérete (points^2-ben mérve) 
    defaultsize:  mekkora legyen a size=None pontok mérete  (points^2-ben mérve)
    colordefault: color=None pontok színe  (nem float értéket kell megadni).
     - None esetén a színciklus szerinti következő szín 
    colorminmax: (min,max)  Csak akkor kell megadni, ha nem megfelelő az aColor-ban megadott float-tömb min-max értéke
    cmap:  
    colorbar:  dict(label,numformat)     Jobb szélen megjelenő colorbar (cmap alkalmazása esetén sem kötelező megjeleníteni)
    '''


    aX,aY,aColor,aSize,aAnnot=unzip(config.aScatterG)       # a FvScatterAdd ebben a tömbbe írta a buborékok paramétereit
    
    aColor=list(aColor)
    aSize=list(aSize)
 
    # Ellenőrzés: van-e legalább kettő nem None eleme az aColor-nak
    nColorDb=0
    for i in range(len(aColor)):
        if aColor[i]==None or pd.isna(aColor[i]): aColor[i]=np.nan
        else: nColorDb+=1
    if nColorDb==0: aColor=colordefault

    if cmap and nColorDb>0 and nColorDb<len(aColor):
        cmap=plt.colors.colormap(cmap)
        cmap.set_bad(color=colordefault,alpha=0.5)
    
    vmin=None
    vmax=None
    if colorminmax:   
        vmin=colorminmax[0]
        vmax=colorminmax[1]

    
    sizemax=0
    for i in range(len(aSize)):
        if not aSize[i]: aSize[i]=defaultsize
        if aSize[i]>sizemax: sizemax=aSize[i]
    for i in range(len(aSize)):
        aSize[i]=(aSize[i]/sizemax)*maxsize

    #RAJZOLÁS
    plt.scatter(aX,aY,s=aSize,c=aColor,cmap=cmap,alpha=alpha)


    # Colorbar rajzolás
    if (nColorDb>0 or colorminmax) and colorbar:
       label,numformat = dgets(colorbar,'label,numformat')
       if numformat: format=mpl.ticker.StrMethodFormatter(FvNumformat(numformat,'x'))
       else: format=None
       plt.colorbar(label=label,format=format)


    # Annotálás
    if aAnnot:
        for i in range(len(aAnnot)):
            if aAnnot[i]: FvAnnotateAdd(aX[i],aY[i],aAnnot[i],color=color_darken(aColor[i],0.5))
    

def FvScatterAdd(x,y,color=None,size=None,annot=None):
    # Egy pont (kör) hozzáadása a későbbi FvScatterPlot() rajzoláshoz
    
    # x,y:  koordináták  (az x dátum (időpont) is lehet, bár scatter diagramoknál float a jellemző)
    # color, size: nem kell normalizálni (a rajzoláskor lesz normalizálva; tetszőleges járulékos float adat)
    # - a size csak pozitív lehet (vagy None) 
    # - ha valamelyik pontnál nincs megadva (None), akkor a FvScatterPlot() híváskor megadott default szín illetve méret érvényesül
    # annot: nem kötelező

    if size and size<=0: return      # csak pozitív érték adható meg (teljesen kimarad a pont)

    config.aScatterG.append((x,y,color,size,annot))


# BAR PLOT

def FvPlotAreaBars(ser_label_to_float,sort='descending',shorten=None,captionfilter=None,numformat='3g',
                    x1=0,step=1,gap=0.2,hide_grid=True):
    '''
    Függőleges oszlopok, annotációkkal
    FvPlot sorozat, vízszintes szakaszokkal (area plot), a szakasz közepén annotációval
    - a címkék csak annotációként jelennek meg, az x tengelyen számértékek vannak
    - pltinit - pltshow keretben kell hívni    
    - subplotban is alkalmazható, akár más FvPlot hívásokkal együtt is

    numformat: ha nem üres string, akkor zárójelben kiírja a számértéket is a label után
    '''

    X=Range(x1,add=step,count=len(ser_label_to_float))
    halfwidth = step/2 - gap/2
           
    if sort=='ascending': ser_label_to_float=ser_label_to_float.sort_values(ascending=True)
    elif sort=='descending': ser_label_to_float=ser_label_to_float.sort_values(ascending=False)

    labels = list(ser_label_to_float.index)
    Y=ser_label_to_float.values
    for i in range(len(ser_label_to_float)):
        serL=pd.Series([Y[i],Y[i]],[X[i]-halfwidth,X[i]+halfwidth])

        label,annot = '',''
        annotcolor=None
        if captionfilter is None or labels[i] in captionfilter or i==0 or i==len(X)-1:    # a széleken lévők mindenképpen
            label=labels[i]    
            if shorten: label=txtshorten(label,shorten)
            if numformat!='': 
                label = label + ' (' + strnum(Y[i],numformat) + ')'
            annot='middle'
            if i==0 or i==len(X)-1: annotcolor='red'
                    
        FvPlot(serL,'original',area=True,label=label,annot=annot,annotcolor=annotcolor)
       
    if hide_grid:
        plt.gca().grid(visible=False)
        plt.gca().get_xaxis().set_ticklabels([])    

              
def FvPlotBarH(ser,alphas=None,annotfontsize=8,annotgap=5,annotformat='2f',alphamin=0.2,
        shorten=None,cutright_=None,alphas_out=False,captionfilter=None):
    '''
    Horizontális sávok nyomtatása (utána még pltshow kell)
        Bal szélen kategóriafeliratok (a ser indexéből), mellettük az értékekeknek megfelelő szélességű sávok.
    Subplotokban is alkalmazható  (a megszokott title, commenttopright_out és hasonló paraméterezésekkel)
    Ha túl nagy a rekordszám (pl. >20), akkor a hist_plot függvény is alkalmazható, annotbyindex=True beállítással

    ser:  kategóriafelirat to value      Alulról felfelé jelennek meg a kategóriák (a ser előzetes rendezésénél ügyelni kell erre)
    alphas: None esetén nincs halványítás (midegyik sáv átlátszatlan)
       - a tömb méretének egyeznie kell a ser méretével
       - pl. a válaszok száma adható meg  (a függvény fogja átkonvertálni a [0:1] tartományba; minél kevesebb válasz, annál halványabb)
    alphas_out:  az alphas értékek hozzáfűzése a bal oldali feliratokhoz (zárójelben)
    annotfontsize: 0 esetén nem jelenik meg annotáció a sávok jobb szélén
    alphamin:   az [alphamin,1] tartományba konvertálja az alphas értékeket
    shorten:  a feliratok szükség szerinti rövidítése (ha hosszabb, akkor a végén ...)
    cutright:  megadható egy minta (string), amit minden felirat végéről le kell vágni (ellenőrzi, hogy valóban ezzel végződik)
    captionfilter: csak a felsorolt feliratok jelenjenek meg, a többinél csak a sáv látszik
        - 
    '''
    alphas=np.array(alphas)             # általában count jellegű  (később át lesz konvertálva a [0:1] tartományba)


    bars_to_annot=None
    if captionfilter is not None:
        captionfilter=[Lics(x) for x in captionfilter]
        
        index_mod=[]
        bars_to_annot = []
        for i,index_value in enumerate(ser.index): 
            caption=index_value
            captionL=Lics(caption)
            annot=True
            if not (captionL in captionfilter): 
                caption = ''
                annot = False
            index_mod.append(caption)
            bars_to_annot.append(annot)
        ser.index=index_mod
        
        
    if shorten:
        index_mod=[]
        for i,index_value in enumerate(ser.index): 
            if index_value!='':
                caption=txtshorten(index_value,shorten)
                if alphas_out and i<len(alphas): caption = caption + ' (' + str(alphas[i]) + ')' 
            else: caption=''
            index_mod.append(caption)
        ser.index=index_mod
    elif cutright_:
        index_mod=[]
        for i,index_value in enumerate(ser.index): 
            if index_value!='':
                caption=cutright(index_value,cutright_,True)
                if alphas_out and i<len(alphas): caption = caption + ' (' + str(alphas[i]) + ')' 
            else: caption=''
            index_mod.append(caption)
        ser.index=index_mod
    elif alphas_out:
        index_mod=[]
        for i,index_value in enumerate(ser.index): 
            if index_value!='':
                caption=index_value
                if alphas_out and i<len(alphas): caption = caption + ' (' + str(alphas[i]) + ')' 
            else: caption=''
            index_mod.append(caption)
        ser.index=index_mod
    
    ser.plot.barh()
    FvAnnotateBarH(fontsize=annotfontsize,gap=annotgap,numformat=annotformat,bars_to_annot=bars_to_annot)

    # alpha értékek transzformálása 0.2 és 1 közé
    try:
        if alphas.min() < alphas.max()*alphamin:
            alphas = alphas + (1+alphamin)*(alphas.max()*alphamin - alphas.min())
        alphas = alphas / alphas.max()
        alphas[alphas<alphamin]=alphamin        # kisebb kilógás még előfordulhat
        FvFadeBarH(alphas)
    except:
        pass

    plt.gca().grid(visible=False)
    plt.gca().get_xaxis().set_ticklabels([])    

def FvAnnotateBarH(fontsize=8,gap=5,numformat='2f',bars_to_annot=None):
    '''
    Horizontális oszlopdiagram annotálása
    gap:  point-ban
    decimal:  hány tizedesjegy jelenjen meg 
    bars_to_annot:  egy boolean list adható meg, összehangolva a sávokkal

    Kiindulópont:  https://stackoverflow.com/questions/28931224/adding-value-labels-on-a-matplotlib-bar-chart
    '''


    rects = plt.gca().patches

    # For each bar: Place a label
    for i_rect,rect in enumerate(rects):
        if bars_to_annot is not None and not bars_to_annot[i_rect]: continue
        # Get X and Y placement of label from rect.
        x_value = rect.get_width()
        y_value = rect.get_y() + rect.get_height() / 2

        # Vertical alignment for positive values
        ha = 'left'
        # If value of bar is negative: Place label left of bar
        if x_value < 0:
            gap *= -1
            ha = 'right'

        formatstring=FvNumformat(numformat)
        label = formatstring.format(x_value)

        plt.annotate(label,(x_value, y_value),     
            xytext=(gap, 0), textcoords="offset points", 
            va='center', ha=ha, fontsize=fontsize)

def FvFadeBarH(alphas):   # alpha értékek beállítása az egyes sávokra
    '''
    alphas:  az elemszáma egyezzen meg a sávok számával
    '''
    rects = plt.gca().patches

    # For each bar: Place a label
    for i,rect in enumerate(rects):
        rect.set(alpha=alphas[i])



# ANNOTATE
def FvAnnotateAdd(x,y,caption,position='right bottom',color=None,annottype=''): 
    '''
    # x,y:  az annotálandó pont helye (eredeti koordinátákban; az x általában dátum)
    #     Ha y irányban kell a felső határ:   plt.ylim()[1]    (kisebb eltolódás előfordulhat)
    # caption: felirat  (több soros is lehet \n jelekkel)
    #     példa:  egy görbe jobb szélső pontja    
    #         caption = '... (' + '{:.1f}'.format(ser[-1]*100) + '%)'
    #         FvAnnotateAdd(ser.index[-1],ser[-1],caption)
    #     példa:  egy görbe közbenső pontja   (a FvPlot annotplus argumentumában is megadható)
    #         datumL='2021.02.01'
    #         caption = '... (' + '{:.1f}'.format(ser.loc[datumL]*100) + '%)'
    #         FvAnnotateAdd((pd.Timestamp(datumL),ser.loc[datumL],caption))
    #     példa:  egy görbe maximumpontja
    #         datumL=ser.idxmax()
    #         caption = '... (' + '{:,.0f}'.format(ser.loc[datumL]) + ')'
    #         FvAnnotateAdd((pd.Timestamp(datumL),ser.loc[datumL],caption,15))          # yoffset is meg van adva
    #  color:  a label betűszíne   (pl. színszó, színkód, stb)
    #     - ha nincs megadva, akkor az utoljára rajzolt volna színe (sötétítve)
    #  type:  szöveges típusjelzés.  Példa: a FvPlot a first,last,middle,max,min típusjelzéseket használja
    #     - a FvAnnotatePlot függvény annot_count argumentumában hivatkozni lehet a típusokra
    '''
    
    if y==np.nan: return        # not-a-number érték nem adható meg

    if not color:
        lastcolor=pltlastcolor()    # utoljára rajzolt vonal színének bekérése
        if lastcolor: color=color_darken(lastcolor,0.5)

        # lines=plt.gca().get_lines()
        # if len(lines)>0: 
        #     color=lines[-1].get_color()        # így lehet bekérni utólag a vonal színét
        #     color=color_darken(color,0.5)
    
    #print('FvAnnotateAdd (utána):  caption:' + caption + '  color:' + str(color))

    if type(x)==int: x=float(x)

    config.aAnnotG.append((x,y,caption,position,color,annottype))

def FvAnnotatePlot(xoffset=8,yoffset=5,fontsize=8,heightfactor=1.3,annot_count=None,annotcallback=None,
                   serbaseline=None,color=None,ymin=None,ymax=None,ystack=None):
    # Diagramok annotálása, az átfedések minimalizálásával
    #   - előfeltétel:  FvAnnotateAdd sorozat
    #   - csak függőleges irányú label-eltolások, a lehető legkisebb mértékben (mind esztétikai, mind funkcionális szempontból
    #       előny, ha a címkék pozícionálása viszonylag egységes)
    #   - nyilakat rajzol a címkéktől az annotált pontig
    #   - függőleges irányban megőrzi az eredeti sorrendet (csak rendkívül erős tömörülés esetén fordulhat elő sorrendcsere)
    #   - ha több rajzterület is van, akkor az aktuális rajzterületre rajzol (rajzterületenként külön annotálás kell)
    # xoffset,yoffset:  a label alapértelmezett pozíciója az annotálandó ponthoz képest, point-ban  (a betűmérethez igazítható)
    #     - mindkét érték legyen pozitív (a position-nak megfelelő előjelváltásokról a függvény gondoskodik)
    #     - kezdőérték, átfedések minimalizálása során eltérhet ettől a tényleges pozíció
    # fontsize:  egyetlen szám (globális beállítás), vagy egy dictionary típusToFontsize hozzárendelésekkel (string-dict formátum is lehet) 
    #     - ha egyetlen szám, akkor a localmax,localmin jelölőkre eggyel kisebb méret érvényesül, minden másra a megadott méret  
    #     - a FvPlot a first,last,middle,max,min,localmax,localmin,egyedi típusneveket alkalmazza;   minden egyéb: "other")
    # heightfactor: milyen szorzóval számolja a sormagasságokat (az eredeti fontmagassághoz képest) 
    # color:  mindegyik címke ezzel a színnel jelenjen meg. 
    #    - ha nincs megadva, akkor a FvAnnotateAdd-ban megadott érték (default: az utoljára rajzolt vonal színe, sötétítve), annak hiányában
    #       a default színciklus következő színe. Lásd még: config.speccolorsG
    # annot_count:  globálisan vagy típusonként megadható, hogy hány annotáció jelenjen meg (az alapvonaltól legtávolabbi jelölők jelennek meg)
    #    - szám, dictionary vagy kódlista-string.    Default:  'first:20//middle:20//last:20//localmax:10//localmin:10//other:20'   
    #    - a FvPlot által alkalmazott típusok:  first,middle,last, localmax,localmin  (más típus is előfordulhat lásd FvAnnotateAdd). Továbbá: 'other'
    #    - a config.linecolorsG-ben megadott label-ekre nem vonatkozik a korlátozás (mindenképpen megjelennek)
    #    - default alapvonal: x tengely  (lásd serbaseline)
    # serbaseline: a annot_count algoritmushoz tartozó alapvonal.  Default: x tengely (y=0 függvény)
    #    - egyetlen szám vagy egy ser adható meg.  Egyetlen szám esetén a baseline egy vízszintes egyenes 
    #    - egy series adható meg, ami legalább két x-re tartalmaz y értéket (nem lineáris esetben sok pont kell). Minden más pontra lineáris interpolációval lesz kiszámolva az y érték.
    #    - localmin esetén csak a baseline alattiak, localmax esetén csak a baseline felettiek maradnak meg
    # ymin,ymax:  egyetlen címke se kerüljön a megadott határvonal alá illetve fölé (csak az egyik érvényesül; nem kötelező megadni)
    # ystack: (y,stacklen)  vagy array of (y,stacklen);   az adott y-hoz tartozó pontok címkéinek lépcsőzetes eltolása stacklen ciklusokban
    #    - példa: x tengely címkézése;  a függőleges címkefeliratok helyett alkalmazható, pl. egy korrelációs táblázatban 
    
    #print('ymax:' + str(ymax)

    if xoffset==None: xoffset=8
    if yoffset==None: yoffset=5

    if fontsize==None: fontsize='localmax,localmin,localsbs:8//other:8'       # None,  8,  '8',  'last:8//other:7', {'last':8,'other':7}
    elif type(fontsize)==int:
        fontsize={'localmax':fontsize-1,'localmin':fontsize-1,'gaussabs':fontsize-1,'other':fontsize}
    
    if type(fontsize)==str:
        try: fontsize=int(fontsize)
        except:
            fontsize=kodlistaToDict(fontsize,bIntValue=True)
            if fontsize=={}: fontsize=={'localmax':7,'localmin':7,'gaussabs':7,'other':8}

    if heightfactor==None: heightfactor=1.3
    
    ystep=(8*heightfactor)*0.2     # label léptetések nagysága az átlagos label-magassághoz képest (fontsize=8 értékkel számol)

    if annot_count==None: annot_count='first,middle,last:30//localmax,localmin:30//gaussabs:30//other:30'
    # string formátum esetén konverzió integer-re vagy dict-re
    if type(annot_count)==str:
        try: annot_count=int(annot_count)
        except:
            annot_count=kodlistaToDict(annot_count,bIntValue=True)
            if annot_count=={}: annot_count=25          # ha rossz volt a formátum, akkor 25 darabos globális limit


    maxcycles=200

    if len(config.aAnnotG)==0: return

    #aRectFix=[]
    #xlim=plt.xlim()
    #xlim=axeskoord(xlim[0],xlim[1],'point');
    #if ymax!=None: 
        
    #    aRectFix.append((xlim[0],ymax,xlim[1],ymax+1000))
    #if ymin!=None: aRectFix.append((xlim[0],ymin-1000,xlim[1],ymin))


    ax=plt.gca()
    dpi=plt.rcParams['figure.dpi']


    # Csoportosítani kell az axindex mező alapján (rajzterületenként külön ciklusok)
    #config.aAnnotG.sort(key = lambda x: x[5])        
    #for axindex,itersub in groupby(config.aAnnotG,lambda x: x[5]):
    #    # A rajzterület aktuálissá tétele (a plt.gca() ezt a rajzterületet adja vissza)
    #    plt.sca(plt.gcf().axes[axindex])
    #    aAnnotsub=list(itersub)


    aAnnot=config.aAnnotG.copy()
    nCount=len(aAnnot)
    aX,aY,aCaption,aPosition,aColor,aType = unzip(aAnnot)

    # FELESLEGES ANNOTÁCIÓK ELHAGYÁSA  (darabszám limit feletti, localmax esetén a baseline alatti, localmin esetén a baseline feletti)
    if annot_count:
        # Eltérések a baseline-hoz képest
        # minden x értékhez kell egy yBaseline (aYBase)
        if serbaseline is None: aYBase=[0]*len(aX)
        elif type(serbaseline) in [int,float]: aYBase=[serbaseline]*len(aX)
        else: aYBase=servalueA(serbaseline,aX)
        aYBase=array(aYBase)

        # Ha a baseline minden pontja >0, akkor szorzófaktor alapján kell rendezni, egyébként távolság alapján
        positive =  (len(aYBase[aYBase<=0])==0) 
        # - pozitív alapgörbe esetén multiplikatív jelleget feltételez: a baseline növekedésével az anomália-határ is arányosan növekedik
        if positive:    
            ymin_,ymax_=plt.ylim()
            epszilon = min(serbaseline.max(),ymax_)  * 0.02        # megnöveli az anomáliahatárt a baseline nagyon kis értékeire

        aYDiff=[0]*len(aY)
        ydiffmax=0      # a legnagyobb eltérés a baseline-tól (abszolút érték)
        for i in range(len(aY)): 
            aYDiff[i]=abs(aY[i]-aYBase[i])
            if positive: aYDiff[i]=aYDiff[i] / (aYBase[i]+epszilon)
            if aYDiff[i]>ydiffmax: ydiffmax=aYDiff[i]
        # az ydiffmax ne legyen nagyobb a diagramon látható ymax-ymin értéknél  
        #    (csak azokat az annotációkat dobja ki, amelyek a látható területhez képest elenyészőek)
        y1_,y2_=plt.ylim()
        if not positive: ydiffmax = Limit(ydiffmax,max=y2_-y1_)


        sortarrays3(aYDiff,aAnnot,aYBase,True)      # csökkenő sorrend, aAnnot és aYBase szinkronizált rendezése

        annot_counter={}
        if type(annot_count)==dict:
            # számlálók létrehozása típusonként
            annot_counter=annot_count.copy()
            for typeL in annot_counter: annot_counter[typeL]=0    # számlálók nullázása
        
        aAnnot_=[]
        for i in range(len(aAnnot)):
            annot=aAnnot[i]
            typeL=annot[5]
            y=annot[1]
            
            if typeL=='localmax':
                # if annot[2]=='Moldova':
                #     print('Moldova')
                if y<=aYBase[i]: continue                # csak a baseline feletti lokális maximumok kellenek
                # 2023-11-24:  kiiktattam, mert túl sok olyan helyzet van, amikor kényelmetlen
                # if positive and aYDiff[i]<=0.05: continue
                # elif not positive and aYDiff[i]<=ydiffmax*0.05: continue    # abszolút eltérés nem lehet túl kicsi
            elif typeL=='localmin':
                if y>=aYBase[i]: continue                # csak a baseline alatti lokális minimumok kellenek
                # 2023-11-24:  kiiktattam, mert túl sok olyan helyzet van, amikor kényelmetlen
                # if positive and aYDiff[i]<=0.05: continue
                # elif not positive and aYDiff[i]<=ydiffmax*0.05: continue    # abszolút eltérés nem lehet túl kicsi

            if typeL==None or typeL=='': typeL='other'
            nDb=annot_counter.get(typeL)
            # ha nincs ilyen típus, akkor 'other'
            if nDb==None and typeL!='other': 
                typeL='other'
                nDb=annot_counter.get('other')
            # Ha nincs limit az adott típusra, akkor mindegyik record megtartandó
            if nDb==None: aAnnot_.append(annot)
            # Ha van darabszám-limit, akkor ellenőrzés
            else:
                nDbMax=annot_count.get(typeL)
                if nDbMax and nDb<nDbMax:
                    annot_counter[typeL]=nDb+1
                    aAnnot_.append(annot)
        if type(annot_count)==int:
            aAnnot_=aAnnot_[:annot_count]
        aAnnot=aAnnot_
            
        # Tömbök incializálása (változhatott az aAnnot)
        nCount=len(aAnnot)
        aX,aY,aCaption,aPosition,aColor,aType = unzip(aAnnot)

    
    if annotcallback:
        aCaption=list(aCaption)
        for i,caption in enumerate(aCaption): aCaption[i]=annotcallback(caption)


    aX_points,aY_points = axeskoordA(aX,aY,'point')
    #print('aX_points:' + str(aX_points))

    def sub_fontsize(labeltype,color):
        if type(fontsize)==dict:
            try: result=int(fontsize[labeltype])
            except:
                try: result=int(fontsize['other'])
                except: result=8
        else: result=fontsize             # fontsize: input argumentum (int vagy dict lehet)
        if color=='kiemelt': result=result+1
        return result


    def labelsize(caption,labeltype,color):     # felirat mérete point-ban (több soros is lehet)
        aSor=caption.splitlines()
        height=0
        width=0
        for sor in aSor:
            w,h,d = mpl.textpath.TextToPath().get_text_width_height_descent(sor, 
                                mpl.font_manager.FontProperties(family="Century Gothic", size=sub_fontsize(labeltype,color)), False)
            height += h*heightfactor     # nem kell a d ("descent"), mert már benne van a h-ban
            if w>width: width=w
        return width,height



    # Induló aRect-ek:
    aRect=[None]*nCount
    aYLabel0=[0]*nCount          # a címkék bal alsó sarkának kiinduló pozíciója (point-ban)
    for i in range(nCount):
        caption=aCaption[i]
        labeltype=aType[i]
        width,height = labelsize(caption,labeltype,aColor[i])
        xoffsetL=xoffset
        yoffsetL=-yoffset - height/2      # default: bottom;   a height/2 azért kell, mert a nyíl középről indul
        #print('yoffsetL:' + str(yoffsetL))
        position=aPosition[i]
        if not position: position='right bottom'
        words=position.lower().split()
        if 'left' in words: 
            xoffsetL=-xoffsetL-width        # jobb szélről indul a nyíl
        if 'top' in words: 
            yoffsetL=-yoffsetL - 4          # koorrekció - túl magasan voltak a maxjelzők, és ez átfedéseseket eredményezett 
        aRect[i]=(aX_points[i]+xoffsetL,aY_points[i]+yoffsetL,aX_points[i]+xoffsetL+width,aY_points[i]+yoffsetL+height)
    
    
    # ystack: az azonos y értékhez tartozó címkék lépcsőzetes eltolása
    if ystack:
        if type(ystack)==tuple: aYStack=[ystack]     # tuple vagy arra of tuple az input
        else: aYStack=ystack
        # print('aYStack:' + str(aYStack))
        for ystack in aYStack:        
            y=ystack[0]                 # (y,stacklen);   az y a címkézendő pontsorozat y koordinátája
            stacklen=ystack[1]
            # Az adott y-hoz tartozó rect-ek kiválasztása
            aStack=[]
            for i in range(len(aY)):
                if aY[i]!=y: continue
                aStack.append((aX[i],i))          # az index is tárolandó
            if len(aStack)>1:               # ha tartozik legalább két címke a megadott y-hoz
                sortrecords(aStack,0)       # nem volt elvárás, hogy az input rendezett legyen x irányban
                width,height=labelsize('Sample')       
                nShift=stacklen-1    # induló eltolás
                if len(aStack)<stacklen: nShift=len(aStack)-1    
                for i in range(len(aStack)):
                    index=aStack[i][1]
                    shift=height*nShift
                    aRect[index]=FvRectShiftY(aRect[index],-height*nShift)     # lefelé tolás, lépcsőzetesen
                    nShift-=1
                    if nShift<0: nShift=stacklen-1


    # aRect-ek léptetése
    aYLabel0=[0]*nCount                 # Az induló y-értékek tárolás
    for i in range(nCount): aYLabel0[i]=aRect[i][1]         # a rect-ek alsó határa
    
    cycle=0
    steps=''
    aStepsize=[ystep,ystep/2,ystep/4,ystep/8,ystep/16,ystep/32,ystep/64]
    for stepsize in aStepsize:
        #print('Stepsize: ' + str(stepsize))
        for cycleL in range(maxcycles):
            # print('cycle:' + str(cycle))
        
            nShiftCount=0
            stepsL=['0']*nCount
            # Alciklus a címkékre
            for i in range(nCount):
                y0=aYLabel0[i]          # az eredeti pozíció point-ban
                # kísérlet felfelé illetve lefelé tolásra
                rect=aRect[i]
                aOverlap=[0]*3
                for dir in range(3):    # 0 jelenlegi helyzet,  1: eltolás felfelé  2: eltolás lefelé
                    if dir==0: rectL=rect
                    elif dir==1: rectL=FvRectShiftY(rect,stepsize)
                    elif dir==2: rectL=FvRectShiftY(rect,-stepsize)
                    # Átfedés a többi címkével
                    for j in range(nCount):
                        if j==i: continue
                        overlap=FvRectOverlapHeight(rectL,aRect[j])
                        #if overlap>aOverlap[dir]: aOverlap[dir]=overlap
                        aOverlap[dir]+=overlap*(1 - (abs(aRect[j][1]-y0)/10000))    # minél messzebb van a másik az eredeti pozíciótól, annál kisebb az átfedés súlya
                        # if cycle==8 and i==7 and overlap>0: 
                        #    print('j:' + str(j) + ' overlap:' + str(overlap) + '  ' + str(rectL) + '  ' + str(aRect[j]) )
                    # Átfedés a fix rect-ekkel
                    #for rect in aRextFix:
                    #    overlap=FvRectOverlapHeight(rectL,rect)
                    #    aOverlap[dir]+=overlap
                    # if dir==0 and aOverlap[0]==0: break     # ha jelenleg nincs átfedés, akkor nem kell megnézni a két eltolási lehetőséget
                    #elif dir==1 and ymax!=None and rectL[3]>ymax: aOverlap[dir]+=rectL[3]-ymax
                    #elif dir==2 and ymin!=None and rectL[1]<ymin: 
                    #    aOverlap[dir]+=ymin-rectL[1]
                    #    print('hahó')
            
                # Javul-e a helyzet valamelyik eltolással
                ystepL=0
                # Ha nincs semmilyen átfedés 
                if aOverlap[0]==0:
                    if stepsize==ystep/64:     # az utolsó körben egy záró korrekció
                        # Ha a címke lejjebb van az eredeti pozíciónál, és felfelé léptetéssel sem lesz semmilyen átfedés, akkor felfelé léptetés
                        if aRect[i][1]+stepsize<aYLabel0[i] and aOverlap[1]==0: ystepL=stepsize
                        # Ha a címke feljebb van az eredeti pozíciónál, és lefelé léptetéssel sem lesz semmilyen átfedés, akkor lefelé léptetés
                        elif aRect[i][1]-stepsize>aYLabel0[i] and aOverlap[2]==0: ystepL=-stepsize

                # Ha a címke felül van, akkor a felfelé tolás preferált
                elif aY_points[i]<=(aRect[i][1]+aRect[i][3])/2:
                    if aOverlap[1]<=aOverlap[0]: ystepL=stepsize       
                    elif aOverlap[2]<aOverlap[0]: ystepL=-stepsize          #  and rect[1]-stepsize>ymin
                # Ha a címke alul van, akkor a lefelé tolás preferált
                else:
                    if aOverlap[2]<=aOverlap[0]: ystepL=-stepsize       
                    elif aOverlap[1]<aOverlap[0]: ystepL=stepsize   


                # Az eltolás végrehajtása
                #if cycle in [0,1]:
                #    print('cycle:' + str(cycle) + ' i:' + str(i) + '  step:' + str(ystepL) + '  rect_now:' + str(aRect[i]))
                #    print('overlap:' + str(aOverlap[0]) + '  fel:' + str(aOverlap[1]) + '  le:' + str(aOverlap[2]) + '\n')
                if ystepL!=0:
                    aRect[i]=FvRectShiftY(rect,ystepL)
                    nShiftCount+=1
                    if ystepL>0: stepsL[i]='F'      # bejegyzés az eltolás táblázatba (step-log)
                    else: stepsL[i]='L'

            steps=steps + '\n' + ''.join(stepsL) + '   step: ' + str(stepsize)
            cycle+=1
            if nShiftCount==0: break
   
    
    # Címkék felső vagy alsó határa (pl. ymax=1.1:  a címkék ne kerülhessenek magasabbra a felső határ 10%-ánál)
    #print('ymax:' + str(ymax))
    if ymax!=None:
        # átszámítás points-ra
        w,h = axesunits('point')
        ymax=ymax*h

        y2=array(unzip(aRect,3))
        y2=y2[aY_points<=h]     # csak a látható címkék érdekelnek
        if len(y2)>0:
            diff=max(y2)-ymax
            if diff>0: 
                for i in range(len(aRect)): aRect[i]=FvRectShiftY(aRect[i],-diff)
    elif ymin!=None:
        # átszámítás points-ra
        w,h = axesunits('point')
        ymin=ymin*h

        y1=array(unzip(aRect,1))
        y1=y1[aY_points>=0]     # csak a látható címkék érdekelnek
        if len(y1)>0:
            diff=ymin - min(y1)
            if diff>0: 
                for i in range(len(aRect)): aRect[i]=FvRectShiftY(aRect[i],diff)

    for i in range(nCount):
        # Teszt: rect kirajzolása  (nem jó, adat-koordináták kellenének)
        #rect = patches.Rectangle((aRect[i][0], aRect[i][1]), aRect[i][2]-aRect[i][0], aRect[i][3]-aRect[i][1], linewidth=1, edgecolor='r', facecolor='none')
        #plt.gca().add_patch(rect)
        
        #if 'Hungary' in aCaption[i]: print('Hungary left, right:' + str(aRect[i][0]) + ' ' + str(aRect[i][2]))
        #elif 'Romania' in aCaption[i]: print('Romania left, right:' + str(aRect[i][0]) + ' ' + str(aRect[i][2]))

        
        rad=0.2         # nyilak görbületének erőssége
        if aY_points[i]>(aRect[i][1]+aRect[i][3])/2: rad=-rad    # alulról induló nyíl esetén negatív a görbület
        # angleB=math.atan(yoffsetL/xoffset)*(180/math.pi)
        # arrowprops=dict(arrowstyle='-|>',shrinkA=0,shrinkB=0,mutation_scale=6,connectionstyle='angle3,angleA=0,angleB=' + str(angleB)))
        
        if aRect[i][2]<aX_points[i]: xoffset=aRect[i][2]-aX_points[i]       # negatív xoffset, haling=right
        else: xoffset=aRect[i][0]-aX_points[i]           # alapesetben pozitív xoffset (halign=left), Ha negatív, akkor halign=right

        yoffsetL=((aRect[i][1]+aRect[i][3])/2) - aY_points[i]    # a nyíl a bal vagy a jobb szél közepéről indul 

        caption=aCaption[i]
        labeltype=aType[i]
        
        fontsizeL=sub_fontsize(labeltype,aColor[i])

        colorL=None
        if aColor[i]=='kiemelt': colorL='red'
        else:
            # annotcolorG
            words=caption.split()   
            if len(words)>0:                        # kiemelt felirat színe (a felirat első szava alapján)
                firstword=words[0]
                colorL=dget(config.annotcolorG,firstword)
            if colorL==None: colorL=aColor[i]       # FvAnnotateAdd-ben megadott egyedi szín
            if colorL==None: colorL=color           # argumentumként megadott közös szín
            if colorL==None: colorL='black'         # default: fekete
        
        FvAnnotateOne(aX[i],aY[i],caption,xoffset=xoffset,yoffset=yoffsetL,fontsize=fontsizeL,color=colorL)
    
    config.aAnnotG=[]
    # print('Annotation cycle:' + str(cycle) + steps)       # debug





def Captionwidth(text,fontsize,font_family='Century Gothic'):     # felirat mérete point-ban
    '''
    point: inch/72     Az axesunits függvénnyel lekérdezhető a koordináta-terület teljes szélessége (viszonyítási alapként)
    Példa: 8 pontos karakterek átlagos szélessége 5 point körüli  (erős szórással)
    '''
    
    w,h,d = mpl.textpath.TextToPath().get_text_width_height_descent(text, 
                        mpl.font_manager.FontProperties(family=font_family, size=fontsize), False)
    return w



def FvAnnotateOne(x,y,caption,xoffset=10,yoffset=10,fontsize=8,color='black',alpha=0.9,alphabbox=0.3):
    # Címke és nyíl megjelenítése
    # Ha xoffset<0, akkor a címke jobbra igazított és a nyíl a jobb szélről indul, egyébként balra igazított és a nyíl bal szélről indul
    # y irányban a nyíl a bal vagy a jobb szél közepéről indul
    
    rad=0.2         # nyilak görbületének erőssége
    if yoffset<=0: rad=-rad    # alulról induló nyíl esetén negatív a görbület

    relpos=(0,0.5)
    ha='left'

    if xoffset<0:
        rad=-rad
        relpos=(1,0.5)
        ha='right'
    
    #print('caption:' + caption + ', x:' + str(x) + ', y:' + str(y) + ' xoffset:' + str(xoffset) + ' yoffset:' + str(yoffset))
    plt.annotate(caption,(x,y),xycoords='data',
                    xytext=(xoffset,yoffset), textcoords='offset points',
                    ha=ha,va='center',fontsize=fontsize, alpha=alpha, color=color,
                    bbox=dict(visible=True,edgecolor=None,color='white',alpha=alphabbox,pad=0.1),
                    arrowprops=dict(arrowstyle='-|>',shrinkA=0,shrinkB=0,relpos=relpos,alpha=0.4,
                                    mutation_scale=fontsize-2,connectionstyle='arc3,rad=' + str(rad)))

def FvAnnotateTwo(xy1,xy2,caption,xoffset=10,yoffset=10,leftright='right',fontsize=8,color='black',alpha=0.9,alphabbox=0.3):
    # egy címke két ponthoz.  Példa: két érték különbségének vagy hányadosának kiírásakor
    # - az xoffset a jobbra eső ponthoz képest értendő (leftright='left' esetén a balra eső ponthoz képest)
    # - az yoffset az első pont x értékéhez képest értendő
    
    xy1_point=axeskoord(xy1[0],xy1[1],'point')
    xy2_point=axeskoord(xy2[0],xy2[1],'point')

    if leftright=='right':
        if xy2_point[0]>xy1_point[0]: 
            xoffset1=xoffset + (xy2_point[0]-xy1_point[0])
            xoffset2=xoffset
        else: 
            xoffset1=xoffset
            xoffset2=xoffset + (xy1_point[0]-xy2_point[0])
    else:
        if xy2_point[0]<xy1_point[0]: 
            xoffset1=xoffset + (xy1_point[0]-xy2_point[0])
            xoffset2=xoffset
        else: 
            xoffset1=xoffset
            xoffset2=xoffset + (xy2_point[0]-xy1_point[0])
    
    yoffset1=yoffset
    yoffset2=(xy1_point[1] + yoffset) - xy2_point[1] 

    FvAnnotateOne(xy1[0],xy1[1],caption=caption,xoffset=xoffset1,yoffset=yoffset1,fontsize=fontsize,
                        color=color,alpha=alpha,alphabbox=alphabbox)
    FvAnnotateOne(xy2[0],xy2[1],caption=caption,xoffset=xoffset2,yoffset=yoffset2,fontsize=fontsize,
                        color=color,alpha=0,alphabbox=0)


# SZÍNKEZELÉS

def color_darken(color, amount=0.5):
    # color:  (r,g,b),  színnév,   float (szürkeárnyalat)
    
    # konvertálás (r,g,b)-re   (ha nem az)
    # if vanbenne(str(type(color)),'float'): c=(color,color,color)
    if isinstance(color,float): c=(color,color,color)
    else:
        try:
            c = mpl.colors.cnames[color]        # ha színnévvel lett megadva
        except:
            c = color
    c = colorsys.rgb_to_hls(*mpl.colors.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def pltlastcolor():       # az utoljára rajzolt volna színe (automatikus színkiosztás esetén fontos)
    try:
        lines=plt.gca().get_lines()
        if len(lines)>0: 
            return lines[-1].get_color()        # így lehet bekérni utólag a vonal színét
        else:
            return plt.cm.tab10.colors[0]
    except: pass    





# ILLESZTÉSEK, MOZGÓÁTLAG

def fn_moving_avg(ser,count,center=True):
    '''
    count:  integer vagy datetime frequency string  (pl. "D", "W", "7D")
        integer esetén az átlagolásba bevonandó pontok száma (az x-irányú távolságtól függetlenül, csak
            a darabszám számít)
    Páratlan count esetén pontosan teljesül, hogy minden egyes ponthoz az adott pont körüli count számú
        pont átlaga íródik be  (a széleken féloldalas lehet az átlagolás)
    Páros count esetén mindenképpen van egy feles jobbra tolódás.
        Példa:  count=4 esetén  két pont előtte, az adott pont, egy pont utána
        Ahhoz, hogy ne legyen eltolódás, az x-értékeket a felezőpontokra kellene áthelyezni, de ez a megoldás
            egyenetlen x-értékek esetén elég különös eredményhez vezetne
        Jobb megoldás resample-t kérni és ezt követően átlagolni, megnövelt pontszámmal
    '''
    if type(count)==int:
        n_periods = count-1     # a count-ban a pontok számát kellett megadni, nem a szakaszokét
                # - a closed='both' miatt végül mégis count számú pontra fog átlagolni

        ser2 = ser.rolling(n_periods, center=center, closed='both', min_periods=1).mean()
    else:
        ser2 = ser.rolling(count, center=center, closed='left', min_periods=1).mean()

    # Ha a count páratlan, akkor center=True esetén nem középre, hanem eggyel jobbra íródik be az átlag
    # Ennek a kellemetlen eltolódásnak a kezelésére csinálok egy balra tolást
    if center and count%2==1:
        Y = ser.values
        # A jobb szélén lévő pontra eső féloldalas átlagot önállóan kell számolni
        count_right = count//2 + 1
        y_last = Y[-count_right:].mean()
        Y = np.append(ser2.values[1:],y_last)        # eltolás balra, jobb szélre beírandó a féloldalas átlag
        ser2 = pd.Series(Y,ser2.index)
    return ser2
            

    
def FvGaussAvg(ser,gausswidth=None,min_periods=None,std=None,trendpoint=None):
    '''  Gauss súlyozású mozgóátlag  (középre centrált számolással)
    ser:  tbl is lehet (DataFrame). Tbl esetén az összes oszlopra kiterjed (pl. minden országra)
    gausswidth:  G, Gauss szélesség (talpponti szélesség, a szórás négyszerese; néhol "effektív szélességnek" is hívom, de korábban G/2 értelmezés is előfordult)
            Megadható a mérési pontok száma (>1 integer) vagy az összes mérési ponthoz viszonyított hányad (0-1 közötti tört, default 0.1).
            Vesszős felsorolás is megadható (string).  Példa: '140,120,100,80,60'  (általában a legnagyobb legelöl; egyre halványabb megjelenítés) 
            Default: a teljes x-szélesség tizede.
            A ténylegesen figyelembe vett pontok száma a gausswidth kétszerese (8*szigma)
        - Dirac deltából gausswidth szélességű gauss lesz (csökkenő csúcsértékkel)
        - Gauss görbe 0.7-es faktorral lesz laposabb, azonos G esetén. Egy Gauss kilengés eltüntetéséhez nagyjából 8G-s mozgóátlag kell.  
            Megfordítva: ha egy nagyívű Gauss érintetlenül hagyása a cél, akkor a mozgóátlag G-je legfeljebb nyolcadrésze lehet az eredetinek.
        - a zajszűrési periódusidő a gausswidth fele (T=G/2,  2*szórás). A <=T periódusú kétirányú 
            kilengések illetve az egymás után <=T-vel következő tüskék eltűnnek  (kb századrészre csökkennek). 
        - a széleken lévő megbízhatatlansági tartomány szintén a gausswidth fele. 
    min_periods:  a széleken meddig csökkenhet a mozgóablak szélessége 
       NEM JAVASOLT MEGADNI    nem teljesen jól definiált a működése
       Default: 1
       "all":  nem megengedett a csökkenés (a görbe nem terjed ki a szélekig)
    std:  (szigma) Az ablakon belüli gauss súlyozás "csúcsossága".  Default: szigma = windowwidth/4  (NEM JAVATOLT MEGADNI)
    trendpoint:  megadható egy (x,y) trendpont. Részt vesz az átlagolásban, de a kimenetben nem lesz benne.
       A bal és a jobb szélen túli pont is lehet. Egy sűrű pontsorozat esetén nem nagyon van hatása
       Akkor merülhet fel, ha valamilyen logikai vagy definíciós összegüggés alapján tudható az érték (pl 1-ben 0-nak kell lennie)
    A belső nan értékek kitöltése lineáris interpolációval (nan értékekkel nem működik a mozgóátlag)
    - a pandas interpolate() függvénye a jobb szélen lévő NaN értékeket is hajlamos kitölteni (az utolsó értékkel)
    '''
    
    serL=ser
    # if not 'DataFrame' in str(type(serL)):         # DataFrame esetén üres táblázatot eredményezne (???)
    #     serL=serL.dropna()


    # ha nincs megadva gausswidth, akkor az összes pont tizede
    if not gausswidth: gausswidth=int(round(len(ser)/10))
    # 0-1 közötti törtszámmal megadott szélesség (az összes pont megadott hányada)
    if gausswidth<1: gausswidth=int(round(len(ser)*gausswidth))

    if not min_periods: min_periods=1   # int(gausswidth)
    # if not min_periods: min_periods=int(gausswidth*1.25)    
    #  2: a teljes gausswidth elmarad a szélekről
    #  1: nincs elhagyás a széleken

    if type(min_periods)==str and min_periods=='all': min_periods=gausswidth   # math.ceil(1.5*gausswidth)
    
    if not std: std=gausswidth/4    # a gausswidth a szórás négszerese
    
    windowwidth=gausswidth*2    # az összes figyelembe vett pont a gausswidth kétszerese

    if trendpoint: 
        serL[trendpoint[0]]=trendpoint[1]

    serL=serL.rolling((windowwidth), min_periods=min_periods, center=True, win_type='gaussian',closed='left').mean(std=std)
        # - 2022.09.29  az std felezése miatt növeltem kétszeresére, a korábbi plotokkal való folytonosság érdekében.
        # - a closed adat mint ha nem működne. A "both" arra utalna, hogy a gördülő ablak bal és jobb szélén lévő pont is figyelembe veendő.
        #   Ehelyett az történik, hogy a széleken lévő érték felezéssel van figyelembe véve
    #print('Utána: ' + str(serL))
    if trendpoint: serL=serL.drop(index=trendpoint[0])
    return serL

def FvGaussTAvg(ser,G=None,T='G/2',leftright='leftright',extend=None,positive=None,fit_method='linear',
            bMirror=False,erlang_params=None):       # lineáris folytatás a széleken
    '''
    A széleken előforduló ingadozásokat szűri ki, illetve a szélekre extrapolált lineáris folytatás is kérhető
    
    G: gauss szélesség (talpponti szélesség, 4 szigma). Periódikus zaj hatékony kiszűréséhez legalább G=2*T szükséges.
    T: a kiszűrendő zaj periódusideje (alapesetben G/2). Nem lehet kisebb 1-nél 
        Hozzávesz a görbéhez T pontot, lineáris illesztéssel a szélső T+T tartomány két átlagértékére
        Ezeket a pontokat is figyelembe veszi a mozgóátlag számításakor, de ez az extension nem jelenik meg az output-ban 
        (lásd még extend, amivel meghosszabbítható a lineáris illesztés)

    extend:  kiterjesztési pontok száma a széleken, az alap-extension továbbfolytatásával  
        - ez az extension a kimenetben is megjelenik. Nagy távolságban a görbe belesimul a szélekre illesztett egyenesekbe
        - tuple vagy egyetlen szám adható meg.
        - ha a szám float, akkor a G-hez viszonyított arányszámként értelmezi (egész esetén is legyen a végén pont)
    
    fit_method:  "linear"  "erlang"
        - linear:  a két szélső G/2 szakaszra átlag-pontot képez, összeköti a két átlagpontot egy egyenessel, majd a Gauss átlagolás
            pontosításához az egyenesre illeszkedő többletpontokat vesz fel (csak a Gauss átlagolás idejére)
        - erlang:  a három szélső G/2 szakaszta átlag-pontokat képez, illeszt rájuk egy erlang görbét, majd a Gauss átlagolása
            pontosításához az Erlangra illeszkedő többletpontokat vesz fel (csak a Gauss átlagolás idejére)
            Lásd még:  erlang_params
        
    positive:  True esetén a lineáris illesztésből adódó negatív értékeket 0-val helyettesíti (a mozgóátlag mindenképpen 0 feletti lesz)
        None: önállóan állapítja meg. Ha a ser minden eddigi értéke pozitív volt, akkor bekapcsolja
    bMirror:  legyen a tükrözés az utolsó G/2 szakasz kilengéseinek kiszűrésére
    erlang_params:  dict(leftwidth,leftwidth_delta,kitevo)    fit_method='erlang' esetén adható meg
        leftwidth:   az illesztendő erlang görbe felfutási szélessége (+/- leftwidth_delta;   nem a pontok számáról, hanem x-szélességről van szó)
        kitevo:      az erlang kitevője
        Default értékek:   leftwidth = 2*G    leftwidt_delta=0.2*G     kitevo=20
    '''

    serL=ser.copy()

    # positive önálló beállítása
    if positive==None:
        positive =  (serL.min()>=0)   # True, ha serL minden eddigi értéke >=0 volt


    # G átszámítása x-pontok számára
    # ha nincs megadva G, akkor az összes pont tizede
    if not G: G=int(round(len(serL)/10))
    # 0-1 közötti törtszámmal megadott szélesség (az összes pont megadott hányada)
    if G<1: G=int(round(len(serL)*G))

    if (not T) or (type(T)==str and T=='G/2'): T=int(G/2)
    if T<1: T=1             # enélkül elszállna, de nem ártana egy figyelmeztetés   (G=1 esetén helyből ide kerül)

    addcountleft,addcountright=(0,0)
    # Extend esetén hozzáadandóak az extension pontok (a G/2 többletpont ilyenkor is kell)
    if extend:
        if type(extend)==tuple and len(extend)==2: 
            addcountleft,addcountright=extend
        elif type(extend)==int:
            addcountleft,addcountright=(extend,extend)
        elif type(extend)==float:               # G-hez képest
            if vanbenne(leftright,'left'): addcountleft=int(extend*G)
            if vanbenne(leftright,'right'): addcountright=int(extend*G)
        else:
            print('FvGaussTAvg, Érvénytelen extend: ' + str(extend))
    addcount=T      # Ez mindenképpen kell  (extend==None esetén is)
    addcountleft+=addcount
    addcountright+=addcount

    datetimeindex = ('timestamp' in str(type(min(serL.index))))
    if datetimeindex:
        serL=datefloat(serL)
    width=max(serL.index)-min(serL.index)
    osztásköz=width/(len(serL)-1)
    xmax=max(serL.index)
    x0=xmax
    # lineáris illesztés a jobb szélre
    if fit_method=='linear':
        if vanbenne(leftright,'right'):
            y1=serL.iloc[-2*T:-T].mean()                    # -2T és -T közötti pontok átlaga
            x1=(serL.index[-2*T] + serL.index[-T-1])/2      # középső pont -2T és -T között
            y2=serL.iloc[-T:].mean()                        # -T -től a végéig lévő pontok átlaga
            x2=(serL.index[-T] + serL.index[-1])/2          # középső pont -T és a vége között
            m=(y2-y1)/(x2-x1)                               
            y0=y1-m*x1          # a két átlegérték-pontot összekötő egyenes értéke x=0-ban
            # Kiterjesztés jobbra
            serRight=pd.Series(dtype=object)        # hozzá lesz fűzve a jobb oldalon
            for j in range(addcountright): 
                x=x0+(j+1)*osztásköz
                y=y0 + x*m              # folytatja a két átlaérték-pontot összekötő egyenest a jobb széle túl
                # az utolsó T adat tükrözése az egyenesre (az átlagolásban ne jelenjen meg semmilyen kilengés)
                if bMirror and j+1<len(serL):
                    ylinear=y0 + serL.index[-(j+1)]*m       # valójában interpoláció kellene; inhomogén x-eloszlás torzulást okozhat
                    y= y - (serL.iloc[-(j+1)] - ylinear) * fn_gauss(j,0,T)    # (T-j)/T 
                if positive and y<0: y=0
                serRight[x]=y
            serL=serL.append(serRight)
        xmin=min(serL.index)
        x0=xmin
        # lineáris illesztés a bal szélre
        if vanbenne(leftright,'left'):
            y1=serL.iloc[:T].mean()
            x1=(serL.index[0] + serL.index[T-1])/2
            y2=serL.iloc[T:2*T].mean()
            x2=(serL.index[T] + serL.index[2*T-1])/2
            m=(y2-y1)/(x2-x1)
            y0=y1-m*x1 
            # Kiterjesztés balra
            for j in range(addcountleft):
                x=x0-(j+1)*osztásköz                    
                y=y0 + x*m
                # az első T adat az utolsó T adat tükrözése (az átlagolásban ne jelenjen meg semmilyen kilengés)
                if bMirror and j<len(serL):
                    ylinear=y0 + serL.index[j]*m
                    y= y - (serL.iloc[j] - ylinear) * fn_gauss(j,0,T)   # (T-j)/T 
                if positive and y<0: y=0
                serL[x]=y
    # Erlang illesztés (három pontos)
    elif fit_method=='erlang':   
        if not erlang_params: 
            leftwidth=2*G * (ser.index[-1]-ser.index[0])/len(ser)
            leftwidth_delta = 0.2*leftwidth
            kitevo=20
        else:
            leftwidth,leftwidth_delta,kitevo=dgets(erlang_params,'leftwidth,leftwidth_delta,kitevo')

        if vanbenne(leftright,'right'):
            y1=serL.iloc[-3*T:-2*T].mean()                    # -3T és -2T közötti pontok átlaga
            x1=(serL.index[-3*T] + serL.index[-2*T-1])/2      # középső pont -3T és -2T között
            y2=serL.iloc[-2*T:-T].mean()                    # -2T és -T közötti pontok átlaga
            x2=(serL.index[-2*T] + serL.index[-T-1])/2      # középső pont -2T és -T között
            y3=serL.iloc[-T:].mean()                        # -T -től a végéig lévő pontok átlaga
            x3=(serL.index[-T] + serL.index[-1])/2          # középső pont -T és a vége között

            erlangparams,diffpercent = fn_erlang_3points(X=[x1,x2,x3],Y=[y1,y2,y3],leftwidth=leftwidth,leftwidth_delta=leftwidth_delta,kitevo=kitevo)
    
            # Kiterjesztés jobbra
            X=Range(x0+osztásköz,add=osztásköz,count=addcountright)
            Y=fn_erlang_multi(X,erlangparams,kitevo=kitevo)
            serL=serL.append(pd.Series(Y,X))
        xmin=min(serL.index)
        x0=xmin
        # erlang illesztés a bal szélre
        if vanbenne(leftright,'left'):
            y1=serL.iloc[:T].mean()
            x1=(serL.index[0] + serL.index[T-1])/2
            y2=serL.iloc[T:2*T].mean()
            x2=(serL.index[T] + serL.index[2*T-1])/2
            y3=serL.iloc[2*T:3*T].mean()
            x3=(serL.index[2*T] + serL.index[3*T-1])/2

            erlangparams,diffpercent = fn_erlang_3points(X=[x1,x2,x3],Y=[y1,y2,y3],leftwidth=leftwidth,leftwidth_delta=leftwidth_delta,kitevo=kitevo)
            
            X=Range(x0-osztásköz,add=-osztásköz,count=addcountleft)
            Y=fn_erlang_multi(X,erlangparams,kitevo=kitevo)
            serL=pd.concat(pd.Series(Y,X),serL)

    serL.sort_index(inplace=True)

    if datetimeindex:
        serL=floatdate(serL)


    serGauss=FvGaussAvg(serL,gausswidth=G)

    # Gausstrend pontok levágása a szélekről (extend esetén is ott van ez a többlet)
    if vanbenne(leftright,'right'): serGauss=serGauss.iloc[:-T]   # serGauss=serGauss[:xmax]
    if vanbenne(leftright,'left'): serGauss=serGauss.iloc[T:]   # serGauss=serGauss[xmin:]

    return serGauss

def FvGaussAvgHistory(ser,G=None,back=0.25,withtrend=True):   # A gauss simítás jobb szélének idősoros ellenőrzése
    '''
    Minden x-re kiszámítja, hogy mit adott volna a FvGaussAvg a jobb szélre (illetve a jobb széltől G*back-re)
    Ciklusban (len(ser) - G)-szer fogja hívni a FvGaussAvg függvényt
    A kiszámított görbe (1-back)*G - nél indul és a jobb szél előtt back*G ponttal fejeződik be
    A ciklusokban nem a teljes ser-rel számol, hanem az utolsó 1.5*G ponttal (a távolabbi pontok már lényegtelenek)
    '''

    # ha nincs megadva G, akkor az összes pont tizede
    if not G: G=int(round(len(ser)/10))
    # 0-1 közötti törtszámmal megadott szélesség (az összes pont megadott hányada)
    if G<1: G=int(round(len(ser)*G))

    aX=[]
    aY=[]
    windowlen=int(1.5*G)
    backlen=int(back*G) 
    for indexright in range(G,len(ser)):
        print(ser.index[indexright])        
        
        if indexright>windowlen: indexleft=indexright-windowlen
        else: indexleft=0
        serL=ser.iloc[indexleft:indexright]
        if withtrend: serGauss=FvGaussTAvg(serL,G,leftright='right')
        else: serGauss=FvGaussAvg(serL,G)
        aX.append(serGauss.index[-backlen])
        aY.append(serGauss.iloc[-backlen])

    return pd.Series(aY,aX)

def FvGaussStdAvg(ser,gausswidth:int,min_periods=None):
    # Szórás számítása Gauss súlyozású mozgóátlaggal (standard deviation)
    # ser:  tbl is lehet (DataFrame). Tbl esetén az összes oszlopra kiterjed (pl. minden országra)
    
    if not min_periods: min_periods=int(round(gausswidth/2))
    return ser.rolling(gausswidth, min_periods=min_periods, center=True, win_type='gaussian').std(std=gausswidth/4)

def FvWeightedAvg(ser,serWeights):
    # ser,serWeights:  ser és list is lehet
    #    ser esetén elvárás, hogy a két sorozat x értékei össze vannak hangolva  (pl. ugyanahhoz a dataframe-hez taroznak)
    # az nan értékeket nem veszi figyelembe (kihagyás, ha akár az érték, akár a súly nan)
    # Két tömb esetén:  np.average(aIn,aWeights)
    if type(ser)==pd.Series: aL=ser.values
    else: aL=ser
    
    if type(serWeights)==pd.Series: aW=serWeights.values
    else: aW=serWeights
        
    sum=0
    sumW=0
    for i in range(len(aL)):
        if np.isnan(aL[i]) or np.isnan(aW[i]): continue
        sum+=aL[i]*aW[i]
        sumW+=aW[i]
    if sumW: return sum / sumW

def fn_gauss_arima_avg(ser,G,positive=None):
    '''
    Gauss mozgóátlag, jobb szél korrekcióval.
    Végrehajt egy ARIMA predikciót a jobb szél utáni G/2 pontra, majd erre a kiterjesztett görbére kér mozgóátlagot

    Egyelőre nem tűnik túlzottan sikeresnek (egy szélsőségesen ingadozó adatsorra próbáltam ki)
        
    ser:  az időközöknek egyenletesnek kell lennie és nem lehet benne üres érték
        Ha az időközök nem egyenletesek, akkor előtte FvLinearResample
    '''

    p=G             # train pontok száma. Milyen hosszúságú mintázatokat tanuljon meg.
    d=2             # legfeljebb négyzetes görbületeket engedjen meg
    q=int(G)      # zajszűrő mozgóablak szélessége 
    
    arima=ARIMA(order=(p,d,q) ,seasonal_order=(0,0,0,0),
                    enforce_stationarity=False,enforce_invertibility=False)

    arima.fit(ser.values)      # viszonylag időigényes

    Y_pred = arima.predict(int(G/2))

    step = (ser.index.max()-ser.index.min()) / (len(ser)-1)
    X_pred = [ser.index.max() + step*(i+1)  for i in range(int(G/2))]
    
    ser_new=pd.Series(Y_pred,X_pred)   # predikció a következő G/2 pontra

    ser_extended = Concat(ser,ser_new)

    serG = FvGaussAvg(ser_extended,gausswidth=G)

    return serG.loc[serG.index<=ser.index.max()]
    
    


def FvLinear(ser,outtype='ser',resample='',extend=''):
    # lineáris regresszió  (legkisebb négyzetek módszere)
    # Alapesetben a kimeneti ser x értékei megegyeznek az input x értékeivel (nincs resample)

    # outtype:  'ser'   
    #          'params':  (meredekség,y0)    ahol y0:  y értéke x=0-ban
    # resample:    ha nem üres és nagyobb 0-nál, akkor egyenletes x-eloszlás az xmin és xmax között (2 esetén csak a két végpont)
    #     Csak a kimenet pontsűrűségét érinti, magára az illesztésre nincs hatása
    #     
    # extend:   hány ponttal bővítse ki a kemeneti görbét a bal illetve a jobb szélen  (előrejelzésre alkalmazható)
    #     Csak resample>0 esetén érvényesül
    #     '0,20':   bal oldalon nincs bővítés, jobb oldalon 20 pont (a unit a resample-ből adódik)


    serL=ser.dropna()       # üres értéket tartalmazó rekordok kidobása (nem tud velük mit kezdeni a spline)
    serL=serL.sort_index()
    # serL=serL.drop_duplicates()   # hibát okozna (????  kidob minden ismétlődő y értéket)
    
    aX=serL.index.array 
    aY=serL.array
    
    datetimeindex = ('timestamp' in str(type(aX[0])))
    if datetimeindex: aX_float=datefloatA(aX)
    else: aX_float=aX

    # Illesztés
    reg = linear_model.LinearRegression(fit_intercept=True)
    
    X=[[aX_float[i]] for i in range(len(aX_float))]     # elemenként kell egy-egy egydimenziós tömb (pl  [[1],[5],[7]])
    # print(X)
    reg.fit(X,aY)       
    
    # Koefficiensek, pontosság
    if outtype=='params': return (reg.coef_[0],reg.intercept_) 

    print('coeff:' + str(reg.coef_) + ' y0:' + str(reg.intercept_) + ' score:' + str(reg.score(X,aY)) )
    # - meredekség (feature-önként egy-egy meredekség;  súlyfaktornak is tekinthető)
    # - y0
    # - illesztés minősége

    if resample==None or resample=='' or resample=='0':
        aY=reg.predict(X)      # eredeti x helyeken
        #if datetimeindex:
        #    for i in range(len(aX)): aX[i]=pd.to_datetime(aX[i],unit='D')
        return pd.Series(aY,aX)
    
    else:
        aL=resample.split(',')
        if len(aL)==0: resample='2'
        else: resample=int(aL[0])

        extend_left=0
        extend_right=0
        if extend:
            aL=extend.split(',')
            if len(aL)==1:
                extend_left=int(aL[0])
                extend_right=int(aL[0])
            elif len(aL)>=2:
                extend_left=int(aL[0])
                extend_right=int(aL[1])
        minx=min(aX_float)
        maxx=max(aX_float)
        unit=(maxx-minx)/resample
        minx -= extend_left*unit
        maxx += extend_right*unit
        resample += extend_left + extend_right

        aX=np.linspace(minx,maxx, num=resample, endpoint=True)
        X=[[aX[i]] for i in range(len(aX))]     # elemenként kell egy-egy egydimenziós tömb (pl  [[1],[5],[7]])
        aY=reg.predict(X)

        if datetimeindex:
            for i in range(len(aX)): aX[i]=pd.to_datetime(aX[i],unit='D')
        return pd.Series(aY,aX)

def FvSmoothSpline(ser,diffByMaxmin=0.01,resample='100,500',extend='',trendpoint=None):
    # Elsőként egy lineráris interpoláció, majd spline illesztés
    # ser:  tetszőleges mintavételsor,   az input x értékekeknek nem kell egyenletesnek lenniük és rendezés sem szükséges
    #    - az nan értékeket nem veszi figyelembe
    #    - ha ismétlődő x értékek fordulnak elő, akkor csak egyet őriz meg közülük
    #    - a kimeneti x értékek egyenletesek és rendezettek lesznek
    # diffByMaxmin:  milyen arányú átlagos eltérés megengedett a függvény min-max intervallumához képest
    #     Addig szaporítja a csomópontok számát, amíg a pontosság megfelelővé válik 
    # resample:    egyenletes x-eloszlás az xmin és xmax között
    #     '0,500':     nincs lineáris resample, a spline resample 500 pontos 
    #     '100,500':   100 pontos lineáris resample, majd 500 pontos spline resample
    #     '200':   megegyező lineáris és out resample
    # extend:   hány ponttal bővítse ki a kemeneti görbét a bal illetve a jobb szélen  (ritkán alkalmazható, a spline előrejelzésre nem nagyon alkalmas)
    #     '0,200':   bal oldalon nincs bővítés, jobb oldalon 200 pont


    serL=ser.dropna()       # üres értéket tartalmazó rekordok kidobása (nem tud velük mit kezdeni a spline)
    serL=serL.sort_index()
    serL=serL.drop_duplicates()   # hibát okozna a lineráris és a spline illesztésnél is
    
    std=serL.std()   # nagyjából a max-min felének felel meg (extrém értékektől eltekintve)

    aX=serL.index.array 
    aY=serL.array

    if diffByMaxmin==None: diffByMaxmin=0.01
    s=2*std*diffByMaxmin
    s=(s**2)*len(aX)       # a UnivariateSpline argumentumaként az eltérés-négyzetek összegének max értékét kell megadni

    if resample==None or resample=='': resample='100,500'
    if type(resample)==int:
        resample_linear=resample
        resample_spline=resample
    elif type(resample)==str:
        aL=resample.split(',')
        if len(aL)==1:
            resample_linear=int(aL[0])
            resample_spline=resample_linear
        elif len(aL)>=2:
            resample_linear=int(aL[0])
            resample_spline=int(aL[1])


    datetimeindex = ('timestamp' in str(type(aX[0])))

    if datetimeindex:
        aX_float=[0]*len(aX)
        for i in range(len(aX)): aX_float[i]=aX[i].timestamp()/(24*60*60)
    else: aX_float=aX

    # Elsőként egy lineáris interpoláció, hogy egyenletes eloszlásúak legyenek az x pontok
    #    Ha nem egyenletes, akkor a ritkább régiókban rosszul viselkedhet a spline (erőteljes "kihasasodások" fordulhatnak elő)    
    if resample_linear>0:
        f=interp1d(aX_float,aY,assume_sorted=True)
        lenL=max(aX_float)-min(aX_float)
        minL=min(aX_float)
        maxL=max(aX_float)
        #minL=min(aX_float) + (lenL/resample_linear)/2       # nem megy teljesen a széléig (a széleken bizonytalanság lehet)
        #maxL=max(aX_float) - (lenL/resample_linear)/2
        aX_floatL=np.linspace(minL,maxL, num=resample_linear, endpoint=True)
        aYL=f(aX_floatL)
    else: 
        aX_floatL=aX_float
        aYL=aY

        #if datetimeindex:
        #    aX=[0]*len(aX_floatL)
        #    for i in range(len(aX)): aX[i]=pd.to_datetime(aX_floatL[i],unit='D')
        #else: aX=aX_floatL
        #plt.scatter(aX,f(aX_floatL))
    
    if trendpoint:
        print(trendpoint)
        aX_floatL=np.append(aX_floatL,datefloat(trendpoint[0]))
        aYL=np.append(aYL,trendpoint[1])


    f=UnivariateSpline(aX_floatL,aYL,k=3,s=s)    
    
    # spline resample
    extend_left=0
    extend_right=0
    if extend:
        aL=extend.split(',')
        if len(aL)==1:
            extend_left=int(aL[0])
            extend_right=int(aL[0])
        elif len(aL)>=2:
            extend_left=int(aL[0])
            extend_right=int(aL[1])
    minx=min(aX_float)
    maxx=max(aX_float)
    unit=(maxx-minx)/resample_spline
    minx -= extend_left*unit
    maxx += extend_right*unit
    resample_spline += extend_left + extend_right

    aX_float=np.linspace(minx,maxx, num=resample_spline, endpoint=True)
    aY=f(aX_float)

    if datetimeindex:
        aX=[0]*len(aX_float)
        for i in range(len(aX)): aX[i]=pd.to_datetime(aX_float[i],unit='D')
    else: aX=aX_float
    serout=pd.Series(aY,aX)
    
    return serout



# GRADIENS, INTEGRÁL, NORMALIZÁLÁS, RESAMPLE

def FvGradient(ser):
    '''
    Lényegesen pontosabb a naív algortimusoknál
    Időtartam:  1000-es pontsorozat esetén kb 0.2 msec
    Az edge_order=2 utólag került be. Pontosabb a széleken, és nem látszik lassulás
        (enélkül az fn_erlang harmadik deriváltjánál a széleken torzulás látszott)
    '''
    return pd.Series(np.gradient(ser.to_numpy(),edge_order=2),ser.index)



def FvIntegrate(ser):
    # A függvénygörbe alatti területet adja vissza  (ha az integrált függvény kell, akkor cumsum)
    # Elviekben resample + átlagszámolással is megoldható (ha nem egyenletes a lépésköz, akkor torz lenne az eredmény)
    # A trapéz-alapú számolás feltehtőleg gyorsabb, mint a resample-hez szükséges lineáris interpolációk
    X=ser.index
    if 'Datetime' in str(type(X)): X=datefloatA(X)
    return trapz(ser.array,X)


def normalizeSer(ser,maxout=1,faktorout=False,robust=False):
    '''
    robust: True esetén az outlier-ek nélkül számítandó a skálázás (az outlierek a [-1:1] tartományon kívülre kerülnek)
    '''

    faktor=1

    if robust:
        values = fn_outliers(ser.to_numpy(),quote='auto',method='quantile',drop=True)
        maxIn=values.max()
        minIn=values.min()
    else:
        maxIn=ser.max()
        minIn=ser.min()
    if abs(maxIn)>abs(minIn) and abs(maxIn)>0: faktor=(maxout/abs(maxIn))
    elif abs(minIn)>0: faktor=(maxout/abs(minIn))

    if faktor!=1: ser=ser*faktor
    
    if faktorout: return ser,faktor
    else: return ser

def FvNormalize(na):    # -> naNormalized
    # [-0.5,0.5] sávba transzformálja a tömböt
    max=np.nanmax(na)   # nan értékeket ne vegye figyelembe
    min=np.nanmin(na)
    if abs(max)>abs(min): na=na*0.5/abs(max)
    else: na=na*0.5/abs(min)
    return na


def FvLinearResample(ser,density=None,count=None,X_out=None,kind='linear'):
    '''
    Egyenletesen elosztott x értékek választása a teljes tartományra. Az y értékek lineáris interpolációval
    Akkor érdemes alkalmazni, ha az x pontok eredeti eloszlása sztochasztikus (pl. nem szisztematikus mérési adatok, korrelációszámítások)
    Megfelelően megválasztott pontsűrűséggel a függvénygörbe simítására is alkalmazható
    density:  az eredeti x-irányú pontsűrűségének hányszorosa legyen a visszaadott görbe pontsűrűség (default:4)
        - ha az eredeti x-értékek is egyenletesen voltak elosztva, akkor változatlanul benne lesznek az X-ben
            (a két szélső pont egyenetlen esetben is benne lesz)
        - count = int((lex(X)-1) * density) + 1
    count:  vagylagos a density-vel - hány pontos legyen a resample 
    X_out:  ha meg van adva, akkor érdektelen a density és a count
        ha a ser x-tartományán kívüli pontokra is tartalmaz x-értékeket, akkor np.nan íródik be
        Akkor érdemes alkalmazni, ha több ser-re is ugyanaz az X pontosorozat szükséges
    kind:  'linear' '2': másodrendű spline   '3': harmadrendű spline, ...    (további lehetőségeket lásd: scipy.interpolate.interp1d)
        - a 'linear'-on kívüli típusok csak rendkívüli esetben (jobb megoldás a resample után egy Gauss mozgóátlag)
    '''

    ser=ser.dropna()        # az üres értékekkel nem tud mit kezdeni
    if len(ser)==0: return None

    aRec=list(zip(ser.index,ser))
    aRec=FvLinearResampleX(aRec,density,count,X_out,kind)
    return SerFromRecords(aRec)

def FvLinearResampleX(aXy:list,density=None,count=None,X_out=None,kind='linear'):   # -> aRecEqualized
    # Egyenletesen elosztott x értékek választása a teljes tartományra. Az y értékek lineáris interpolációval
    # Akkor érdemes alkalmazni, ha az x pontok eredeti eloszlása sztochasztikus (pl. nem szisztemaikus mérési adatok, korrelációszámítások)
    # Megfelelően megválasztott pontsűrűséggel a függvénygörbe simítására is alkalmazható (messze nem olyan hatékony, mint a spline)
    # density:  az eredeti x-irányú pontsűrűségének hányszorosa legyen a visszaadott görbe pontsűrűsége (default: 4) 
    #    - ha az eredeti x-értékek is egyenletesen voltak elosztva, akkor változatlanul benne lesznek az X-ben
    #    - count = int((lex(X)-1) * density) + 1
    # count:  vagylagos a density-vel - hány pontos legyen a resample 
    # kind:  'linear' '2': másodrendű spline   '3': harmadrendű spline, ...    (további lehetőségeket lásd: scipy.interpolate.interp1d)

    if len(aXy)==0: return []

    if isinstance(count,float): count=int(count)
    
    aXy.sort(key = lambda x: x[0])
    aX,aY=unzip(aXy)
    nLen=len(aXy)
    aX=list(aX)         # módosításra lesz szükség, ezért át kell térni list-re (eredetileg tuple)
    
    datetimeindex = ('timestamp' in str(type(aX[0])))
    if datetimeindex:
        aX_float=[0]*len(aX)
        for i in range(len(aX)): aX_float[i]=aX[i].timestamp()/(24*60*60)
    else: aX_float=aX

    xmin=min(aX_float)
    xmax=max(aX_float)
   

   
    # unicitás biztosítása  (előfeltétele az interpolációnak)
    xlast=None
    for i,x in enumerate(aX_float):
        if xlast and x<=xlast: 
            x=xlast+(xmax-xmin)/(len(aX)*1000000)       # eltolás az átlagos x-távolság milliomod részével
            aX_float[i]=x
        xlast=x
    
    # interpoláció
    f=interp1d(aX_float,aY,assume_sorted=True,kind=kind)
    
    if X_out is None:
        if not count:
            if not density: density=4
            count=int((len(aX_float)-1)*density)+1  
            # - ha aX_float egyenletes volt, akkor mindegyik érték benne marad a kimenetben is

        X_out=np.linspace(xmin,xmax, num=count, endpoint=True)
        X_out_middle=X_out
        X_out_left=[]
        X_out_right=[]

    else:
        X_out=array(X_out)
        X_out_left=X_out[X_out<aX_float[0]]
        X_out_right=X_out[X_out>aX_float[-1]]
        X_out_middle=X_out[(X_out>=aX_float[0]) & (X_out<=aX_float[-1])]

    Y_out=f(X_out_middle)
    # túllógó részeken np.nan
    if len(X_out_left)>0:  Y_out=Concat([np.nan]*len(X_out_left),Y_out)
    if len(X_out_right)>0:  Y_out=Concat(Y_out,[np.nan]*len(X_out_right))

    if datetimeindex:  X_out=floatdate(X_out)


    return list(zip(X_out,Y_out))         # a zip összekapcsolja a két tömböt (elempárok)


def fn_maxpos_precise(ser,maxpos,halfwidth=2,resample=100):     # diszkrét maxhely pontosítása a környező pontok alapján
    '''
    Visszaadja a pontosított maxhelyet. Legfeljebb a szomszédos pont feléig tolódhat el
    ser:  elvárás, hogy az x legyen unique és rendezett
       Ha a maxhely egyenlő valamelyik szomszédos ponttal, akkor nagyjából a kettő között középen, de függ a további szomszédoktól is
    maxpos:  diszkrét maxhely (nem iloc-index, hanem a ser-ben előforduló x-érték)
        ser.index.get_loc() függvénnyel kéri be a függvény az iloc-indexet (legközelebbi találat)
    halfwidth: hány szomszédot vegyen figyelembe előtte illetve utána  (2 esetén 5 pontos környezet)
        Elvárás, hogy a maxhely körül legyen ennyi pont.
    resample:  a teljes 5 pontos környezetre vonatkozik  (100 esetén átlagosan 25 pont a pontok között)
        
    return: pontosított maxhely.  
    Algoritmus:  resample a kiinduló maxhely körüli 5 pontra, Gauss mozgóátlag, G = 2 * mean_period, maxhely keresés
    '''
    
    i_maxpos=ser.index.get_loc(maxpos)

    # Előfordulhat, hogy azonos értékek vannak a maxpos körül. Fel kell deríteni az esetleges plató határait
    nLen=len(ser)
    Y = ser.values
    Y_max=Y[i_maxpos]
    i_maxpos_right=i_maxpos
    while i_maxpos_right+1<nLen and Y[i_maxpos_right+1]==Y_max:  i_maxpos_right+=1 
    i_maxpos_left=i_maxpos
    while i_maxpos_left-1>=0 and Y[i_maxpos_left-1]==Y_max:  i_maxpos_left-=1 

    # ha nincs elég szomszéd, akkor hibaüzenet és visszaadja az eredeti maxpos értéket
    if i_maxpos_left<halfwidth or i_maxpos_right>=len(ser)-halfwidth:
        print('fn_maxpos_precise   A kiinduló maxpos érték nem lehet közelebb a szélekhez halfwidth-nél')
        return maxpos
    
    ser_surround = ser.iloc[i_maxpos_left-halfwidth:i_maxpos_right+halfwidth+1]      # a záróérték is benne van
    x_width=(ser_surround.index[-1]-ser_surround.index[0])
    # delta_avg=x_width/(2*halfwidth)    # 5 pont esetén 4 intervallum (nem kell)

    # pltinit()
    # FvPlot(ser_surround,'scatter')
    
    ser_surround = FvLinearResample(ser_surround,count=resample)
    ser_surround = FvGaussAvg(ser_surround,gausswidth=0.5)   
    i_maxposL=np.argmax(ser_surround.values)
    maxpos=ser_surround.index[0] + x_width * i_maxposL/resample
    
    # FvPlot(ser_surround,'original',annot='max',annotcaption='x')
    # pltshow()
        
    return maxpos



def FvEventCountPerDay(tblEvents,query=None):
    ''' Időponttal (dátummal) rendelkező esemény-rekordok számlálása naponként 
    tblEvents:  datetime index az elvárás (nem kell unique-nak lennie az index-nek, tetszőleges további kiegészítő adatai lehetnek)
    query:  előzetes szűrés az eseményekre   pl.  "Eseménytípus=='reg'"
    return:  ser,  nap-felbontású idősor (a kulcsa unique)
                   az értéke az adott napra eső események száma
    '''
    if query:
        tblEvents=tblEvents.query(query)
    dates=tblEvents.index.array
    ser=pd.Series([1]*len(dates),dates)
    ser=ser.resample('D').count()
    return ser


def fn_outliers(arr, quote=0.01, method='isolation forest',out_='values',drop=False):    # outlier-ek egy értékkészletben
    '''
    arr:  összes előforduló érték.  Egydimenziós numpy tömb vagy list
        - az nan értékek nem minősülnek outlier-nek
    quote:  
        max ekkora hányad lehet az outlierek aránya  
            - 'quantile' esetén maximum ekkore lehet az outlier-ek aránya (akár 0 is lehet; ha nagyobb lenne ennél, akkor IsolationForest szűkítés)
        'auto':  automatikusan megállapított hányad
    method:
        'LOF':      LocalOutlierFactor      Sűrűség-alapú outlier keresés
        'isolation_forest':  izoláltság alapján
        'qauntile':   ilyenkor is megadható quote (nem kötelező)
            - ha nincs megadva quote, akkor a standard kirtérium: a középtartománytól 1.5 szélességnél távolabbi pontok
    out_:
        'values'        outlier értékek  (drop esetén a nem-outlier értékek)
        'both'          két tömböt ad vissza   non_outliers, outliers   (drop érdektelen)
        'mask':         azonos méretű tömb, true/false értékekkel (drop=true True/False váltás)
                        Series esetén:  ser.iloc[mask]  -  csak a True rekordok maradjanak meg
        'indexes':      indexek felsorolása         (drop esetén a nem-outlierek)
    drop:  True esetén a nem outlier értékeket adja vissza

    return:  outlier értékek  (egydimenziós numpy tömb)
        - drop=True esetén a nem-outlier értékek
    '''
    arr=array(arr)
    
    # arr[np.isnan(arr)] = np.nanmedian(arr)     # nem dobom ki az nan értékeket, mert out_=indexes esetén elcsúszást okozna
    arr = arr[~np.isnan(arr)]       # nan értékek kidobandók

    if len(arr)==0: 
        return []

    if out_=='both': bDrop=False        # ilyenkor érdektelen az arg-ban megadott érték
        

    if method=='quantile':
        faktor=1.5
        iqr_quantile=0.5
        # faktor=1
        # iqr_quantile=0.7

        q3=np.quantile(arr,(1+iqr_quantile)/2)
        q1=np.quantile(arr,(1-iqr_quantile)/2)
        iqr=q3-q1       # interquantile range,  a középső 50% pont szélessége
        min_=q1 - 1.5*iqr
        max_=q3 + 1.5*iqr

        # Ha meg van adva konkrét hányad, és a quantile több outlier-t adna a kvótánál, akkor további finomítás
        if type(quote)==float:
            quote_ = len(arr[(arr<min_) | (arr>max_)]) / len(arr)
            if quote_ > quote:          # kisebb minden további nélkül lehet (sőt előfordulhat, hogy egyetlen outlier sincs)
                if iqr==0: iqr=(max(arr) - min(arr)) * quote    # hasraütés, de mindenképpen >0 kell
                faktor_min=faktor
                faktor_max = max((max(arr)-q3)/iqr,(q1-min(arr))/iqr)
                iter_count=0 
                while abs(quote_ - quote)/quote > 0.01  and  iter_count<20:
                    faktor=(faktor_min + faktor_max)/2
                    min_=q1 - faktor*iqr
                    max_=q3 + faktor*iqr
                    quote_ = len(arr[(arr<min_) | (arr>max_)]) / len(arr)
                    if quote_>quote:    faktor_min=faktor
                    else:               faktor_max=faktor
                    iter_count += 1
            if drop: 
                mask=(arr>=min_) & (arr<=max_)
            else: 
                mask=(arr<min_) | (arr>max_)
        else:
            if drop: 
                mask=(arr>=min_) & (arr<=max_)
            else: 
                mask=(arr<min_) | (arr>max_)

    elif beginwith(method.lower(),'lof|local'):
        x=arr.reshape(len(arr),1)
        y = LocalOutlierFactor(contamination=quote).fit_predict(x)
        if drop:    mask=  (y==1)
        else:       mask=  (y==-1)        # -1 jelzi az outlier-eket

    elif beginwith(method,'isolation'):
        x=arr.reshape(len(arr),1)
        y = IsolationForest(contamination=quote).fit_predict(x)          # 1 százalék
        if drop:    mask=  (y==1)
        else:       mask=  (y==-1)        # -1 jelzi az outlier-eket

    if out_=='mask': return mask
    elif out_=='values': return arr[mask]
    elif out_=='both': return arr[~mask],arr[mask]
    elif out_=='indexes': return range(0,len(arr))[mask]

def fn_outlier_limits(arr):             # pl. plot-hoz,  xmin,xmax    vagy   ymin,ymax
    nonoutliers = fn_outliers(arr,method='quantile',drop=True)
    return nonoutliers.min(),nonoutliers.max()

def ser_dropoutliers_x(ser):    # x irányban kidobja a széleken lévő sporadikus pontokat (pl. kooreláció diagramban, Gauss illesztéshez)
    mask = fn_outliers(ser.index.to_numpy(),method='quantile',out_='mask',drop=True)
    return ser.iloc[mask]






# RECT
def FvRectOverlap(rectA:tuple,rectB:tuple):  #  ->float  (az átfedő terület nagysága)
    # rect[0]:x_left   rect[1]:y_bottom   rect[2]:x_right   rect[3]:y_top
    if rectA[2]<=rectB[0] or rectA[0]>=rectB[2] or rectA[3]<=rectB[1] or rectA[1]>=rectB[3]: return 0
    x = max(rectA[0], rectB[0])         # nagyobbik left
    y = max(rectA[1], rectB[1])         # nagyobbik bottom
    w = min(rectA[2], rectB[2]) - x     # kisebbik right - nagyobbik left
    h = min(rectA[3], rectB[3]) - y     # kisebbik top - nagyobbik bottom
    return (w*h)   

def FvRectOverlapHeight(rectA:tuple,rectB:tuple):  #  ->float  (az átfedés magassága; a szélessége érdektelen)
    if rectA[2]<=rectB[0] or rectA[0]>=rectB[2] or rectA[3]<=rectB[1] or rectA[1]>=rectB[3]: return 0
    return min(rectA[3], rectB[3]) - max(rectA[1], rectB[1])

def FvRectArea(rect:tuple): # ->float     (a rect területe)
    return (rect[2]-rect[0])*(rect[3]-rect[1])

def FvRectShiftY(rect:tuple,deltay:float):  # ->tuple   
    return (rect[0],rect[1]+deltay,rect[2],rect[3]+deltay)





# DATETIME
# - datefloat és floatdate átkerült az ezhelper-be

def ser_reset_datetimeindex(ser):       
    # datemindex kidobása és helyébe egyszerű sorszámozott index
    # Visszaadja az eredeti kezdődátumot és a kalkulált lépésközt (float)
    # Ha a dátumindex lépésköze konstans volt, akkor a restore művelettel teljesértékűen vissza lehet majd állítani
    #   a datetime indexet
    date0=None
    date_step=None
    datefloat0=None
    if 'date' in str(type(ser.index)): 
        datefloat0=datefloat(ser.index[0])
        date_step=(datefloat(ser.index[-1])-datefloat0) / (len(ser)-1)
        ser=ser.reset_index(drop=True)      # nem kell datetime index, egyszerű sorszámozás
    return ser,datefloat0,date_step

def ser_restore_datetimeindex(ser,datefloat0,date_step):
    # Visszaállítja a dátumindexet az autosorszámozott index helyett
    # date0: induló dátum (float vagy datetime)
    # date_step: dátum lépésköz (float)
    if not datefloat0 or not date_step: return ser

    X = ser.index.values
    ser.index = floatdate(datefloat0 + X*date_step)

    return ser



# KOORD TRANSZFORMÁCIÓK

def axeskoord(x,y,  unit='0-1'):                # Data to axeskoord   (axeskoord:  a rajzolási tartomány határai [0,1]-nek felelnek meg)
    # a diagram rajzolási területéhez viszonyított relatív koordinátákat adja vissza,    [0-1] vagy inch vagy point mértékegységben
    # Fontos: a diagram-terület átméretezésével érvénytelenné válhatnak az inch-ben és point-ban visszaadott koordináták
    # x: dátum is lehet  (szöveges vagy pd.Timestamp)
    # A speciális skálázásokat is tudja kezelni  (pl. log)
    # unit:  '0-1', 'point', 'inch' (=72 point)
    x=datefloat(x)
    # koordináták [0-1] mértékegységgel  (axespercent)
    xy=plt.gca().transLimits.transform(plt.gca().transScale.transform((x,y)))    # figyelembe veszi a log skálázást is
    xunit,yunit=axesunits(unit)       # unit='0-1' esetén (1,1) a result
    return (xy[0]*xunit,xy[1]*yunit)

def axeskoordA(aX,aY,unit='0-1'):
    # a diagram rajzolási területéhez viszonyított relatív koordinátákat adja vissza [0-1], inch vagy point mértékegységben
    # Fontos: a diagram-terület átméretezésével érvénytelenné válhatnak az inch-ben és point-ban visszaadott koordináták
    # aX: dátumokat is tartalmazhat  (szöveges vagy pd.Timestamp)
    # A speciális skálázásokat is tudja kezelni  (pl. log)
    # unit:  '0-1', 'point', 'inch'
    xunit,yunit=axesunits(unit)       # '0-1' esetén (1,1) a result;   a bbox teljes szélessége/magassága point-ban / inch-ben
    #print('xunit:' + str(xunit))
    xlim=plt.xlim()         # dummy hívás;  enélkül a transform műveletek nem működnek jól (vélhetően inicializál valamit)
    aXOut=[0]*len(aX)
    aYOut=[0]*len(aY)
    for i in range(len(aX)):
        x=datefloat(aX[i])
        xy=plt.gca().transLimits.transform(plt.gca().transScale.transform([x,aY[i]]))    # figyelembe veszi a log skálázást is
        # - elvileg mindkét koordinátának 0-1 közé kellene esnie, mert a limitek a min-max értékek alapján lettek beállítva
        #print('x:' + str(x) + ' x_transzf:' + str(xy[0]))
        aXOut[i]=xy[0]*xunit
        aYOut[i]=xy[1]*yunit
    return (aXOut,aYOut)

def axesunits(unit='point'):        # a koordináta-tér méretei point-ban, inch-ben (viszonyítható a fontok méretéhez)
    # Ha az a kérdés, hogy hány karakter fér ki vízszintesen a rajzterületen, akkor a charwidth=fontsize*5/8 ökölszabály alkalmazható
    #   nCharb = axesunits[0] / (fontsize*5/8)

    if unit=='0-1': return (1,1)
    bbox = plt.gca().get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())       # a koordináta tengelyek befoglaló rect-je
    if unit=='inch': return (bbox.width,bbox.height)
    elif unit=='point': return (bbox.width*72,bbox.height*72)       



# LOSS FÜGGVÉNYEK
#  (korábban az ez_sklearn-ben volt)

def fn_smape(ytrue,ypred,resolution=None,vektoravg='avg'):      # symmetric mean absolute precentege error
    '''
    Az sklearn mean_absolute_percentage_error függvényének szimmetrikus változata
    A két szám átlagához viszonyított abszolút eltérés
    Kis ytrue tartományban sem száll el, mindig [0,2] értéket ad visssza   (ha 1-re normált indikátor kell, akkor egyszerűen osztható kettővel)
    A resolution megadásával az ytrue=0 értékeket is viszonylag jól kezeli (pl. kvantált ytrue esetén)

    Az interpretálása is meglehetősen kézenfekvő. Valójában ez felel meg jobban az  "átlagosan 20%-os az eltérés" kifejezésnek
        ("hibahatár", "tűrésküszöb", ...)

    abs(diff) / mean
    abs(ytrue-ypred) / (abs(ytrue) + abs(ypred))/2

    MAPE_lower = 1 - (2-SMAPE)/(2+SMAPE)     
    MAPE_upper = (2+SMAPE)/(2-SMAPE) - 1
    
    0%     teljes egyezés
    20%    lefelé 18%-os, felfelé 22% eltérés
    33%    lefelé 29%-os, felfelé 40%-os eltérés 
    50%    lefelé 40%-os, felfelé 67%-os eltérés
    66%    lefelé 50%-os, felfelé 100%-os eltérés
    100%   lefelé 67%-os, felfelé 200%-os eltérés
    120%   lefelé 75%-os, felfelé 300%-os eltérés 
    150%   lefelé 86%-os, felfelé 600%-os eltérés
    200%   végtelenhez tartó szorzó a két szám között
        Csak akkor lehet 200%, ha valamelyik érték 0 vagy eltér az előjelük  (előjel eltérés esetén mindig 200%)

    Vektorokra is működik. Alapesetben egyszerű átlagolás, de kérhető a számlálók és a nevezők külön felösszegzése, arányképzéssel
        (nagyobb az outlier-tűrése, mint az átlagolós változatnak)
    Csak >0 adatsorokra ad értelmes eredményt
    
    resolution:  az ennél kisebb eltérések csak részlegesen veendők figyelembe
        Hozzáadódik az ytrue és az ypred értékhez is a resolution fele  (epszilon = resolution/2)
        - elsősorban az ytrue=0, ypred>0 esetén fontos, mert enélkül egészen kis eltérés esetén is 200% lenne az eredmény
        - kezeli azt az esetet is, amikor ytrue és ypred is 0
        - a darabszám jellegű ytrue értékek eleve kvantáltak. Ilyenkor az elemi lépésközt kell megadni resolution-ként
        - másik megközelítés:  mérési hiba
        - ökölszabály:   a maximális mért érték 1 százaléka vagy ezreléke    ("1 ezrelékes mérési pontosság";  ekkora hibát még nem tekintünk valódi hibának)

    vektoravg:    vektorok esetén átlag ('avg')  vagy szummák hányadosa  ('sum'),   'vektor' esetén vektort ad vissza
        Ha az input float (nem array), akkor érdektelen
    '''
    
    if type(ytrue) == list: ytrue=np.array(ytrue)
    if type(ypred) == list: ypred=np.array(ypred)

    if resolution and resolution>0:
        epszilon=resolution/2
        ytrue[ytrue<epszilon]=epszilon
        ypred[ypred<epszilon]=epszilon

    számláló = np.abs(ytrue - ypred)
    nevező = (np.abs(ytrue) + np.abs(ypred))/2       


    if vektoravg=='vektor': return számláló / nevező
    elif vektoravg=='avg': return np.mean(számláló / nevező, where = (nevező!=0))
    elif vektoravg=='sum': return np.sum(számláló) / np.sum(nevező)

    
def fn_diffpercent_global(ytrue,ypred,relate_to='ytrue_mean'):   # az eltéréseket a globális (0-k nélküli) átlagértékhez viszonyítja
    '''
    Többnyire ez a leginkább plauzibilis mutató, de csak a szigorúan >0 értékkészletek esetén (y_true>0)
    Függvényillesztés minőségének vizuális megállapításakor is jól alkalmazható.
    A nagy értékeknél és a kis értékeknél is ugyanaz a viszonyítási alap
       A MAPE, SMAPE ezzel szemben minden pontra egyedi viszonyítási alapot ad.
    
    ytrue:  alapesetben ytrue>=0 az elvárás
        ha minden értéke 0, akkor kötelezően np.nan a result (a nullák nélküli átlaghoz viszonyít)
    ypred:   lehetnek negatív értékei is
    relate_to:  
        'ytrue_mean':  az ytrue átlagértékéhez viszonyít
        'ytrue_max':   az ytrue maxértékéhez viszonyít
        - megadható egyedi érték is.  Példa: az ytrue csak egy résztartománya a teljes görbének, és a teljes görbe átlaga vagy maximuma a mérvadó
    '''
    if type(ytrue) == list: ytrue=np.array(ytrue)
    if type(ypred) == list: ypred=np.array(ypred)
    
    # Viszonyítási alap:  az ytrue értékek átlaga, csak a >0 értékeket figyelembe véve
    if relate_to=='ytrue_mean': relate_to = ytrue.mean(where=ytrue>0)
    elif relate_to=='ytrue_max': relate_to = ytrue.max()
        
    if relate_to==0 or relate_to==np.nan: return np.nan

    # abszolút eltérések átlaga
    diff_mean = abs(ypred-ytrue).mean()
             
    return diff_mean / relate_to 





# MATEMATIKAI FÜGGVÉNYEK

def fn_gauss(x,x0,szigma):
        ''' 1-re normált gauss'''
        return math.exp(-((x-x0)**2)/(2*szigma**2))

def FvGauss(x,x0,G,ymax=1):     # gauss görbe, az x array vagy ser is lehet
    ''' Gauss görbe.  Az integrálja nem feltétlenül 1 '''
    szigma=G/4
    # if type(x)==list or type(x)==np.ndarray:
    #     return ymax*[math.exp((-(xL-x0)**2) / (2*szigma**2)) for xL in x]
    # else:
    if type(x)==list: x=array(x)
    return ymax*np.exp((-(x-x0)**2) / (2*szigma**2))

def fn_gauss_multi(X,*params,outtype='sum'):
    '''
    FONTOS: ha a params egy list, akkor a függvény hívásakor ne felejtds elé írni a *-t 
    
    X:       list, array vagy egyetlen float.      Dátumok is megadhatók (float-ra konvertálja)
    params:  maxpos,maxvalue,G, ..., maxpos,maxvalue,G
        - négyes csoportokban kell megadni az elemi felfutások paramétereit
        - G=4*szigma,   a lognormal és az Erlang leftwidth paraméteréhez hasonló (a "talppont" a teljes bal-szélesség felénél van)
    outtype: "sum"       "array"  (rész-erlang idősorok tömbje)
        - Y  vagy array of Y
    '''
    if len(params)==0:
        return array([0]*len(X))
    
    X=datefloat(X)      

    surge_count = len(params) // 3

    idosor = array([0.]*len(X))     #
    idosorok = []

    for i in range(surge_count):
        i_first=3*i
        maxpos=params[i_first]
        maxvalue=params[i_first+1]
        G=params[i_first+2]

        try:
            idosorL = FvGauss(X,x0=maxpos,G=G,ymax=maxvalue)
            idosor += idosorL
            idosorok.append(idosorL)
        except Exception as e:		  # tetszőleges exception leírásának megjelenítése
            print(e)


    if outtype=='sum': return idosor
    elif outtype=='array': return idosorok

def fn_gauss_fit(X,Y,params0,lower,upper,scale):
    '''
    Gauss illesztés az X,Y pontsorozatra.
    Több-felfutásos illesztés is kérhető   (a params0 elemszáma határozza meg a felfutások számát)

    params0: a paraméterek kezdőértéke   
        - egy vagy több paraméter-hármas adható meg, szekvenciálisan   (3,6,9, ... lehet az elemszám)
            maxpos,maxvalue,G
    lower_bounds: alsó határok paraméterenként
    upper_bounds: felső határok paraméterenként
    x_scale:  a paraméterek nagyságrendje (jellemző értéke)
    '''

    Y_pred=None
    last_params=None
    def f_gauss_multi(X,*params):
        nonlocal Y_pred,last_params
        # params:  maxpos,maxvalue,leftwidth,modus, ...
        last_params=params
        Y_pred = fn_gauss_multi(X,*params) 
        return Y_pred

    try:
        popt,pcov = curve_fit(f_gauss_multi,X,Y,p0=params0,bounds=(lower,upper),
                        method='dogbox',ftol=1e-8,xtol=1e-8,maxfev=None,verbose=0,x_scale=scale)
    except Exception as e:
        if (nincsbenne(str(e),'The maximum number of function evaluations is exceeded') and
        nincsbenne(str(e),'Residuals are not finite in the initial point')):
            print('ERROR   ' + str(e))
            raise e
        else:
            popt=last_params

    return popt



def fn_erlang(x,startpos,maxpos,kitevo):    # 1-re normált Erlang függvény
        '''
        Hatványfüggvénnyel indul (kitevő), és exponenciális lecsengéssel zárul
        
        A maxpos a baloldali szélesség. A jobb oldali (effektív) szélesség >maxpos
        Minél kisebb a kitevő, annál aszimmetrikusabb a görbe, tehát annál nagyobb a jobboldali szélesség
        A kitevőt csak metaszinten érdemes hangolni
        
        x: list vagy array is lehet  (a visszaadott érték array lesz)
        maxpos:  a startpos-hoz viszonyítva.  Más néven: bal oldali szélesség
        kitevo:  [1,150)  
            - minél nagyobb, annál szimmetrikusabb
            - 1 esetén erősen asszimmetrikus, lineáris indulással
            - a felső határt a lebegőpontos aritmetika határolja be (overflow)


        '''
        if type(x)==list: x=np.array(x)
       
        if type(x)==np.ndarray:
            if maxpos==0: return [0]*len(x)

            x_erlang=x-startpos
            # Ha az összes elem <=0, akkor return 0
            if (x_erlang<=0).all(): return [0]*len(x)   # kötelezően 0 a startpos előtt

            xL=(kitevo/maxpos)*x_erlang
            
            ymax=(kitevo/math.e)**kitevo  
            # ymax=kitevo**kitevo  *  math.exp(-kitevo)     # 16 esetén 0-t ad (nem tudom miért)


            # try:
            result = np.where(x_erlang>=0,(1/ymax) * (xL**kitevo) * np.exp(-xL),0)
            # except RuntimeWarning as e:
            #     print(f'Hiba történt: {e}')
        
            return result
            
            # # Ha az összes elem <=0, akkor return 0
            # if (x_erlang<=0).all(): return [0]*len(x)   # kötelezően 0 a startpos előtt
            
            # # Ha van legalább egy <0 elem, akkor ciklus kell
            # if (x_erlang<0).any():
            #     return [fn_erlang(x_,startpos,maxpos,kitevo) for x_ in x]
            #     # y=np.array([0]*len(x))
            #     # for i in range(len(x)):
            #     #     z=fn_erlang(x[i],startpos,maxpos,kitevo)
            #     #     y[i] = z
            #     # return y
            # else: 
            #     xL=(kitevo/maxpos)*x_erlang
                
            #     # ymax=(kitevo/math.e)**kitevo  
            #     ymax=kitevo**kitevo  *  math.exp(-kitevo)

            #     return (1/ymax) * (xL**kitevo) * np.exp(-xL)

           
        else:
            x_erlang=x-startpos

            if maxpos==0: return 0
            if x_erlang<=0: return 0   # kötelezően 0 a startpos előtt

            xL=(kitevo/maxpos)*x_erlang
            
            
            # ymax=(kitevo/math.e)**kitevo  
            ymax=kitevo**kitevo  *  math.exp(-kitevo)

            return (1/ymax) * (xL**kitevo) * np.exp(-xL)

def fn_erlang_multi(X,*params,outtype='sum',kitevo='fit_one'):
    '''
    FONTOS: ha a params egy list, akkor a függvény hívásakor ne felejtds elé írni a *-t 
    
    params:  kitevo1,maxpos1,maxvalue1,leftwidth1, ..., kitevoN,maxposN,maxvalueN,leftwidthN
        - ha kitevo=[value] vagy "fit_one", akkor lásd lejjebb
    outtype: "sum"       "array"  (rész-erlang idősorok tömbje)
        - Y  vagy array of Y
    kitevo:  ha szám, akkor mindegyik részerlangra ez érvényesül,   
        - számérték (float):    params = maxpos1,maxvalue1,leftwidth1, ..., maxposN,maxvalueN,leftwidthN
        - "fit_all":            params = kitevo1,maxpos1,maxvalue1,leftwidth1, ..., kitevoN,maxposN,maxvalueN,leftwidthN
        - "fit_one":            params = kitevo,  maxpos1,maxvalue1,leftwidth1, ..., maxposN,maxvalueN,leftwidthN
    '''

    if len(params)==0:
        return array([0]*len(X))
    
    X=datefloat(X)

    bKitevo = (kitevo!=None)

    if kitevo=='fit_one': 
        kitevo=params[0]
        erlang_count = (len(params)-1) // 3
        kitevo_type=1
    elif kitevo=='fit_all': 
        erlang_count = len(params) // 4
        kitevo_type=2
    else:
        erlang_count = len(params) // 3
        kitevo_type=0

    idosor = array([0.]*len(X))     #
    idosorok = []

    for i in range(erlang_count):
        if kitevo_type==2: 
            i_first=4*i
            kitevo=params[i_first]
            maxpos=params[i_first+1]
            maxvalue=params[i_first+2]
            leftwidth=params[i_first+3]
        else:
            if kitevo_type==1: i_first=3*i + 1
            else: i_first=3*i
            maxpos=params[i_first]
            maxvalue=params[i_first+1]
            leftwidth=params[i_first+2]

        startpos=maxpos-leftwidth
        maxpos=leftwidth        # itt relatív kell

        try:
            idosorL=maxvalue * array(fn_erlang(X,startpos,maxpos,kitevo))
            idosor += idosorL
            idosorok.append(idosorL)
        except Exception as e:		  # tetszőleges exception leírásának megjelenítése
            print(e)


    if outtype=='sum': return idosor
    elif outtype=='array': return idosorok

def fn_erlang_fit(X,Y,params0,lower_bounds,upper_bounds,x_scale,kitevo):
    '''
    Erlang illesztés az X,Y pontsorozatra.
    Multierlang illesztés is kérhető   (a params0 határozza meg az elemi erlangok számát)

    params0: a paraméterek kezdőértéke
        - a kitevo-tól függően erlangonként 3 vagy 4 paraméter (lásd ott)
    lower_bounds: alsó határok paraméterenként
    upper_bounds: felső határok paraméterenként
    x_scale:  a paraméterek nagyságrendje (jellemző értéke)

    kitevo:  ha szám, akkor mindegyik részerlangra ez érvényesül,   
        - számérték (float):    params = maxpos1,maxvalue1,leftwidth1, ..., maxposN,maxvalueN,leftwidthN
        - "fit_all":            params = kitevo1,maxpos1,maxvalue1,leftwidth1, ..., kitevoN,maxposN,maxvalueN,leftwidthN
        - "fit_one":            params = kitevo,  maxpos1,maxvalue1,leftwidth1, ..., maxposN,maxvalueN,leftwidthN
    '''

    Y_pred=None
    last_params=None
    def f_erlang_multi(X,*params):
        nonlocal Y_pred,last_params
        # params:  maxpos1,maxvalue1,leftwidth1, ..., maxposN,maxvalueN,leftwidthN
        last_params=params
        Y_pred = fn_erlang_multi(X,*params,kitevo=kitevo) 
        return Y_pred

    try:
        popt,pcov = curve_fit(f_erlang_multi,X,Y,p0=params0,bounds=(lower_bounds,upper_bounds),
                        method='dogbox',ftol=1e-8,xtol=1e-8,maxfev=None,verbose=0,x_scale=x_scale)
    except Exception as e:
        if (nincsbenne(str(e),'The maximum number of function evaluations is exceeded') and
        nincsbenne(str(e),'Residuals are not finite in the initial point')):
            print('ERROR   ' + str(e))
            raise e
        else:
            popt=last_params

    return popt

   
def fn_erlang_3points(X,Y,leftwidth,leftwidth_delta,kitevo=20):    # három pontos erlang illesztés
    '''
    Megpróbálja egy erlanggal majd két erlanggal. A jobbik illesztéssel kiszámítja az erlang paramétereket.
    
    X,Y:  három értékpár.  Az X-nek nem kell egyenletesnek lennie, de legyen növekvő és unique
    leftwidth:  Erlang-felfutási szélesség (x-irányban)
    leftwidth_delta:  max ekkora eltérés engedhető meg
    
    return:  erlangparams,diffpercent
    '''
    if type(X)==list: X=array(X)
    if type(Y)==list: Y=array(Y)


    # Proxy függvény az fn_erlang_multi-hoz
    # - azért szükséges, mert megszakított illesztési folyamat esetén is szükségem van arra az info-ra, hogy
    #    mennyire volt sikeres az illesztés
    Y_pred=None
    last_params=None
    fit_count=0
    def f_erlang_multi(X,*params):
        nonlocal Y_pred,fit_count,last_params
        # params:  maxpos1,maxvalue1,leftwidth1, ..., maxposN,maxvalueN,leftwidthN
        last_params=params
        Y_pred=fn_erlang_multi(X,*params,kitevo=kitevo)
        fit_count+=1
        return Y_pred

    def f_erlang_fit(X,Y,params0,lower_bounds,upper_bounds,x_scale,max_count):
        try:
            popt,pcov = curve_fit(f_erlang_multi,X,Y,p0=params0,bounds=(lower_bounds,upper_bounds),
                            method='dogbox',x_scale=x_scale,maxfev=max_count)  
        except Exception as e:
            if nincsbenne(str(e),'The maximum number of function evaluations is exceeded'):
                print('ERROR   ' + str(e))
                raise e
            else:
                # Alapesetben azért kerül ide, mert elérte a maximális lépésszámot
                # az Y_pred tartalmazza az utolsó lépés illesztett pontsorát
                diffpercent=fn_diffpercent_global(Y,Y_pred)
                return last_params,diffpercent
                    # - a fit_count (tehát az f_erlang_multi hívásainak száma) lényegesen nagyobb
                    #    a max_count-nál (ezek szerint egy fit-hez több függvényhívás tartozik, kb 30)

        erlangparams=popt
        Y_fitted=fn_erlang_multi(X,*popt,kitevo=kitevo)
        diffpercent=fn_diffpercent_global(Y,Y_fitted)

        return erlangparams,diffpercent


    # első próbálkozás: egyetlen erlang illesztése
    max_index=np.argmax(Y)
    if max_index==1:     # Ha a maxhely középen van
        maxvalue=Y.max()
        maxpos=X[1]
        maxpos_bounds=[X[0],X[2]]
        maxvalue_bounds=[0,5*maxvalue]
    elif max_index==0:     # Ha a maxhely a bal szélen van
        maxvalue=Y.max()*2
        maxpos=X[0] - leftwidth/2
        maxpos_bounds=[X[0]-2*leftwidth,X[0]]
        maxvalue_bounds=[0,10*maxvalue]
    elif max_index==2:     # Ha a maxhely a bal szélen van
        maxvalue=Y.max()*2
        maxpos=X[2] + leftwidth/2
        maxpos_bounds=[X[2],X[2]+2*leftwidth]
        maxvalue_bounds=[0,10*maxvalue]
    leftwidth_bounds=[leftwidth-leftwidth_delta,leftwidth+leftwidth_delta]

    params0=[maxpos,maxvalue,leftwidth]
    lower_bounds=[maxpos_bounds[0],maxvalue_bounds[0],leftwidth_bounds[0]]
    upper_bounds=[maxpos_bounds[1],maxvalue_bounds[1],leftwidth_bounds[1]]
    x_scale=[maxpos,maxvalue,leftwidth]

    max_count=100

    erlangparams1,diffpercent1 = f_erlang_fit(X,Y,params0,lower_bounds,upper_bounds,x_scale,max_count)

    # Csak akkor próbálkozik két Erlang-gal, ha a középső a legkisebb
    if np.argmin(Y)!=1:
        return erlangparams1,diffpercent1

    # Második próbálkozás:  2 Erlang görbe
    X_scale=X.mean()
    Y_scale=Y.mean()

                    # maxpos,maxvalue,leftwidth
    params0=        [X[0]-leftwidth/2,Y[0]*2,leftwidth,             X[2]+leftwidth/2,Y[2]*2,leftwidth]
    lower_bounds=   [X[0]-2*leftwidth,0,leftwidth-leftwidth_delta,  X[2],0,leftwidth-leftwidth_delta]
    upper_bounds=   [X[0],20*Y[0],leftwidth+leftwidth_delta,        X[2]+2*leftwidth,20*Y[2],leftwidth+leftwidth_delta]
    x_scale=        [X_scale,Y_scale,leftwidth,                     X_scale,Y_scale,leftwidth]

    erlangparams2,diffpercent2 = f_erlang_fit(X,Y,params0,lower_bounds,upper_bounds,x_scale,max_count)

    if diffpercent2<diffpercent1:
        return erlangparams2, diffpercent2
    else:
        return erlangparams1,diffpercent1

# def f_teszt_erlang_3points():       # 
#     X=[18926.5, 18936.5, 18946.5]
#     Y=[222.1464, 597.9651000000001, 757.1555999999999]
#     kitevo=20

#     X_out=Range(18900,19000,add=1)

#     pltinit()
#     FvPlot(pd.Series(Y,X),'scatter')

#     erlangparams,diffpercent = fn_erlang_3points(X,Y,leftwidth=40,leftwidth_delta=10,kitevo=kitevo)
#     Y_out=fn_erlang_multi(X_out,*erlangparams,kitevo=kitevo)
#     FvPlot(pd.Series(Y_out,X_out),'original',label='40')

#     erlangparams,diffpercent = fn_erlang_3points(X,Y,leftwidth=100,leftwidth_delta=10,kitevo=kitevo)
#     Y_out=fn_erlang_multi(X_out,*erlangparams,kitevo=kitevo)
#     FvPlot(pd.Series(Y_out,X_out),'original',label='100')

#     erlangparams,diffpercent = fn_erlang_3points(X,Y,leftwidth=200,leftwidth_delta=10,kitevo=kitevo)
#     Y_out=fn_erlang_multi(X_out,*erlangparams,kitevo=kitevo)
#     FvPlot(pd.Series(Y_out,X_out),'original',label='200')


#     pltshow()


def fn_lognormal_orig(x,sigma,mu=0):        # a lognormal eloszlás sűrűség függvénye (PDF)
    '''
    x:      list is lehet
    mu:     a kiinduló gauss eloszlás csúcshelye   
        - ennek a gauss eloszlásnak a logaritmusa a lognormal függvény
    sigma:  a gauss eloszlás szórása

    '''
    if type(x)==list: x=array(x)

    return (
            (1 / (x * sigma * np.sqrt(2 * np.pi)))  *  np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
            )

def fn_lognormal(X,maxpos,maxvalue,leftwidth,modus,mu=0):
    '''
    X:          list, array vagy egyetlen float is lehet
    maxpos:     a tetőzés helye
    maxvalue:   csúcsérték
    leftwidth:  bal oldali szélesség
        - közelítő megfeleltetés az Erlang függvény leftwidth paraméterével:  (lognormal illesztése Erlang-ra)
            leftwidth_erlang = leftwidth_lognormal * 0.68   (a faktor kis mértékben függ a modus-tól)
    modus:      (0:1) közötti float.  Jellemző értékek   0.8, 0.9, 0.95   Minél nagyobb annál "csúcsosabb" a görbe
        - közelítő megfeleltetés az Erlang függvény hatványkitevőjével:  (lognormal illesztése Erlang-ra)
                power = 1 / ((1-modus)*2.34)
                modus = 1 - 1/(2.34*power)
    '''

    if type(X)==list: X=array(X)

    x0 = maxpos-leftwidth
    X_rel = X-x0

    # Ha az összes elem <=0, akkor return 0
    if (X_rel<=0).all(): return [0]*len(X)   # kötelezően 0 a startpos előtt

    sigma = np.sqrt( mu - np.log(modus))        # wikipedia  

    # maxvalue a normált eloszlásban:   (x=modus)
    maxvalue_n = fn_lognormal_orig(x=modus,sigma=sigma,mu=mu)
    y_factor = maxvalue / maxvalue_n

    X_rel = X_rel * (modus/leftwidth)      # koordinátatranszformáció

    result = np.where(X_rel>0, y_factor * fn_lognormal_orig(X_rel,sigma,mu),
                               0           # 0 ha x<=x0
                     )

    return result

def fn_lognormal_multi(X,*params,outtype='sum'):
    '''
    FONTOS: ha a params egy list, akkor a függvény hívásakor ne felejtds elé írni a *-t 
    
    X:       list, array vagy egyetlen float.      Dátumok is megadhatók (float-ra konvertálja)
    params:  maxpos,maxvalue,leftwidth,modus, ..., maxpos,maxvalue,leftwidth,modus
        - négyes csoportokban kell megadni az elemi felfutások paramétereit
    outtype: "sum"       "array"  (rész-erlang idősorok tömbje)
        - Y  vagy array of Y
    '''
    if len(params)==0:
        return array([0]*len(X))
    
    X=datefloat(X)      

    surge_count = len(params) // 4

    idosor = array([0.]*len(X))     #
    idosorok = []

    for i in range(surge_count):
        i_first=4*i
        maxpos=params[i_first]
        maxvalue=params[i_first+1]
        leftwidth=params[i_first+2]
        modus=params[i_first+3]

        try:
            idosorL = fn_lognormal(X,maxpos,maxvalue,leftwidth,modus)
            idosor += idosorL
            idosorok.append(idosorL)
        except Exception as e:		  # tetszőleges exception leírásának megjelenítése
            print(e)


    if outtype=='sum': return idosor
    elif outtype=='array': return idosorok

def fn_lognormal_fit(X,Y,params0,lower,upper,scale):
    '''
    Lognormal illesztés az X,Y pontsorozatra.
    Több-felfutásos illesztés is kérhető   (a params0 elemszáma határozza meg a felfutások számát)

    params0: a paraméterek kezdőértéke   
        - egy vagy több paraméter-négyes adható meg, szekvenciálisan   (4,8,12, ... lehet az elemszám)
            maxpos,maxvalue,leftwidth,modus
    lower_bounds: alsó határok paraméterenként
    upper_bounds: felső határok paraméterenként
    x_scale:  a paraméterek nagyságrendje (jellemző értéke)
    '''

    Y_pred=None
    last_params=None
    def f_lognormal_multi(X,*params):
        nonlocal Y_pred,last_params
        # params:  maxpos,maxvalue,leftwidth,modus, ...
        last_params=params
        Y_pred = fn_lognormal_multi(X,*params) 
        return Y_pred

    try:
        popt,pcov = curve_fit(f_lognormal_multi,X,Y,p0=params0,bounds=(lower,upper),
                        method='dogbox',ftol=1e-8,xtol=1e-8,maxfev=None,verbose=0,x_scale=scale)
    except Exception as e:
        if (nincsbenne(str(e),'The maximum number of function evaluations is exceeded') and
        nincsbenne(str(e),'Residuals are not finite in the initial point')):
            print('ERROR   ' + str(e))
            raise e
        else:
            popt=last_params

    return popt


# Összevont illesztési függvények
def fn_multi(X,corefunction,*params):
    if corefunction=='erlang': return fn_erlang_multi(X,*params)
    elif corefunction=='lognormal': return fn_lognormal_multi(X,*params)
    elif corefunction=='gauss': return fn_gauss_multi(X,*params)

def fn_fit(X,Y,corefunction,params0,lower,upper,scale):
    if corefunction=='erlang': return fn_erlang_fit(X,Y,params0,lower,upper,scale)
    elif corefunction=='lognormal': return fn_lognormal_fit(X,Y,params0,lower,upper,scale)
    elif corefunction=='gauss': return fn_gauss_fit(X,Y,params0,lower,upper,scale)




def fn_fűrész(x,T):
    # [-1,1] közötti érték, cikluson belül lineárisan növekvő
    x_ciklus=int(x/T)
    x=x%T
    m=2/(T-1)   
    return -1 + m*x

def fn_noise(szigma):
    # (-szigma,szigma) közötti gauss zaj  (outlier-ek előfordulhatnak)
    return szigma * np.random.randn()




# tbl=Readcsv(r"C:\Users\Zsolt\Downloads\decompose_erlangs G_all leftw_free flatten.csv",format='hu')
# tblL=tbl.loc[tbl.country=='Hungary']
# ser = serfromtbl(tblL,'leftwidth',indexcol='grad2_start',aggfunc='mean',orderby='index')
# ser=FvLinearResample(ser,count=500)
# FvPlot(ser,'gauss',G=20,label='gauss_future',annot='max right',color='0.5')
# pltinit()
# for day in Range(datefloat('2022-01-01'),add=20,count=5):
#     progress(str(day))
#     ser_cut = ser.loc[ser.index<day]
#     ser_arima = fn_gauss_arima_avg(ser_cut,G=20)
#     FvPlot(ser_arima.loc[ser_arima.index>=day-20],'original',label='arima',annot='right')
#     serG = FvGaussAvg(ser_cut,gausswidth=20)
#     FvPlot(serG[serG.index>=day-20],'original',label='gauss',annot='right',color='last')
# pltshow()




# ser=pd.Series([1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4])
# # ser=FvLinearResample(ser,10)
# ser2=fn_moving_avg(ser,4,center=True)
# ser3=FvGaussAvg(ser,4)
# ser4=FvLinearResample(ser,10)
# ser4=fn_moving_avg(ser4,40)
# plotnow([ser,ser2,ser3,ser4],plttype='scatter original')
    

# X=Range(0,1000)
# ser1=pd.Series(fn_erlang_multi(X,200,100,200,kitevo=9),X)
# ser2=pd.Series(fn_erlang_multi(X,400,100,200,kitevo=9),X)

# plotnow([ser1,ser2])
# exit()
