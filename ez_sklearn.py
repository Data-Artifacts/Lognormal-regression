# SKLEARN, KERAS, PROPHET, ARIMA  ALGORITMUSOK

import pandas as pd
import numpy as np
from numpy import array

from ezcharts import *
from ezdate import *
from ezplot import *

from itertools import product


from sklearn.preprocessing import RobustScaler              # a pontok túlnyomó többsége kerüljön a [-1,1] tartományba

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score,mean_absolute_percentage_error,mean_absolute_error

from sklearn.preprocessing import StandardScaler            # nullára centrálás és skálázás a szórással ([-1,1]-be kerül a többség)
from sklearn.preprocessing import MinMaxScaler              # [0-1] tartományba transzformálja
from sklearn.preprocessing import MaxAbsScaler              # csak skálázás
from sklearn.preprocessing import RobustScaler              # a pontok fele kerüljön a [-1,1] tartományba

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from keras.models import Sequential,load_model,save_model
from keras.layers import Dense      # ,LSTM
from keras.callbacks import EarlyStopping


from prophet import Prophet
from prophet.plot import plot_yearly, plot_weekly, add_changepoints_to_plot

from pmdarima.arima  import auto_arima,ARIMA
import pmdarima as pmd

from statsmodels.tsa.statespace.sarimax import SARIMAX

# from ezplot import *


# r"C:\Users\Zsolt\OneDrive\Python\Projects\sklearn\Adatsorok\Tigáz Kürt" + '\\'




def fn_load_ts(dataname='wineind',outtype='ser'):   # gyakran használt idősorok
    '''
    outtype:  'ser':     datetime - floatvalues   
              'prophet': tbl['ds','y']
    '''
    if dataname=='wineind':
        # auto_arima demo adatbázisa   Forrás: Kaggle
        # 1980.01 - 1994.08   Australian total wine sales by wine makers
        from pmdarima.datasets import load_wineind
        ser = load_wineind(True)         # year-month - float 
        # pmd.plot_acf(ser.values,lags=np.arange(100))        # autokorreláció
        # ser=load_wineind(as_series=True)        # stringek kerülnek az indexbe (pl. "Jan 1980")
        # ser.index=pd.to_datetime(ser.index)     # legyen inkább float, mert a datetime-nál néha nem megy a diagramban a nagyítás
    elif dataname=='msft':
        # auto_arima,  forrás: Kaggle
        from pmdarima.datasets import load_msft
        tbl = load_msft()         # autosorszámozott betöltés
        ser = serfromtbl(tbl,'Open','Date')
        # ser=tbl['Open']
    elif beginwith(dataname,'tigáz_'):
        dir_ =r"C:\Users\Zsolt\OneDrive\Python\Projects\sklearn\Adatsorok\Tigáz Kürt" + '\\'
        if beginwith(dataname,'tigáz_hőmérséklet'):
            tbl=pd.read_csv(dir_ + dataname + '.csv',sep=';',encoding='cp1250',decimal=',')
            tbl['dátum']=pd.to_datetime(tbl['dátum'])
            ser = serfromtbl(tbl,'Tény','dátum')
        elif beginwith(dataname,'tigáz_all_'):
            dataname=dataname.replace('all_','')
            tbl=pd.read_csv(dir_ + 'all\\' + dataname + '.csv',sep=';',encoding='cp1250')
            tbl['dátum']=pd.to_datetime(tbl['dátum'])
            ser = serfromtbl(tbl,'kWh','dátum')
        else:
            tbl=pd.read_csv(dir_ + dataname + '.csv',sep=';',encoding='cp1250')
            tbl['dátum']=pd.to_datetime(tbl['dátum'])
            ser = serfromtbl(tbl,'kWh','dátum')
    elif dataname=='wiki_views':
        # Egy amerikai focistáról szóló wiki honlap látogatottsága  (prophet)
        tblProphet = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')
            # 0          ds  string  "2007-12-10"  "2016-01-20"
            # 1           y   float   5.262690189   12.84674689   
        ser=serfromtbl(tblProphet,'y','ds')
    else:
        return

    ser.name=dataname

    if outtype=='ser':
        return ser
    
    elif outtype=='prophet':
        return tblProphet



def fn_sklearn_loss(loss_function,Y_true,Y_pred,sample_weight=None):   # 'mae' 'mape', 'mse'
    '''
    stringnév és közvetlen függvény is jó
    
    '''    
    if type(loss_function)==str:
        if loss_function=='mae': return mean_absolute_error(Y_true,Y_pred,sample_weight=sample_weight)
        elif loss_function=='mape': return mean_absolute_percentage_error(Y_true,Y_pred,sample_weight=sample_weight)
        elif loss_function=='mse': return mean_squared_error(Y_true,Y_pred,sample_weight=sample_weight)
    else:
        return loss_function(Y_true,Y_pred,sample_weight=sample_weight)


# fn_smape(array([2,3]),array([0.1,0.2]),1,'avg')



def fn_tune_bygrid(fn_loss,d_params):       # gridsearch, általános többváltozós loss-függvényre
    '''
    A hangolás kétkörös:
    - az első körben a közösen hangolandó paraméterekre fut le egy gridsearch
    - második körben elszigelet hangolási kísérletek, egy-egy paraméter módosításával (a többi változótól független hangolás)
        A második körös hangolásokat egy beágyazott listával lehet kérni
    
    fn_loss:   általában fn_ts_loss() egy paraméterezhető modellel és adott tesztpontokkal. 
               Az fn_ts_loss, az fn_ts_test néma hívása, és a tesztpontokra számított loss-ok és loss_base-ek átlagát adja vissza

    d_params={
              'learning_rate_init':[1e-3,1e-4, [3e-3,3e-4],[3e-4,3e-5]],       // az utolső helyen lévő beágyazott list jelzi az elszigetelt hangolási kérés
              'alpha':[0.1,0.01, [0.3,0.03],[0.03,0.003]]                     // legalább egy első körös hangolási érték mindenképpen kell 
             }
    
    return:   params_tuned, loss_min, tblLosses
        - params_tuned:     dict 
        - loss_min:         a params_tuned-hoz tartozó loss
        - tblLosses:        a hangolás lépései
            oszlopok:   [params_names], 'loss', 'mod', 'loss_prev', 'sec'
                    - a 'mod' és a 'loss_prev' csak izolált (második körös) hangolási lépések esetén tartalmaz értéket
                        A hangolt param megnevezése és a hangolás előtti loss (mihez képest kellene javulást elérni)
                    - 'sec' =  időtartam
    '''

    # Második körös hangolási kérések kivétele
    nEmbedded=0
    d_params1={}
    d_params2={}
    for param_name,value_list in d_params.items():
        for value in value_list:
            if type(value)==list:   # beágyazott lista
                if not d_params2.get(param_name): 
                    d_params2[param_name]=[]
                    d_params2[param_name].append(value)
                    nEmbedded+=len(value)   # csak az első lista elemszámát nézem (több elágazásos sublista is lehet)
                else:
                    d_params2[param_name].append(value)
            else:   
                if not d_params1.get(param_name): d_params1[param_name]=[]
                d_params1[param_name].append(value)
    

    # Első körös hangolása, grid-szekvencia
    all_params = [dict(zip(d_params1.keys(), v)) for v in product(*d_params1.values())]
    nAll=len(all_params) + nEmbedded        
    nStep=1
    losses,losses_pred,losses_base,secs=[],[],[],[]
    loss_min=None
    for params in all_params:
        t=stopperstart()

        loss,loss_base = fn_loss(**params)      # keresztvalidáció futtatása általában 3-5 tesztpontra

        # loss_sort             # a loss_base változhat az egymást követő hangolások során, ezért a loss_sort a mérvadó
        if loss_base==0:
            if loss==0: loss_sort=1
            else: loss_sort=-1
        else: loss_sort = loss/loss_base
        losses.append(loss_sort)

        if loss_min is None or loss_sort<loss_min: loss_min=loss_sort   # csak a progress-hez kell

        loss_pred = Round(loss,3)
        losses_pred.append(loss_pred)

        loss_base = Round(loss_base,3)
        losses_base.append(loss_base)

        sec=Round(stopper(t,False),3)
        secs.append(sec)

        progress('Tune  ' + str(nStep) + '/'  + str(nAll) + '   ' + 'loss_sort=' + strnum(loss_sort,'3g') + '  ' + 
                        'loss=' + strnum(loss,'3f') + '   ' + 'loss_base=' + strnum(loss_base,'3f') + '  ' +
                        'sec=' + strnum(sec,'3f') +  '   ' + 'loss_min=' +  strnum(loss_min,'3f'))
        nStep+=1

        # print('Tune   loss=' + strnum(loss,'3f') + '   ' + 'sec=' + strnum(sec,'3f') + '   ' + str(params))
    tblLosses = pd.DataFrame(all_params)
    tblLosses['loss']=losses
    tblLosses['loss_pred']=losses_pred
    tblLosses['loss_base']=losses_base
    tblLosses['sec']=secs
    tblLosses['mod']=None           # második körben lesz használva
    tblLosses['loss_prev']=None     # második körben lesz használva

    # optimális rekord beolvasása
    params_tuned = tblLosses.copy().sort_values(by='loss').iloc[0:1].to_dict('records')[0]
    loss_min = params_tuned.pop('loss')
    params_tuned.pop('loss_pred')
    params_tuned.pop('loss_base')
    params_tuned.pop('sec')        # a 'sec' adatot is ki kell szedni a dict-ből
    params_tuned.pop('mod')        # a 'mod' adatot is ki kell szedni a dict-ből
    params_tuned.pop('loss_prev')        # a 'loss_prev' adatot is ki kell szedni a dict-ből


    # második körös hangolások  (a többi változótól független)
    paramsL=params_tuned
    for param_name,value_list in d_params2.items():
        # melyik érték győzőtt az első körben (index)
        param_value = params_tuned[param_name]
        param_value_list = d_params1[param_name]
        index = param_value_list.index(param_value)
        if index >= len(d_params2[param_name]): index=-1    # ha nincs megfelelő index, akkor az utolsó

        # a paraméter beállítása, majd loss számítás
        param_value_list2 = d_params2[param_name][index]
        for value2 in param_value_list2:
            paramsL = params_tuned.copy()
            paramsL[param_name] = value2
            
            t=stopperstart()
            loss,loss_base = fn_loss(**paramsL)      # keresztvalidáció futtatása általában 3-5 tesztpontra

            # loss_out
            if loss_base==0:
                if loss==0: loss_sort=1
                else: loss_sort=-1
            else: loss_sort = loss/loss_base

            loss = Round(loss,3)
            sec=Round(stopper(t,False),3)


            paramsL['loss']=loss_sort
            paramsL['loss_pred']=loss
            paramsL['loss_base']=loss_base
            paramsL['sec']=sec
            paramsL['mod']=param_name
            paramsL['loss_prev']=loss_min       # paraméterenként frissítődik
            tblLosses=tblLosses.append(paramsL,ignore_index=True)
            progress('Tune  ' + str(nStep) + '/'  + str(nAll) + '   ' + 'loss_sort=' + strnum(loss_sort,'3g') + '  ' +
                        'loss=' + strnum(loss,'3f') + '   ' + 'loss_base=' + strnum(loss_base,'3f') + '  ' +
                        'sec=' + strnum(sec,'3f') + '   ' + 'mod=' + param_name + '   ' + 'loss_min=' +  strnum(loss_min,'3f'))


            nStep+=1
            # print('Tune   loss=' + strnum(loss,'3f') + '   ' + 'sec=' + strnum(sec,'3f') + '   ' + 
            #             'mod=' + param_name + '   ' +  str(paramsL))

        # optimális rekord beolvasása
        params_tuned = tblLosses.copy().sort_values(by='loss').iloc[0:1].to_dict('records')[0]
        loss_min = params_tuned.pop('loss')
        params_tuned.pop('loss_pred')
        params_tuned.pop('loss_base')
        params_tuned.pop('sec')        # a 'sec' adatot is ki kell szedni a dict-ből (csak a modell paraméterei maradhatnak)
        params_tuned.pop('mod')        # a 'mod' adatot is ki kell szedni a dict-ből
        params_tuned.pop('loss_prev')        # a 'loss_prev' adatot is ki kell szedni a dict-ből

    
    cols_to_front(tblLosses,'loss,sec,mod,loss_prev')     

    return params_tuned, loss_min, tblLosses


# def fn_tune_visualize(tblLosses):
#     pass



def fn_tune_bygroping(fn_loss,params,paramname,add,addsub=None,maxcalls=100,lossstart=None,xmin=None,xmax=None):   # 2023.11
    '''
    Tapogatózó keresés a paramname paraméterre:  megnézi az aktuális érték felett és alatt. Ha valamelyik irány jobb, 
        akkor addig lépked az adott irányban amíg javulás van.   Ha az addsub is meg van adva, akkor pontosító keresés
    fn_loss(**params)=loss:  az argumentumai a hangolandó paraméterek (ezek egyike a most hangolt paraméter)
        - return:  loss  (pl. egy predikció eltérése a tényértéktől)
            Ez a függvény tartalmazza a teljes "üzleti logikát" (általában machine és deep learning modellekre támaszkodik)
        - általában egy proxy függvény, ami meghív egy predikciós függvényt és kiszámítja a loss-t
        - viszonylag időigényes lehet a hívása (a maxcount-tal szabályozható, hogy legfeljebb hány hívás megengedett)
    params:  egy dictionary az fn_loss összes hangolható paraméterével (inicializáltnak kell lennie)
        Ha egyetlen hangolható paramétere van az fn_loss-nak, akkor egy egymezős dict
    paramname:  a hangolandó paraméter megnevezése  (szerepelnie kell a dict-ben)
        A paraméter float vagy int lehet
    add:    lépésenként hozzáadandó illetve levonandó érték
    addsub: (opcionális, <add)  Ha elkészült az első körös hangolás, akkor még egy pontosító kör  
    maxcalls: az fn_loss hívások maximális száma (az opcionális losstart számítás nélkül)
        - 2 hívás mindenképpen szükséges (próbálkozás felette és alatta)
    lossstart:  az induló értékkel számított loss (ha ismert egy előző hangolási lépésból, akkor érdemes megadni)
    xmin,xmax:   limitek a hangolandó érték számára  (x>=xmin, x<=xmax)
            
    return: params,loss,stepcount         Az adott paraméter hangolás utáni értékével. 
        - stepcount: sikeres hangolási lépések száma.  Ha 0, akkor a params nem változott.
    '''

    x=params[paramname]

    # kiinduló loss számolása
    if not lossstart: loss_out = fn_loss(**params)
    else: loss_out=lossstart
    print('TUNE START     ' + paramname + '=' + strnum(params[paramname]) + 
                  '  loss=' + strnum(loss_out,'2%'))
    
    callcount=0     # fn_loss hívásainak száma
    stepcount=0     # sikeres lépések száma           
    
    params_out=params.copy()

    def f_call_fn_loss(params):
        nonlocal callcount
        callcount += 1
        lossL=fn_loss(**params)
    
        if lossL<loss_out:
            print('Call fn_loss   ' + paramname + '=' + strnum(params[paramname]) + 
                  '  loss=' + strnum(lossL,'2%') + '  success')
        else:
            print('Call fn_loss   ' + paramname + '=' + strnum(params[paramname]) + 
                  '  loss=' + strnum(lossL,'2%'))
        return lossL


        
    def f_succeded(x,loss):
        nonlocal params_out,loss_out,stepcount
        # print('TUNE  ' + paramname + '=' + strnum(x) + '  loss=' + strnum(loss,'%'))
        params_out=params.copy()
        loss_out=loss
        stepcount +=1


        
    def f_circle(add,xmin,xmax):
        nonlocal params,x
        # próbálkozás felette és alatta
        if not xmax or x+add<=xmax:
            params[paramname]=x+add
            loss_upper=f_call_fn_loss(params)
        else: loss_upper=loss_out
        if not xmin or x-add>=xmin: 
            params[paramname]=x-add
            loss_lower=f_call_fn_loss(params)
        else: loss_lower=loss_out
        
        if loss_lower<loss_out and loss_lower<loss_upper:
            x=x-add
            params[paramname]=x
            f_succeded(x,loss_lower)
            # további próbálkozások
            while (not xmin or x-add>=xmin) and (not maxcalls or callcount<maxcalls):
                x=x-add
                params[paramname]=x
                lossL=f_call_fn_loss(params)
                if lossL>=loss_out: break
                f_succeded(x,lossL)
        elif loss_upper<loss_out and loss_upper<loss_lower:
            x=x+add
            params[paramname]=x
            f_succeded(x,loss_upper)
            # további próbálkozások
            while (not xmax or x+add<=xmax) and (not maxcalls or callcount<maxcalls):
                x=x+add
                params[paramname]=x
                lossL=f_call_fn_loss(params)
                if lossL>=loss_out: break
                f_succeded(x,lossL)

    # Első kör
    f_circle(add,xmin,xmax)

    if addsub:
        params=params_out.copy()
        x=params[paramname]
        f_circle(addsub,max(x-add + addsub/1000,xmin),min(x+add - addsub/1000,xmax))         # a két addsub/1000 azért kell, mert nem megengedett az egyenlőség


    return params_out,loss_out,stepcount
    

def fn_tune_bylist(fn_loss,list_x,tupleout=False,plot_label=None):    # hiperparaméter hangolás, egyváltozás, ciklus a listára
    xmin=None
    ymin=None
    records=[]
    for x in list_x:
        y=fn_loss(x)
        if ymin==None or y<ymin: 
            ymin=y
            xmin=x
        records.append((x,y))

    if plot_label:
        X,Y=unzip(records)
        plotnow(Y,X,labels=plot_label,title='Hangolás')

    if tupleout: 
        return xmin,ymin,records
    else: 
        return xmin


def fn_tune_byhalving(fn_loss,xmin,xmax,xstart,steps,q=0.5,tupleout=False,plot_label=None):   # hiperparaméter hangolás, felezésekkel
    '''
    Egyszerű, egyváltozós minimumhely keresés, időigényes fn_loss-függvénnyel
    Csak viszonylag jól viselkedő függvényekre működik.
    Feltételezi, hogy van valami előzetes tipp a megfelelő választásra (start)

    fn_loss:  egyváltozós függvény, float argumentummal, float return-nel  (loss jellegű)
    min, max:  float
    start:  float, min-max között
    steps:  hány felező lépéssel keresse a minimumhelyet
    q:   ha feltehető, hogy a minimumhely start közelében van, akkor 0.5-nél kisebb szorzó is felmerülhet
    tupleout:  True esetén  (x,fn_loss(x),xmin,xmax,records) a return, egyébként x
        ahol xmin és xmax azt jelzi, hogy ezek között a határok között bárhol lehet a minimumhely
    plot_lable:  ha meg van adva, akkor plot megjelenítése az x,y értékekre

    return: tuple(x,xmin,xmax)
    '''
    irány='max'
    x=xstart
    y=fn_loss(xstart)


    records=[]
    for step in range(steps):
        if irány=='max':
            x_ = x + (xmax-x)*q
            y_ = fn_loss(x_)
            # ha jó az irány
            if y_ < y:
                xmin=x
                x=x_
                y=y_
            # ha rossz az irány
            else:
                xmax=x_
                irány='min'
        elif irány=='min':
            x_ = x - (x-xmin)*q
            y_ = fn_loss(x_)
            # ha jó az irány
            if y_ < y:
                xmax=x
                x=x_
                y=y_
            # ha rossz az irány
            else:
                xmin=x_
                irány='max'
        records.append((x_,y_))
        if plot_label:
            print('Hangolás: ' + plot_label + '=' + strnum(x_,'3f') + ', loss=' + strnum(y_,'1%'))

    if plot_label:
        X,Y=unzip(records)
        plotnow(Y,X,labels=plot_label,title='Hangolás')


    if tupleout: return x,y,xmin,xmax,records
    else: return x


    


    


def plot_model_test(modelname,model_fitted,X_test,y_test,G=0.01,modelparams='all'):   # trained model grafikus tesztelése
    tbl=pd.DataFrame()
    tbl['Original']=y_test
    tbl['Predicted'] = model_fitted.predict(X_test)
    tbl=tbl.sort_values(['Original','Predicted'],ignore_index=True)
    if modelparams=='all': modelparams=str(model_fitted.get_params())
    else:
        params=model_fitted.get_params()
        titleitems=[]
        for paramname in modelparams.split(','): titleitems.append(paramname + '=' + str(params[paramname]))
    tblinfo(tbl,'plot',G=G,normalize=0,
        suptitle=modelname + ' tesztelés',
        title=', '.join(titleitems))     



def plot_models_test(models,X,y,split='80-20',sortby='y',G=0.01,suptitle='Modellek tesztelése',annotpos='middle'):   # több modell grafikus tesztelése
    '''
    Több model tesztelése  (eltérhet a model típusában vagy paraméterezésében),  ugyanarra az adathalmazra (train-test )
    models:  [(felirat,model),(felirat,model),...]
    X,y:     20%-os train-test split,   pl. X=tbl[inputcols].values   y=tbl[outcol].values
    split:   '80-20': random-kiválasztásos trian-test split
             '80+':  rendezés az első input változóra, az első 80%-ra train, majd teszt a következő 20%-ra
                     (idősoros predikció a jobb szélre; elsősorban egymezős inputra)
             None:  train és teszt megegyezik
    sortby:  'y' - rendezés a kimeneti értékek szerint
             - kérhető egy input adatra való rendezés is. Ehhez meg kell adni az X tömbön belüli indexet (integer) és egy feliratot.
               Példa:  sortby = '0,Length'
    '''
    
    x_testfirst=None
    if split=='80-20':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    elif endwith(split,'\+'):        # pl. "80+":  az első input-mezőre rendezve a bal oldali 80% a tanító állomány
        # Xy összehangolt rendezése
        tblXy=pd.DataFrame(X)
        tblXy['y']=y
        tblXy=tblXy.sort_values(0)     # a kontruktor sorszámozott oszlopnévvel hozta létre az oszlopokat
        y=tblXy['y']
        tblXy=tblXy.drop('y',axis='columns')
        X=tblXy.to_numpy()
        
        lentrain_arány=int(split[:-1])/100      # '+' levágása a végéről, százalék
        lentrain=int(len(X)*lentrain_arány)
        X_train=X[:lentrain]
        X_test=X                  # A teljes X bekerül
        y_train=y[:lentrain]
        y_test=y                  # A teljes y bekerül
        if X.shape[1]==1: x_testfirst=X_test[lentrain][0]
        else: x_testfirst=lentrain          # ilyenkor egyszerű sorszámozás van az x-tengelyen
    else:
        X_train=X
        X_test=X
        y_train=y
        y_test=y

    tbl=pd.DataFrame()
    tbl['Original']=y_test

    if X.shape[1]==1:   # egyetlen input-feature esetén a feature a diagram x-tengelyére kerül
        tbl.index=X_test.flatten()
    else:        # skálázás csak többmezős input esetén kell
        scaler = RobustScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    annotitems=['Scores']

    if sortby=='y': sortby=['Original']
    else:      # '0,Weight' formátum (rendezés egy input oszlop szerint)
        index,colname = splitfirst(sortby,',',default_second='Input')
        try:
            tbl[colname]=list(unzip(X_test))[int(index)]
            sortby=[colname]
        except:
            pass
    
    # Tanítás és predikció 
    for modelrec in models:
        (modelname,model) = modelrec
        print(modelname)
        model.fit(X_train, y_train)
        annotitems.append('"' + modelname + '": ' + strnum(model.score(X_test,y_test),'3g'))
        tbl[modelname] = model.predict(X_test)
        sortby.append(modelname)
    


    if X.shape[1]>1:    # többmezős input
        tbl=tbl.sort_values(by=sortby,ignore_index=True)
        plttype='gausst'
        plotstlye='firstarea'
    else:     # egymezős input
        plttype='scatter regausst'
        plotstyle='line'
    

    if annotpos=='middle': annotpos=str(int((tbl.index.max()-tbl.index.min())/2))     # RangeIndex objektumnak nincs mean() metódusa

    y_bands=None
    if x_testfirst: x_bands=[{'x1':x_testfirst,'x2':'end'}]
    
    tblinfo(tbl,'plot',G=G,normalize=0,annotpos=annotpos,plttype=plttype,plotstyle=plotstyle,
        suptitle=suptitle,xy_texts=[{'x':0.1,'y':0.75,'caption':'\n'.join(annotitems)}],
        y_bands=y_bands)     




# ÁLTALÁNOS MODELLEK
def f_keras_model(X_train,Y_train,nLayers=3,nNodes=100,verbose=0,epochs=200,optimizer='adam',loss_function='mse',
        validation=None,validation_freq=1,min_delta=0,patience=10,sample_weight=None,
        scaler=None,outtype='model'):         # egyszerű szekvenciális model, tanítással
    '''
    Hátrány a KNN-hez képest:  lassú a tanítása
    Előny a KNN-hez képest:  pontosabb és túl tud terjeszkedni a train értékkészletén
      - a KNN csak a train értékkészleten belül tud mozogni, csak közbenső jellegű válaszra képes
    Előny az sklearn MLPRegressor-hoz képest:
      - a kimenet többváltozós is lehet (az sklearn regressor csak egy output feature-t kezel)
      - nem kell hangolni a learning_rate_init, alpha, tol paramétereket

    X_train:  kétdimenziós tömb (nparray vagy list)
    Y_train:  ha egyváltozós a kimenet, akkor vektor is megadható, egyébként kétdimenziós tömb
    nLayers:  hány layer legyen a modellben  (mindegyik layer 100 elemű)

    optimizer: 'adam', 'sgd'
    validation:  
        [0:1] szám:  az X_train mekkora hányadára validáljon (0 esetén a train-re validál)
        (X_test,Y_test):  közvetlenül megadott validációs adatsor
    min_delta:  leállás, ha tartósan ez alá süllyed a loss csökkenése (a leállás általában 5-10-szer ekkora
        loss környékén következik be, mert a 10 lépésen keresztül kell teljesülnie)
    sample_weight:  a loss számításkor alkalmazandó súlyozás. A rekordok számának megfelelő méretű számtömb.
    scaler:   'StandardScaler'   'MinMaxScaler'   'MaxAbsScaler'  'RobustScaler'
        - az input adatok standardizálása (mindegyikre ugyanazzal a módszerrel)
        - a modell utólagos használatakor is érvényesül az input-ra (nem kell külön szerializálni a scaler-t)

    outtype:  'model',  
        'model_hist':  model,history      history.history(dict(loss,val_loss))
            tbl_plot(pd.DataFrame(history.history))   
            érdemes szerializálni csv-be

    
    A modell használata:
        Y = f_keras(X,model)

    Szerializálás:
        model.save(path)
        load_model(path,compile=False)          # a compile akkor legyen False, ha biztos nem volt Keras verzióváltás
    
    '''

    validation_split=0
    validation_data=None
    if validation is not None:
        if type(validation)==float: validation_split=validation
        else: validation_data=validation

    if type(X_train)==list: X_train=array(X_train)
    if type(Y_train)==list: Y_train=array(Y_train)
    if len(X_train.shape)!=2:  
        print('ERROR  f_keras_model   Az X_train inputnak két-dimenziósnak kell lennie')
        return
    X_dim=X_train.shape[1]

    if len(Y_train.shape)==1: Y_dim=1
    elif len(Y_train.shape)==2: Y_dim=Y_train.shape[1]
    else:
        raise Exception('f_keras_model   Az Y_train egyváltozós esetben egydimenziós tömb, többváltozós esetben kétdimenziós')

    
    model = Sequential()
    if scaler:
        if scaler=='StandardScaler': scaler=StandardScaler().fit(X_train)
        elif scaler=='MinMaxScaler': scaler=MinMaxScaler().fit(X_train)
        elif scaler=='MaxAbsScaler': scaler=MaxAbsScaler().fit(X_train)
        elif scaler=='RobustScaler': scaler=RobustScaler().fit(X_train)
        model.add(Lambda(lambda x: scaler.transform(x), input_shape=(X_dim,)))
    model.add(Dense(nNodes, input_dim=X_dim, activation='relu'))
    for i_layer in range(1,nLayers):
        model.add(Dense(nNodes, activation='relu'))
    # Kimeneti layer
    model.add(Dense(Y_dim, activation='linear'))
    
    model.compile(optimizer='adam', loss=loss_function, weighted_metrics=[])
        # - optimizer:  'sgd' merülhet még fel  (gradient decent optimizer)
        # - loss:  'mse', 'mae', 'mape'  cosine_similarity

    monitor='loss'
    if validation is not None: monitor='val_loss'

    # Model tanítása
    history=model.fit(X_train,Y_train, epochs=epochs, batch_size=32,verbose=verbose,
            validation_split=validation_split,shuffle=True,
            validation_data=validation_data,validation_freq=validation_freq,
            callbacks=[EarlyStopping(monitor=monitor,min_delta=min_delta,patience=patience,restore_best_weights=True)],
        )
    '''
        - shuffle:  a felosztás előtt legyen egy keverés  (nem arra utal, hogy minden egyes epoch-ban új felosztás legyen)

        - a result egy History objektum

        - a monitor='val_loss' a profi megoldás, mert ellenőrzi az overfittinget is
        - patience:  leállás, ha egymás után ennyiszer nem javul a loss
        - min_delta:  ha ennél kisebb a javulás, akkor ne tekintse javulásnak
           Ehhez tudni kell a loss értékkészletét, ami nem mindig ismert előre.    
        - restore_best_weights:  vannak olyan vélemények, melyek szerint félrevezető eredményt adhat a bekapcsolása
        - start_from_epoch:  ennyi epoch mindenképpen fusson le, csak ezt követően induljon az ellenőrzés
    '''
    if outtype=='model_hist': 
        return model,history
    else:
        return model


def f_keras_train_and_save(tbl_train,inputcols,outcols,fname_fix='model',train_count=1,
            verbose=1,save_hist=True,hist_subtitle='',
            scaler=None,validation=0.1,weightcol=None,
            nLayers=3,nNodes=100,
            epochs=500,patience=300,loss_function='mae'):
    '''
    Keras model tanítása, mentése, history plot mentése
    A modell használata:
        model = load_model(path,compile=False)          # a compile akkor legyen False, ha biztos nem volt Keras verzióváltás
        Y = f_keras(X,model)        # az X oszlopainak az inputcols-hoz kell igazodnia (egy- vagy többrekordos is lehet)

    tbl:  szerepelnie kell benne az inputcols,outcols,weightcol oszlopoknak (más oszlopai is lehetnek)
        - az input és output oszlopokban nem lehet üres érték (előtte el kell tüntetni)
        - általában nem kell standardizálni az input oszlopokat, de gyorsíthatja a tanítást (lásd: scaler)
        - az input mezők outlier-eit érdemes lehet egy limit-értékkel korlátozni, mert enélkül a train nehezen konvergálhat
            példa:       tbl.loc[tbl.formfaktor>50,['formfaktor']]=50
        - ha szűréssel lettek kiválasztva a validation rekordok, akkor train-ben nem szerepelhetnek
    inputcols:  list, vagy felsorolás
    outcols:  több oszlopos is lehet
    fname_fix:  a modell fájlnevének fix része      downloads\fname_fix yyyy-mm-dd hh-mm.keras
        példa:  "model boostwidth"
    train_count:  ha >1, akkor többször is lefuttatja a train-t és a legjobbat választja
    verbose: 1 esetén epocs-szintű adatok is megjelennek
    save_hist:  True esetén menti a histogramot képfájlként
    hist_subtitle:  a histogram felett megjelenő leírás
    validation:    csak addig megy a tanítás, amíg a validációs rekordokra is csökken a loss (túltanítás elkerülése)
        float[0:1]:   0.1 esetén a tbl_train 10%-a lesz a validációs adatsor
        tbl_val:  közvetlenül megadott validációs rekordok
            - az oszlopai megegyeznek a tbl_train-nel, de általában kevesebb rekord
            - általában valamilyen szűrésssel állítható elő a kiinduló tbl-ből
    weightcol:   opcionális    A rekordok súlyozása a loss számításakor (a fontosabb rekordok nagyobb súllyal 
            befolyásolják a loss-t). Előzetesen kell gondoskodni az oszlop  felvételéről és feltöltéséről a tbl-ben
    patience:   a leálláshoz legalább ennyi olyan epoch kell, ahol a validation loss legkisebb értéke már nem csökkent tovább
    loss_function: 'mse', 'mae', 'mape'  cosine_similarity   (függvénynév vagy rövidnév adható meg)
    scaler:   'StandardScaler'   'MinMaxScaler'   'MaxAbsScaler'  'RobustScaler'
        - az input adatok standardizálása (mindegyikre ugyanazzal a módszerrel)
        - a modell utólagos használatakor is érvényesül az input-ra (nem kell külön szerializálni a scaler-t)
    
    return: model_best          - train_count>1 esetén a legjobb
    '''
    
    if type(inputcols)==str: inputcols=inputcols.split(',')
    if type(outcols)==str: outcols=outcols.split(',')

    X_train=tbl_train[inputcols].to_numpy()
    Y_train=tbl_train[outcols].to_numpy()
    sample_weights_train = None
    if weightcol: sample_weights_train=tbl_train[weightcol].values   

    if type(validation)==pd.DataFrame:
        tbl_val=validation
        X_val=tbl_val[inputcols].to_numpy()
        Y_val=tbl_val[outcols].to_numpy()
        sample_weights_val = None
        if weightcol: sample_weights_val=tbl_val[weightcol].values
        validation=(X_val,Y_val,sample_weights_val)

    loss_min=None
    model_best,history_best,epochs_last_best = None,None,None
    print('KERAS train    ' + hist_subtitle)
    for i_train in range(train_count):                        
        progress('train ' + str(i_train+1))
        model,history = f_keras_model(X_train,Y_train,nLayers=nLayers,nNodes=nNodes,verbose=verbose,optimizer='adam',loss_function=loss_function,
                            sample_weight=sample_weights_train,
                            validation=validation,scaler=scaler, 
                            min_delta=0,patience=patience,epochs=epochs,
                            outtype='model_hist')

        # loss_out = model.evaluate(X_train, Y_train, sample_weight=sample_weights_train)
        # - nem jó értéket ad (??)

        epochs_last = len(history.history['loss']) - patience
        loss_last = history.history['loss'][-patience-1]       # ez valóban jó értéket ad, de nincs súlyozva
        print_out = 'TRAIN  epochs=' + str(epochs_last) + '   loss=' + strnum(loss_last)
        if validation is not None:
            loss_val_last = history.history['val_loss'][-patience-1]   # ez csak a validációs halmazra érvényes (súlyozott)
            print_out += '   loss_val=' + strnum(loss_val_last)
        print(print_out)

        if loss_min==None or loss_last<loss_min:
            loss_min = loss_last
            model_best = model
            history_best = history
            epochs_last_best = epochs_last 

    model=model_best
    history = history_best
    
    tbl_history = pd.DataFrame(history.history)
    
    fname_now = fn_now('to_path')
    
    fname = fname_fix + ' ' + fname_now
    path=downloadspath(fname,'keras')

    save_model(model,path)
    print('Keras model saved: ' + path)

    
    # Eredmény kiírása, history plot
    Y_true = Y_train
    Y_pred = f_keras(X_train,model)

    if weightcol:
        weighted_loss=fn_sklearn_loss(loss_function,Y_true,Y_pred,sample_weight=sample_weights_train)
        loss=fn_sklearn_loss(loss_function,Y_true,Y_pred)
        # weighted_loss=mean_absolute_error(Y_true,Y_pred,sample_weight=sample_weights_train)
        # loss=mean_absolute_error(Y_true,Y_pred)
        print('RESULT     weighted_loss=' + strnum(weighted_loss) + '   loss=' + strnum(loss))
        commenttopright_in =('LOSS' + '\n' + 
                             'weighted_loss = ' + strnum(weighted_loss) + '\n' + 
                             'loss = ' + strnum(loss) + '\n' +
                             'epoch_last = ' + strnum(epochs_last_best) 
                            )
    else:
        loss=fn_sklearn_loss(loss_function,Y_true,Y_pred)
        # loss=mean_absolute_error(Y_true,Y_pred)
        print('RESULT     loss=' + strnum(loss))
        commenttopright_in =('LOSS' + '\n' + 
                             'loss = ' + strnum(loss) + '\n' +
                             'epoch_last = ' + strnum(epochs_last_best) 
                            )

        

    # Tocsv(tbl_history,fname,format='hu',indexcol=True,numbering=False)

    if save_hist:
        f_keras_plot_train_history(tbl_history,
            title = 'Train history:  ' + fname_fix,
            subtitle=hist_subtitle,
            comment=commenttopright_in,
            to_file=fname)

    return model,tbl_history


def f_keras_plot_train_history(tbl_history,title=None,subtitle=None,comment=None,to_file=None):
    '''
    Megjeleníti vagy png fájlba írja az f_keras_train_and_save() által visszaadott train_history-t
    Közvetlenül az f_keras_train_and_save() függvény save_hist argumentumával is kérhető.
    Akkor érdemes önállóan hívni, ha a comment-be speciális tesztelési információk kellenek.
    '''
    
    tbl_plot(tbl_history,suptitle = title,
        title=subtitle,
        commenttopright_in=comment,
        annot='gaussmin last',
        to_file=to_file)



    
def f_keras(X,model):       # betanított modellre támaszkodó függvény
    '''
    Return:
        - egyrekordos input és egyváltozós kimenet esetén egyetlen számot ad vissza
        - egyrekordos input és többváltozós kimenet esetén egy vektort ad vissza
        - többrekordos input és egyváltozós kimenet esetén egy vektort ad vissza
        - többrekordos input és többváltozós kimenet esetén kétdimenziós tömböt ad vissza

    X:  Lehet egydimenziós is (egyetlen rekord). Ilyenkor az output is egyetlen szám (többváltozós kimenet esetén vektor)
        Lehet kétdimenziós tömb  (nparray vagy list)
            Rekordok, rekordonként általában több input-változóval (input features)
    model:  betanított modell
        pl. az f_keras_model() függvénnyel, vagy egy mentett modell beolvasása a load_model(path,compile=False) függvénnyel
    '''
    if type(X)==list: X=array(X)

    x_dim = len(X.shape)
    if x_dim==1: X=X.reshape(1,X.shape[0])      # egyrekordos input

    Y=array(model(X))
    if Y.shape[1]==1: 
        # Ha egyetlen rekord volt az input:
        if x_dim==1: return Y.reshape(-1)[0]
        # Többrekordos input esetén
        else: return Y.reshape(-1)
    else:
        # Ha egyetlen rekord volt az input:
        if x_dim==1: return Y.reshape(-1)
        # Többrekordos input esetén
        else: return Y



def f_knn_train(tbl,inputcols,outcols,idcols=None,scale='robust',n_neighbors=10):  # train, tbl inputtal
    '''
    inputcols: általában több oszlop.   Vesszős felsorolás vagy list
    outcols:  egy vagy több oszlop
    idcols:  az intput rekordok azonosítói (nem kötelező)
        - az f_knn_disctances() függvényhez szükséges (részletes információk a szomszédokról), magának
            a modellnek nem integráns része (egyedi attributumként tárolódik)
        - ha nincs megadva, akkor az input oszlopokat tekinti rekord-azonosítóknak
        - lehtenek benne input oszlopok és az intputtól független oszlopok is (szöveges oszlopok is)
    scale:  'robust', 'standard'.   Ha üres, akkor nincs skálázás
        - a scaler tárolódik a modellben is, és az f_knn()  automatikusan alkalmazza
    n_neighbors:   hány szomszédot adjon vissza, illetve hány szomszédra képezzen súlyozott átlagot
    '''
    if type(inputcols)==str: inputcols=inputcols.split(',')
    if type(outcols)==str: outcols=outcols.split(',')

    X_train=tbl[inputcols].to_numpy()
    Y_train=tbl[outcols].to_numpy()

    model = f_knn_model(X_train,Y_train,scale=scale,n_neighbors=n_neighbors)

    if not idcols:
        model.ids_train = X_train
    else:
        if type(idcols)==str: idcols=idcols.split(',')
        ids_train = tbl[idcols].to_numpy()
        model.ids_train = ids_train
    
    return model

def f_knn_model(X_train,Y_train,scale='robust',n_neighbors=10):   # train, közvetlen adatokkal
    '''
    A train szinte azonnali és a predict is nagyon gyors.
    Hátránya:  a visszadott érték nem léphet túl a megtanított inputokhoz tatozó értékkészleten
      (nem képes a megtanult határokon kívüli általánosításra)

    X_train:  kétdimenziós tömb (nparray vagy list)
    Y_train:  egydimenziós tömb (nparray vagy list)
    scale:  'robust', 'standard'.   Ha üres, akkor nincs skálázás
        - a scaler tárolódik a modellben is, és az f_knn()  automatikusan alkalmazza
    
    Használata:
        Y = f_knn(X,model)

    Szerializálás:
        model.save(path)
        load_model(path,compile=False)          # a compile akkor legyen False, ha biztos nem volt Keras verzióváltás
    '''

    scaler=None
    if scale=='robust':
        scaler=RobustScaler().fit(X_train)
        X_train=scaler.transform(X_train)
    elif scale=='standard':
        scaler=StandardScaler().fit(X_train)
        X_train=scaler.transform(X_train)


    knn = KNeighborsRegressor(n_neighbors=n_neighbors,weights = "distance")
    model = knn.fit(X_train,Y_train)

    setattr(model,'scaler',scaler)
    setattr(model,'Y_train',Y_train)    # pl. az f_knn_neighbors_Y() használja
    return model
     
def f_knn(X,model):             # betanított modellre támaszkodó függvény
    '''
    Return:
        - egyrekordos input esetén egyetlen számot ad vissza
        - többrekordos input esetén vektort ad vissza

    X:  Lehet egydimenziós is (egyetlen rekord). Ilyenkor az output is egyetlen szám (többváltozós kimenet esetén vektor)
        Lehet kétdimenziós tömb  (nparray vagy list)
            Rekordok, rekordonként általában több input-változóval (input features)
        Nem kell standardizálás, az esetleges scaler a modell argumentumaként van tárolva
    model:  betanított modell
        pl. az f_knn_model() függvénnyel, vagy egy mentett modell beolvasása a load_model(path,compile=False) függvénnyel
    '''
    if type(X)==list: X=array(X)

    x_dim = len(X.shape)
    if x_dim==1: X=X.reshape(1,X.shape[0])      # egyrekordos input

    try: 
        scaler=model.scaler
        if scaler is not None:
            X=scaler.transform(X)
    finally: 
        pass

    Y = model.predict(X)     # vektor  (rekordonként egy érték)


    # Ha egyetlen rekord volt az input:
    if x_dim==1: return Y[0]
    # Többrekordos input esetén
    else: return Y

def f_knn_distances(x,model):     # Leghasonlóbb input rekordok, y-értékkel és távolsággal
    '''
    Akkor is érdemes alkalmazni, ha végül keras lesz a választás. Feltérképezhetőek a hasonló rekordok.
    
    x:  egyetlen input rekord
    model:  f_knn_train-nel állítható elő
        - az f_knn_train-ben megadhatóak a rekordokat azonosító adatok
        - ha nem volt megadva az idcols, akkor az X_train rekordok lesznek a rekordazonosítók
    
    return:  ids_train (vagy X_train) megfelelő sorai, kiegészítve az "y" és a "distance" oszloppal (utolsó két helyen)
      - a lista rendezett (leghasonlóbb áll legelöl), a tételszáma megegyezik a tanítás n_neighbors paraméterével  
      - a distance a skálázott adatokra vonatkozik (pl. scale="robust" esetén)
      - az f_knn()  az y értékek távolsággal súlyozott átlagát adja vissza
    '''
    if type(x)==list: x=array(x)

    if x.ndim==1: X=[x]
    else: X=x

    try: 
        scaler=model.scaler
        if scaler is not None:
            X=scaler.transform(X)
    finally: 
        pass
    
    distances,indexes=model.kneighbors(X,model.n_neighbors)
    distances=distances[0]      # distances:  shape=(1,n_neighbors)
    indexes=indexes[0]          # indexes:    shape=(1,n_neighbors)

    neighbors = np.concatenate(
        [model.ids_train[indexes],
         npcol(model.Y_train[indexes]),     # az Y_train egyedi attributumként van tárolva a modell-ben
         npcol(distances),
         ],    axis=1)

    return neighbors

def f_knn_neighbors_Y(x,model):     # csak a leghasonlóbb input rekordok y-értékeit adja vissza
    '''
    Akkor hasznos, ha nem a súlyozott átlagra van szükség, hanem valamilyen más statisztikai adatra
       (pl. érték a nagyság szerinti sorrend 90%-ánál)
    Nem kell megadni az Y_train-t, mert egyedi attribútumként tárolva van a modellben
    '''
    if type(x)==list: x=array(x)

    if x.ndim==1: X=[x]
    else: X=x

    try: 
        scaler=model.scaler
        if scaler is not None:
            X=scaler.transform(X)
    finally: 
        pass
    
    distances,indexes=model.kneighbors(X,model.n_neighbors)
    indexes=indexes[0]          # indexes:    shape=(1,n_neighbors)

    return model.Y_train[indexes]


def f_keras_versus_knn():
    
    # Egy egyszerű aritmetikai függvény
    X_train=np.random.rand(1000, 2) * 10       # 0,10 közötti véletlenszámok
    Y_train=array([x1**2 + x2    for x1,x2 in X_train])     


    print('KERAS és KNN összehasonlítása')
    print('Egyszerű kétváltozós aritmetikai függvény: y=x1**2+x2')
    print('Tanítás:  1000 random input, x1 és x2 is [0-10] tartományban ')

    t=stopperstart()
    model_keras=f_keras_model(X_train,Y_train,verbose=1,epochs=100,validation_split=0.1,min_delta=0.01)
    stopper(t,msg='keras train process time')

    t=stopperstart()
    # scaler=StandardScaler().fit(X_train)
    model_knn=f_knn_model(X_train,Y_train,scale='robust')
    stopper(t,msg='knn train process time')

    print('\nTesztelés egyedi értékekre')
    for x1,x2 in [(5,2),(1,1),(11,2),(100,100),(-2,-6)]:
        x=[x1,x2]
        print('(' + str(x1) + ',' + str(x2) + ')=' + strnum(x1**2+x2) +
              '  keras: ' + strnum(f_keras(x,model_keras),'5g') + 
              '  knn: ' + strnum(f_knn(x,model_knn),'5g'))

    # knn esetén a leghasonlóbbak listázása
    print('Leghasonlóbb inputok: [2,5]')
    neighbors = f_knn_distances([2,5],model_knn,X_train,Y_train)
    print(neighbors)


    # Tesztelés
    print('\nTesztelés 100 inputra, a tanítási tartományon belül')
    X=np.random.rand(100,2)*10     # a tanítási input tartományon belüli tesztelés ( [0:10] )
    Y_true=array([x1**2 + x2    for x1,x2 in X])     
    t=stopperstart()
    Y_keras=f_keras(X,model_keras)
    stopper(t,msg='keras process time')
    t=stopperstart()
    Y_knn=f_knn(X,model_knn)
    stopper(t,msg='knn process time')
    print(' Mean absolute error:' +
          '  keras=' + strnum(mean_absolute_error(Y_true,Y_keras)) +
          '  knn=' + strnum(mean_absolute_error(Y_true,Y_knn)) )
    
    print('\nTesztelés 100 inputra, a tanítási tartományon kívül ([10:100])')
    X=np.random.rand(100,2)*90+10     # a tanítási input tartományon kívüli tesztelés ( [100:1000] )
    Y_true=array([x1**2 + x2    for x1,x2 in X])     
    Y_keras=f_keras(X,model_keras)
    Y_knn=f_knn(X,model_knn)
    print(' Mean absolute error:' +
          '  keras=' + strnum(mean_absolute_error(Y_true,Y_keras)) +
          '  knn=' + strnum(mean_absolute_error(Y_true,Y_knn)) )
    
# f_keras_versus_knn()
# exit()






# IDŐSOR PREDIKCIÓ

def fn_ts_test(models,timesers,test_points='last10',pred_len=7,exogs=None,finalpred=True,b_tune=False,
                    plots=True,plotdir=None,xlsout=True,xlsdir='downloads',
                    verbose=2,timeser_comments=None,loss_type='diffpercent'):  # idősoros predikció teszteléssel
    '''
    Idősoros predikciós algoritmusok tesztelése. Speciális crossvalidation, idősorokra.
    Több idősor és a több model is megadható. Mindegyik kombinációra elvégzi a teszteket és a záró-időpontra vonatkozó predikciót
    Kimenet 
      - tblResults (lásd lent;  a "loss_mean" oszlop tartalmazza az átlagos hibát, az "R2" oszlop tartalmazza a score-t)
      - xls fájl és plot-ok (az utóbbi elmaradhat; idősoronként és modellenként egy-egy plot; file-ba is írható)
    
    Előfeltétel: létre kell hozni a megfelelő ts osztály(oka)t (timeser-regressor)
        A ts-osztályok nem pontosan ugyanazt az interfészt implementálják, mint az általános sklearn modellek. Általában egy sklearn model-ből 
            érdemes kiindulni. Igazítani kell az alább definiált interfész elvárásokhoz, és meg kell oldani az X,Y inputok előállítását a timeser-ből
        - van egy konstuktora a paraméterezéshez  (átalában a mögötte álló sklearn osztály paraméterei, továbbá az X,Y input képzéséhez szükséges paraméterek)
        - implementálja a pred függvényt:      model.pred(timeser_in,pred_len) = ser_pred
        - implementálja a tune függvényt:      model.tune(timeser_in,pred_len)      # hiperparaméter optimalizáció, adott ser-re 
        - implementálja a tofile és fromfile függvényt          Model-paraméterek fájlba írása és beolvasása
        - van "name" és "tuned" adata       (tuned=False esetén a jelen függvény hívja a model.tune függvényt)
        - opcionális adatok:  "G" - plotnál használható gauss-simítási paraméter
        - lehetnek további kimeneti adatai és eljárásai is, de ezeket a jelen függvény nem használja

    models:  osztály vagy osztálynevek (list).     Példa:  ts_ARIMA, ts_MLP, ts_Gauss, ts_Prophet, ts_KNN
        - a model-osztályoknak implentálniuk kell legalább az alábbi függvényeket:  fit(X,Y)  pred(X)
        - a pred több időpontra is legyen képes visszaadni előrejelzést (pred_len>1 esetén)
        - mindegyiknek legyen name adata   (ha nincs, akkor "Model1"... elnevezéssel jelenik meg a kimenetben)
    timesers:  Series cagy Series-lista.  Az index kötelezően datetime típusú, az adatok típusa float
        - az idősorok x-tengelye legyen egyenletes eloszlású  (datetime)
        - ne legyenek adathiányok az idősorban  (a hiányzó értékek előzetesen feltöltendőek)
        - mindegyik Series-nek legyen name adata (ha nincs akkor "Series1"... elnevezés jelenik meg)
    test_points:  tesztelési időpontok
        - ha None, akkor csak a záró-időpontra számol előrejelzést, és a kimenetben a loss adatok üresek lesznek
        - 'last10':  az utolsó 10 tanítási pontra
        - 'last10_5:  utolsó 10 napra, 5 napos lépésekkel
        - 'last10_5_sun:  utolsó 10 napra, 5 napos lépésekkel, az első tesztnap legyen vasárnap
        - megadható egyetlen x-érték, lista vagy Range objektum (first,last,add,count)
        - a "<1" float számokat a timeser hosszához viszonyított arányszámként értelmezi   (eltérő hosszúságú timeser-ek esetén ezt érdemes alkalmazni)
        - általában 5-10 egymástól független tesztidőpontot érdemes megadni, növekvő train-tartománnyal
        - az utolsó lehetséges tesztpont:  len(timeser) - pred_len - 1
        - mindegyik train-re loss számítás, a végén pedig átlagos loss-értékek számítása
    pred_len:  hány időpontra kell előrejelzés  (a tesztpont-toktól illetve a záróponttól számítva, időegységenként)
    exogs:  exogén adatsorok, amelyek előrejelzést is tartalmaznak (a zárónap utánai egy vagy több napra is van adat)
        - ha több timesers van, akkor mindegyikhez meg kell adni (lehet None).  list of dict
          idősoronként:   dict(name to (f_days,ser)):      f_days: előrejelzett napok
        - a base-idősorral korreláló más idősorok (pl. hömérséklet, másik base adatsor előrejelzéssel vagy eltolással, ...) 
    finalpred:  számoljon-e előrejelzést a záró-időponta is  (ha test_points=None, akkor kötelezően True)
    plots:  False,  True,   'tofile' -  írja fájlba a képfájlt   pl  "ts_KNN tigáz_13422.png"
    plotdir:  ha nem None, akkor a megadott mappába írja 'tofile':   "ts_pred_model(n).png" fájlba írja
    timeser_comments:  dict  sername to comment    (elsősorban a plot-ban jelenik meg)
    verbose:  0: egyáltalán ne legyen print    1: csak záró print (model-timeser kombinációnként)   2: közbenső printek is
        - a modelleknek is lehet saját verbose adata. A jelen beállítás ezt nem érinti.
    loss_type:
        'diffpercent':      eltérések az átlaghoz viszonyítva  (0-k nélküli átlag)
        'percent':          eltérések a true értékhez viszonyítva (smape)
    
    return:  tblResults       ser+model párosításonként egy-egy rekord
        'model':            modelname
        'idősor':           sername,
        'test_count':       len(test_points),
        'len_pred':         pred_len,
        'loss_mean':        loss-átlag a tesztelési pontokra
        'loss_max':         maximális loss a tesztelési pontokra
        'loss_cum':         kumulált loss-értékek átlaga a tesztelési pontokra  (pred_len>1 esetén a görbe alatti terület hibája) 
        'loss_base':        záróértékkel való becslés hibája a tesztelési pontokra 
        'R2':               összehasonlítás a záróértékkel való becsléssel  (0-1)-jobb a base-nél,  0-base-zel azonos, <0: rosszabb a base-nél)
                            Ezt az értéket érdemes score-ként kezelni
        'train_score_mean': a modell scoreértéke a tanítási pontokra
        'pred_final':       zárópontra vonatkozó előrejelzés  (None, ha finalpred=False)
        'train_score_final' zárópontra vonatkozó előrejelzés tanítási score-ja  (loss érték csak a korábbiakból extrapolálható)
        'elapsed':          a számítások időtartama [sec]
        'model_params':     a model részletes paraméterei

    '''

    if not exogs: exogs=[None]*len(timesers)

    if type(timesers)!=list: timesers=[timesers]
    if type(models)!=list: models=[models]
    if type(exogs)!=list: exogs=[exogs]

    test_points_arg=test_points


    records=[]      # ser+model párosításonként egy-egy rekord
    for i_ser,ser in enumerate(timesers):

        tune_points = None
        ser_tune = None

        if test_points_arg is None: 
            test_points=[]
        elif type(test_points_arg)==str and beginwith(test_points_arg,'last'):  # last10_5_sun
            test_points = fn_lasttestpoints_in_season(ser,pred_len,test_points_arg)

            # tune-pontok
            if b_tune:
                # Ha "last..." tesztpontok lettek megadva, akkor a tune-pontok is hasonlóak lesznek,
                #   de a teszt előtti tartományban, és a tesztpontok számának felével
                testpoint_first=array(test_points).min()
                ser_tune=ser.iloc[:testpoint_first]
                before,after = splitfirst(test_points_arg,'_')
                test_points_arg2 = 'last' + str(int(len(test_points)/2)) + '_' + after
                tune_points = fn_lasttestpoints_in_season(ser_tune,pred_len,test_points_arg2)

        else:
            if type(test_points_arg) not in [list,range]: test_points=[test_points]
            else: test_points=test_points_arg

        # legkisebb és legnagyobb testpoint  (a tune-hoz és a plot-hoz szükséges; a tune csak korábbi pontokra mehet)
        testpoint_first=array(test_points).min()
        if testpoint_first<1: testpoint_first=int(len(ser)*testpoint_first)
        testpoint_last=array(test_points).max()
        if testpoint_last<1: testpoint_last=int(len(ser)*testpoint_last)



        try: sername=ser.name
        except: sername='timeseries' + str(i_ser)
        
        timeser_comment=None
        if timeser_comments is not None:
            timeser_comment=timeser_comments.get(sername)

        
        exogs_=exogs[i_ser]

        # A modellek tuned adatának nullázása
        for i_model,model in enumerate(models):
            model.tuned=False


        # Első kör: hangolások az adott ser-hez (ha szükséges)
        for i_model,model in enumerate(models):
            model.tune_time=None
            if not model.tuned and b_tune:
                # Elsőként megpróbálja beolvasni a szerializált beállításkészletet (todo)

                # A tune-pontoknak az első tesztpont elé kell esnie
                if ser_tune is None: ser_tune=ser.iloc[:testpoint_first]
                # - last típusú tesztpontok esetén a tune_points is elő van már állítva (test_points előtt és feleannyi)

                # Hívja a modell tune metódusát  (általában fn_tune_bygrid, de lehet egyedi is)
                print('TUNE start   model=' + model.name + '   ' + 'timeser=' + sername)
                t=stopperstart()
                params_tuned, loss_min, tbl_tunesteps = unpack(model.tune(ser_tune,pred_len,tune_points,exogs_),3)
                model.tune_time = stopper(t,False)

                if tbl_tunesteps is not None:
                    tbl_tunesteps=tbl_tunesteps.sort_values(by='loss')
                    if verbose>=2: 
                        print('TUNE  ' + model.name + '  ' + ser.name + '\n' + str(tbl_tunesteps))       # a teljes táblázat kiírása
                    if verbose>=1: 
                        print('TUNE result   loss min=' + strnum(loss_min,'1%') + '   sec=' + strnum(model.tune_time,'3f') + 
                                '   Optimal: ' +  dgets(params_tuned,'all',True))
                    
                    if xlsout:
                        # Közös adatok hozzáadása a lépésekhez (gyűjtőfájlba fognak kerülni, ezért szükségesek a kontextuális adatok is)
                        tbl_tunesteps['created']=fn_now()
                        tbl_tunesteps['idősor']=sername
                        tbl_tunesteps['rank']=Range(1,len(tbl_tunesteps))

                        cols_to_front(tbl_tunesteps,'created,idősor,rank')

                        Tocsv(tbl_tunesteps,'tigáz TUNE ' + model.name,dir=xlsdir,append=True,format='hu')     # mentés excel fájlba
                        # - minden egyes hangolási lépést beír a közös gyűjtőfájlba (modellenként egy-egy gyűjtőfájl)


                # Az optimális paraméterek beállítása az aktuális példányban
                set_attrs(model,params_tuned)
                model.tuned=True



        # Második kör: teszt- és éles predikciók végrehajtása
        for i_model,model in enumerate(models):
            t_start=stopperstart()

            try: modelname = model.name
            except: modelname='model' + str(i_model)
            
            # pltinit
            if plots:
                pltinit(suptitle=modelname,height=0.9,bottom=0.08,nrows=2,height_ratios=(1,5),hspace=0.17,
                        sharex=False,sharey=False)

                # Eredeti görbe a felső átnézeti diagramra
                pltinitsub('next',title=sername)
                FvPlot(ser,plttype='original',annotate='')
                
                # Fűtési szezon a felső átnézeti diagramra
                ser_heating = pd.Series(array(fn_heatingseason(ser))*ser.max()/5,ser.index)
                FvPlot(ser_heating,plttype='original',annotate='last',label='fűtési szezon',area=True)

                # test-range a felső átnézeti diagramra
                value=ser.max()       # legyen magasabb a fűtési szezon jelzőnél
                test_width=testpoint_last - testpoint_first
                lower=testpoint_first - int(test_width*0.2)
                if lower<0: lower=0
                upper=testpoint_last + int(test_width*0.2)
                if upper>len(ser)-1: upper=len(ser)-1
                testrange = [value  if i>=lower and i<=upper  else 0   for i in range(len(ser))]
                FvPlot(pd.Series(testrange,ser.index),plttype='original',annotate='last',label='test range',area=True)

                # tune_range a felső átnézeti diagramra
                if tune_points is not None:
                    value=ser.max()       # legyen magasabb a fűtési szezon jelzőnél
                    tunepoint_first=array(tune_points).min()
                    tunepoint_last=array(tune_points).max()
                    tune_width=tunepoint_last - tunepoint_first
                    lower=tunepoint_first - int(tune_width*0.2)
                    if lower<0: lower=0
                    upper=tunepoint_last + int(tune_width*0.2)
                    if upper>len(ser)-1: upper=len(ser)-1
                    tunerange = [value  if i>=lower and i<=upper  else 0   for i in range(len(ser))]
                    FvPlot(pd.Series(tunerange,ser.index),plttype='original',annotate='last',label='tune range',area=True)

                pltshowsub()



                # részletező diagram:
                pltinitsub('next')
                # Eredeti görbe és Gauss fact
                FvPlot(ser,plttype='original',annotate='',colors={'color':'gray','alpha':0.5})
                try: 
                    G=model.G
                    if G and G>0: 
                        fact_gauss = FvGaussTAvg(ser,G=G,positive=True,leftright='right')
                        # FvPlot(fact_gauss,plttype='original',label='fact',annotate='last',colors={'alpha':0.5})
                    else: G=None
                except:
                    G=None
                
                # Vasárnapok jelzése pontokkal
                # Az első vasárnap megkeresése
                index_first=6 - ser.index[0].weekday()      # a vasánap 6-nak felel meg
                serL=ser.iloc[index_first::7]       # minden 7. nap
                FvPlot(serL,plttype='scatter',annotate='last',label='vasárnapok',colors={'color':'gray','alpha':0.5})


            scores=[]
            losses_pred=[]
            losses_cum=[]
            losses_base=[]
            loss_max=0
            # R2s=[]

            for test_point in test_points:
                if test_point<=1: len_train=int(len(ser)*test_point)+1
                else: len_train=int(test_point)+1

                # Van-e elegendő pont a loss számításhoz
                if len(ser)-len_train<pred_len: 
                    print('- HIBA  ' + modelname + '  ' + sername + '  teszt pont=' + str(test_point) + ': ' + 
                          'Nincs elegendő pont az idősorban a loss számításhoz, ezért ez a tesztpont kimarad')
                    continue

                
                ser_mean=ser.iloc[:len_train].mean()
                resolution=ser.iloc[:len_train].max() / 100         # a 0 körüli értékek miatt fontos
                
                arr_= ser.iloc[:len_train].to_numpy()
                ser_mean_without0 = arr_.mean(where=arr_>0)

                def f_loss(y_fact,y_pred):
                    if loss_type=='diffpercent':
                        return (abs(array(y_fact)-array(y_pred))).mean()/ser_mean_without0
                    elif loss_type=='percent':
                        return fn_smape(y_fact,y_pred,resolution)



                # PREDIKCIÓ
                # model.pred(ser.iloc[:len_train],pred_len,exogs_)
                try:
                    model.pred(ser.iloc[:len_train],pred_len,exogs_)
                except Exception as e:
                    print('ERROR  Sikertelen pred  ' + model.name + ' ' + sername + '  teszt pont=' + str(test_point) + '\n' +
                            '   A tesztpont kimarad a számításból.' + '\n' +
                            str(e)  )
                    continue

                pred=model.ser_pred
                # Ha a model pred függvénye a tanítási tartományra is vissszadott értékeket, akkor levágás
                if len(pred)>pred_len: pred=pred.iloc[-pred_len:]

                try:
                    pred_gauss=model.ser_pred_gauss          # nem minden modell ad vissza ilyet
                except:
                    pred_gauss=pd.Series()
                score=model.score
                
                # Loss számítás
                loss_last=min(len_train+pred_len,len(ser))

                y_fact = ser.iloc[len_train:loss_last].values
                y_pred=pred.values     
                y_last = [ser.iloc[len_train-1]]*len(y_fact)  # predikció az előző napi értékkel


                loss_pred=f_loss(y_fact,y_pred)
                losses_pred.append(loss_pred)

                # if loss_pred>loss_max: loss_max=loss_pred

                loss_cum=f_loss(y_fact.cumsum(),y_pred.cumsum())
                losses_cum.append(loss_cum)

                loss_base = f_loss(y_fact,y_last)
                losses_base.append(loss_base)

                if loss_base==0: 
                    if loss_pred==0: R2=0
                    else: R2=-1
                else: R2 = 1 - (loss_pred/loss_base)        # # nem teljesen szabványos, mert a négyzetes eltérésekkel kellene képezni a hányadost
                # R2 = 1 - (((y_fact - y_pred)**2).mean()  /  ((y_fact - y_last)**2).mean())
                # R2s.append(R2)

                scores.append(score)


                # prediktált görbe nyomtatása (a train utolsó pontjától)
                if plots:
                    pred_plot=pred
                    # felveszem elé a ser utolsó tanítási pontját
                    pred_plot.loc[ser.index[len_train-1]]=ser.iloc[len_train-1]
                    pred_plot=pred_plot.sort_index()

                    # előjel = '+'
                    # if R2<0: előjel='-'
                    label='test (R2=' + strnum(R2,'3g') + ')'
                    FvPlot(pred_plot,plttype='scatter original',label=label,annotate='last',colors={'alpha':0.5})


                    if len(pred_gauss)>0 and model.G:
                        plot_len=min(len_train + pred_len + 10,len(ser))    # 10 ponttal továbbmegy
                        FvPlot(pred_gauss.iloc[len_train-int(model.G/2):plot_len],plttype='original',label='',annotate='',colors={'color':pltlastcolor(),'alpha':0.2})

                if verbose>=1:
                    print_ = ('model=' + modelname + ', idősor=' + sername + ', test_point=' + strnum(test_point,'3f') + ', ' +
                        'loss=' + strnum(loss_pred,'1%') + ', loss_cum=' + strnum(loss_cum,'2%') + ', loss_base=' + strnum(loss_base,'2%') + ', ' +
                        'R2=' + strnum(R2,'2f') + ', score=' + strnum(score,'2f'))
                    if verbose==1:  progress(print_)
                    else: print(print_)

            pred_final=None
            pred_score=None

            if finalpred:
                len_train=len(ser)

                # PREDIKCIÓ
                try:
                    model.pred(ser,pred_len,exogs_)
                except:
                    print('ERROR  Sikertelen záró predikció  ' + model.name + ' ' + sername)
                    continue

                pred_final=model.ser_pred
                # Ha a model pred függvénye a tanítási tartományra is vissszadott értékeket, akkor levágás
                if len(pred_final)>pred_len: pred_final=pred_final.iloc[-pred_len:]

                try:
                    pred_gauss=model.ser_pred_gauss          # nem minden modell ad vissza ilyet
                except:
                    pred_gauss=pd.Series()
                pred_score=model.score


                # prediktált görbe nyomtatása (a train utolsó pontjától)
                if plots:
                    pred_plot=pred_final
                    # felveszem elé a ser utolsó tanítási pontját
                    pred_plot.loc[ser.index[-1]]=ser.iloc[-1]
                    pred_plot=pred_plot.sort_index()

                    label='pred_final'
                    FvPlot(pred_plot,plttype='scatter original',label=label,annotate='last',colors={'alpha':0.5})

                    if len(pred_gauss)>0 and G:
                        plot_len=min(len_train + pred_len + 10,len(ser))    # 10 ponttal továbbmegy
                        FvPlot(pred_gauss.iloc[len_train-int(G/2):plot_len],plttype='original',label='',annotate='',colors={'color':pltlastcolor(),'alpha':0.2})

                if verbose>=2:
                    print('model=' + modelname + ', idősor=' + sername + ', final_score=' + strnum(pred_score,'2f'))


            try: 
                loss_pred_mean=np.array(losses_pred).mean()
                loss_max=np.array(losses_pred).max()
            except: 
                loss_pred_mean=np.nan
                loss_max=np.nan
            
            try: loss_cum_mean=np.array(losses_cum).mean()
            except: loss_cum_mean=np.nan
            
            try: loss_base_mean=np.array(losses_base).mean()
            except: loss_base_mean=np.nan
            
            if loss_base_mean==0: 
                if loss_pred_mean==0: R2=0
                else: R2=-1
            else: R2 = 1 - (loss_pred_mean/loss_base_mean)        # # nem teljesen szabványos, mert a négyzetes eltérésekkel kellene képezni a hányadost
            
            # try: R2 = 1 - (loss_pred_mean/loss_base_mean)       # nem teljesen szabványos, mert a négyzetes eltérésekkel kellene képezni a hányadost
            # except: R2=np.nan

            try: score_mean=np.array(scores).mean()
            except: score_mean=np.nan

            pred_time = stopper(t_start,False)

            rec={'created':fn_now(),
                 'model':modelname,'idősor':sername,'test_points':len(test_points),'len_pred':pred_len,
                 'loss_mean':Round(loss_pred_mean,3),'loss_max':Round(loss_max,3),'loss_cum':Round(loss_cum_mean,3),
                 'loss_base':Round(loss_base_mean,3),'R2':Round(R2,3),
                 'train_score':Round(score_mean,3),
                 'pred_final':pred_final,'train_score_final':Round(pred_score,3),
                 'tune_sec':Round(model.tune_time,3),'test_sec':Round(pred_time,3),'model_params':model.get_params()}
            records.append(rec)


            title = dgets(rec,'model,idősor,test_count,len_pred')
            commenttopright = (dgets(rec,'R2[4g],loss_mean[3%],loss_max[3%],loss_base[3%]',True) + '\n' +  
                               dgets(model.get_params(),'all',True))
            if timeser_comment: commenttopright += '\n' + timeser_comment
            
            print_ = ('model=' + modelname + ', idősor=' + sername + ', ' +
                'R2=' + strnum(R2,'3g') + ', ' +
                'loss=' + strnum(loss_pred_mean,'1%') + ', loss_max=' + strnum(loss_max,'1%') + ', ' +
                'loss_cum=' + strnum(loss_cum_mean,'2%') + ', loss_base=' + strnum(loss_base_mean,'2%') + ', ' +
                'train_score=' + strnum(score_mean,'3g'))
            if verbose>=2:
                print(print_)
            elif verbose>=1:
                # progress
                print('TEST result  '  + print_)

            if plots:
                # Csak az a rész látszódjon, ahol a tesztpontok vannak (mindkét oldalon 20%-os ráhagyással)
                xmin,xmax = None, None
                test_width=testpoint_last - testpoint_first
                min_point=testpoint_first - int(test_width*0.2)
                max_point=testpoint_last + int(test_width*0.2)
                if min_point>0:  xmin=ser.index[min_point]
                xmax=ser.index[max_point]

                commenttoprightL=(
                        'R2 = 1 - loss_pred / loss_base\n' +
                        '(pozitív esetén jobb, negatív esetén rosszabb a base-nél)\n' +
                        'Base: becslés az utolsó napi értékkel')

                pltshowsub(title=title,xmin=xmin,xmax=xmax,xtickerstep='date',
                           commenttopright=commenttoprightL)        # x2=len_train+loss2+10

                to_file=None
                if plots=='to_file':
                    to_file=sername + ' ' + modelname

                pltshow(commenttopright=commenttopright,to_file=to_file,pngdir=plotdir)
    
    tblResults=pd.DataFrame.from_records(records)

    if xlsout:
        fname='tigáz PRED'
        # # Ha egyetlen modellre lett hívva, akkor írja be a fájlnévbe a model nevét
        if len(models)==1 and models[0].name: fname += ' ' + models[0].name
        # # Ha egyetlen timeseriesre lett hívva, akkor írja be a fájlnévbe a timeser nevét
        # if len(timesers)==1 and timesers[0].name: fname += ' ' + timesers[0].name
        Tocsv(tblResults,fname_fix=fname,dir=xlsdir,append=True,format='hu')
        # - minden teszt bekerül a táblázatba új rekordként  (adatbázis-szerű kezelés;  downloads mappa)
    
    return tblResults


def fn_ts_loss(model,timeser,test_points=Range(0.7,0.9,count=3),pred_len=7,exogs=None):     # tune-ra használható
    '''
    Az fn_ts_test speciális hívása, egyetlen modellre és egyetlen timeser-re (nem állít elő xls-t, nincs plot és verbose=0))

    exogs:  dictionary   (exogname: (len,ser))

    Return:  a teszt-pontokra számított loss-ok átlaga
    '''
    model.tuned=False
    tblResults=fn_ts_test(model,timeser,test_points,pred_len,finalpred=False,plots=False,xlsout=False,verbose=0,exogs=exogs)
    ser_results=tblResults.iloc[0]
    return ser_results['loss_mean'],ser_results['loss_base']




class ts_Gauss:
    '''
        - van egy konstuktora a paraméterezéshez  (átalában a mögötte álló sklearn osztály paraméterei, továbbá az X,Y input képzéséhez szükséges paraméterek)
        - implementálja a pred függvényt:      model.pred(timeser,pred_len) =  ser_pred
        - implementálja a tune függvényt:      model.tune(timeser,pred_len)      # hiperparaméter optimalizáció, adott ser-re 
        - implementálja a tofile és fromfile függvényt          Model-paraméterek fájlba írása és beolvasása
        - van "name" és "tuned" adata       (tuned=False esetén a jelen függvény hívja a model.tune függvényt)
        - opcionális adatok:  "G" - plotnál használható gauss-simítási paraméter
    '''

    def __init__(self, Gmax,Gmin,Gsteps=4, T='Gmin/2',falloffstart='Gmin/2',lecsengés=0.5,tuned=False):
        self.Gmax=Gmax
        self.Gmin=Gmin
        self.G=Gmin
        self.Gsteps=Gsteps
        self.T=T
        self.falloffstart=falloffstart
        self.lecsengés=lecsengés
        self.tuned=tuned
        
        self.name='ts_Gauss'


    @classmethod
    def fn_TimescaledPred(cls,ser,Gmax,Gmin,Gsteps=2,T='Gmin/2',train_len=1,falloffstart='Gmin/2',lecsengés=0.5,prints='',**printparams):
        '''
        Történeti okokból külön függvénybe került (valójában a pred függvényről van szó)
        Részletező plot-műveletekre is alkalmas

        ser:  elvárás az x egyenletes növekedése  (előtte resample-re illetve a hiányzó értékek pótlására lehet szükség)
        Gmax:  (integer vagy <=1 float) Hosszú távú trend. Maximum a teljes tanítási hossz lehet. Ha <=1 float, akkor a teljes hossz szorzója.
        Gmin:  (integer) legalább a T kétszerese (inkább háromszorosa)
        GSteps:  (integer) Hányféle G értékre készítsen mozgóátlagot.   >=2 számot kell megadni
            - a GMax a tanítási állomány teljes hossza (x-tartomány)
            - a Gmax és a Gmin között harmonikus értékek  (ugyanaz a szorzófaktor mindegyik lépésköznél)
        T: periódusidő  (integer)
            - 0 vagy None esetén nem számol periodicitással (Gmin időskálájú predikciót ad vissza, tehát elnyomja a Gmin/2-nél rövidebb periódusidejű zajokat)
            - 'Gmin/2':  a Gmin a periódusidő kétszerese (ez az alapeset)
            - megadható egyedi érték is, de csak integer lehet
            - ha a periódusidő nem integer (pl. az év napjainak száma), akkor előtte kell egy resample a ser-re, ami módosítja az időegységet
        train_len: (integer vagy <=1 float)   a tanítási pontok száma (a görbe bal oldalán). Float esetén a ser hosszához képest.
        falloffstart:   'Gmin/2'        xlast - Gmin/2 (konstans)     Ez látszik a legjobbnak      
                        'G/2'           xlast - G/2 (dinamikus)
                        'lasttrain'     xlast (konstans)              Túlzott kilengéseket okozhat
            - ettől a ponttól indul a fél-gauss lecsengés (a szigma mindenképpen G/2)
        lecsengés:  a G hányszorosa legyen a lecsengés-gauss szigmája
        prints:  'plot', 'gausst', 'stationars', 'ciklusok'     Több is felsorolható (vesszővel vagy szóközzel)
            - 'plot':  nyomtatás segédvonalakkal.  Printparams:  print_arima   print_multigausst
            - 'gausst': FvPlot hívások a gausst illesztésekre (nem differenciális Gauss, hanem közvetlen). Szürke és átlátszó vonalak. pltinit-show keretben kell hívni
        printparams:  a nyomtatásokhoz szükséges segédparaméterek 

        return: dict('G','pred','pred_gauss','fact_gauss')       Részletesen lásd a függvény végén
        '''

        if train_len<=1: train_len=int(len(ser)*train_len)
        else: train_len=int(train_len)

        ser_train=ser[:train_len].copy()

        # aG
        if Gmax<=1: Gmax=int(len(ser_train)*Gmax)
        aG=[None]*Gsteps
        for i in range(Gsteps): aG[i]=int(round(Gmax - (Gmax-Gmin)/(Gsteps-1)*i))

        # aGV
        GminV=Gmin
        aGV=aG.copy()
        # if type(GminV)==str:
        #     if GminV=='2Gmin': GminV=2*Gmin
        #     elif GminV=='Gmin': GminV=Gmin
        # faktor=(Gmax/GminV)**(1/(Gsteps-1))
        # aGV=[Gmax]*Gsteps
        # for i in range(1,Gsteps): aGV[i]=int(round(aGV[i-1]/faktor))

        xlast=ser_train.index[-1]
        
        def f_gauss_lecsengés_params(Gmax,Gmin):
            extendleft=0
            if falloffstart=='lasttrain':
                x0=ser_train.index[-1]   # konstans érték minden G-re
                aX0=[x0]*len(aG)        # ettől a ponttól indul a Gauss lecsengés
                extendright=int(Gmax)
            elif falloffstart=='Gmin/2':
                x0=ser_train.index[-int(Gmin/2)]   # konstans érték minden G-re (a Gmin határozza meg)
                aX0=[x0]*len(aG)        # ettől a ponttól indul a Gauss lecsengés
                extendright=int(Gmax-Gmin/2)
            elif falloffstart=='G/2':
                aX0=[ser_train.index[-int(G/2)] for G in aG]
                extendright=int(Gmax/2)
            else:
                print('Érvénytelen falloffstart argumentum')
            return aX0,extendright,extendleft
        
        aX0,extendright,extendleft=f_gauss_lecsengés_params(Gmax,Gmin)
        aX0V,extendrightV,extendleftV=f_gauss_lecsengés_params(Gmax,GminV)

        def f_gauss_lecsengés(ser,x0,szigma):
            def f_apply(rec):
                x=rec.name
                if x>=x0:
                    rec.y = rec.y * fn_gauss(x,x0,szigma)
                return rec
            return serapply(ser,f_apply)

        aGauss=[]
        aGaussLabel=[]
        aGaussPred=[]
        aStationar=[]
        aStationarLabel=[]
        aStationar.append(ser_train.copy())
        aStationarLabel.append('fact')

        fact_gauss = FvGaussTAvg(ser.copy(),G=Gmin,positive=True,leftright='right')
        variance_fact = FvGaussTAvg((ser-fact_gauss)**2,G=GminV,positive=True,leftright='right').apply(math.sqrt) 
        fact_upper = fact_gauss + variance_fact


        # Átlag stacionarizálása
        for i,G in enumerate(aG):
            # Az utolsó stacionarizált függvény gauss mozgóátlaga (csökkenő ablakszélességgel)
            aGauss.append(FvGaussTAvg(aStationar[-1],G=G,extend=(extendleft,extendright),positive=True,leftright='right'))
            aGaussLabel.append('GaussMean G=' + str(G))
            # Gauss mozgóátlag kivonása
            aStationar.append(aStationar[-1]-aGauss[-1].iloc[extendleft:-extendright])
            aStationarLabel.append('Stationar diff ' + str(i))

            # Lecsengés
            if i==0: 
                aGaussPred.append(aGauss[-1])
                pred_gauss=aGaussPred[-1]       # kumulatív
            else: 
                aGaussPred.append(f_gauss_lecsengés(aGauss[-1],aX0[i],lecsengés*G))
                pred_gauss=pred_gauss + aGaussPred[-1]    # kumulált

        gaussvariance_fact=FvGaussTAvg(aStationar[-1]**2,G=GminV,positive=True,leftright='right').apply(math.sqrt)
        
        # Variancia stacionarizálása
        aGaussVariance=[]
        aGaussVarianceLabel=[]
        aGaussVariancePred=[]
        for i,G in enumerate(aGV):
            # Az utolsó stacionárius függvény négyzetének Gauss simítása, majd négyzetgyök (variancia mozgóátlaga)
            aGaussVariance.append(FvGaussTAvg(aStationar[-1]**2,G=G,extend=(extendleft,extendright),positive=True,leftright='right').apply(math.sqrt))
            aGaussVarianceLabel.append('GaussVariance G=' + str(G))
            # A stacionárius függvény skálázása a variancia inverzével
            aStationar.append( aStationar[-1]/aGaussVariance[-1].iloc[extendleft:-extendright] )
            aStationarLabel.append('Stationar scale ' + str(i))

            # Lecsengés
            if i==0: 
                aGaussVariancePred.append(aGaussVariance[-1])
                gaussvariance_pred=aGaussVariancePred[-1]
            else: 
                aGaussVariancePred.append(f_gauss_lecsengés(aGaussVariance[-1]-1,aX0V[i],lecsengés*G) + 1)         # 1 körüli függvény
                gaussvariance_pred=gaussvariance_pred*aGaussVariancePred[-1]


        # Részletes predikció
        pred=pred_gauss.copy()        # ha nincs periodicitás, akkor ez lesz a kimenet
        pred_upper=pred + gaussvariance_pred

        pred_variance=gaussvariance_pred.values


        stationar=aStationar[-1]      # az utolsó stacionarizálást követő görbe
        if T and T>0:
            if type(T)==str and T.lower() in ['gmin/2','g/2']: T=int(Gmin/2)
            # Periódikus mintázat előállítása
            # Le kell vágni az elejéről a csonka ciklust
            cut=len(stationar)%T
            stationarL=stationar[cut:].values
            # A jövőbeli ciklusok nem lesznek teljesen egyformák. A stationarL folyamatosan bővül
            ciklus_pred_db=math.ceil((len(pred)-len(stationarL))/T)
            for i_ciklus_pred in range(ciklus_pred_db):
                ciklus_db=int(len(stationarL)/T)         # folyamatosan nő
                # Súlyfaktorok előállítása gauss függvénnyal (ciklusok súlyozása)
                # - minél régebbi, annál kevésbé érvényesül (elavulás)
                szigma=10    # hangolható paraméter. Az utolsó 5-10 ciklust veszi figyelembe (5-ről 10-re emeltem)
                x0 = 0      # 1 is felmerült: eggyel korábbi ciklus legyen a legjobban súlyozva, mert az utolsó ciklus méréseire még előfordulhat korrekció
                weights1=[fn_gauss(x,x0,szigma) for x in range(ciklus_db)]  # az utolsótól visszafelé

                # weights2 (NINCS HASZNÁLVA): minél jobban korrelál az előtte lévő félciklussal, annál jobban érvényesül
                weights2=[None]*(ciklus_db)
                utolsófélciklus=stationarL[-int(T/2)-1:]   # beveszem az előző pontot is, mert a görbéhez a bevezető szakasz is hozzátartozik
                for i in range(ciklus_db):
                    előzőfélciklus=stationarL[-(i+1)*T-int(T/2)-1:-(i+1)*T]
                    if len(előzőfélciklus)<int(T/2)+1: weights2[i]=0   # nincs előző félciklusa, kimarad a súlyozásból
                    else:
                        corr_félciklus=corr(előzőfélciklus, utolsófélciklus)
                        weights2[i]=corr_félciklus
                        if weights2[i]<0: weights2[i]=0     # nagyon egzotikus esetben fordulhat elő
                weights=np.array(weights1)   
                # weights=(np.array(weights1) * (np.array(weights2))**2)   # a korrelációt átskálázom, mert általában túl közel van 1-hez
                # weights=(np.array(weights1) + np.array(weights2))/2
                # if i_ciklus_pred<2:
                #     Toexcel(pd.DataFrame({'gauss':weights1,'prehalfmatch':weights2,'eredő':weights}),'weights')
                # Súlyozott átlag a ciklusok napjaira
                ciklus=[None]*T
                for iZárónap in range(-T,0):    # -T, -T+1, -T+2,  -1
                    ciklus[iZárónap+T] = np.average(stationarL[iZárónap::-T],weights=weights)  # iZárónap, iZárónap-T, iZárónap-2T, ...

                if 'ciklusok' in prints and i_ciklus_pred==0:
                    utolsófélciklusL=pd.Series(utolsófélciklus)
                    # utolsófélciklusL.index=utolsófélciklusL.index + int(T/2)
                    pltinit(nrows=5,ncols=3,indexway='col')
                    for i in range(len(weights)):
                        másfélciklus=stationarL[-(i+1)*T-int(T/2)-1:len(stationarL)-i*T]
                        if len(másfélciklus)<T+ int(T/2)+1: break  # ha nincs már előtte félciklus
                        pltinitsub('next',title=str(i) + '. periódus visszafelé, előzménnyel')
                        FvPlot(pd.Series(másfélciklus),plttype='scatter original',annotate='')    # ,colors={'alpha':weights[i]
                        FvPlot(utolsófélciklusL,plttype='original',label='utolsó félciklus',annotate='last')
                        pltshowsub(commenttopright='prehalf_similarity=' + strnum(weights2[i],'3f') + '\n' +
                                                'weight=' + strnum(weights[i],'3f'))
                    pltinitsub('next')
                    pltinitsub('next',title='Összes periódus')
                    for i in range(len(weights)):
                        serL=pd.Series(stationarL[-(i+1)*T-1:len(stationarL)-i*T])
                        serL.index=serL.index + int(T/2)
                        FvPlot(serL,plttype='original',annotate='',colors={'color':'navy','alpha':weights[i]**2})
                    FvPlot(utolsófélciklusL,plttype='original',label='utolsó félciklus',annotate='last')
                    pltshowsub()
                    pltinitsub('next',title='Súlyozott átlag')
                    ciklusL=pd.Series(ciklus)
                    ciklusL[-1]=utolsófélciklusL.iloc[-1]
                    ciklusL=ciklusL.sort_index()
                    ciklusL.index=ciklusL.index + int(T/2)+1
                    FvPlot(ciklusL,plttype='original',annotate='')
                    FvPlot(utolsófélciklusL,plttype='original',label='utolsó félciklus',annotate='last')
                    pltshowsub() 
                    pltshow()

                stationarL=np.append(stationarL,ciklus)



            # A csonka ciklus viszaírása az elejére (0 értékekkel)
            stationarL=np.insert(stationarL,0,[0]*cut)

            # A periodikus mintázat hozzáadása a kimenethez (csak a predikciós tartományban)
            
            for i in range(len(ser_train),len(pred)):
                pred.iloc[i] = pred.iloc[i] + stationarL[i] * pred_variance[i]




        # Stacionárius görbék nyomtatása
        if 'stationars' in prints:
            pltinit(nrows=(len(aStationar)+1)//2,ncols=2,indexway='vert',sharey=False)
            for i,(serL,labelL) in enumerate(zip(aStationar,aStationarLabel)): 
                pltinitsub(axindex=i,title=labelL)
                FvPlot(serL,plttype='original',annotate=None)
                pltshowsub()
            pltshow()

        # Differenciális Gauss görbék nyomtatása
        if 'steps' in prints:
            aGaussL=aGauss + [fact_gauss] + aGaussVariance + [gaussvariance_fact]
            aGaussLabelL=aGaussLabel + ['GaussMean PRED'] + aGaussVarianceLabel + ['GausssVariance PRED']
            aGaussPredL=aGaussPred + [pred_gauss] + aGaussVariancePred + [gaussvariance_pred]
            aStationarL=aStationar[0:len(aG)] + [ser] + aStationar[len(aG):2*len(aG)] + [aStationar[len(aG)]]
            aBandLeft=aX0 + [aX0[0]]
            aBandLeft=aBandLeft + aBandLeft
            aBandRight=list(np.array(aX0) + np.array(aG))  + [aX0[0] + aG[0]]
            aBandRight=aBandRight + aBandRight
            pltinit(nrows=(len(aGaussL)+1)//2,ncols=2,indexway='vert',sharey=False,
                suptitle='Stacionarizálás')
            for i,(serL,labelL,serPredL,serStationarL) in enumerate(zip(aGaussL,aGaussLabelL,aGaussPredL,aStationarL)): 
                pltinitsub(axindex=i,title=labelL)
                FvPlot(serL,plttype='original',label='original',annotate='last',colors={'color':'gray','alpha':0.5})
                FvPlot(serPredL,plttype='original',label='pred',annotate='last')   # exponenciális lecsengéssel
                FvPlot(serStationarL,plttype='original',label='stationar',annotate='last')
                # if i in [0,len(aG)]: FvPlot(ser,plttype='original',label='nogauss',annotate='last',colors={'color':'gray','alpha':0.5})

                pltshowsub(y_bands=[{'x1':aBandLeft[i],'x2':aBandRight[i]},{'x1':xlast,'x2':xlast+1}])
            pltshow(commenttopright='T=' + str(T) + '\n' + 'Gmax=' + str(Gmax) + '\n' + 'Gmin=' + str(Gmin))

        # Gauss illesztések nyomtatása (csak FvPlot)
        if 'gausst' in prints:    
            for G in aG:
                serL=FvGaussTAvg(ser_train,G=G,extend=(extendleft,extendright),positive=True,leftright='right')
                FvPlot(serL,plttype='original',label='gausst' + str(G),annotate='last',colors={'color':'gray','alpha':0.1})



        return {'G':aG[-1],             # legkisebb G érték  (plot-hoz használható)
                'ser_pred':pred,        # (ser) predikció. A train-időszakban megegyezik a pred_gauss-del, utána a részletes predikciót tartalmazza
                'ser_pred_upper':pred_upper,  # (ser)  A predikciós görbe felső variancia-határa
                'ser_pred_gauss':pred_gauss,  # (ser) Gmin időskálájú mozgóátlag a train-adatokra és a predikciós időszakra
                'ser_pred_variance':pred_variance,   # (ser) Predikált görbe varianciája
                'ser_fact_gauss':fact_gauss,  # (ser) Gmin mozgóátlag a teljes input görbére
                                        # - nem feltétlenül azonos a pred_gauss train-időszakra eső részével, mert a pred_gauss több lépéses 
                                        #    mozgóátlagolással, majd a differenciális gauss-komponensek visszaösszegzésével jön létre
                'ser_fact_upper':fact_upper,  # (ser) az input görbe varianciája
                'ser_stationar':stationar,    # (ser) stacionarizált görbe
                }



    def pred(self,ser,pred_len,exogs):
        # autosorszámozott indexet igényel

        ser,datefloat0,date_step = ser_reset_datetimeindex(ser)
        
        result=ts_Gauss.fn_TimescaledPred(ser,self.Gmax,self.Gmin,self.Gsteps,self.T,falloffstart=self.falloffstart,
                    lecsengés=self.lecsengés)
        
        # kell még egy score-számítás a tanítási pontokra
        
        self.score=1     # a train-pontokra teljes egyezést ad, mert invertálható dekompozícióról van szó

        self.ser_pred=ser_restore_datetimeindex(result['ser_pred'],datefloat0,date_step)
        self.ser_pred_upper=ser_restore_datetimeindex(result['ser_pred_upper'],datefloat0,date_step)
        self.ser_pred_gauss=ser_restore_datetimeindex(result['ser_pred_gauss'],datefloat0,date_step)
        self.ser_pred_variance=result['ser_pred_variance']
        self.ser_fact_gauss=ser_restore_datetimeindex(result['ser_fact_gauss'],datefloat0,date_step)
        self.ser_fact_upper=ser_restore_datetimeindex(result['ser_fact_upper'],datefloat0,date_step)
        self.ser_stationar=ser_restore_datetimeindex(result['ser_stationar'],datefloat0,date_step)
        self.G=result['G']

        return self.ser_pred
        

    def tune(self,ser,pred_len,test_points,exogs):
        if 'Date' in str(type(ser.index)): ser=ser.reset_index(drop=True)      # nem kell datetime index, egyszerű sorszámozás

        params_grid={
            'Gmax': [self.Gmin*2, [self.GMin*1.5,self.GMin*2.5,self.GMin*3]],
            'lecsengés': [0.5,  [0.2, 0.4, 0.6, 0.8, 1]],
            }

        if test_points is None: test_points='last5_5_sun'

        def fn_loss(**modelparams):
            test_pointsL = list(array(test_points) - np.random.randint(0,6))
            return fn_ts_loss(ts_Gauss(**modelparams),ser,test_pointsL,pred_len)

        return fn_tune_bygrid(fn_loss,params_grid)




    def get_params(self):   # releváns modell-paraméterek (megjelenítésekben használható)
        return {'Gmax':self.Gmax,'Gmin':self.Gmin,'T':self.T,'Gsteps':self.Gsteps,'lecsengés':self.lecsengés}


class ts_Prophet:
    def __init__(self, changepoint_prior_scale=0.05, seasonality_prior_scale=10, holidays_prior_scale=10, 
                    seasonality_mode='multiplicative',yearly_seasonality='auto',weekly_seasonality='auto',daily_seasonality='auto',
                    w_exogs=0, 
                    tuned=False):
        self.name='ts_Prophet'

        self.changepoint_prior_scale=changepoint_prior_scale
        self.seasonality_prior_scale=seasonality_prior_scale
        self.holidays_prior_scale=holidays_prior_scale
        self.seasonality_mode=seasonality_mode
        self.yearly_seasonality=yearly_seasonality
        self.weekly_seasonality=weekly_seasonality
        self.daily_seasonality=daily_seasonality

        self.w_exogs=w_exogs       # 1 esetén figyelembe veendő a hőmérsékleti adatsor

        self.tuned=tuned        


        
    def pred(self,ser,pred_len,exogs):
        m = Prophet(changepoint_prior_scale=self.changepoint_prior_scale,
                yearly_seasonality=self.yearly_seasonality, weekly_seasonality=self.weekly_seasonality,daily_seasonality=self.daily_seasonality,
                seasonality_prior_scale=self.seasonality_prior_scale,
                holidays_prior_scale=self.holidays_prior_scale,seasonality_mode=self.seasonality_mode)
                # growth='logistic')          # ne mehessen 0 alá

        m.add_country_holidays(country_name='HU')           # python holydays package
        

        # A fit-hez kell egy input _train, ds és y oszlopokkal     (ds: datetime-series)
        tbl_train=pd.DataFrame({'ds':ser.index,'y':ser.values})   
        # tbl_train['floor']=0

        # exogén idősorok hozzáadása  (regressors, hőmérsékleti adatsor)
        ser_celsius=None
        if self.w_exogs>0 and exogs is not None:
            for name,adatok in exogs.items():     #  name to (forecast_days,ser)  
                if name=='hőmérséklet':
                    forecast_days,ser_celsius = adatok
                    if forecast_days>=pred_len:     # csak akkor használható, ha az exogén idősor jövőbeli értékei a teljes pred_len-re rendelkezésre állnak
                        ser_celsius=ser_celsius - ser_celsius.min()
                        tbl_train = merge_ser(tbl_train,ser_celsius,'celsius','ds')
                        m.add_regressor('celsius', standardize='auto',prior_scale=0.5, mode='additive')
        
        m.fit(tbl_train)

        # Előrejelzés pred_len napra
        tbl_future = m.make_future_dataframe(periods=pred_len)
        # tbl_future['floor']=0
        # A tbl_future táblába is be kell írni az exogén idősorok jövőbeli értékeit
        if ser_celsius is not None and forecast_days>=pred_len:
            tbl_future = merge_ser(tbl_future,ser_celsius,'celsius','ds')

        tbl_forecast = m.predict(tbl_future)

        self.ser_pred=serfromtbl(tbl_forecast,'yhat','ds')
        self.score=1     # a train-score nem értelmezhető (vagy nem fontos)

        return self.ser_pred

    def tune(self,ser,pred_len,test_points,exogs):

        params_grid={
            'changepoint_prior_scale':  [0.05,               ],             # ritkán változott, ezért nincs hangolás
            'seasonality_prior_scale':  [10,                 [7,15]],
            'holidays_prior_scale':     [10,                 [7,15]],
            'weekly_seasonality':       ['auto',             [2,4]],        # 3 a default
            'seasonality_mode':         ['multiplicative',   ['additive']],  
            'w_exogs':                  [0,                  [1]]           # rosszabb lesz a celsius idősorral
            }

        if test_points is None: test_points='last5_5_sun'

        def fn_loss(**modelparams):
            test_pointsL = list(array(test_points) - np.random.randint(0,5))
            return fn_ts_loss(ts_Prophet(**modelparams),ser,test_pointsL,pred_len,exogs=exogs)

        return fn_tune_bygrid(fn_loss,params_grid)




    def get_params(self):   # releváns modell-paraméterek (megjelenítésekben használható)
        return get_attrs(self,'weekly_seasonality,' +
                         'changepoint_prior_scale,seasonality_prior_scale,holidays_prior_scale,' +
                         'seasonality_mode,w_exogs')


class ts_MLP:
    def __init__(self, points_in=14,G=0,hidden_layer_sizes=(10,10,10,10,10),learning_rate_init=0.00005,alpha=0.1,tol=1e-5,
                    heat_season=1,weekdays=0,weekend=1,season=4,w_exogs=1,w_level=0,epszilon=0.1,
                    tuned=False):
        self.name='ts_MLP'

        self.points_in=points_in
        self.G=G
        self.hidden_layer_sizes=hidden_layer_sizes
        self.learning_rate_init=learning_rate_init
        self.alpha=alpha
        self.tol=tol
        self.heat_season=heat_season          # True esetén mellékel egy fűtési szezonalitás-jelzőt az input-hoz (input pontonként 0 vagy 5, átlagérték)
        self.weekdays=weekdays          # True esetén mellékel egy weekday jelzőt az input-hoz (input zárópontra  0,..,6)
        self.weekend=weekend          # True esetén mellékel egy hétvég jelzőt az input-hoz (input zárópontra, 0 vagy 5)
        self.season=season          # True esetén mellékel egy évszak-jelzőt az input-hoz (input zárópontra, 0,1,2,3)
        self.w_exogs=w_exogs
        self.w_level=w_level        # mennyire érzékeny a görberészlet szintjére (pl. a nulla-közeli szakaszok érdemben eltérően viselkednek)
        self.epszilon=epszilon

        self.tuned=tuned        # egyelőre nincs hangoló algoritmus


        
    def pred(self,ser,pred_len,exogs):
        mlp = MLPRegressor(solver='adam',hidden_layer_sizes=self.hidden_layer_sizes,max_iter=2000,
            learning_rate_init=self.learning_rate_init,learning_rate='constant',
            early_stopping=True,verbose=False,alpha=self.alpha,n_iter_no_change=10,tol=self.tol,activation='relu')

        ser_pred,score = fn_ts_pred_bylastpoints(mlp,ser,self.points_in,pred_len,1,self.G,'sklearn',
                                                self.heat_season,self.weekdays,self.weekend,self.season,exogs,
                                                self.w_exogs,self.w_level,self.epszilon)


        self.ser_pred = ser_pred
        self.score=score     # train-score


        return self.ser_pred


    def tune(self,ser,pred_len,test_points,exogs):
        # if 'Date' in str(type(ser.index)): ser=ser.reset_index(drop=True)      # nem kell datetime index, egyszerű sorszámozás



        if pred_len<3: points_in_values=    [7,             [5,9]]
        else: points_in_values=             [pred_len,      [2*pred_len,3*pred_len]] 

        params_grid={
            'points_in': points_in_values,
            'G':            [0,                     ],
            'hidden_layer_sizes':[(10,10,10,10,10)  ],                                # , [(12,12,12,12,12)]
            'learning_rate_init':[1e-3,1e-4,        [3e-3,3e-4],[3e-4,3e-5]],
            'alpha':        [0.1,0.01,              [0.2,0.05],[0.003,0.001]],
            'tol':          [1e-4                   ],          # [1e-4,                  [3e-4,3e-5]],
            'heat_season':  [1,                     [0]],
            'weekdays':     [0                      ],
            'weekend':      [1,                     [0]],
            'season':       [4                      ],
            'w_exogs':      [1,                     [2]],
            'w_level':      [1,                     [0]],
            'epszilon':     [0.1,                   ],
            }

        params_grid={
            'points_in': points_in_values,
            'G':            [0,                     ],
            'hidden_layer_sizes':[(10,10,10,10,10)  ],                              
            'learning_rate_init':[1e-3,1e-4,        [3e-3,3e-4],[3e-4,3e-5]],
            'alpha':        [0.1,0.01,              [0.2,0.05],[0.003,0.001]],
            'tol':          [1e-4                   ],        
            'heat_season':  [1,                     [0]],
            'weekdays':     [0                      ],
            'weekend':      [1,                     [0]],
            'season':       [4                      ],
            'w_exogs':      [1,                     [2]],
            'w_level':      [1,                     [0]],
            'epszilon':     [0.1,                   ],
            }



        # params_grid={
        #     'points_in':    points_in_values,
        #     'G':            [0,                     [2]],
        #     'hidden_layer_sizes':[(10,10,10,10),    [(10,10,10,10,10)]],
        #     'learning_rate_init':[1e-3,             [1e-4,3e-5]],
        #     'alpha':        [0.1,                   [0.01,0.001]],
        #     'tol':          [3e-5,                  [1e-4,1e-5]],
        #     'heat_season':  [True,                  [False]],
        #     'weekdays':     [False,                 [True]],
        #     'weekend':      [True,                  [False]],
        #     'season':       [False,                 [True]],
        #     'w_exogs':      [True,                  [False]],
        #     }

        # test_points=Range(0.6+fn_noise(0.02),0.9+fn_noise(0.02), count=5)
        # test_points='last5'

        if test_points is None: test_points='last5_5_sun'

        def fn_loss(**modelparams):
            test_pointsL = list(array(test_points) - np.random.randint(0,6))
            return fn_ts_loss(ts_MLP(**modelparams),ser,test_pointsL,pred_len,exogs)
            # - új példányt hoz létre a megadott paraméterekkel (annyiszor, ahány paraméter kombináció van)

        return fn_tune_bygrid(fn_loss,params_grid)



    def get_params(self):   # releváns modell-paraméterek (megjelenítésekben használható)
        return get_attrs(self,'points_in,G,hidden_layer_sizes,learning_rate_init,alpha,tol')


class ts_ARIMA:
    def __init__(self, p=2,d=1,q=1,P=2,D=1,Q=1,m=1,w_exogs=0,tuned=False):
        self.name='ts_ARIMA'

        self.p=p        # autoregression order.  Hány visszamenőleges napot vegyen figyelembe  (lineáris illesztés)
        self.d=d        # differential order.  Hány deriválás kell ahhoz, hogy stacionáriussá váljon.  Default: 1
        self.q=q        # moving average order.  Zajszűréshez szükséges mozgóablak szélessége
        self.P=P        # szezonális komponens, autoregression order
        self.D=D        # szezonális komponens, differential order
        self.Q=Q        # szezonális komponens, moving average order
        self.m=m        # 1 esetén nincs szezonális komponens.   
        
        self.w_exogs=w_exogs       # 1 esetén figyelembe veendő a hőmérsékleti adatsor


        self.tuned=tuned        


        
    def pred(self,ser,pred_len,exogs):
        # autosorszámozott indexet igényel
        ser,datefloat0,date_step = ser_reset_datetimeindex(ser)

        if not self.w_exogs or exogs is None:
            arima=ARIMA(order=(self.p,self.d,self.q) ,seasonal_order=(self.P,self.D,self.Q,self.m),
                            enforce_stationarity=False,enforce_invertibility=False)

            arima.fit(ser)
            ser_new=pd.Series(arima.predict(pred_len))
            ser_old=pd.Series(arima.predict_in_sample())
            ser_pred=ser_old.append(ser_new,True)

        else:
            len_ser=len(ser)
            endog=ser.to_numpy()
            
            ser_celsius=None
            if exogs:
                for name,adatok in exogs.items():     #  name to (forecast_days,ser)  
                    if name=='hőmérséklet':
                        forecast_days,ser_celsius = adatok
            exog=ser_celsius.to_numpy()

            sarimax=SARIMAX(endog=endog, exog=exog[:len(endog)], 
                            order=(self.p,self.d,self.q) ,seasonal_order=(self.P,self.D,self.Q,self.m), 
                            trend='c',
                            enforce_invertibility=False, enforce_stationarity=False)
            sarimax_fitted=sarimax.fit(disp=False)
            y=sarimax_fitted.predict(start=len_ser,end=len_ser+pred_len-1,exog=exog[len_ser:len_ser+1])
            
            y[y<0]=0        # előfordulhatnak negatív értékek

            ser_new=pd.Series(y)
            ser_pred=ser.append(ser_new,True)

        # Dátumindex visszaállítása
        ser_pred = ser_restore_datetimeindex(ser_pred,datefloat0,date_step)

        self.ser_pred = ser_pred
        self.score=np.nan        # train-score

        return self.ser_pred

    def tune(self,ser,pred_len,test_points,exogs):
        if 'Date' in str(type(ser.index)): ser=ser.reset_index(drop=True)      # nem kell datetime index, egyszerű sorszámozás

        # A pdqn hangolás mindenképpen kell
        arima = auto_arima(ser, start_p=2, max_p=3, max_d=2, start_q=2, max_q=3, m=self.m,
                        trace=False,
                        error_action='ignore',  # don't want to know if an order does not work
                        suppress_warnings=True,  # don't want convergence warnings
                        stepwise=True)  # set to stepwise
        # p:  points_in  hány pontot vegyen figyelembe a görbe végén a következő napo(k) számításához
        # d:  hányadik deriválttal tehető nagyjából stacionáriussá;   hányadrendű polinom
        # q:  zajszűrő mozgóátlag ablakszélessége

        p,d,q=arima.order
        P,D,Q,m=arima.seasonal_order
        self.p=p
        self.d=d
        self.q=q
        self.P=P
        self.D=D
        self.Q=Q
        self.m=m
        print('ARIMA  Optimal  order=' + str(arima.order) + '   seasonal_order=' + str(arima.seasonal_order))

        #  return get_attrs(self,'p,d,q,P,D,Q,m'), None, None


        # grid-search az exogén idősorok ki illetve bekapcsolására
        params_grid={
            'p':        [self.p     ],
            'd':        [self.d     ],
            'q':        [self.q     ],
            'P':        [self.P     ],
            'D':        [self.D     ],
            'Q':        [self.Q     ],
            'm':        [self.m     ],
            'w_exogs':  [0,         [1]],
            }

        # params_grid={
        #     'p': [2,3,4],
        #     'd': [1,2],
        #     'q': [1,2],
        #     'P': [1,2],
        #     'D': [1],
        #     'Q': [1],
        #     'm': [self.m],
        #     }

        def fn_loss(**modelparams):
            test_pointsL = list(array(test_points) - np.random.randint(0,6))
            return fn_ts_loss(ts_ARIMA(**modelparams),ser,test_pointsL,pred_len,exogs)
            # - új példányt hoz létre a megadott paraméterekkel (annyiszor, ahány paraméter kombináció van)

        return fn_tune_bygrid(fn_loss,params_grid)



    



    def get_params(self):   # releváns modell-paraméterek (megjelenítésekben használható)
        return get_attrs(self,'p,d,q,P,D,Q,m,w_exogs')


class ts_KNN:
    def __init__(self, points_in=14, G=2, n_neighbors=10, weights='distance', leaf_size=30, 
                    heat_season=1,weekdays=0,weekend=1,season=4,w_exogs=1,w_level=1,epszilon=0.1,
                     tuned=False):
        self.name='ts_KNN'

        self.points_in=points_in
        self.G=G                            # gauss simítás
        self.n_neighbors=n_neighbors
        self.weights=weights                # 'distance'  'uniform'
        self.leaf_size=leaf_size
        
        self.heat_season=heat_season          # True esetén mellékel egy fűtési szezonalitás-jelzőt az input-hoz (input pontonként 0 vagy 5, átlagérték)
        self.weekdays=weekdays          # True esetén mellékel egy weekday jelzőt az input-hoz (input zárópontra  0,..,6)
        self.weekend=weekend          # True esetén mellékel egy hétvég jelzőt az input-hoz (input zárópontra, 0 vagy 5)
        self.season=season          # True esetén mellékel egy évszak-jelzőt az input-hoz (input zárópontra, 0,1,2,3)
        self.w_exogs=w_exogs
        self.w_level=w_level        # mennyire érzékeny a görberészlet szintjére (pl. a nulla-közeli szakaszok érdemben eltérően viselkednek)
        self.epszilon=epszilon

        self.tuned=tuned        

        
    def pred(self,ser,pred_len,exogs):
        knn = KNeighborsRegressor(n_neighbors=self.n_neighbors, weights=self.weights, leaf_size=self.leaf_size)

        ser_pred,score = fn_ts_pred_bylastpoints(knn,ser,self.points_in,pred_len,1,self.G,'sklearn',
                                                self.heat_season,self.weekdays,self.weekend,self.season,
                                                exogs,self.w_exogs,self.w_level,self.epszilon)

        self.ser_pred = ser_pred
        self.score=score     # train-score


        return self.ser_pred


    def tune(self,ser,pred_len,test_points,exogs):
        # if 'Date' in str(type(ser.index)): ser=ser.reset_index(drop=True)      # nem kell datetime index, egyszerű sorszámozás

        if pred_len<3: points_in_values=    [9,             [7,11]]
        else: points_in_values=             [2*pred_len,      [pred_len,3*pred_len]] 

        params_grid={
            'points_in': points_in_values,
            'G':            [0,                     ],
            'n_neighbors':  [10,                    [8,12]],
            'leaf_size':    [20                     ],
            'weights':      ['distance',            ['uniform']],                  # érzéketlen volt a "uniform" váltásra
            'heat_season':  [1,                     [0]],
            'weekdays':     [0,                     ],
            'weekend':      [1,                     [2,0]],
            'season':       [4,                     [0]],
            'w_exogs':      [1,                     [0]],
            'w_level':      [1,                     [0]],
            'epszilon':     [0.1,0.03,              [0.2],[0.01]],
            }

        if test_points is None: test_points='last5_5_sun'


        def fn_loss(**modelparams):
            test_pointsL = list(array(test_points) - np.random.randint(0,6))
            return fn_ts_loss(ts_KNN(**modelparams),ser,test_pointsL,pred_len,exogs)
            # - új példányt hoz létre a megadott paraméterekkel (annyiszor, ahány paraméter kombináció van)

        return fn_tune_bygrid(fn_loss,params_grid)




    def get_params(self):   # releváns modell-paraméterek (megjelenítésekben használható)
        return get_attrs(self,'points_in,G,n_neighbors,weights,heat_season,weekdays,weekend,season,w_exogs,w_level,epszilon')



class ts_LSTM:
    def __init__(self, points_in=7,G=0,layers=(20),learning_rate_init=0.001,tol=1e-5,
                    heat_season=True,weekdays=False,weekend=True,season=False,w_exogs=True,w_level=0,epszilon=0.1,
                    tuned=False):
        self.name='ts_LSTM'

        self.points_in=points_in
        self.G=G
        self.layers=layers
        self.learning_rate_init=learning_rate_init
        self.tol=tol
        self.heat_season=heat_season          # True esetén mellékel egy fűtési szezonalitás-jelzőt az input-hoz (input pontonként 0 vagy 5, átlagérték)
        self.weekdays=weekdays          # True esetén mellékel egy weekday jelzőt az input-hoz (input zárópontra  0,..,6)
        self.weekend=weekend          # True esetén mellékel egy hétvég jelzőt az input-hoz (input zárópontra, 0 vagy 5)
        self.season=season          # True esetén mellékel egy évszak-jelzőt az input-hoz (input zárópontra, 0,1,2,3)
        self.w_exogs=w_exogs
        self.w_level=w_level        # mennyire érzékeny a görberészlet szintjére (pl. a nulla-közeli szakaszok érdemben eltérően viselkednek)
        self.epszilon=epszilon

        self.tuned=tuned        # egyelőre nincs hangoló algoritmus



        
    def pred(self,ser,pred_len,exogs):
        
        model = Sequential()
        for i,layer_size in enumerate(self.layers):
            if i==len(self.layers)-1:  # utolsó LSTM layer
                model.add(LSTM(50, activation='relu'))
            elif i==0:        # első LSTM layer (de nem utolsó)
                model.add(LSTM(layer_size, activation='relu', return_sequences=True, input_shape=(self.points_in, 1)))
            else:      # közbenső LSTM layer-ek
                model.add(LSTM(layer_size, activation='relu', return_sequences=True))
        model.add(Dense(pred_len))
        model.compile(optimizer='adam', loss='mse')
        

       
        # mlp = LSTMRegressor(layers=self.layers,max_iter=2000,
        #     learning_rate_init=self.learning_rate_init,learning_rate='constant',
        #     early_stopping=False,verbose=False,alpha=self.alpha,n_iter_no_change=10,tol=self.tol,activation='relu')

        ser_pred,score = fn_ts_pred_bylastpoints(model,ser,self.points_in,pred_len,1,0,'keras',
                                                self.heat_season,self.weekdays,self.weekend,self.season,self.exogs,
                                                self.w_exogs,self.w_level,self.epszilon)


        self.ser_pred = ser_pred
        self.score=score     # train-score


        return self.ser_pred


    def tune(self,ser,pred_len,test_points,exogs):
        # if 'Date' in str(type(ser.index)): ser=ser.reset_index(drop=True)      # nem kell datetime index, egyszerű sorszámozás

        if pred_len<3: points_in=[3,5]
        else: points_in=np.array([1,2])*pred_len


        params_grid={
            'points_in': points_in,
            'layers':[(20,),(20,10)],
            'learning_rate_init':[1e-3,1e-4],
            'heat_season': [True],
            'weekdays': [True],
            'weekend': [False],
            'season': [False],
            'tol': [1e-5],
            }

        if test_points is None: test_points='last5_5_sun'

        def fn_loss(**modelparams):
            test_pointsL = list(array(test_points) - np.random.randint(0,6))
            return fn_ts_loss(ts_MLP(**modelparams),ser,test_pointsL,pred_len,exogs)
            # - új példányt hoz létre a megadott paraméterekkel (annyiszor, ahány paraméter kombináció van)

        return  fn_tune_bygrid(fn_loss,params_grid)




    def get_params(self):   # releváns modell-paraméterek (megjelenítésekben használható)
        return get_attrs(self,'points_in,G,layers,learning_rate_init')





def fn_ts_pred_bylastpoints(model,ser,train_points,pred_points,T=1,G=0,model_type='sklearn',   # predikció adott számú előzménypont alapján
        heat_season=1,weekdays=0,weekend=1,season=4,exogs=None,w_exogs=1,w_level=2,epszilon=0.1):   
    '''
    Tanítási minták:  train_points + pred_points szélességű időablak a teljes lefedhető tartományra (naponként eltolva)

    model:  pl. MLPRegressor,  KNeighborsRegressor   (legyen fit, pred és score metódusa)
    model_type:  'sklearn', 'keras'
    ser: autosorszámozott indexet igényel
    train_point:  hány előzmény-pontot vegyen figyelembe a tanításkor (általában nagybb a pred_point-nál)
    pred_proint:  hány utózmány-pontra kell a predikció 
    T: ha nagyobb mint 1, akkor a train és a pred pontok a megadott méretű intervallummal követik egymást
    G: a ser görbe előzetes gauss-simításának ablakszélessége (4szigma)   0 esetén nincs simitás
    heat_season: >0 esetén mellékel egy fűtési-szezon jelzőt az input-hoz (input pontonként 0 vagy 1, átlagérték)
    
    exogs:  dict(name to (f_days,ser))
        - általában jövőbeli pontokat tartalmaz (az exog series hosszabb az alapértelmezett series-nél)
    '''

    # autosorszámozott indexet igényel
    ser,datefloat0,date_step = ser_reset_datetimeindex(ser)

    if weekend>0:
        holidays_=holidays.country_holidays('HU',years=Range(2010,date.today().year))



    if G>0: serG=FvGaussTAvg(ser,G,positive=True)
    else: serG=ser

    n_in = train_points
    n_out = pred_points

    if heat_season:
        season_labels=fn_heatingseason(ser)


    def f_t_in(iStart):
        return [iStart + (j+1)*T for j in range(n_in)]      #  T, 2T, ..., n_in*T
    def f_t_out(iStart):
        return [iStart + (n_in+j+1)*T for j in range(n_out)]    #  (n_in + 1)*T, ..., (n_in + n_out)*T 

   
    mainap=len(ser)-1

    y_ser=ser.values
    y_serG=array(serG.values)
    if G>0: 
        y_serG=np.roll(y_serG,-1)
        y_serG[-1]=y_serG[-2]

    ymax=serG.max()
    

    # epszilon = ser.max() / 1000     # a skálázás miatt kell egy kis mértékű eltolás (zéró osztó elkerülése)
    epszilon = serG.max() *epszilon     # minél nagyobb, annál inkább additív jellegű a görbe (minél kisebb, annál inkább multiplikatív karakterű)
            # - az additív/multiplikatív jelleg a csúcsossággal függ össze

    def f_x(i_minta):
        x = np.array([y_serG[t] for t in f_t_in(i_minta)]) + epszilon
        scale=x[-1]     # az epszilon miatt mindenképpen >0
        if scale==0: scale=1        # kivéve:  ymax=0  (minden érték 0)

        x = x /scale

        x=x[:-1]        # az utolsó elem kötelezően 1 lenne

        # x=[np.array(x)]       # vektorként adom át   NEM LEHETSÉGES  (hibaüzenet a fit-nél)

        t_last_x=i_minta + n_in*T
        t_first_y=i_minta + (n_in+1)*T
        x_last=y_serG[t_last_x]

        # szintjelző  (nem csak a görbe alakja lehet fontos;  pl. nulla körül egészen más viselkedés valószínűsíthető)
        if w_level>0:
            if x_last>ymax/10: level=w_level        # 10% felett
            else: level=0           # 10% alatti
            x = np.append(x,level)


        # szezonalitás jelző  (1 ha fűtési szezon)
        if heat_season>0:
            season_label=np.array([season_labels[t] for t in f_t_in(i_minta)]).mean()
            # season_label=season_labels[t_first_y]
            x = np.append(x,season_label*heat_season)         # nem túlzottan fontos, hogy az egyes napok melyik sorszámnak felelnek meg
            

        # Hétnapja jelző
        if weekdays>0:
            x = np.append(x,floatdate(t_first_y,datefloat0,date_step).weekday()/6*weekdays)      

        # Hétvége jelző, és karácsony jelző
        if weekend>0:
            # x2=np.array([y_serG[t] for t in f_t_in(i_minta)])

            date_first_y=floatdate(t_first_y,datefloat0,date_step)
            weekend_ = date_first_y.weekday() in [5,6]    # hétfő = 0
            if not weekend_:
                weekend_=date_first_y in holidays_                
            x = np.append(x,int(weekend_)*weekend)

            # ünnepnap = date_first_y in holidays_
            # x = np.append(x,int(ünnepnap)*weekend)

            # month_day = date_first_y.strftime('%m-%d')
            # karácsony=month_day in ['12-24','12-25','12-26']
            # x = np.append(x,int(karácsony)*weekend)
                




        # Évszak jelző
        if season>0:
            m=floatdate(datefloat0 + t_first_y*date_step).month
            if m in [12,1,2]: label=1
            elif m in [3,4,5]: label=2
            elif m in [6,7,8]: label=3
            elif m in [9,10,11]: label=4
            x = np.append(x,label/4*season)         

        # Exogén idősorok előrejelzett adatai
        if w_exogs and exogs:
            for name,adatok in exogs.items():     #  name to (forecast_days,ser)  
                forecast_days,ser_exog = adatok

                # Két nap beírása (utolsó input nap, első forecast nap)
                # date_last=floatdate(datefloat0 + t_last_x*date_step)
                # x2=[ser_exog[fn_dateadd(date_last,day)] for day in range(forecast_days+1)]
                # x=np.array(list(x) + list(x2))
                
                # maxmin=ser_exog.max()-ser_exog.min()
                avg_diff = 2        # becsült érték (számolni kellene). A hőmérséklet abszolút értéke átlagosan ennyivel változik
                date_last=floatdate(datefloat0 + t_last_x*date_step)
                last=ser_exog[date_last]
                for day in range(1,forecast_days+1):
                    x=np.append(x,(ser_exog[fn_dateadd(date_last,day)]-last)/avg_diff*w_exogs)  



        return x,scale

    # minta_db=len(ser_histG) - ((n_in+n_out)*T)                  
    minta_db = mainap - ((n_in+n_out)*T)          # egy napos eltolásokkal a teljes felhasználható tartományra 
    X_train=[0]*(minta_db)
    Y_train=[0]*(minta_db)
    for i_minta in range(minta_db):  #  (n_in-1)*T+T1,len_ser-((n_in-1)*T+T2)):
        X_train[i_minta],scale = f_x(i_minta)

        Y_train[i_minta]=np.array([y_serG[t] for t in f_t_out(i_minta)]) + epszilon
        Y_train[i_minta]=Y_train[i_minta]/scale

        if n_out==1: Y_train[i_minta]=Y_train[i_minta][0]
    # - egy-egy train-szakasz 6T szélességű

    
    # Train
    if model_type=='sklearn':
        model.fit(X_train,Y_train)
        score=model.score(X_train,Y_train)
    elif model_type=='keras':
        X_train=np.array(X_train)
        X_train=X_train.reshape(X_train.shape[0],train_points,1)
        Y_train=np.array(Y_train)
        model.fit(X_train,Y_train,epochs=1000,validation_split=0.9,shuffle=False,
                  callbacks=[EarlyStopping(monitor='loss',mode='min',min_delta=0.001,patience=10,restore_best_weights=True)],
                  verbose=0)
        # score=model.score(X_train,Y_train)
        score=np.nan



    # Predikció:  
    nStart = mainap - n_in*T         # a mai nap az utolsó nap az input-ban
    # if G==0: nStart = mainap - n_in*T         # a mai nap az utolsó nap az input-ban
    # else: nStart=minta_db-1 + n_out*T            # a tanítási tartomány utáni n_out pontra

    x_pred,scale=f_x(nStart)
    # x_pred = np.array([y_serG[t] for t in f_t_in(nStart)]) + epszilon
    # scale=x_pred[-1]
    # x_pred = x_pred/scale
    
    # x_pred=np.append(x_pred,nStart%7)         # nem túlzottan fontos, hogy az egyes napok melyik sorszámnak felelnek meg

    if model_type=='sklearn':
        Y_pred=model.predict([x_pred])
    elif model_type=='keras':
        x_pred=x_pred.reshape(x_pred.shape[0],train_points,1)
        Y_pred=model.predict([x_pred])

    
    # KNN RÉSZLETEZÉS
    # distances,indexes = model.kneighbors([x_pred], 8, True)
    # arr_=[]
    # for i,index_ in enumerate(indexes[0]):
    #     arr_.append(str(fn_dateadd('2017.04.01',int(index_))) + ':' + strnum(distances[0,i],'4g'))
    # print(', '.join(arr_))


    
    y_pred=Y_pred[0]*scale  - epszilon

    if type(y_pred)!=np.ndarray: y_pred=np.array([y_pred])
    
    y_pred[y_pred<0] = 0       # nem zárható ki negatív érték. Lecserélés epszilon/2-re
    # y_pred[y_pred<0] = epszilon/2       # nem zárható ki negatív érték. Lecserélés epszilon/2-re

    ser_pred = pd.Series(y_pred,f_t_out(nStart))

    # Dátumindex visszaállítása
    ser_pred = ser_restore_datetimeindex(ser_pred,datefloat0,date_step)


    return ser_pred, score


def fn_heatingseason(ser):
    '''
    Általánosítható, bár van néhány speciális jellemzője
    0 közeli szakaszok kijelölése   (a max érték tizede alatti pontok átlagának kétszerese, legalább 28 pont)
    return:  0 - nyári időszak   1 - fűtési időszak   (array)

    időtartam:  a FvGaussAvg miatt tizedmásodperc körüli lehet    
    '''
    # Nullás időszakok keresése (legalább 7 nap hosszú)
    # count=0
    # i_first=0
    # for i,value in enumerate(ser.values):
    #     if value > 0:
    #         if count>=28:
    #             season[i_first:i+1]=[0]*(i-i_first+1)
    #         count=0
    #     elif value==0:
    #         if count==0: i_first=i
    #         count+=1
    # nyári időszakok keresése  (alacsony fogyasztás, legalább 28 nap)
    ser_max=ser.quantile(0.95)
    if ser_max==0: ser_max=ser.max()
    if ser_max==0: return [0]*len(ser)         # nincs fűtési szezon

    # 0-nál nagyobb, de 10% alatti pontok átlagának kétszerese lesz a küszöb
    serL=ser.loc[ser.values>0]
    if len(serL)==0: return [0]*len(ser)
    serL = serL.loc[serL<ser_max/10]
    if len(serL)==0: return [0]*len(ser)

    limit=serL.mean()*2
    season=[1]*len(ser)         # default: fűtési szezon (=1)

    if limit:       # előfordulhat, hogy egyáltalán nincs a max 10%-ánál kisebb pont
        # 14 napos gauss mozgóátlag            
        serG=FvGaussAvg(ser,14)
        valuesG=serG.values
        count=0         # limit alatti napok számlálója
        i_first=0       # az első limit alatti nap
        for i,value in enumerate(valuesG):
            if value>limit or i==len(valuesG)-1:    # ha túllépi a limitet, akkor nullázás
                if count>=28:   # legalább 28 limit alatti nap
                    if i==len(valuesG)-1 and value<=limit: i_upper=i+1
                    else: i_upper=i
                    season[i_first:i_upper]=[0]*(i_upper-i_first)  # visszamenőleg 0-kat írok be
                count=0
            else:
                if count==0: i_first=i      # az első limit alatti nap (28 ilyen nap kell még)
                count+=1
    
    return season


def fn_lasttestpoints_in_season(ser,pred_len,test_points_arg='last10_5_sun'):
    paramsL = cutleft(test_points_arg,'last').strip()    # 10_5_sun
    b_sunday=False
    if endwith(paramsL,'_sun'):
        b_sunday=True
        paramsL=cutright(paramsL,'_sun')    # 10_5
    count,step = splitfirst(paramsL,'_','1')
    
    def f_firstpoint(ilast):
        first_point=(ilast-3) - pred_len - int(count)*int(step)
        # - azért (i_last-3), mert a 14 napos Gauss mozgóátlag a végén 3-4 nullás értéket még bevehet szezonokba
        if b_sunday:
            weekday_=ser.index[first_point].weekday()
            if weekday_!=6: first_point = first_point-weekday_-1
        if first_point<20: first_point=20        # legyen valamennyi hely a points_in-nek is
        return first_point 

    # Keresse meg az utolsó olyan fűtési szezont, amibe beleférnek a teszt-pontok
    #   Ha nincs ilyen, akkor adja vissza a leghosszabbat
    serL=ser.copy()
    maxlen=None
    ilast_maxlen=None
    while True:
        maxstep=len(serL) - int(0.5*len(ser))       # legfeljebb a ser 50%-ig mehet vissza a season keresésben
        if maxstep<=0: break
        ser_heating = pd.Series(array(fn_heatingseason(serL)),serL.index)
        ilast=serlastvalue(ser_heating,maxstep=maxstep,out_='ilast')
        # Ha nincs további fűtési szezon, akkor leállás
        if not ilast:  break
        else:
            ifirst=None
            for i in range(ilast,int(0.5*len(ser)),-1):
                value=ser_heating.iloc[i]
                if pd.isna(value) or value==0:
                    ifirst=i
                    break
            # Ha belefér, akkor OK
            if not ifirst or f_firstpoint(ilast)>=ifirst: break  # ilast be van állítva
            elif maxlen is None or ilast-ifirst+1 > maxlen: 
                maxlen=ilast-ifirst+1
                ilast_maxlen=ilast
            
            # Keressen tovább az eddig megtalált fűtési szezonok előtt
            serL=serL.iloc[:ifirst-1]

    # Ha nem volt olyan fűtési szezon, amibe belefért, akkor válassza a leghosszabbat
    if not ilast: ilast=ilast_maxlen
    
    # Ha így sem volt találat, akkor keresse az utolsó nem nulla értéket (jobb híján)
    if not ilast:
        ilast=serlastvalue(ser,maxstep=0.8*len(ser),out_='ilast')  # akár 80%-kal is visszamehet
    
    first_point=f_firstpoint(ilast)

    return Range(first_point,count=int(count),add=int(step))





