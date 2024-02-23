from datetime import datetime

username ='zsolt'       # 'józsef', 'péter'
tigáz_kürt_dir = r"C:\Users\Zsolt\OneDrive\Python\Projects\sklearn\Adatsorok\Tigáz Kürt"



aAnnotG=[]
aScatterG=[]

linecolorsG={'Hungary':{'color':'orange','linewidth':1.5, 'alpha':1},
             'World':{'color':'0.7','linewidth':4,'alpha':0.3},
             'Europe':{'color':'blue','linewidth':1.5,'alpha':1},
             'European Union':{'color':'blue','linewidth':1.5,'alpha':1},
             'Regisztrációk':{'color':'orange','linewidth':1.5, 'alpha':1},
             'Átlag':{'color':'orange','linewidth':1.5, 'alpha':1}}
annotcolorG={'Hungary':'red',
             'European Union':'blue'}

serplotlastG=None        # a FvPlot által utoljára rajzolt görbe 
tblinfoplotsG=None       # a tblinfo 'plot' által kirajzolt series objektumok
normfaktorlastG=None     # a FvPlot által utoljára alkalmazott normfaktor

d_honapok={'január':1,'február':2,'március':3,'április':4,'május':5,'június':6}


