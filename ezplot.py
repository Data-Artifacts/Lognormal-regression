# 2023 09   Proxy az ezcharts modul plot-függvényeire, github publikáció
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from numpy import array
import pandas as pd

import math

from sklearn.datasets import load_iris


from ezhelper import *
from ezcharts import *

matplot_init()




def tbl_plot(tbl_or_ser,cols='',groupby='',orderby='',query='',
             G=0.1, resample=True, normalize=None, trend=None, trend_extend=None,
             plotstyle='lines', points=None,
             annot='localmax2 last', annotcaption='default', annotlength=None, annotcount=30, annot_baseline=None, legend=None,
             suptitle=None,title=None,xtitle=None,ytitle=None,
             ynumformat=None,xnumformat=None,
             xmin=None,xmax=None,ymin=None,ymax=None,
             width=None,height=None,left=None,right=None,top=None,bottom=None,
             y_bands=None,x_bands=None,xy_circles=None,xy_texts=None,
             commenttopright=None,commentbottomright=None,commenttopright_in=None,commenttopleft_in=None,commentfontsize_add=None,
             to_file=None,
             titles=None,xlabel=None,ytitles=None,ylabel=None,ylabels=None):           # alternatív argumentumnevek
    '''
    A táblázat egy vagy több oszlopához és/vagy tételcsoportjához tartozó vonaldiagram(ok) rajzolása, közös diagramba vagy aldiagramokba
    Szolgáltatások:
        - annotációk:  dinamikusan pozícionált és nyilakkal ellátott vonal-feliratok (legend helyett), szélsőérték-helyek annotálása 
        - a görbék Gauss simítással jelennek meg (hangolható a simítás mértéke)
        - kérhető az eredeti pontok rajzolása is (dinamikus pontméretezéssel vagy KDE pontsűrűséggel), 
        - trend-számítás a széleken, 
        - az x tengely dátumtengely is lehet
        - egyenetlen x-sorozat esetén resample
        - képernyős megjelenítés helyett fájlba írás is kérhető (pdf fájl)

    tbl: pandas DataFrame vagy Series,   Y: list,  (X,Y): tuple of lists
        - Series,Y,(X,Y):  a cols argumentumban kötelező megadni egyetlen oszlopnevet


    cols: oszlopnevek felsorolása (rövidítések is megengedettek).
        Oszloponként egy-egy görbe, alapesetben ugyanabban a diagramban
            Több aldiagram is kérhető, "//" határolással.   Példa:  "width,height//area,mass"
        Szintaxis:  kisbetű-nagybetű eltérés érdektelen, vesszős felsorolás
            Gyorsírással: az elejére írt "**" jelzi, hogy "bárhol a szövegben" jellegű keresési feltételről van szó (több oszlop is lehet az eredmény)
            A végén NOT feltételrész adható meg ("kivéve" értelmezéssel)
        - "case, total_case, deaths, total_deaths"     - mezőnevek felsorolása (kibetű-nagybetű eltérés érdektelen, de egyébként pontos találat kell)
        - ["Case","Total_case","Deaths","Total_deaths"]- ha vesszők vagy szóközök is előfordulnak az oszlopnevekben, akkor listát kell megadni
        - "**case"                                     - minden olyan mező, amiben szerepel a "case" szövegrész (>>"case","total_case")
        - "**case total,death"                        - "case" és "total" is van benne VAGY "death" van benne (>>"total_case","deaths","total_deathes")
        - "**total NOT case":                          - a "NOT" után "kivéve" értelmezésű szűrés adható meg (>>"total_deaths")
        - ""                                           - összes számoszlop
        - "type=float,int"                             - összes számoszlop ("float","int",text","date" adható meg)
    groupby:   egy csoportosító mező által meghatározott tételcsoportok görbéit rajzolja (pivot jellegű)
        Példa:  cols="case", groupby="country:**hun,cze"  -  a két görbe: Hungary és Czechia, esetszámok naponta (a szintaxist lásd a cols-nál)
                    - a cols-ban ilyenkor is lehet több mező (a görbék száma:  cols_db * group_db)
                cols="case", groupby="country"   - görbe az összes előforduló országnévre (10-20 ország felett nehezen áttekinthető)
                    - a query argumentumben szűrés adható meg az országszintű adatokra (pl. continent="Europe")
                cols="case", groupby="country:**hun,cze//germany,denmark"     - több aldiagram
    query:  alapszűrő (nem kötelező)  
        'Dátum>="2021.01.01"'                           # itt case-sensitive mezőnevek
        'Continent=="Europe"'
        'Year>1900 and Year<=2020'
        'Continent.str.startswith("Eu",case=False)'
    orderby:  oszlopnév vagy oszlopnevek felsorolása (lehet gyorsírásos is, lásd cols, NOT nélkül)). 
            A legvégére "desc" írható (sajnos több oszlopos esetén is csak globálisan adható meg)

        

    G:  A görbék Gauss-simításának erőssége (Gauss mozgóátlag)   [0,1] közötti érték vagy értékfelsorolás
        0 esetén nincs Gauss simítás (az eredeti pontokat összekötő egyenesek jelennek meg)
        Minél nagyobb, annál inkább elmosódnak a részletek és egyre jobban kirajzolódik a nagy időskálájú trend.   Példa: 0.05 - enyhe simítás,  0.5 - erős simítás (default: 0.1)
        Több G érték is megadható stringben. Példa: "0.1,0.09,0.08"  (egyre halványabb vonalak, eltérő G értékekkel).
        Egynél nagyobb egész szám is megadható. Ilyenkor a megadott számú mérési pont határozza meg a Gauss-szélességet
            (pl. egy napi mintavételezésű idősor esetén 7 napos Gauss simítás (G=14 esetén kiszűrődnek a heti kilengések)
        Szakmai háttér:
        - a "G" a mozgóátlagolás Gauss-eloszlásának talpponti vagy effektív szélessége a diagram teljes szélességhez viszonyítva (4*szigma).
        - egy "végtelen" keskeny tüskéből a simítást követően G szélességű Gauss-görbe lesz
        - a zajszűrés erősségét is a G határozza meg: a zajszűrési periódus a G fele. Az ilyen vagy ennél kisebb periódusú, kétirányú kilengések lényegében eltűnnek (kb századrészre csökkennek). 
        - a széleken lévő megbízhatatlansági tartomány szintén a G fele (a széleken féloldalas az átlagolás)
        - az átlagolásban ténylegesen figyelembe vett pontok száma a G kétszerese (8*szigma)
    resample:  True esetén a hiányzó közbenső pontok pótlása lineáris interpolációval 
        Default G>0 esetén:
            Egyenletes x-irányú eloszlás, az eredetihez képest négyszeres pontsűrűséggel.
            Akkor javasolt bekapcsolni, ha a pontok x irányban nem egyenletesek illetve hiányosak. A Gauss simítás akkor ad jó eredményt, ha az x-irányú eloszlás közelítőleg egyenletes.
        Default G=0 esetén:   a hiányzó értékek (y=NaN) pótlása interpolációval. A hiányzó értékek ne szakítsák meg a görbét. 
    trend:  True, "right", "linear", "right erlang"      Lokális trend érvényesítése a széleken lévő G/2 tartományra.
        True:  lineáris, kétoldali trend-érvényesítés
        Alkalmazható kulcsszavak (a sorrend érdektelen, köztük szóköz):  
            "left" "right"  - ha csak a bal vagy a jobb szélen erősítendő a lokális trend 
            "linear"  "erlang"  -  az illesztés jellege a széleken lévő G illetve 1.5*G tartományra
                (két- illetve hárompontos illesztés, G/2 szélességű tartományok átlagos értékeire)
        Szakmai háttér:  idősorok esetén a Gauss-simítás a széleken lévő G/2 tartományban nem ad végleges értéket 
           (a pár nappal későbbi információk alapján még változhat a végső görbe). 
           A Gauss-simítás alapesetben a vízszintes trendhez való lassú visszatérés felé hat (egy emelkedő egyenes 
           szélein enyhe elhajlást ad a vízszintes felé). Ha a legfrissebb trend hangsúlyozása a cél, akkor érdemes 
           lineáris trendet hozzáadni a jobb oldali G szélességű szakasz alapján. A széleken érvényes lokális trend
           érvényesítése csökkentheti a mozgóátlag-algoritmus alapkarakteréből fakadó "hajlam" érvényesülését.
           "Erlang" trend hozzáadása akkor javasolt, ha sok-kompenensű makro-folyamatról van szó
           (pl. járványgörbék). Az ilyen "statisztikai karakterű" görbék Erlang-görbével jellemezhető felfutások 
           összegeként írhatók le, ezért G-hez mérhető idősávban az Erlang trend viszonylag megbízható predikciót
           adhat.
    trend_extension:  a görbe továbbfolytatása a megadott szélességű tartományban (prediktív jellegű)
        A szélességet a G-hez viszonyított arányszámként kell megadni. Példa 0.5: 0.5*G szélességű kiterjesztés
        Előfeltétel:  a "trend" argumentum be van kapcsolva  (ott adható meg az illesztés típusa is)
    normalize:  False:   Nincs normálás  
        True:    1-re normálás (vonalaknként külön-külön)  A korrelációk és trendek skálafüggetlen felderítéséra alkalmas.
        float:   a vonalak max-értéke mire skálázódjon
        "sharey":  az aldiagramok y-tengelyeinek legyenek ugyanazok a határai. A határokat a legkiugróbb min-max értéket
            mutató vonal határozza meg (skálázás nincs; akkor hasznos, ha mindegyik vonal hasonló karakterű)
    
    plotstyle:
        'firstline':    az első görbe vonaldiagram (színezéssel), a többi area (szürke vonal és háttérszínnel)
        'firstarea':    az első görbe area, a többi színezett vonal
        'area':         mindegyik görbe area  (szürke vonal és háttérszín)
        'line':         mindegyik görbe vonaldiagram (automatikus színválasztással)
    annot:  annotáció-típusok felsorolása   pl. "max last"   (a sorrend érdektelen, szóköz határolás)
        Az annotálások egy feliratból és egy hajlított nyílból állnak. A nyíl a görbe egy pontjára mutat.
          A program automatikusan úgy tolja el a feliratokat, hogy ne legyen átfedés közöttük.
          A nyilak tájolása az annotáció típusától függően eltérő lehet.
        annot='' esetén nincsenek annotációk (ha szükséges, legend kérhető helyette)
        Lehetséges annotáció típusok:
        'last'  'first':            a görbe jobb illetve bal szélső pontjára mutat
        'left' 'middle' 'right'':   x-irányban 10%, 50%, 90% pozícióknál
        'max' 'min':                a vonal szélsőértékeinél (ha a szélsőérték hely nem esik egybe valamelyik fentivel) 
        'localmax' 'localmin':      lokális minimimum ill. maximumértékeknél (ugyanannak a görbének több ilyen helye is lehet)
                                    Közvetlenül a típusfelirat után egy szám is írható: max ennyi annotálás jelenjen meg
                                    Erősen ingadozó görbék esetén az x-szélesség 5%-ánál közelebbi szélsőértékek közül csak egyhez jelenik meg annotálás.
        'maxabs'                    0 feletti és 0 alatti max- ill. minhelyek annotálása (itt is megadható max darabszám)
        'gaussabs', 'gaussmax'      A görbék trendvonalához képest mért legnagyobb kilengések annotálása.
                                    Trendvonal: a megjenő vonal Gauss-simításának kétszeresével simított görbe.
                                    A "gaussabs" kétirányú, a "gaussmax" csak a felső kilengéseket annotálja
        konkrét x-érték             konkrét x érték megadása:  annotáció ehhez a ponthoz (több is megadható szóközzel elválasztva)
                                    Ha az x float, akkor számértéket kell megadni (stringben), ha dátum, akkor yyyy-MM-dd formátum
                                    Nem elvárás, hogy a megadott x-értékek közvetlenül szerepeljenek a pontsorozatokban (ha nem szerepel, akkor interpoláció)
        Alapértelmezett annotáció:
            - "localmax2 last"
        Dictionary is megadható, görbénként eltérő annot paraméterrel. 
            példa:  {"Europe":"right","other":"localmax"}      (a "Europe" a görbe megnevezése, a groupby vagy a cols argumentum alapján)
    annotcaption:    annotációk felirat-sablonja   (default: "label"  vagy "y", az annotáció típusától függően)
        - alapesetben a last, first, left, middle, right, localmax, localmin, és "konkrét x" annotálásoknál a 
            vonalhoz tartozó label jelenik meg. A label alapesetben a colname vagy a groupname (ld. cols, groupby argumentumok).
            A colname-ek helyett egyedi label-ek is megadhatók a tblstat fájlban (tblstat_path argumentum, ami általában egy xlsx fájlra mutat)
            A maxabs, gaussabs, gaussmax annotálásoknál alapesetben az y_orig érték jelenik meg 
        - egyetlen sablon megadása esetén az összes típusra ez a sablon érvényesül
            Shorthands:  "label"   "y"    "y%"   "x"
            VAGY feliratsablon a következő helyettesítőjelekkel:   {label}  {x}  {y} {y%}  {y_orig} {y_orig%}   (az y_orig a skálázás előtti y érték)
            Példa:  "max érték: {y}"
        - megadható egy dictionary, annot-típusonként eltérő feliratokkal
            Példa:  {"last":"label", "max" : "y",  "localmax":"Lokális maximum: {y}"}
        - megadható egy függvény ami visszaadja a konkrét felirat:   
            - egyedi formázások, vagy "kod to caption" konverziók csak így valósíthatók meg
            - a függvény prototípusa:  def f(rec):       return caption,   belül: rec.type, rec.label,rec.x,rec.y,rec.y_orig
    annotcount:
        - legfeljebb hány annotálás jelenhet meg a diagramon illetve az egyes aldiagramokon (minden annottípust figyelembevéve)
    annotlength:
        Az annotálásokban megjelenő "label" adat legfeljebb hány karakteres lehet (None esetén nincs levágás). 
        Ha hosszabb ennél, akkor ... a label végén.   
        Az annotáció teljes hossza ennél hosszabb is lehet, mert a label-en kívül más is lehet benne
    annot_baseline:  a localmin, localmax, maxabs annotációk közül csak a baseline alatti illetve feletti annotációk maradnak meg (default:0)
            - megadható egyetlen szám (konstans függvény) vagy egy ser.  Default: 0 (y=0 konstans függvény, vízszintes egyenes)
            - közös az összes megjelenített görbére (ellentétben a gaussabs, gaussmax annotácóval, ahol a baseline görbénként eltérő)
            - példa:  országok járványgörbéi egy közös diagramban (normalizálva). annot_baseline = görbék átlaga  beállítással elérhető,
                     hogy csak az átlagörbéhez képest érdemben kiugró localmax, localmin értékekhez jelenjen meg annotáció
            
    points:  bool   True esetén az eredeti xy pontok megjelenítése a diagramon (scatter plot)
        - nagy tömegű pont esetén csökkenő pontméret és növekvő átlátszóság
        - pontsorozatonként külön színek (ha van vonalszínezés, akkor azzal megegyezően)
    kde: bool   True esetén KDE-plot, azaz a pontsűrűség rajzolása (szürke-árnyalatos)
        - ha a diagramhoz több pontsorozat tartozik, akkor közös mindegyikre
        
    suptitle:   a diagram-ablak közös címe
    titles:   a diagram(ok) felett megjelenő címfelirat  
        - ha nincs megadva, akkor egyetlen görbe esetén a colname vagy groupname, több görbe esetén a cols-ban vagy a groupby-ban megadott kifejezés.
        - egyetlen szöveg is megadható  (aldiagramok esetén mindegyik aldiagramnak ugyanaz lesz a címsora)
        - aldiagramok esetén általában listaként kell megadni, összehangolva az aldiagramok számával 
    xtitle:    az x tengely felirata  
    ytitle:    az y tengely felirata (több aldiagram esetén list is megadható, length = aldiagramok száma)
        - ha nincs megadva, akkor csoportosító érték (groupby) vagy az oszlop megnevezése (cols) 

    ynumformat: y tengely jelölőinek számformátuma. Ha több aldiagram van, akkor listaként is megadható (pl. [None,"%"])
        None:   automatikus formázás
        '':     ne jelenjenek meg jelölők
        '%': [0,1] bázistartományú mezőkre, a százalék 1 tizedesjegyig   pl. 0.1234 > "12.3%"
        '%%': [0,100] bázistartományú mezőkre, a százalék 1 tizedesjegyig  pl. 12.34 > "12.3%"
        '0%':  az elején a decimális digitek száma adható meg
        'date':     dátumok jelenjenek meg a tengelyen (ha a tbl kulcsmezője date típusú, akkor ez az alapértelmezett)  
        'month':    hónapnevek jelenjenek meg (ritkán szükséges; akkor javasolt, ha a "date" nem megfelelő)
        'year':     csak az évhatárok jelenjenek meg
    xnumformat: x tengely jelölőinek számformátuma (ritkán szükséges; dátumtengely esetén érdektelen)
        
    x1,x2,y1,y2:   koordináta határok   (x esetén dátumstring is megadható;   féloldalasan is megadható;  xmin,xmax,ymin,ymax is jó)
        - több aldiagram esetén mindegyik aldiagramra érvényesül
    
    commenttopright:       a képtér jobb felső sarkában megjelenő szöveg (szürke, kis betűméret). "\n" jelek lehetnek benne, általában max 3 sor
    commentbottomright:    a képtér jobb alsó sarkában megjelenő szöveg (szürke, kis betűméret). "\n" jelek lehetnek benne, általában max 3 sor
    commenttopright_in:    a diagramterület jobb felső sarkában megjelenő szöveg (több soros is lehet) Alárendelt diagramok esetén lista is megadható
    commenttopleft_in:     a diagramterület bal felső sarkában megjelenő szöveg (több soros is lehet)
    commentfontsize_add:   alapesetben a comment-ek 7-es méretűek        

    y_bands, x_bands:  Függőleges / vízszintes sávok rajzolása  (aldiagramok esetén az összesre)
        list of dict(koord,koord2,irány='vert',color='green',alpha=0.07,caption='',align='left'/'bottom',fontsize=7)
        Példa:   x_bands = [{'koord':120,'koord2':130}]       (fontos a szögletes zárójel)
        - koord,koord2: mettől meddig (vert esetén x-értékek).
        - koord lehet 'start', 'begin',   koord2 lehet 'end'.   Dátumtengely esetén stringként is megadható pl "2022-11-09"
        align:  a felirat pozícionálása
                'top' 'bottom':  vert esetén függőleges tájolással a bal szélen, felül vagy középen
                'left' 'right':  horz esetén vízszintes tájolással a felső szélen, balra vagy jobbra igazítva
                'topcenter':     vert esetén vízszintes tájolással, felül középen (széles sávok esetén alkalmazható)
    xy_circles:  Annotációs körök rajzolása (a diagram-görbéktől független;  aldiagramok esetén mindegyikre)        
        list of dict(x,y,size,color,caption)
    xy_texts:    Magyarázó feliratok     (diagramvonalaktól független; aldiagramok esetén mindegyikre),           
        list of dict(x,y,caption,ha='left',va='top',fontsize=fontsize0-1,alpha=0.5,color=default,transform='gcf'/'gca')

    '''

    if type(tbl_or_ser)==pd.DataFrame: tbl=tbl_or_ser
    elif type(tbl_or_ser)==pd.Series: tbl=pd.DataFrame(tbl_or_ser,columns=[cols])
    elif type(tbl_or_ser)==list: tbl=pd.DataFrame(pd.Series(tbl_or_ser),columns=[cols])
    elif type(tbl_or_ser)==tuple:
        X,Y = tbl_or_ser
        tbl=pd.DataFrame(pd.Series(Y,X),columns=[cols])
    else:
        print('ERROR  tbl_plot  Érvénytelen typusú tlb_or_ser argumentum (DataFrame, Series, Y, (X,Y) lehet)')
        return

    # Alternatív argumentumnevek
    if title and not titles: titles=title
    if xlabel and not xtitle: xtitle=xlabel
    if ytitle and not ytitles: ytitles=ytitle
    if ylabel and not ytitles: ytitles=ylabel
    if ylabels and not ytitles: ytitles=ylabels

    plotparams={}

    if annotcaption=='default':
        annotcaption={'maxabs':'y', 'gaussabs':'y', 'guassmax':'y', 'other':'label'}

    dset(plotparams,G=G,resample=resample,trend=trend,trend_extend=trend_extend,normalize=normalize,
         annotate=annot,annotcaption=annotcaption,annotlength=annotlength,annot_count=annotcount,annot_baseline=annot_baseline,legend=legend,
         points=points,
         suptitle=suptitle,title=titles,xtitle=xtitle,ytitles=ytitles,
         ynumformat=ynumformat,xnumformat=xnumformat,
         x1=xmin,x2=xmax,y1=ymin,y2=ymax,
         width=width,height=height,left=left,right=right,top=top,bottom=bottom,
         y_bands=y_bands,x_bands=x_bands,xy_circles=xy_circles,xy_texts=xy_texts,
         to_file=to_file,
         commenttopright=commenttopright,commentbottomright=commentbottomright,
         commenttopright_in=commenttopright_in,commenttopleft_in=commenttopleft_in,
         commentfontsize_add=commentfontsize_add
        )

    tblinfo(tbl,'plot',cols=cols,groupby=groupby,query=query,orderby=orderby,
            plotstyle=plotstyle,**plotparams)


def plotline(ser,label,
             G=0.1,trend='right',trend_extend=0.5,
             annot='localmax last',                 # GitHub proxy
             show_line=True,show_points=True,area=None,lineformats=None,normalize=None):    
    '''
    Pontsorozat rajzolása vonal- vagy területdiagrammal.  pltinit-pltshow keretben kell hívni, ugyanarra a diagramra több pontsorozat rajzolható
    '''
    params={}
    plttype=[]
    if show_points: plttype.append('scatter')
    if show_line: 
        if G>0 and G<1: 
            if trend: 
                plttype.append('regausst')
                dset(params,trend=trend,extend=trend_extend)
            else: plttype.append('gauss')
        else: plttype.append('original')
    plttype=' '.join(plttype)


    FvPlot(ser,plttype=plttype,gausswidth=G,label=label,annotate=annot,normalize=normalize,area=area,
            colors=lineformats,**params)


def f_tbl_plot_test():


    tbl = load_iris(as_frame=True).data

    tbl_plot(tbl,annot='max localmax2 last',annotlength=15,resample=True,trend='right',trend_extend=0.5)
    tbl_plot(tbl,plotstyle='lines',annot='maxabs')




def f_plotline_test():
    tbl = load_iris(as_frame=True).data
    # tblinfo(tbl)
    ser1=tbl['sepal length (cm)']
    ser2=tbl['sepal width (cm)']

    suptitle='Function tests: "plotline"'
    comment='Iris database from sklearn'

    # teszt
    pltinit(suptitle=suptitle)
    # FvPlot(ser1,plttype='scatter spline',colors={'color':'red','faderight':[100,105,110,115,120,125,130,135]})
    FvPlot(ser1,plttype='spline')
    FvPlot(ser1,plttype='original')
    pltshow(commenttopright=comment)
    exit()


    title="localmax, localmin"
    pltinit(suptitle=suptitle,title=title)
    plotline(ser1,label='sepal_length')             # default: annot="localmax last"
    plotline(ser2,label='sepal_width',annot="localmin last")
    pltshow(commenttopright=comment)

    title="max and min"
    pltinit(suptitle=suptitle,title=title)
    plotline(ser1,label='sepal_length',annot="min max last")
    plotline(ser2,label='sepal_width',annot="min max last")
    pltshow(commenttopright=comment)

# f_tbl_plot_test()
# f_plotline_test()

