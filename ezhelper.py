''' Alapszintű helper függvények:  stingkezelés (pl. text2int, d_annotate, d_getByLemma), tömbök és szótárak kezelése, fájlrendszer, debug
''' 

from datetime import datetime,date,timedelta
from time import process_time,perf_counter
import os
from pathlib import Path
from itertools import groupby,product
import string
import math
import numpy as np
from numpy import array
import pandas as pd
import re
import time
from random import randint
import random
import pickle
import glob







# ============================================================================ #
# # STOPPER
# ============================================================================ #

def stopperstart():
    '''
    Használat:   t=stopperstart()       stopper(t)
    '''
    return process_time()

def stopper(process_time_start,bPrint=True,msg='process time'):
    '''
    Előtte:  t=stopperstart()       stopper(t)     
    A nettó processzor-időt adja vissza sec-ban (sleep nélkül)
    Másik lehetőség:   perf_counter()   -   a teljes eltelt idő sec-ban (általában lényegesen nagyobb)
    '''
    sec = process_time()-process_time_start
    if bPrint: print(msg + ': ' + str(sec))
    return sec


# ============================================================================ #
# # PATH MŰVELETEK
# ============================================================================ #

def fileexists(path):
    return os.path.exists(path)

def fn_path(fname,ext,dir='downloads'):             # a Path sajnos foglalt
    if ext: fname=fname + '.' + ext
    if dir=='downloads':
        return os.path.join(dir_downloads(),fname)
    else:
        return os.path.join(dir,fname)
    
def nextpath(fixpart,ext,dir='downloads'):
    """
    A következő (még nem foglalt) sorszám nélküli vagy sorszámozott fájlnevet adja vissza.
    Gyors kereső-algoritmus, nagy tömegű fájl esetén is.

    fixpart:   a fájlnév fix része. A path az alábbi lesz:  [dir]\[fixpart] (1).[ext]
    dir:  default: az aktuális felhasználó Downloads mappája
        - '':  a python fájl mappájába kerül
        - r'\adatok':    relatív elérési út is megadható a py fájl mappájához képest
        - teljes path is megadható (a végére nem kell perjel vagy backslash)
    
    return:   [dir]\[fixpart] (1).[ext]         // sorszám csak akkor, ha van már ilyen mintázatú fájlnév
    """

    if not dir: dir='downloads'

    # Elsőként meg kell nézni, hogy van-e már sorszám nélküli változat
    path_pattern=fixpart
    if ext: path_pattern+='.' + ext
    if dir.lower()=='downloads': dir=str(Path.home() / "Downloads")
    if dir!='': path_pattern=dir + "\\" + path_pattern
    
    if not os.path.exists(path_pattern):
        return path_pattern

    # Sorszámozott
    path_pattern=fixpart + ' (%s)'
    if ext: path_pattern+='.' + ext
    if dir.lower()=='downloads': dir=str(Path.home() / "Downloads")
    if dir!='': path_pattern=dir + "\\" + path_pattern

    i = 1

    # First do an exponential search
    while os.path.exists(path_pattern % i):
        i = i * 2

    # Result lies somewhere in the interval (i/2..i]
    # We call this interval (a..b] and narrow it down until a + 1 = b
    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2 # interval midpoint
        a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)

    return path_pattern % b

def lastpath(fixpart,ext=None,dir='downloads'):
    """
    A legnagyobb sorszámú fájlnevet tartalmazó path-t adja vissza. Ha nincs még sorszámozott, akkor a
        sorszám nélküli.
    Nem exponenciális a sorszám keresése, ezért nagy tömegű sorszámozott fájl esetén kevésbé gyors, mint a nextpath

    fixpart:   a fájlnév fix része. A sorszámozott path:  [dir]\[fixpart] (1).[ext]
        - kisbetű-nagybetű eltérésre nem érzékeny
    dir:  default: az aktuális felhasználó Downloads mappája
        - '':  a python fájl mappájába kerül
        - r'\adatok':    relatív elérési út is megadható a py fájl mappájához képest
        - teljes path is megadható (a végére nem kell perjel vagy backslash)
    
    return:   [dir]\[fixpart] (1).[ext]        
    """
    if not dir: dir='downloads'

    path_pattern=fixpart + '*'
    if ext: path_pattern+='.' + ext
    if dir.lower()=='downloads': dir=str(Path.home() / "Downloads")
    if dir!='': path_pattern=dir + "\\" + path_pattern


    paths = glob.glob(path_pattern)     # fájlnevek lekérdezése
    index_max=-1
    path_out=None
    for path in paths:
        mappa,fname_ext = os.path.split(path)
        fname,extL = os.path.splitext(fname_ext)

        if fname==fixpart and index_max==-1: path_out=path
        else:
            found = re.search(r"\((\d+)\)", fname)
            if found:
                index = int(found.group(1))
                if index>index_max: 
                    index_max=index
                    path_out=path
    
    if path_out: return path_out
    
def downloadspath(fname,ext):
    if ext: fname=fname + '.' + ext
    return os.path.join(dir_downloads(),fname)


    
def dir_downloads():
    return str(Path.home() / "Downloads")

def progress(text,rowlen=120):
    '''
    A progress sorozat végén érdemes egy print hívással zárni

    rowlen:  a várható leghosszabb kiírandó szöveg
        - ha egy hosszabb szöveg után egy rövidebb jön, akkor a korábbi vége nem törlődne
        - ne legyen túl nagy, mert kis ablakszélesség esetén kétsorossá válhat a szöveg, és ezzel a sor elejére
           lépés nem a szöveg legelejére viszi a kurzort
        - shorten funkció jelenleg nem érvényesül
    '''
    text='... ' + text      # azért is kell valami az elejére, mert egy cursor-téglalap jelenik meg az első karakterhelyen
    if rowlen is not None and rowlen>0: 
        # if len(text)>rowlen: text=txtshorten(text,rowlen)
        # else: 
        text=text.ljust(rowlen)
    print(text, end='\r',flush=False)            # a kurzor a kiírás után azonnal visszaugrik az elejére

def Serialize(obj,fname_fix='serial',dir='downloads',numbering=True):
    if numbering: path=nextpath(fname_fix,'pkl',dir)
    else: 
        path=fname_fix + '.pkl'
        if dir.lower()=='downloads': dir=str(Path.home() / "Downloads")
        if dir!='': path=dir + "\\" + path
    
    try:
        with open(path, 'wb') as file:
            pickle.dump(obj, file)
        print('Szerializálás végrehajtva: ' + path)
    except Exception as e:
        print('ERROR    A szerializálás nem sikerült  (' + path + ')' + '\n' + str(e))

def ReadSerialized(path):
    try:
        with open(path, 'rb') as file:
            return pickle.load(file)
        print('Szerializált objektum beolvasva: ' + path)
    except Exception as e:
        print('ERROR    A szerializálás objektum beolvasása nem sikerült  (' + path + ')' + '\n' + str(e))

    


# ============================================================================ #
# # ÁLTALÁNOS
# ============================================================================ #

def notnone(v1,v2,v3=None,v4=None,v5=None):      # az első not None argumentum (általában változónevek a végén egy konstans)
    # Ha mindegyik None, akkor None lesz a return
    if v1 is not None: return v1
    elif v2 is not None: return v2
    elif v3 is not None: return v3
    elif v4 is not None: return v4
    elif v5 is not None: return v5

def stringif(str,condition):
    if condition: return str
    else: return ''



# ============================================================================ #
# # SZÁM MŰVELETEK
# ============================================================================ #

def Round(n, decimals=0):
    ''' Ez felel meg a standard kerekítési szabálynak (>=0.5 esetén felfelé egyébként lefelé; negatív esetén az abszolút értékre vonatkozik)
    A python round() függvénye bizonyos pontokon eltér ettől (pl. 2.5 esetén 2-t ad a kerekítés, más esetekben viszont rendben van).
    Ennek az a törekvés a magyarázata, hogy a python a lehető legkisebb eredő kerekítési hibát próbálja meg elérni.

    Csak a decimális oldalon kerekít (nem digit-számról van szó)
    '''
    if n==None: return None
    multiplier = 10 ** decimals
    rounded_abs=np.floor(abs(n)*multiplier + 0.5) / multiplier
    return math.copysign(rounded_abs, n)

def Limit(number,min=None,max=None):        # minmax egy sorban
    '''
    min,max:  ha float kimenet kell, akkor legyen float   pl. min=1.
    '''
    
    if min and number<min: return min
    else:
        if max and number>max: return max
        else: return number

# Limit(19.99,20,30)


# ============================================================================ #
# # STRING MŰVELETEK
# ============================================================================ #

def kodto(kod,kodlista,default=''):         # ELAVULT:  lásd dget
    ''' Szerializált dictionary, nem túl nagy kódlistákra 
    kodlista:  "kod:felirat//..."       A kódokban nem lehet írásjel, a felirat-ban nem lehet '//'
    Ha tömegesen kell hívni és/vagy nagy a kódlista, akkor érdemes átfordítani dictionary-re (kodlistaToDict)
    '''
    if not kod: return default
    if not kodlista: return default
    if type(kod)!=str: kod=str(kod) 
    kereső=kod + ':'
    nLen=len(kereső)
    aSor=kodlista.split('//')
    for sor in aSor:
        if sor[:nLen]==kereső: return sor[nLen:]
    return default

def kodlistaToDict(kodlista,sep='//',bIntValue=False):
    ''' "kod:value//kod:value//..."  formátumú gyorsírásos kódlisták konverátálása dictionary-re
    - a kódokban ne legyen írásjel (csak alfanum és underline), a value-ban nem lehet '//'
    - a kód helyén megadható vesszős kód-felsorolás (a felsorolt kódok ugyanazt az értéket kapják)
    - a kód és a felirat között ':' vagy '=' jel is állhat 
    A result üres dictionary is lehet, ha a kodlista arg nem értelmezhető kódlistaként
    Utána:   dict[kod]  vagy  dict.get(kod)   hívások    (a dict indexelt ezért nagy listákra is gyors)
    '''
    aRec=kodlista.split(sep)
    d={}
    for rec in aRec:
        kod,value = splitfirst(rec,':')
        if value==None: 
            kod,value = splitfirst(rec,'=')    # '=' határolójel is lehet a kód és a value között
            if value==None: continue;       # ha nincs benne kettőspont vagy '=', akkor kihagyja
        kod=kod.strip()     # space levágás a szélekről
        kodok=kod.split(',')    # vesszős felsorolás lehet (mindegyik ugyanazt a value-t adja vissza)
        for kod in kodok:
            kod=kod.strip()
            if kod=='': kod='other'
            if bIntValue:
                try: d[kod]=int(value)
                except: pass            # ha nem integer, akkor kihagyja
            else: d[kod]=value
    return d


    
def beginwith(strIn,samples,replace=None):   # -> found_sample vagy ''     replace esetén a cserével létrejövő string
    ''' Gyors változat:  strIn.startswith(with)    boolean
    
    samples:  minták | jeles felsorolása   példa:  'ki|be'
      FIGYELEM:  pont, zárójel és egyéb regex karakterek elé kötelezően '\' kell   (pl. r'\.')
    return:  üres vagy a talált minta
    process_time: mikrosec körül
    '''
    m=re.search(r'\A(' + samples + ')',strIn)
    if m==None: 
        if replace!=None: return strIn
        else: return ''
    else: 
        found=m.group()
        if replace!=None: return replace + strIn[len(found):]
        else: return found 

def endwith(strIn,samples,replace=None):   # -> found_sample vagy ''     replace esetén a cserével létrejövő string
    ''' Gyors változat:  strIn.endswith(with)    boolean
    
    samples:  minták | jeles felsorolása   példa:  'tól|től'
      FIGYELEM:  pont, zárójel és egyéb regex karakterek elé kötelezően '\' kell   (pl. r'\.')
    return:  üres vagy a talált minta
    process_time: mikrosec körül
    '''
    m=re.search('(' + samples + r')\Z',strIn)
    if m==None: 
        if replace!=None: return strIn
        else: return ''
    else: 
        found=m.group()
        if replace!=None: return strIn[:len(strIn)-len(found)] + replace
        else: return found 

def vanbenne(strIn,samples,bWholeword=False):      # egyszerűbben: strIn.find(str_sub) == -1
    ''' samples:  minták | jeles felsorolása   példa:  'dika|dike'   (vagy kapcsolat)
    return:  üres string vagy a talált minta  (ha több minta is jó, akkor a strIn-ben előrébb álló minta)
    process_time: mikrosec körül
    '''
    if bWholeword:
        m=re.search(r'\b(' + samples + r')\b',strIn)
    else:
        m=re.search(r'(' + samples + ')',strIn)
    if m==None: return ''
    else: return m.group()

def nincsbenne(strIn,samples,bWholeword=False):
    return not vanbenne(strIn,samples,bWholeword)

def cutleft(strIn,strCut):
    # Nem ellenőrzi, hogy valóban a strCut-tal kezdődik-e, egyszerűen levágja az elejéről a strCut hosszának megfelelő karaktereket
    return strIn[len(strCut):]

def cutright(strIn,strCut,check=False):   # Levágás a jobb szélről
    '''
    check:  True esetén ellenőrzi, hogy valóban a strCut-tal végződik-e
            False esetén csak a strCut hossza számít
    '''
    if check and not endwith(strIn,strCut): return strIn
    else: return strIn[:-len(strCut)]

def txtshorten(text,maxlen,rightlen=0):   # rövidítés a végén "..."-tal
    # ... a végére, ha túl hosszú. A result szélessége garantáltan<=maxlen. 
    # rightlen:  mennyit tartson meg a jobb oldalon  (elvárás: < maxlen-3)
    # példa:    '1234567890',10  >>  '1234567890'
    #           '1234567890',9   >>  '123456...'
    #           '1234567890',9,3 >>  '123...890'
    #           '1234567890',9,7 >>  '...567890'
    #           '1234567890',2   >>  '..'
    
    if maxlen==None or len(text)<=maxlen: return text
    if maxlen<3: return '.'*maxlen



    if rightlen>maxlen-3: rightlen=maxlen-3
    if rightlen<0: rightlen=0

    leftlen=maxlen-3-rightlen
    if leftlen<0: leftlen=0

    if rightlen>0: return text[:leftlen] + '...' + text[-rightlen:]
    else: return text[:leftlen] + '...'

def trim(str):
    ''' strip a széleken + ismétlődő space-k törlése belül 
    Minden whitespace-re kiterjed  (a kimenet belsejében a whitespace karakterek közül csak space maradhat, ismétlődés nélkül)
    Ha csak a széleken kell, akkor str.strip()  str.lstrip()  str.rstrip()   (ezek is minden whitespace-re vonatkoznak)
    '''
    return ' '.join(str.split())


trans_ékezet = str.maketrans('áéíóöőúüű','aeiooouuu')
def Lics(s):        # list.sort, pandas.sort_values, pandas.sort_index:   key=Lics
    if type(s)==str:
        return s.lower().translate(trans_ékezet)
    elif type(s)==pd.Series and s.dtype=='object':
        return (s.str.lower()).str.translate(trans_ékezet)
    else:
        return s


def clean(str,accentlevel='soft',bDropSpaces=False):
    ''' lower, hosszú ékezet helyett rövid ékezet (a-á, e-é, o-ö, u-ü nincs összevonva), írásjelek törlése  (betűk és számjegyek maradnak)
    bDropSpaces:   True esetén a whitespace-ek teljes törlése, False esetén csak trim (a belső szóközök megmaradnak, az írásjelek helyett space)
    accentlevel: '': nincs összevonás,  'soft':  a-á, e-é, o-ö, u-ü nincs összevonva,    'hard':  erős összevonás
    '''
    if accentlevel=='hard':
        s1='áéíóöőúüű'
        s2='aeiooouuu'
    elif accentlevel=='soft':
        s1='íóőúű'
        s2='ioöuü'
    else:
        s1=''
        s2=''
    
    str=str.lower()
    if bDropSpaces:
        str=str.translate(str.maketrans(s1,s2,string.punctuation + string.whitespace))      # a harmadik arg az elhagyandó karakterek listája
    else:
        str=str.translate(str.maketrans(string.punctuation,' '*len(string.punctuation)))   # írásjelek helyett szóközök
        str=str.translate(str.maketrans(s1,s2))
        str=' '.join(str.split())
        return str

def skipaccents(str,level='soft'):
    ''' Ékezetek elhagyása
    level:
       'soft':  csak a hosszú ékezeteket cseréli rövidre,  a-á, e-á, o-ö, u-ü eltérés megmarad
       'hard':  minden ékezetet elhagy
    '''
    if level=='soft':
        return str.translate(str.maketrans('íóőúűÍÓŐÚŰ','ioöuüioöuü'))
    elif level=='hard':
        return str.translate(str.maketrans('áéíóöőúüűÁÉÍÓÖŐÚÜŰ','aeiooouuuaeiooouuu'))
    else:
        return str

def skippunctuations(str,tochar=''):
    # írásjelek törlése vagy lecserélése egy megadott karakterre (pl. szóköz)
    # írásjelek:   !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~.     - az underline is benne van
    if len(tochar)==1: return str.translate(str.maketrans(string.punctuation,tochar*len(string.punctuation)))
    else: return str.replace(string.punctuation,'')
    

def tokenize(strIn,delimiter,options=[]):     # tokenizálás:  split függvény a python-ban, hide_quot, hide_arg lehetőséggel
    '''
    delimiter: több karakteres is lehet
    options:  kulcsszavak felsorolása (list)
        strip:  strip a kimeneti lista stringjeire
        hide_quot: idézőjeles szövegrészben ne keressen határolójelet 
        hide_arg:  zárójeles szövegrészekben ne keressen haátrolójelet

    '''    

    if strIn=='' or delimiter=='': return strIn

    for option in options:
        if option=='hide_quot': strIn,contents_quot=hide_quot(strIn)
        elif option=='hide_arg': strIn,contents_arg=hide_arg(strIn)

    tokens=strIn.split(delimiter)

    for option in reversed(options):
        if option=='hide_quot': tokens=[replace_back(token,contents_quot,'""')  for token in tokens]
        elif option=='hide_arg': tokens=[replace_back(token,contents_arg,'()')  for token in tokens]

    if 'strip' in options:  tokens=[x.strip() for x in tokens]

    return tokens


def splitfirst(str,separator=None,default_second=''):
    '''  Hívása:  first,second = splitfirst(str,',') 
    Ha nincs benn határolójel, akkor a második változóba '' értéket ad vissza
    '''
    array=str.split(separator,1)        # max 1 split (csak ez első határolójel érdekli)
    if len(array)>1: return array[0],array[1]
    elif len(array)==1: return array[0],default_second
    else: return '',''

def splitrand(str,token='|'):
    # felsorolásból visszaad egy random item-et
    aL=str.split(token)
    if aL.size==0: return ''
    elif aL.size==1: 
        return aL[0]
    else:
        return aL[randint(0,aL.Size-1)]

def splitToSentences(str,bChat=False):
    ''' A string darabolása mondatokra (a mondatok szélein strip, üres mondat nem kerül be)
    bChat: True esetén nagybetű nélkül is lehet mondatkezdet (pont+2szóköz vagy kettőnél több szóköz), és minden sordobás új mondatot indít.
    Példa: (tesztelés)
        'Első mondat. Pont utáni nagybetű mondathatárt jelez. ' +
        'A rövidítések - pl. "ez egy kft." - nem indítanak új mondatot. ' +
        'Szám is előfordulhat a mondat végén: esetek száma 12. ' +
        'A számok környékén pontok lehetnek: 2012.10.12-én utazott el. A hőmérséklet 12.3 fok volt Pesten és ' +
        '10.4 fok Debrecenben.  3. és 5. mondat. ' + 
        '(a mondat néha írásjellel kezdődik) ' +
        'Felkiáltójel után is új mondat! A kérdőjel is fontos ?  Nyilván ... ' +
        'Zárójeles felkiáltójel (!) vagy kérdőjel (?) a mondaton belül is lehet. ' +
        'Több felkiáltójel is előfordulhat !!! ' +
        'A felkiáltójel után kisbetű is jöhet! ez is új mondat' +
        '\nÚj sor nagybetűvel: legyen új mondat (pont nélkül is)' +
        '\n- új sor gondolatjellel: mindig új mondat' +
        '\núj sor kisbetűvel: ne legyen új mondat' +
        '\n\nkét sordobás után akkor is kezdődjön új mondat, ha az előző végén nem volt pont. ' +
        'PROBLÉMÁS A CSUPA NAGYBETŰ SORSZÁMMAL: 12. JELÖLT. ' +
        'Királynevek is problémásak: XXI. Károly.'
    '''
    sentences=[]
    while True:
        # következő mondathatároló keresése (a következő mondat első karaktere is benne van)
        if bChat:
            match=re.search(r'\.\s*[A-ZÁÉÍÓÖÚÜŰ([#&@{/\]|' +       # pont után nagybetű vagy írásjel
                            r'\.\s+[\d\-]|' +                      # pont után szám vagy kötőjel (közte legalább egy szóköz)
                            r'\.\s{2,}[\w\d([#&@{/\-]|' +          # pont után min2 szóköz (kisbetű is jó)
                            r'\?\s*[\w\d([#&@{/\-]|' +             # kérdőjel után betű-szám-írásjel (kisbetű is jó)
                            r'!\s*[\w\d([#&@{/\-]|' +              # felkiáltójel után betű-szám-írásjel
                            r'\s{3,}|' +                           # min3 szóköz
                            '\n' + r'[\w\d([#&@{/\-]',             # minden sordobás  (kivéve, ha meglepő írásjel jön utána, pl, ")" - ennek technikai oka van)
                            str)
        else:
            match=re.search(r'\.\s*[A-ZÁÉÍÓÖÚÜŰ([#&@{/\-]|' +      # pont után nagybetű vagy írásjel
                            r'\.\s+\d|' +                          # pont után szám (közte legalább egy szóköz)
                            r'\d\.\s{2,}[\w\d([#&@{/\-]|' +        # szám+pont után min2 szóköz
                            r'\?\s*[\w\d([#&@{/\-]|' +             # kérdőjel után betű-szám-írásjel (kisbetű is jó)
                            r'!\s*[\w\d([#&@{/\-]|' +              # felkiáltójel után betű-szám-írásjel
                            '\n' + r'[A-ZÁÉÍÓÖÚÜŰ\d\-]|' +         # sordobás után nagybetű-szám-kötőjel
                            '\n\n' + r'[\w\d([#&@{/\-]',           # két sordobás
                            str)
        if not match: break
        # A határolójel utolsó karaktere ne kerüljön be (a következő mondat első karaktere)
        #   - a végén lévő végő felesleges szóközöket (és sordobásokat) a strip levágja
        sentence=str[:match.end()-1].strip()       # a széleken lévő sordobások és szóközök nélkül
        if sentence: sentences.append(sentence) 
        # Maradék:  a határolójel utolsó karaktere maradjon (a következő mondat első karaktere)
        str=str[match.end()-1:]
    sentences.append(str.strip())       # az utolsó mondat hozzáadása (mindegy mi áll utána)
    return sentences

def Sortördelés(text,maxlength=100):    # \n jelek beszúrása
    words=text.split()
    line=''
    lines=[]
    for word in words:
        if len(line) + len(word) + 1 > maxlength:
            lines.append(line)
            line=word
        elif line=='': line=word 
        else: line += ' ' + word
    lines.append(line)
    return '\n'.join(lines)


def hide_arg(strIn,delimiters='()',replace='default', except_level=1):
    '''
    Lecseréli a zárójeles szövegrészeket sorszámozott jelekre  (pl (_0_), (_1_), ...) és visszadja a lecserélt tartalmakat
    Ügyel a beágyazott zárójelekre is (a felső szintű zárójelpárókat cseréli le)
    Maguk a zárójelek nem kerülnek be a kiemelt contentek-be   (a helyettesítőjel szélein viszont alapesetben megjelennek)
    Párosítatlan zárójelek esetén hibaüzenet vagy exception geenerálása kérhető

    Utána általában   replace_back(strOut,contents)

    delimiters:  2 karakteres stringet kell megadni   (cska 1-karakteres haátrolójeleket kezel)
    replace:  default:  '(_{i}_)'   '[_{i}_]'    ....    (a két delimiter a széleken, kivéve '{}': ilyenkor 'brace_{i}_brace')
        - megadható teljes sablon is
    except_level:
        0: nincs exception vagy msg, mindenképpen visszad valamit
        1: mindenképpen visszaad valamit, de hivaüzenetet jeleníthet meg
        2: raise Exception

    return:  strOut, contents
    '''
          

    bException=False
    if type(delimiters)==str:
        if len(delimiters)!=2: bException=True
    else: bException=True
    if bException:
        if except_level<=1: 
            print('ERROR  hide_arg  A delimiters két karakteres string lehet')
            return strIn
        elif except_level>1: raise Exception('hide_arg  A delimiters két karakteres string lehet')

    # A két delimiter nem lehet megegyező
    if delimiters[0]==delimiters[1]:
        if except_level<=1: 
            print('ERROR  hide_arg  A két határolójel nem lehet megegyező')
            return strIn
        elif except_level>1: raise Exception('hide_arg  A két határolójel nem lehet megegyező')


    if replace=='default':
        if delimiters[0]=='{':  replace='brace_{i}_brace'
        else: replace=delimiters[0] + '_{i}_' + delimiters[1]


    # zárójelen belüli szakaszok határainak begyűjtése
    bUncoupled=False
    sections=[]
    chars=list(strIn)
    szamlalo=0
    for i,char in enumerate(chars):
        if char==delimiters[0]: 
            if szamlalo==0: start=i 
            szamlalo+=1
        elif char==delimiters[1]: 
            if szamlalo>0: 
                szamlalo-=1
                if szamlalo==0 and start: sections.append((start,i))
            else: bUncoupled=True
    if szamlalo>0: bUncoupled=True
    if bUncoupled:
        if except_level==1: print('ERROR  hide_arg  Párosítatlan zárójelek')
        elif except_level>1: raise Exception('hide_arg  Párosítatlan zárójelek')

    # zárójelezett szakaszok cseréje helyettesítőjelre
    strOut=''
    contents=[]
    after_last=0
    for i,section in enumerate(sections):
        before,after = section
        strOut += strIn[after_last:before] + replace.format(i=i)   
        contents.append(strIn[before+1:after])          # a zárójel nem kerül be a content-be
        after_last=after+1
    strOut += strIn[after_last:]
    
    return strOut,contents

def hide_quot(strIn,delimiter='"',replace='default',except_level=1):
    '''
    Lecseréli az idézőjelezett szövegrészeket sorszámozott jelekre  (pl "_0_" "_1_", ...) és visszadja a lecserélt tartalmakat
    (alapesetben kettős idézőjelek)
    Maguk az idézőjelek nem kerülnek be a kiemelt contentek-be   (a helyettesítőjel szélein viszont alapesetben megjelennek)
    Páratlan számú idézőjel esetén hibaüzenet vagy exception generálása kérhető

    Utána általában   replace_back(strOut,contents)

    delimiters:  '"'   "'"      1 karakteres stringet kell megadni   (csak 1-karakteres határolójeleket kezel)
    replace:  default:  '"_{i}_"'   "'_{i}_'"  
        - megadható teljes sablon is
    except_level:
        0: nincs exception vagy msg, mindenképpen visszaad valamit
        1: mindenképpen visszaad valamit, de hivaüzenetet jeleníthet meg
        2: raise Exception

    return:  strOut, contents
    '''

    bException=False
    if type(delimiter)==str:
        if len(delimiter)!=1: bException=True
    else: bException=True
    if bException:
        if except_level<=1: 
            print('ERROR  hide_quot  A delimiter egy karakteres string lehet')
            return strIn
        elif except_level>1: raise Exception('hide_quot  A delimiter egy karakteres string lehet')


    if replace=='default':
        replace=delimiter + '_{i}_' + delimiter


    # az idézőjeleken belüli szakaszok határainak begyűjtése
    sections=[]
    chars=list(strIn)
    start=None
    for i,char in enumerate(chars):
        if char==delimiter: 
            if not start: start=i 
            else:
                sections.append((start,i))
                start=None
    if start: 
        if except_level==1: print('ERROR  hide_quot  Párosítatlan idézőjelek')
        elif except_level>1: raise Exception('hide_quot  Párosítatlan idézőjelek')

    # az idézőjeles szakaszok cseréje helyettesítőjelre
    strOut=''
    contents=[]
    after_last=0
    for i,section in enumerate(sections):
        before,after = section
        strOut += strIn[after_last:before] + replace.format(i=i)   
        contents.append(strIn[before+1:after])              # az idézőjel nem kerül be a content-be
        after_last=after+1
    strOut += strIn[after_last:]
    
    return strOut,contents

def replace_back(strIn,contents,delimiters='()'):
    '''
    hide_quot és hide_arg után alkalmazható az eredeti értékek visszaírására

    contents:  list os string.   Csak a határolójelek közötti részt tartalmazza, ezért a függvény automatikusan hozzáfűzi a határolókat
    delimiters:  '()', '[]', '{}', '""', "''"       2-karakteres stringet kell megadni
    '''

    if delimiters[0]=='{' and delimiters[1]=='}': search='brace_{i}_brace'
    else: search=delimiters[0] + '_{i}_' + delimiters[1]

    for i,content in enumerate(contents):
        strIn=strIn.replace(search.format(i=i),delimiters[0] + content + delimiters[1])
    return strIn


def Firstword(str):       # első szó
    '''
    Beveszi az írásjeleket is (kivéve whitespace)
    Ha egy magában álló pont van az elején, akkor azt is szónak tekinti
    '''
    return str.strip().split()[0]

def Lastword(str):
    return str.strip().split()[-1]



# ============================================================================ #
# # STR, OLDTIMER
# ============================================================================ #

def stringadd(str,stradd,sep=','):
    ''' NE HASZNÁLD   Régi vágású stringadd, nem túl sok részstringre;  NINCS INPLACE VÁLTOZAT, lines = stringadd(lines,line,'\n') utasítás kell.
    Korábbi programkódok migrációjához lehet szükséges 
      (a régi módszer bizonyos esetekben kevesebb programsort igényelt, könnyebben át lehetett tekinteni, de kevésbé rugalmas és lassabb volt)
    A Python stringek esetén nem oldható meg az "inplace" (var argumentumos) értékadás   (bytearray objektummal sem) 
    A standard python megoldás összehasonlítása a régi típusú stringadd algoritmussal:
        lines=[]                                lines=''
        for ...:   lines.append(line)           for ...:   stringadd(lines,line,'\n')     // a python-ban csak a lines=stringadd(lines,line,'\n') működik
        '\n'.join(lines)                        lines
    '''
    if str: return str + sep + stradd 
    else: return stradd

def joinlines(lines):
    return '\n'.join(lines)         # régi vágású helper függvény




# ============================================================================ #
# # SZÁM - TEXT KONVERZIóK
# ============================================================================ #

def isint(txt):
    try: 
        int(txt)
        return True
    except:
        return False

def text2int(text,bSerialOk=True,bCleaned=False,tupleout=False):
    ''' 
    text:  számjegyek, szövegesen kiírt szám vagy sorszám, római szám ('nulla', 'százhuszonöt',  max 999 billió)
       Példa:  "123"  "MCMLII", "háromezertizenkettő", "százhamincadik", "tizenkettes"  
       Általában egyetlen szó, de állhat több szóból is.
       case-insensitive, accent-insensitive, whitespace és írásjel érdektelen, szöveges változatnál szótövesít
       Érdemi elgépelések nem megengedettek (nem fuzzy jellegű)
       Nem lehet benne idegen szó (csak számjegyek, számszavak, római szám karakterek, ragok)
    bSorszamOk:  megengedett-e sorszám is  (az out ebben az esetben is egy szám lesz)
    bCleaned:   előzetes szabványosítás megtörtént-e  (egyetlen szó, lower, ékezet-összevonás).  30-40% gyorsulás érhető el 
    tupleout:  True esetén   (n,type)  a result, ahol type = 'szám', 'rómaiszám', 'sorszám'

    Result:  0-999 billió   -1 ha nem ismerhető fel számként
    process_time: 15 microsec körül   (ebből a fele idő a clean)

    '''

    def sub(nOut,tipus=''):
        if tupleout: return nOut,tipus
        else: return nOut


    if text=='': return sub(-1)

    # a szélekről mindenképpen el kell tüntetni a whitespace-eket és pontokat (pont megengedett a végén)
    text=text.strip('. ')
    if text=='': return sub(-1)

    # ha számjeggyel vagy '-' jellel kezdődik
    c=text[0]
    if c=='-' or c in string.digits:
        try:
            nOut=int(text)
            return sub(nOut,'szám')
        except:
            return sub(-1)


    # Próbálja meg római számként értelmezni (csak nagybetűket fogad el, és pont lehet a végén)
    nOut=romaiszam2int(text)
    if nOut>0: return sub(nOut,'rómaiszám')

    if not bCleaned:
        text=clean(text,'hard',True)        # accent hard, ne maradjon whitespace   (viszonylag időigényes, kb 5 mikrosec)
        if text=='': return sub(-1)

    typeout='szám'
    # Ha sorszám is megengedett
    if bSerialOk and ('dik' in text or 'els' in text or endwith(text,'s|stol|sig|son|sen|sban|sben')):
        # lemmatizálás  (nem minden rag, elsősorban a dátumokban előforduló számokra van kiélezve)
        if 'dik' in text: text=endwith(text,'dika|dike|diki|dikai|dikei|dikan|diken|dikaig|dikeig|dikatol|diketol|dikos|dikes|dikas','dik')
        elif 'els' in text: text=endwith(text,'elsotol|elson|elsoig|elseje|elsejen|elsejei|elsejeig|elsejetol','elso')
        else: text=endwith(text,'stol|sig|son|sen|sban|sben','s')    # huszastól << huszas
        
        d={ 'elso':'egy','egyedik':'egy','egyes':'egy','masodik':'ketto','kettedik':'ketto','kettes':'ketto',
            'harmadik':'harom','harmas':'harom','negyedik':'negy','negyes':'negy',
            'otodik':'ot','otos':'ot','hatodik':'hat','hatos':'hat','hetedik':'het','hetes':'het',
              'nyolcadik':'nyolc','nyolcas':'nyolc','kilencedik':'kilenc','kilences':'kilenc',
            'tizedik':'tiz','tizes':'tiz','huszadik':'husz','huszas':'husz','harmincadik':'harminc','harmincas':'harminc',
              'negvenedik':'negyven','negyvenes':'negyven','otvenedik':'otven','otvenes':'otven',
              'hatvanadik':'hatvan','hatvanas':'hatvan','hetvenedik':'hetven','hetvenes':'hetven',
              'nyolcvanadik':'nyolcvan','nyolcvanas':'nyolcvan','kilencvenedik':'kilencven','kilencvenes':'kilencven',
            'szazadik':'szaz','szazas':'szaz','ezredik':'ezer','ezres':'ezer','milliomodik':'millio','millios':'millio',
              'milliardodik':'milliard','milliardos':'milliard','billiomodik':'billio','billios':'billio'}
        for key,value in d.items():
            if text.endswith(key): 
                text=text[:-len(key)]+value
                typeout='sorszám'
                break


    nOut=0
    if text in ['null','nulla','zero']: return sub(0,'szám')

    numwords=[('billio',1000000000000),('milliard',1000000000),('millio',1000000),('ezer',1000),('szaz',100)]
    for numword,numvalue in numwords:
        nPos=text.find(numword)
        if nPos>-1:
            # előtte 1-999 szám állhat (kivéve a "száz" előtt: 1-9)
            nelottemax=999
            if numvalue==100: nelottemax=9

            elotte=text[:nPos]
            if elotte=='': nelotte=1
            else: nelotte=text2int(elotte)          # rekurzív hívás

            if nelotte<1 or nelotte>nelottemax: return sub(-1)
            nOut+=nelotte*numvalue
            text=text[nPos+len(numword):]       # a felhasznált rész levágása

    if text:
        dTizesek={'tizen':10,'tiz':10,'huszon':20,'husz':20,'harminc':30,'negyven':40,'otven':50,'hatvan':60,'hetven':70,'nyolcvan':80,'kilencven':90}    
        key,out = d_getByLemma(text,dTizesek,True)
        if out: 
            nOut+=out
            text=text[len(key):]      # a felhasznált rész levágása


    if text:
        dEgyesek={'egy':1,'ketto':2,'ket':2,'harom':3,'negy':4,'ot':5,'hat':6,'het':7,'nyolc':8,'kilenc':9}
        nEgyesek=dEgyesek.get(text)
        if nEgyesek==None: return sub(-1)
        nOut+=nEgyesek

    if nOut==0: return sub(-1)
    
    return sub(nOut,typeout)

def strint(number,bEzres=True):
    if pd.isnull(number): return ''
    return strnum(int(number),',').replace(',',' ')

def strnum(number,format='3g',varname='',excel=False):
    ''' str.format függvény paraméterezése.  
    format
        'int'  =' '       integer, ezres határolással (szóköz határolás sajnos nincs, csak vessző vagy underline)
        'float'='4g'         4 digit (záró nullák nincsenek). Az e+00 akkor jelenik meg, ha a szám >= e+05 vagy < e-04
        '%'    ='1%'         1.0 bázisú százalék, "12.5%" formátum
        '%%'   ='1f%'        100 bázisú százalék, "12.5%" formátum

        ' '   ='{x:,}        integer, ezres határolással (szóköz határolás, a vessző le lesz cserélve) 
        '3f'  ='{x:,.3f}'    fix decimal digits (0-k lehetnek a végén, e+00 kitevő semmiképpen, ezres határoló)
        '5g'  ='{x:,.5g}'    az összes digit száma van megadva (lehet kevesebb is, záró nullák nincsenek). Az e+00 akkor jelenik meg, ha a szám >= e+05 
        '4e'  ='{x:,.4e}'    mindenképpen e+00 formátum. A decimális digitek száma van megadva (0-k lehetnek a végén)
        '2%'  ='{x:,.2%}'    1.0 bázisú százalék (szorzás százzal).  A decimális digitek száma van megadva (0-k lehetnek a végén)
        '2f%' ='{x:,.2f}%'   nincs szorzás (közvetlen százalékos adatok). A decimális digitek száma van megadva (0-k lehetnek a végén)

    varname: általában üres;  a diagram-jelölők formázásakor 'x' kell.
    excel:  magyar nyelvű excel-be íráskor tizedespont helyett tizedesvessző
    '''

    if pd.isnull(number): return ''

    if format=='int': format=' '
    elif format=='f': format='4g'
    elif format=='g': format='4g'
    elif format=='float': format='4g'
    elif format=='%': format='1%'
    elif format=='%%': format='1f%' 

    if format[-1]=='g':
        digits=int(format[0])
        # Az egész részt csak e9-től írja át kitevős formátumra
        if abs(number)>=10**(digits-1) and abs(number)<=1e9: 
            number=Round(number)
            if abs(number)<1e4: format=''           # 4 digitnél még ne legyen ezres tagolás (pl. irsz)
            else: format=','

    if format=='': str_out=str(number)
    else: str_out=FvNumformat(format).format(number).replace(',',' ')    # ezres határoló legyen szóköz, vessző helyett

    if str_out[-2:]=='.0': str_out=str_out[:-1]

    if excel: str_out=str_out.replace('.',',')      # tizedespont helyett tizedesvessző

    return str_out


def FvNumformat(format='4g',var=''):
    ''' str.format függvény paraméterezése.  
    format
        Shorthand-ek:
        'int'  ='{x:,}'       integer, ezres határolással (szóköz határolás sajnos nincs, csak vessző vagy underline)
        'float'='4g'          4 digit (záró nullák nincsenek). Az e+00 akkor jelenik meg, ha a szám >= e+05 vagy < e-04
        '%'    ='1%'          1.0 bázisú százalék, "12.5%" formátum
        '%%'   ='1f%'         100 bázisú százalék, "12.5%" formátum

        ','    ='{x:,}'       integer, ezres határolással 
        '3f'   ='{x:,.3f}'    fix decimal digits (0-k lehetnek a végén, e+00 kitevő semmiképpen, ezres határoló)
        '5g'   ='{x:,.5g}'    az összes digit száma van megadva (lehet kevesebb is, záró nullák nincsenek). Az e+00 akkor jelenik meg, ha a szám >= e+05 
        '4e'   ='{x:,.4e}'    mindenképpen e+00 formátum. A decimális digitek száma van megadva (0-k lehetnek a végén)
        '2%'   ='{x:,.2%}'    1.0 bázisú százalék (szorzás százzal).  A decimális digitek száma van megadva (0-k lehetnek a végén)
        '2f%'  ='{x:,.2f}%'   nincs szorzás (közvetlen százalékos adatok). A decimális digitek száma van megadva (0-k lehetnek a végén)

    var: általában üres;  a diagram-jelölők formázásakor 'x' kell.

    Hívás:  strnum(number,'3g') = FvNumformat('3g').format(number)   
    '''

    if format[:3]=='{' + var + ':': return format                             # ha teljes formátum van megadva
    elif format in ['int',',']: return '{' + var + ':,}'                      # ezres tagolás
    elif format in ['f','g','float']: return '{' + var + ':,.4g}'
    elif format=='%': return '{' + var + ':,.1%}'
    elif format=='%%': return '{' + var + ':,.1f}%'
    elif format[-2:]=='f%': return '{' + var + ':.' + format[:-2] + 'f}%'     # közvetlen százalék (bázis=100)
    else: return '{' + var + ':,.' + format + '}'                             # pl. '2f'  '3g'   '5e',  '0%'

# number=1.234567
# for i in range(-5,10):
#     szám=number * 10**i
#     szám=Round(szám,4)
#     print(str(szám) + ': ' + '{label} y={y:,.9g}'.format(label='hahó',y=szám)   )
#     # print(str(szám) + ': ' + FvNumformat('3g').format(szám))
#     # print(str(szám) + ': ' + strnum(szám,'float'))
#     # print(str(szám) + ': ' + FvNumformat('float').format(szám))
# print('{label} ({x:,.2%})'.format(label="aaa",x=12.345))
    
    # TESZT, python számformátumok
    #'{:,}'.format(1234)                 # 1,234                 csak vessző vagy underline választható ezres határolónak
    #'{:,}'.format(1234.34)              # 1,234.34

    #'{:,.3f}'.format(12345678.12)       # 12,345,678.120        decdigit,   semmiképpen nincs e-kitevő,   0-k kerülhetnek a végére
    #'{:,.3f}'.format(1)                 # 1.000
    #'{:,.0f}'.format(1)                 # 1

    #'{:,.3g}'.format(1.1233)            # 1.12          összesített digitszám, kerekítéssel    
    #'{:,.3g}'.format(-112.7)             # -113           
    #'{:,.3g}'.format(1123.7)            # 1.12e+03      a kitevő csak akkor jelenik meg, ha a digitszám nem elegendő a teljes megjelenítéshez
    #'{:,.10g}'.format(1123545.7)        # 1,123.7       
    #'{:,.6g}'.format(1123.7)            # 1,123.7       megelégszik kevesebb digittel is (nem ír 0-t a végére)
    #'{:,.5g}'.format(99999)             # 99,999       megelégszik kevesebb digittel is (nem ír 0-t a végére)

    #'{:,.6e}'.format(1123.7)            # 1.123700e+03      decdigit (fix, 0 kerülhet a végére),  e+nn mindig megjelenik
    #'{:,.6e}'.format(1)                 # 1.000000e+00       

    #'{:%}'.format(0.983)                 # 98.300000%       decdigit (default: 6,  0 kerülhet a végére)
    #'{:.0%}'.format(0.987)               # 99%              szorzás százzal (1.0 a bázis;  a % a kapcsos zárójelen belül)

    #'{:.0f}%'.format(98.7)               # 99%              nincs szorzás (a % a kapcsos zárójelen kívül van)


def romaiszam2int(text):
    ''' Ha nem római szám, akkor 0
    Csak nagybetűk megengedettek'''

    roman_numerals = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}
    result = 0
    try:
        for i,c in enumerate(text):
            if (i+1) == len(text) or roman_numerals[c] >= roman_numerals[text[i+1]]:
                result += roman_numerals[c]
            else:
                result -= roman_numerals[c]
    except:
        result=0

    return result


# ============================================================================ #
# # SZÁM - DÁTUM KONVERZIÓK
# ============================================================================ #

def datefloat(x): 
    # x:  ha int vagy float (akár stringként is), akkor nincs konverzió
    #     ha string (de nem értelmezhető számnak), akkor megpróbálja dátumként értelmezni és unix float-ra konvertál
    #     ha timestamp, akkor unix-float ra konvertál
    #     egyébként megpróbálja a közvetlen float konverziót
    #    lehet string (pl. '2021.01.02','20210102')   vagy pd.Timestamp is
    #    Fordított művelet dátum esetén:   date = pd.to_datetime(aX_float[i],unit='D')
    if type(x) in [float,int]: 
        return float(x)
    elif type(x)==str: 
        if len(x)!=10:      # elsőként megnézi, hogy számként értelmezhető stringről van-e szó (kivéve: 20220101)
            try: return float(x)      
            except: pass
        return pd.to_datetime(x).timestamp()/(24*60*60)       
    elif type(x)==list or type(x)==np.ndarray:
        aOut=[0]*len(x)
        for i in range(len(x)): aOut[i]=datefloat(x[i])
        return aOut
    elif type(x)==pd.Series:
        x.index=datefloat(x.index.values)
        return x
    else:
        type_=str(type(x))
        if 'datetime' in type_:     # np.datetime64
            return pd.Timestamp(x).timestamp()/(24*60*60)     # érthetetlen, hogy miért nem tudja automatikusan a pandas ezt a konverziót

        elif 'Timestamp' in type_:
            return x.timestamp()/(24*60*60)     # érthetetlen, hogy miért nem tudja automatikusan a pandas ezt a konverziót
        else: 
            return float(x)

def datefloatA(aX):    #  NEM KELL  Helyette datefloat      date to unix-float  tömbökre
    '''
    ELAVULT:   hívható közvetlenül a datefloat  (kezeli a tömböket és a Series objektumokat is)
    '''
    
    # date: lehet string (pl. '2021.01.02','20210102')   vagy pd.Timestamp is
    aOut=[0]*len(aX)
    for i in range(len(aX)): aOut[i]=datefloat(aX[i])
    return aOut

def floatdate(x,datefloat0=None,date_step=None):   # unix-float to datetime
    '''
    x: array, list, Series is lehet  (Series esetén az indexet konvertálja dátumra, feltéve hogy float típusú)
    FIGYELEM:  Series esetén inplace jellegű
    
    datefloat0:  kezdődátum (unix_float)   Pl:  ser_reset_dateimeindex output-ja
        - ha datefloat0=None, akkor egyszerűen unix-float-ra konvertál
    date_step:  lépésköz napokban számítva
    '''
    
    if datefloat0 and date_step:
        x=datefloat0 + x*date_step      # tömbökre is működik, de list-re kellene egy kisebb átalakítás
        return pd.to_datetime(x,unit='D')

    elif type(x)==pd.Series:
        x.index=floatdate(x.index.values)
        return x
    
    else:
        result=pd.to_datetime(x,unit='D')       # list vagy array input esetén DatetimeIndex típusú
        if type(x)==list: return list(result)
        elif type(x)==np.ndarray: return array(result)
        else: return result

    

def isdatetime(x):
    return isinstance(x,(datetime,date))



# ============================================================================ #
# # INTERVALLUMOK
# ============================================================================ #

def isinintervals(strIn,intervals):
    '''
    strIn:  egyelőre csak szövegre jó (szöveges összahasonlítás)
    intervals:  pl. '2022.05.12,2022.06.01-2022.06.30'
    '''
    aIntervals=intervals.split(',')
    for interval in aIntervals:
        bottom,top=splitfirst(interval,'-')
        if (strIn==bottom or (top and strIn>=bottom and strIn<=top)):
            return True
    
    return False

def embeddinginterval(intervals):   # Befoglaló intervallum
    '''
    Bevoglaló intervallumot adja vissza
    intervals:  pl. '2022.05.12,2022.06.01-2022.06.30'
    return:  min,max
    '''
    min=''
    max=''
    aIntervals=intervals.split(',')
    for interval in aIntervals:
        bottom,top=splitfirst(interval,'-')
        if min=='' or bottom<min: min=bottom
        if not top: top=bottom
        if max=='' or top>max: max=top

    return min,max



# ============================================================================ #
# # LISTÁK, TÖMBÖK
# ============================================================================ #

def Range(first,last=None,add=None,count=None,space='linear'):    # int vagy float számsorozat, a last is beleértve, add vagy count paraméterezéssel 
    '''
    Az np.arange és az np.lincpace összevonása közös platformra. Alapaesetben a last is bekerül a listába.
    A return nem array, hanem mindig list
    Float, integer és dátum értékekre is jó (add megadása esetén, ha lehet, akkor int értékeket ad vissza)
      Dátum megadható strinként vagy datettime-ként is
    last:  ez is bekerül a számsorozatba, ha a határra esik.  (eltérés a range és az np.arange függvénytől)
        - ha nincs megadva, akkor add és count együttes megadásával értelmes a sorozat
    add, count:  0 azonos jelentésű a None-nal.  Ha mindkettő None, vagy last=first, akkor a [first] lesz a return
    space: 'linear' (default;  ha meg van adva az add, akkor mindenképpen 'linear')
        'geom':   mértani sorozat a megadott kezdő és végponttal (a count alapján meghatározott fix faktor)
        'geom_1_5':  5-szörös legyen a pontsűrűség a last-nál (a last súlya 5-szörös)
                - csak a két szám aránya számít, de kényelmesen lehet pl egymás után kérni 'geom_1_5' és 'geom_5_3' range-eket (1,5,3 a pontok súlya)
                - csak >0 float adható meg, de lehet tizedestört is  (pl. "geom_1_0.2"  =  "geom_5_1")
                - a weight-arány csak kellően magas count-ra érvényesül
                    példa: count=3, esetén a két intervallum aránya csak a weight-arány négyzetgyöke
                - a normál 'geom'  a 'geom_[last]_[first]'-nek felel meg
        'random'  count db véletlenszám a megadott határok között (add=None esetén érvényesül)
            - növekvő rendezés
            - ha first és last integer, akkor integer sorozatot ad vissza (ismétlődések lehetnek benne)
            - ha 
        'random_unique'   count db véletlenszám a megadott határok között, ismétlődések nélkül
            - akkor érekes az unicitás, ha first és last integer
    Példa:  Range(10,16,2)        >   [10,12,14,16]
            Range(10,16,count=3)  >   [10,13,16]                Range(10,16,0,3)  is jó
            Range(10,add=2,count=3) > [10,12,14]
            Range(10,count=3)     >   [10,11,12]                add=1
            Range(10)             >   [10]
            Range(10,12)          >   [10,11,12]                add=1-nek tekinti (ha mindegyik integer, float esetén [first,last] a return)
    '''

    bDate = (type(first)==str or isinstance(first,datetime))
    if bDate:
        first=datefloat(first)
        if type(last)==str or isinstance(last,datetime): last=datefloat(last)

    def f_floatdate(result):
        if bDate: result=floatdate(result)
        return result  
    
    if last==first: return f_floatdate([first])
    if last==None:
        if add and count:
            return f_floatdate([first + i*add for i in range(count)])
        elif count:         # add=1
            return f_floatdate([first + i for i in range(count)])
        else:
            return f_floatdate([first])

    if not add and not count:
        if type(first)==int and type(last)==int:
            if last>=first: add=1
            else: add=-1
        else:
            return f_floatdate([first,last])

    if add: 
        out=list(np.arange(first,last,add))
        if len(out)==0: return f_floatdate([first])
        if last==out[-1]+add: out.append(last)
        return f_floatdate(out)

    elif count: 
        if space=='linear':     # számtani sorozat
            return f_floatdate(list(np.linspace(first,last,count)))
        elif beginwith(space,'geom'):    # mértani sorozat
            geom,weight1,weight2 = unpack(space.split('_'),3)        # pl. 'geom_1_5'
            if weight1 and weight2:
                weight1=float(weight1)
                weight2=float(weight2)
                ticks=np.geomspace(weight1,weight2,count)
                if weight2>weight1:
                    ticks = - (ticks - (weight1+weight2))       # tükrözés
                ticks=np.sort(ticks)
                return f_floatdate(list(first + (ticks-ticks[0]) *(last-first) / abs(weight2-weight1)))
            else:
                return f_floatdate(list(np.geomspace(first,last,count)))
        elif space=='random':   # véletlen választás
            if type(first)==int and type(last)==int:
                return f_floatdate(list(np.sort(np.random.randint(first,last,size=count))))
            else:
                return f_floatdate(list(np.sort(first + (last-first)*np.random.rand(count))))
        elif space=='random_unique':   # véletlen választás
            if type(first)==int and type(last)==int:
                if count > last - first + 1:  count = last - first + 1 
                unique_integers = random.sample(range(first,last+1),count)
                return f_floatdate(list(np.sort(unique_integers)))
            else:
                return f_floatdate(list(np.sort(first + (last-first)*np.random.rand(count))))



    return f_floatdate([first,last])

                
def unpack(list_in,count):      # nem ad hibaüzenetet, ha a list hossza nem egyezik a fogadóváltozók számával (None default)
    '''
    Hívása:
    adat1,adat2,adat3 = unpack(list,3)         Nem ad hibaüzenetet, ha a list csak kételemű, hanem None értéket tesz a hiányzó adat helyébe
    '''
    if len(list_in)>count: list_out=list_in[:count]
    else:
        if len(list_in)<count: list_out= list_in + [None]*(count-len(list_in))
        else: list_out=list_in 
    return list_out

def Rank(value,inlist_sorted,side='left'):         # [0:1] értéket ad vissza; hol helyezkedik el az értékkészletben
    '''
    value:   több érték is megadható (list vagy array)
        - számok és stringek is megadhatóak (elvárás: a value és az inlist azonos típusú)
        - ha a lista miden eleménél kisebb, akkor kötelezően 0 lesz az eredmény  (ha mindegyiknél nagyobb, akkor 1)
    inlist_sorted:  list vagy array is lehet.  Kötelező az előzetes rendezés (ascending)
    side:  'left','right','middle'     Értékegyezés esetén az elé vagy a mögé eső pozíció alapján
            - 'middle' esetén 'left' és 'right' számolás, majd átlagolás

    return:  [0:1] float    (több value esetén a return is list)
    '''
    if side=='middle':
        rank1 = np.searchsorted(inlist_sorted,value,side='left')
        rank2 = np.searchsorted(inlist_sorted,value,side='right')
        rank=(rank1+rank2)/2
    else:
        rank = np.searchsorted(inlist_sorted,value,side=side)
    return rank/(len(inlist_sorted))

def Limit_by_quantile(arr,quantile=0.05):   # az elemek felső és alsó ... százalékánál limitálja az értéket
    '''
    arr:  numpy array vagy lista
        - DataFrame oszlop esetén:     tbl[colname]  =   Limit_by_quantile(tbl[colname].values,0.05)
    quantile:  (0:1) közötti float.
        - tuple is megadható:  első és felső limit    Példa  (0.05,0.95)
        - tuple esetén a limitálás lehet egyoldalú is    Példa:  (0.05,None)
    '''
    if type(quantile)==tuple:
        quantile_min,quantile_max=quantile
    else:
        quantile_min=quantile
        quantile_max=1-quantile
    
    if quantile_min is not None:
        limit_min = np.quantile(arr,quantile_min)
        arr[arr<limit_min] = limit_min
    if quantile_max is not None:
        limit_max = np.quantile(arr,quantile_max)
        arr[arr>limit_max] = limit_max
    return arr

def Unique_top(arr,outtype='arrays',max_len=None,label_len=None,delimiter=', '):    # leggyakoribb értékek, tételszámokkal
    '''
    unique értékkészlet, elöl a leggyakoribbak
    - az azonos elemszámúakon belüli sorrend az eredeti előfordulás sorrenje

    arr:  numpy tömb vagy lista is lehet
        - ha a listában számok és stringek is előfordulhatnak, akkor stringeket csinál a számokból is
    outtype:
        'arrays':  arr_unique, counts
        'recs':    ((unique,count))
        'str':     '"value1":count1, "value2":count2, ...'         label_len-ben adható meg a levágási hossz
    max_len:   legfeljebb ennyi uniqe érték kell a lista elejéről
    label_len:   csak "str" esetén érdekes;  a felsorolásban max ekkora hosszban jelenhetnek meg az értékek
    delimiter:   csak "str" esetén érdekes;  a felsorolás határolójele   (lehet pl. '\n' is)
    
    return:  arr_unique,counts
    '''
    if label_len==None: label_len=1000
    
    arr_unique, counts = np.unique(arr, return_counts=True)
    sorter=np.argsort(counts)[::-1]
    
    arr_unique = arr_unique[sorter]
    counts = counts[sorter]

    if max_len:
        arr_unique=arr_unique[:max_len]
        counts=counts[:max_len]

    if outtype=='str':
        joins=[]
        for i in range(len(arr_unique)): joins.append('"' + str(arr_unique[i])[:label_len] + '":' + strint(counts[i]))
        return delimiter.join(joins)
    elif outtype=='arrays':    
        return arr_unique,counts
    elif outtype=='recs':    
        return zip(arr_unique,counts)

   

# ARRAY OF LISTS
def arrayoflists(shape):    # üres python-listákkal inicializált numpy tömb 
    '''
    Akkor hasznos, ha a listák tetszőleges objektumokat tartalmazhatnak, és nem tudható előre az elemszámuk
    Az inicializálást követően hívható a tömb elemeire az append függvény.
    FIGYELEM:  a függvény a dimenziók szorzatának megfelelő számú listát inicializál. 
        Példa:  shape=(100,100,100) esetén 10^6 listát kell inicializálni, ami 2-3 másodpercet is igénybe vehet
    
    shape: lehet int vagy tuple is.  Kételemű tuple esetén array of array of lists
        Példa:  
            aOut=arrayoflists((2,3))
            aOut[0,0].append('akármi')
            ...
    '''
    aOut = np.empty(shape=shape, dtype=object)
    shape=aOut.shape
    if len(shape)==1:
        for i in range(shape[0]): aOut[i]=[]
    elif len(shape)==2:
        for i in range(shape[0]):
            for j in range(shape[1]):
                aOut[i,j]=[]
    elif len(shape)==3:
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    aOut[i,j,k]=[]
    elif len(shape)==4:
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    for l in range(shape[3]):
                        aOut[i,j,k,l]=[]
    else:
        print('arrayoflists   Max 4 dimenziós tömbök inicializálhatók a függvénnyel')
        return None

    return aOut


# LIST OS STRINGS
def Find_in_list(s,list_in):      # több szavas keresés; a lista legjobban hasonlító elemét adja vissza 
    '''
    'anywhere' és 'clean' jellegű keresés (hard), beginwith preferencia, de a szavak sorrendje egyébként nem számít
    
    s:  fontosak a szóhatárok (minden nem alphanum szóhatárnak számít). A szavak sorrendje érdektelen
    list_in:   list of strings
        - nincs indexelés, ezért sok milliós listákra nem javasolt
        - a listaelemek jellemzően egy vagy néhány szavasak
    '''
    s_clean=clean(s,'hard')      # lower, ékezet, írásjelek cseréje space-re, trim   (undeline-t is lecseréli)
    words=s_clean.split()
    if len(words)==0: return

    found,found2 = [],[]
    for s_inlist in list_in:
        s_inlist_clean=clean(s_inlist,'hard')
        lendiff=len(s_inlist_clean)-len(s_clean)
        if lendiff<0: continue          # a listában lévő string nem lehet rövidebb a kereső-stringnél

        bOk=True
        for word in words:
            if s_inlist_clean.find(word)==-1:
                bOk=False
                break
        if bOk: 
            # Ha beginwith is teljesül, akkor kiemelt találat
            if s_inlist_clean.startswith(words[0]): 
                found.append((s_inlist,lendiff))
            else:
                found2.append((s_inlist,lendiff))

    # begin találatok
    if len(found)>0:
        return sortrecords(found,1,outindex=0)[0]
    # anywhere találatok
    if len(found2)>0:
        return sortrecords(found2,1,outindex=0)[0]

def Str_in_list(s,list_in,case='lower',pos='any'):       # a lista-beli találat-stringet adja vissza (szabványosításra használható)
    '''
    Egyszavas keresés
    case:       ha None vagy üres, akkor nincs karkatercsere
        'nocase' vagy 'lower'   lower
        'accent' vagy 'lics':   lower + ékezetes betűk lecserélése (á és é is)
        'clean':                lower, ékezetek, írásjelek ('á' 'é' space marad)
    pos:        ha None vagy '' vagy 'equal', akkor teljes találat kell (case-t leszámítva)
        'anywhere'
        'like' vagy 'begin'
        'end'
    '''
    def f_case(v):
        if not case: return v
        if case in ['lower','nocase']: return v.lower()
        elif case in ['lics','accent']: return Lics(v)
        elif case=='clean':   return clean(v,'soft',False)
        return v

    s_lower = f_case(s)

    found,found2 = [],[]
    for s_inlist in list_in:
        s_inlist_lower=f_case(s_inlist)
        lendiff=len(s_inlist_lower)-len(s_lower)
        if lendiff<0: continue          # a listában lévő string nem lehet rövidebb a kereső-stringnél

        if pos in ['any','anywhere']:
            if s_inlist_lower.find(s_lower)!=-1: 
                found2.append((s_inlist,lendiff))
            if s_inlist_lower.startswith(s_lower):
                found.append((s_inlist,lendiff))
        elif pos in ['like','begin']:
            if s_inlist_lower.startswith(s_lower):
                found.append((s_inlist,lendiff))
        elif pos=='end':
            if s_inlist_lower.endswith(s_lower):
                found.append((s_inlist,lendiff))
        else:   
            if s_inlist_lower==s_lower:  return s_inlist
    # begin vagy end találatok   (any esetén is preferált)
    if pos in ['any','anywhere','begin','like','end'] and len(found)>0:
        return sortrecords(found,1,outindex=0)[0]
    # anywhere találatok
    if len(found2)>0:
        return sortrecords(found2,1,outindex=0)[0]




# ARRAY OF RECORDS
def unzip(aRecord,nCol=None):    # X,Y=unzip(aXy)  list os lists -re is jó
    '''
    Oszlopok kiemelése egy aRecs vagy egy list os lists objektumból
        aX,aY=unzip(aXy)
        aZ=unzip(aXyz,2)        - a nCol 0-bázisú
    Akkor is működik, ha a rekordok hossza eltérő - ilyenkor a legrövidebb rekord elemszáma határozza
        meg a visszaadott oszlopok számát  (a hosszabb rekordok többlet-elemei kimaradnak)
    A visszaadott tömbök valójában tuple-ök. Ha a tömbelemek módosítására van szükség, akkor a list(aX) művelettel át kell térni listára.
    Inverz művelet:   aRec=list(zip(aX,aY))
    '''

    if nCol: return list(zip(*aRecord))[nCol]
    else: return zip(*aRecord)

def npcol(colvalues):     # kétdimenziós tömböt állít elő, egyetlen oszloppal
    '''
    példa:  [1,2,3]     >>    [[1],[2],[3]]
    '''
    colvalues=array(colvalues)
    return colvalues.reshape(len(colvalues),1)



def sortrecords(aRec,byindex,reverse=False,outindex=None):
    # inplace rendezés a byindex sorszámú oszlop szerint
    # outindex:  visszaadja a rendezett aRec megadott sorszámú oszlopát (list)
    # FIGYELEM:  az np.nan értékekre rosszul rendez (ne legyen nan a rendezési oszlopban)
    #  - stringoszlop esetén az '' rendezési érték a végére kerül (itt nincs gond)
    aRec.sort(key = lambda x: x[byindex],reverse=reverse)
    if outindex!=None: return list(zip(*aRec))[outindex]

def sortarrays(aSorter,aToSort,reverse=False):
    # Az aSorter értékei alapján rendezi mindkét tömböt (inplace jellegű)
    aRec=list(zip(aSorter,aToSort))
    aRec.sort(key = lambda x: x[0],reverse=reverse)
    aSorter2,aToSort2=unzip(aRec)
    for i in range(len(aSorter2)): aSorter[i]=aSorter2[i]
    for i in range(len(aToSort2)): aToSort[i]=aToSort2[i]

def sortarrays3(aSorter,aToSortA,aToSortB,reverse=False):
    # Az aSorter értékei alapján rendezi mindhárom tömböt (inplace jellegű)
    aRec=list(zip(aSorter,aToSortA,aToSortB))
    aRec.sort(key = lambda x: x[0],reverse=reverse)
    aSorter2,aToSortA2,aToSortB2=unzip(aRec)
    for i in range(len(aSorter2)): aSorter[i]=aSorter2[i]
    for i in range(len(aToSortA2)): aToSortA[i]=aToSortA2[i]
    for i in range(len(aToSortB2)): aToSortB[i]=aToSortB2[i]

def sortarrays_byfirst(*arrays,reverse=False):    
    '''
    Az első tömb értékei alapján rendezi mindegyik tömböt (inplace jellegű)
    Megjegyzés:  ha csak egy olyan ciklusra van szükség, amiben az indexek az egyik résztömb
       szerint vannak rendezve, akkor elegendő lehet az np.argsort használata
    példa:   sortarrays_byfirst(aSorter,a1,a2,a3)
    '''
    aRec=list(zip(*arrays))
    aRec.sort(key = lambda x: x[0],reverse=reverse)
    arrays2=list(unzip(aRec))
    for k in range(len(arrays2)):
        for i in range(len(arrays[k])): arrays[k][i]=arrays2[k][i]


def grouprecords(aRec,keylambda):
    # Rekordok csoportosítása egy számított kulcs szerint
    # - a kulcs általában egy kategória-mező:   keylambda = lambda x:x[0]
    # - a kategóriák képezhetők egy vagy több folytonos értékkészletű mező alapján is
    #   Példa: egy datetime mező kategorizálása aszerint, hogy melyik évbe esik    keylambda = lambda x: x[2].year
    # Előfeltétel:  from itertools import groupby
    
    # aRec:  list of tuple
    # keylambda:  átlalában egy lambda függvény, ami minden rekordhoz hozzárendeli a rendezési kulcsértéket

    # return:  két-elemű iterátor
    #   for key, records in grouprecords(aRec,lambda x:x[0])
    #      print(key)
    #      print(list(records))                     - a records szintén egy iterátor
    #      for rec in records: print(rec)

    aRec.sort(key=keylambda)        
    return groupby(aRec,keylambda)


    
# ============================================================================ #
# # DICTIONARY
# ============================================================================ #

def ddef(**fields):
    ''' Dictionary létrehozása, függvény-argumentumokkal (key=value argumentumok)
    Előnye a normál szintaxishoz képest:  áttekinthetőbb és nem kell idézőjelezni az adatneveket (csak egy-szavas adatnevekre jó)
    Példa:
        dict1 = { 'suptitle':'Felső sor', 'title':'Alsó sor' }
        dict1 = ddef( suptitle='Felső sor', title='Alsó sor' )
    '''
    return dset({},**fields)
    # return fields           # ez egyszerűbb lenne

def ddef2(fields,values):       #  fields,values  (a fields vesszős felsorolás is lehet)
    '''
    fields:  vesszős felsorolás vagy list
    values:  list
    '''
    if type(fields)==str: fields=fields.split(',')

    dict_out={}
    for i in range(len(fields)): 
        dict_out[fields[i]]=values[i]
    return dict_out

def d_fromrec(tbl,row=0,cols='all'):    # dict egy tábla-rekord adataiból
    '''
    tbl:  lehet egy query eredménye (pl. kérem az első location="Hungary" rekord adatait egy dict-be)
    cols: list vagy vesszős felsorolás
    '''
    if type(cols)==str:  cols=cols.split(',')
    return tbl.iloc[row][cols].to_dict()


def dprint(fields,values,delimiter=', '):
    return dgets(ddef2(fields,values),fields='all',tostring=True,delimiter=delimiter)   


def dget(dict_ser_list_fn_value, key, default=None, index=None):    # egyetlen értéket ad vissza dict-ből, ser-ből, kódlistából, list-ből, kettős default kezeléssel
    '''
    Kettős default kezelés:
        - ha nincs találat, akkor elsőként a dict-ben lévő 'default' kulcsra keres
        - ha nincs megadva default a dict-ben, akkor a jelen függvényben megadott default lesz a return
   
    dict_ser_list_fn_value:  dict, series, kódlista-string, list, function vagy egyetlen érték (pl. string vagy szám)
        - dict,series,kódlista:   key='default' kulcsértékkel megadható a default érték  (nem kötelező megadni)
        - series esetén az első találatot adja vissza (nem garantált a kulcsok unicitása)
        - function:   f(key) = value
        - list:   a megadott indexű item  (ha nincs megadva index, vagy nincs ilyen index, akkor az utolsó listaelem)
        - value:  egyszerű kimeneti változótípus esetén közvetlenül megadható egy konstans érték is (pl. float, int, str)
            Ha a kimeneti adat egy dict,ser,function, vagy '//' jelet és kettőspontot is tartalmazó string, akkor a közvetlen értéket 
            egyelemű tuple-ben kell magadni:    
                (f_codelist_teáor,)             
                (dict_teáor,)
            A tuple jelzi, hogy ne próbálkozzon key vagy index alapján való kereséssel, hanem közvetlenül a megadott összetett adatot adja vissza.
    key:  keresési kulcsérték (mezőnév is lehet). Az adattípus többféle lehet (kivéve None)
    default:  ez lesz a return, ha nincs közvetlen találat és nincs 'default' találat

    return:  egyetlen érték (None csak akkor, ha nincs default).  Az adattípus többféle lehet. Semmiképpen nem ad hibaüzenetet
        '''
    
    if dict_ser_list_fn_value is None: return default

    # ha egyelemű tuple, akkor közvetlenül a value lett megadva (a tuple jelzi, hogy nem kell key vagy index alapú keresés)
    if type(dict_ser_list_fn_value)==tuple and len(dict_ser_list_fn_value)==1:
        return dict_ser_list_fn_value[0]

    # string esetén közvetlen érték vagy kódlista is lehet. Ha van benne "//" és ":" akkor kódlistának tekinti és key alapján keres
    if type(dict_ser_list_fn_value)==str and vanbenne(dict_ser_list_fn_value,'//') and vanbenne(dict_ser_list_fn_value,':'):
        dict_ser_list_fn_value = kodlistaToDict(dict_ser_list_fn_value)         # nem túl gyors megoldás, de eleve csak kisméretű kódlistákra javasolt
   

    result=None

    if type(dict_ser_list_fn_value)==dict:
        result=dict_ser_list_fn_value.get(key)
        if result is None:
            result=dict_ser_list_fn_value.get('default')
    elif type(dict_ser_list_fn_value)==pd.Series:
        result=dict_ser_list_fn_value.get(key)
        if result is None:
            result=dict_ser_list_fn_value.get('default')
        # Ha több találat van, akkor az első találat kell
        if type(result)==pd.Series:
            result=result.iloc[0]
    elif callable(dict_ser_list_fn_value):         
        result=dict_ser_list_fn_value(key)
    elif type(dict_ser_list_fn_value) in [list,np.array]:
        if index is None or index>=len(dict_ser_list_fn_value): index=-1         # ha nincs ilyen index, akkor az utolsó
        elif index<0: index=0
        result=dict_ser_list_fn_value[int(index)]
    else:
        result=dict_ser_list_fn_value           # pl. egyetlen szám, string, ...

    if result is None: result=default

    return result

def dgets(dict,fields,tostring=False,delimiter=', '):   # többmezős unpacking vagy stringbe írás
    # Többmezős unpacking dictionary-re   (nincs ilyen közvetlen művelet)
    #      Példa:   a,b,c = dgets(dict,'a,b,c')
    #  fields:  'all', lista vagy felsorolásos string (vessző határolással)
    #      - 'all':  összes mező
    #      - ha hiányzik valamelyik mezőnév a dictionary-ből, akkor None kerül a változóba  (nincs hibaüzenet)
    #      - default értéket nem kezel, a fogadóváltozók None értéket is kaphatnak
    #      - tostring esetén a mezőnevek után megadható speciális számformázás
    #        Példa:   'szám1[4f],text1,százalék1[1%]    (float esetén a default formázás: [3f] - ezt nem kell megadni)
    # tostring:  True esetén stringbe írja a megadott mezőneveket és értékeket (pl  textadat1="érték1", számadat2=14  )
    # delimiter:  határolójel stringbe írás esetén   példa:  '\n' - új sorokba
    if type(fields)==str: 
        if fields=='' or fields=='all':
            fields=dict.keys()
        else:
            fields=fields.split(',')
    if not tostring:
        if len(fields)==1 or (len(fields)==2 and fields[1]==''): return dict.get(fields[0])           # ilyenkor ne tuple-t adjon vissza
        elif len(fields)>0: return tuple(map(dict.get,fields)) 
    else:
        adatok=[]        
        for i,field in enumerate(fields):
            előtte,utána = splitfirst(field,'[')
            numformat=None
            if utána and utána[-1]==']': 
                field=előtte
                numformat=utána[:-1]

            value=dict.get(field,'')
            if numformat: value=strnum(value,numformat)
            else:
                str_type=str(type(value))
                if vanbenne(str_type,'int'): value=strint(value)
                elif vanbenne(str_type,'float'): value=strnum(value,'4g')
                elif vanbenne(str_type,'bool'): value=str(value)
                else: value='"' + str(value) + '"'
            adatok.append(field + '=' + value)
        return delimiter.join(adatok)
dgetfields=dgets


def dset(dict,**fields):            # dict adatok beállítása függvény-argumentumokkal 
    '''
    fields:  fieldname1=value1,fieldname2=value2, ....    
      - megadható közvetlenül vagy egy dict objektumként
      - közvetlen megadás esetén a mezőnevekhez nem kell idézőjel   (több szavas mezőnevekre nem működik)
      - ha valamelyik mezőnév hiányzik, akkor automatikus felvétel
      - None értékadás nem törli automatikusan a mezőt (None érték íródik be)
    '''
    aField=list(fields.keys())
    aValue=list(fields.values())
    for i in range(len(aField)): 
        dict[aField[i]]=aValue[i]
    return dict

def d_add(dict1,dict2):          # két dict összefésülése  (ütközés esetén dict2 az erősebb)
    dict_out=dict1.copy()         # fontos, mert enélkül a dict1 is változna
    dict_out.update(dict2)
    return dict_out
        
def dsetsoft(dict,**fields):        # inplace jellegű
    '''
    Soft:
      - csak akkor módosít, ha nincs még ilyen mező a dict-ben vagy None az értéke.
      - None értéket semmiképpen nem ír be
    fields:  fieldname1=value1,fieldname2=value2, ....    
      - megadható közvetlenül vagy egy dict objektumként
      - közvetlen megadás esetén a mezőnevekhez nem kell idézőjel   (több szavas mezőnevekre csak egy módosító dict működik)
      - ha valamelyik mezőnév hiányzik, akkor automatikus felvétel
    '''
    aField=list(fields.keys())
    aValue=list(fields.values())
    for i in range(len(aField)):
        if aValue[i]==None or dgets(dict,aField[i])!=None: continue
        dict[aField[i]]=aValue[i]

def d_addhard(dict,key,value):
    ''' Akkor is hozzáadja az értéket a dict-hez, ha van már ilyen key, de ilyenkor sorszámozza a kulcsot.
    return:  a felvett tétel kulcsa  (sorszámozásban térhet el az input key-től)
    Nagy tömegű sorszámozás esetén is viszonylag gyors, mert az eddigi lemgmagasabb szorszámú kulcsot binárisan keresi.
    '''
    
    if key=='': return ''
    # ellenőrzés: van-e már ilyen kulcs
    keyout=key
    if dict.get(keyout):
        i = 2

        # First do an exponential search
        while dict.get(key+str(i)):
            i = i * 2

        # Result lies somewhere in the interval (i/2..i]
        # We call this interval (a..b] and narrow it down until a + 1 = b
        a, b = (i // 2, i)
        while a + 1 < b:
            c = (a + b) // 2 # interval midpoint
            a, b = (c, b) if dict.get(key+str(c)) else (a, c)

        keyout=key+str(b)

    dict[keyout]=value

    return keyout

def d_searchTo(strIn,d_keresőszavakToValue,searchtype='wholeword'): # -> list of values  
    ''' Adott kimeneti értékeket tartalmazó dictionary. Mindegyik kimeneti értékhez egy vagy több keresőszó tartozik. 
    Ha legalább egy keresőszóra van találat, akkor a return-be bekerül a kimeneti érték
    A keresés case-insensitive (a keresőszavak kisbetűvel íthatók), a keresőszavak között VAGY kapcsolat 
    - extra esetben egy-egy keresőszón belül szóköz vagy más írásjel is lehet, de ilyenkor általában r-stringben kell megadni a keresőszavakat
        és a regex foglalt karakterei elé backslash kell (kötőjel-hez nem kell backslash).    Példa: r'borsod-abaúj|zemplén' 
        Párosítatlan zárójelek hibát okozhatnak a kereséskor.
        A pont elé backslash kell (ennek hiányában bármit elfogad a pont helyén)

    strIn:   több szavas szöveg is lehet (akár több mondatos is)
    d_keresőszavakToValue:
        'borsod|abaúj|zemplén': 'Borsod-Abaúj-Zemplén megye',       - több keresőszó is megadható (a keresőszavak viszont alapesetben egyszavasak)
        'vas.':'Vas megye',                                         - a '.' jelzi, hogy csak wholeword jó  (searctype='wholeword.')
    searchtype:  
        'wholeword':    a keresőszavak elejére és végére is \b kerül   (minden keresőszóra érvényesül)
        'wordbegin':    a keresőszavak elejére \b kerül    (minden keresőszóra érvényesül)
        'anywhere':     nincs \b beírás  (keresés bárhol a strIn szavain belül;  explicit \b jelek megadható a keresőmintákban, de ilyenkor r-string kell)
        'like*':        ha * van egy keresőszó végén, akkor wordstart, egyébként wholeword (keresőszavanként eltérhet)
        'wholeword.':   ha pont van egy keresőszó végén, akkor wholeword, egyébként wordstart  (keresőszavanként eltérhet)
    '''
    values=[]
    # Ciklus a kimeneti értékekre illetve a hozzájuk tartozó keresőszavakra
    for key,value in d_keresőszavakToValue.items():
        try: key=str(key)       # keresőszavak
        except: continue
        if searchtype=='wholeword': key=r'\b(' + key + r')\b'
        elif searchtype=='wordbegin': key=r'\b(' + key + ')'
        elif searchtype=='like*':           # viszonylag lassú az átalakítás
            keresőszavak=key.split('|')
            keresőszavak_out=[]
            for keresőszó in keresőszavak:
                if keresőszó[0]=='*': keresőszó=keresőszó[1:] 
                else: keresőszó=r'\b' + keresőszó
                if keresőszó[-1]=='*': keresőszó=keresőszó[:-1] 
                else: keresőszó=keresőszó + r'\b'
                keresőszavak_out.append(keresőszó)
            key='|'.join(keresőszavak_out)
        elif searchtype=='wholeword.': key=r'\b(' + key.replace('.',r'\b') + ')'

        if re.search(key,strIn,flags=re.IGNORECASE): values.append(value)
    return values

def d_getByLemma(strIn,d_lemmaToValue,tupleout=False,bWholesamples=True):  # Keresés lemmákat tartalmazó dict-ben
    ''' A dict alapesetben szótöveket tartalmaz, de teljes-egyezéses minták is megadhatók
      - szótő: általában nyelvtani értelemben, de valójában like mintákról van szó
      - a szótövek több szavasak is lehetnek
    A lehetséges szótövek közül a leghosszabb érvényesül (a strIn a talált szótővel kezdődik)
    Ha van találat, akkor az adott szótőhöz tartozó value-t adja vissza (a tupleout-tal kérhető a kulcs,values pár is return-ként)
    Ha nincs találat, akkor None
    strIn: általában egyetlen szó   (szavanként, vagy néhány szavas ablakokkal kell hívni a függvényt)
    d_lemmaToValue:   {[szótő]:[value],...}         A szótő string, a value lehet szám is vagy akár egy sub dictironary
       - a szótőben csak számok, betűk  (több szavas minta esetén space is)
       - a szótő végén lehet egy pont:  teljes-egyezéses minta   (ha nincs pont, akkor beginwith egyezés is jó)
       - nagy lista esetén érdemes cash-elni egy modulváltozóban
       - a strIn-t és a dictionary-t ugyanúgy kell szabványosítani (pl. a "clean" függvénnyel) 
       
    tupleout: True esetén return (szótő,value).    A szótő lehet több szavas  (az esetleges lezáró pont nélkül adja vissza a függvény)
              False esetén return value 
    bWholesamples:  vannak-e teljes-egyezéses minták is a d_lemmaToValue-ben (pont a keresőszó végén) 
    process_time:   2 microsec     50-es dict, átlagos szó (nem talál)     Feltehetően nagy dict-re is gyors (hash index)
    '''
    
    result=None

    tosearch=strIn

    if bWholesamples:
        tosearch=tosearch.replace(' ','.') + '.'
        
        #result=d_lemmaToValue.get(tosearch + '.')
        #if result: 
        #    if tupleout: return tosearch,result
        #    else: return result

    # Keresés hosszrövidítésekel (a leghosszabb keresőminta érvényesül)
    while len(tosearch)>1:
        result=d_lemmaToValue.get(tosearch)
        if result: 
            if tupleout: 
                tosearch=tosearch.replace('.',' ')
                if tosearch[-1]==' ': tosearch=tosearch[:-1]
                return tosearch,result
            else: return result
        tosearch=tosearch[:-1]

    #if result:
    #    if tupleout: return tosearch,result
    #    else: return result
    #else:
    if tupleout: return None,None
    else: return None

def d_annotate(strIn,lookup,special='[szám] [tólig]',max_words_in_samples=2,accentlevel='soft'):    # -> pattern,invalues,outvalues
    ''' Annotálás jellegű művelet. Milyen mintázathoz sorolható a szöveg, és mik a mintázatban lévő helyettesítőjelek (entitások) konkrét értékei.  
    A beazonosítható szövegrészek helyébe szögletes zárójelben az entity-nevét írja, az eredeti és a szabványosított értéket pedig 
        menti az invalues és outvalues tömbbe (a két tömb indexelése a pattern-be kerülő "[...]" helyettesítőjelekhez igazodik) 
    Számít a szavak sorrendje, írásjelek érdektelenek (kivéve pont), kisbetű-nagybetű és ékezet-eltérés érdektelen (lásd accentlevel)

    strIn:  általában több szavas kifejezés vagy mondat.    
        Nem kell előzetesen szabványosítani (a jelen függvényben: lower, trim, skipaccents, skippunctuations)
    lookup:   {'[startsamples]':'[entity],[value]', ...}
        - [startsamples]: több keresőminta is megadható | határolással.  Példa:  'múlt|elmúlt|előző|legutóbbi'
                A sorrend érdektelen, mindig a lehetséges leghosszabb találat érvényesül ("tavasszal" akkor is talál, ha a "tavasz" minta előrébb áll)
                A keresőminták több szavasak is lehetnek (trimmelve kell megadni)
                Ha egy keresőminta végén pont van, akkor csak wholeword találat megengedett  (egyébként beginwith találat is jó)
        - [entity]:  entity-név (pl. "évszak"). Kötelezően egyszavas  ('_' lehet benne)  
                Ugyanaz az entitás-név több startsamples-höz is megadható.  Tetszőleges számú entitásnév lehet a lookup-ben.
                "none" entitást kell megadni, ha csak szabványosításról illetve szinonimák összevonásáról van szó
                  Példa:  "korábbi|előző":none,korábbi.     Ebben az esetben csere a szabványosított értékre, az outvalues-be nem kerül bele
                "stopword" entitás-névvel kell felsorolni az elhagyandó szavakat (pl. névelők)
        - [value]:  ez az érték kerül be az outvalues-be (az adott entity névvel). Szabványosított / kivonatolt érték
                Ha nincs megadva, akkor a talált startsample lesz a szabványosított érték
                Lehet szám, dátum, string is.  String esetén több szavas is lehet (trimmelve kell megadni)
    special:  példa: '[szám] [tólig]'   speciális beazonosítási eljárások felsorolása (a határolójel és a sorrend érdektelen)
        [szám]:  a szöveges vagy numerikus szám-szavak annotálása "[szám]" entity-névvel (sorszámok és római számok is)
        [tólig]:  a -tól -től -ig ragok leválasztása külön szóként (egybeírt és kötejeles írásmód esetén is)
    max_words_in_samples:  maximum hány szavas minták vannak a lookup-ben
        - az ennél hosszabb (ennél több szóból álló) minták nem fognak találatot adni 
    accentlevel:     '': nincs összevonás,  'soft':  a-á, e-é, o-ö, u-ü nincs összevonva     'hard':  erős összevonás
      

    return   pattern,invalues,outvalues
                pattern:    példa:  "[entity1] egyébszó [entity2]"
                invalues:   az annotált részek eredeti értéke  (a fenti pattern-példához 2-elemű lista tartozik)
                outvalues:  az annotált részek kimeneti értéke  (a fenti pattern-példához 2-elemű lista tartozik)
    '''

    invalues=[]
    outvalues=[]

    # key-felsorolások kibontása (ha nincs még kibontva;  a lookup egy globális objektum, amit egyszer kell csak kibontani)
    if not lookup.get('d_compiled'):
        dict_compiled={'d_compiled':True}     # a d_compiled adat helyből beállítva
        for key,value in lookup.items():
            subkeys=key.split('|')          # több keresőminta is felsorolható '|' határolással
            for subkey in subkeys: 
                if subkey: 
                    subkey=subkey.lower()
                    subkey=skipaccents(subkey,accentlevel)
                    # írásjelek helyett szóközök (kivéve '.')
                    #strL=strL.translate(str.maketrans(dropchars,' '*len(dropchars)))

                    subkey=subkey.replace(' ','.')       # a több szavas keresőmintákban "." kell a szóközök helyett (technikai oka van)
                    dict_compiled[subkey]=value
        lookup=dict_compiled
    

    strL=strIn

     # ".-"  helyet "-"     Példa:   "5.-én": ne tekintse sorszámnak  (egy helyesírási hiba kijavításáról van szó)
    strL=strL.replace('.-','-')

    # szeparátor írásejelek helyett egységesen "vessző", külön szóban. Nem annotált szóként tagoló szerepe lesz.
    #   vessző, pontosvessző, kettős-perjel, függőleges-tagolóvonal
    strL=strL.replace(',',' vessző ')
    strL=strL.replace(';',' vessző ')
    strL=strL.replace(r'//',' vessző ')
    strL=strL.replace(r'|',' vessző ')

    # írásjelek helyett szóközök (kivéve '.')
    dropchars=string.punctuation.replace('.','')        # a ponton kívüli összes írásjel
    strL=strL.translate(str.maketrans(dropchars,' '*len(dropchars)))
    # pontok helyett pont + space  (külön szavak legyenek, de a szóvégi pontok örződjenek meg. A számok esetén fontos, hogy van-e pont a végén)
    strL=strL.replace('.','. ')
    
#    strL=strL.lower()   
    strL=skipaccents(strL,accentlevel)
    
    if '[tólig]' in special:
        strL=re.sub(r'(-jétöl|-átol|-étöl|-tol|-töl|tol|töl)\b',r' tol',strL)   # szóköz kerül elé, külön szó lesz
        strL=re.sub(r'(-jéig|-áig|-éig|-ig|ig)\b',r' ig',strL)


    # trim
    words=strL.split()

    # magában álló pontok elhagyása (nem volt előtte alfanum, érdektelen)
    wordsout=[]
    for word in words:
        if word=='.': continue
        wordsout.append(word)
    words=wordsout
    
    # Keresés szavanként (vagy pár szavas csúszóablakkal) a mintákat tartalmazó dict-ben  (like: karakter-elhagyások a végéről)
    wordsout=[]
    nLen=len(words)
    i=0
    while i<nLen:
        if '[szám]' in special:
            word=words[i]     # a számoknál csak egyszavas keresés van
            if word!='hét':   # a "hét" szó helyébe ne írjon számot (összetéveszthető a "hét" időhatározóval)
                              # Később még ellenőrzöm, hogy ha [időtartam] a következő szó is, akkor [szám]-ra módosuljon az értelmezése (pl. "hét nap")
                n=text2int(word,True,False)
                if n>-1:
                    # ha számjegyekből áll a szám, és a következő szó egy rag, akkor olvassza be a számba
                    #  ragot
                    if word.isnumeric() and i<nLen-1:
                        wordnext=words[i+1]
                        if wordnext in ['án','én','ján','jén','i','ai','ei','jei','os','as','es','ös',
                                'dikán','dikén','diki','dikei','dikai','dikos','dikas','dikes',
                                't','et','at','dikát','dikét',          # "5-öt" nem kezelhető, mert számmal összetéveszthető 
                                'diká','diké']:     # tól, től ig le lett vágva
                            word=word + wordnext
                            i+=1

                    invalues.append(word)
                    outvalues.append(n)
                    wordsout.append('[szám]')
                    i+=1
                    continue

        # max_words_in_samples szóra kell keresni (ha van még ennyi szó)
        i2 = i + max_words_in_samples
        if i2>nLen: i2=nLen
        tosearch=' '.join(words[i:i2]).lower()
        
        key,foundvalue=d_getByLemma(tosearch,lookup,True)
        if foundvalue:
            # hány szavas a találat (ennyi szó lesz lecserélve a helyettesítőjellel)
            nFoundWords=len(key.split())
            
            entity,value=splitfirst(foundvalue,',')
            if entity=='stopword': wordout=''
            else:
                if value=='': value=key
                if entity!='none': 
                    # Ha az előző szó is [időtartam] volt "hét" input-értékkel, akkor előző szó legyen számként annotálva (pl. "hét nap" "[szám] [időtartam]"
                    #   Ellenpélda: "utolsó hét az évben"   "az év" külön minta
                    if (entity=='időtartam' and len(wordsout)>0 and wordsout[-1]=='[időtartam]' 
                        and invalues[-1]=='hét' and not words[i] in ['a','az']):
                        wordsout[-1]='[szám]'
                        outvalues[-1]=7
                    if entity=='szám': value=int(value)
                    invalues.append(' '.join(words[i:i+nFoundWords]).lower())
                    outvalues.append(value)
                    wordout='[' + entity + ']'
                # Ha a keresőmintához "none" lett megadva entitásként, akkor nem helyettesítőjellel, hanem a szabványosított értékkel
                #    kell lecserélni a találatot eredményező szövegrészt  (a value több szavas is lehet) 
                else:
                    wordout=value
            i += nFoundWords

        else: 
            # Ha nincs találat, akkor egyetlen szó kerül be a wordsout-ba (változtatás nélkül), majd lépés a következő szóra
            wordout=words[i]
            i+=1



        if wordout!='': wordsout.append(wordout)
    



    #for word in words:
    #    if word=='.': continue   # a . előtt nem volt alfanum (érdektelen)

    #    if bNum:
    #        if word!='hét':   # a "hét" szó helyébe ne írjon számot (összetéveszthető a "hét" időhatározóval)
    #                          # Később még ellenőrzöm, hogy ha [időtartam] a következő szó is, akkor [szám]-ra módosuljon az értelmezése (pl. "hét nap")
    #            n=text2int(word,True,False)
    #            if n>-1:
    #                invalues.append(word)
    #                outvalues.append(n)
    #                wordsout.append('[szám]')
    #                continue

    #    key,foundvalue=d_getByLemma(word,lookup,True)
    #    if foundvalue:
    #        entity,value=splitfirst(foundvalue,',')
    #        if entity=='stopword': wordout=''
    #        else:
    #            if value=='': value=key
    #            if entity!='none': 
    #                # Ha az előző szó is [időtartam] volt "hét" input-értékkel, akkor előző szó legyen számként annotálva (pl. "hét nap" "[szám] [időtartam]"
    #                if entity=='időtartam' and len(wordsout)>0 and wordsout[-1]=='[időtartam]' and invalues[-1]=='hét':
    #                    wordsout[-1]='[szám]'
    #                    outvalues[-1]=7
    #                if entity=='szám': value=int(value)
    #                invalues.append(word)
    #                outvalues.append(value)
    #                wordout='[' + entity + ']'
    #            else:
    #                wordout=value
    #    else: wordout=word



    #    if wordout!='': wordsout.append(wordout)
    
        
    pattern=' '.join(wordsout)
    return pattern,invalues,outvalues

   
    # B VÁLTOZAT:  mintánként scan a teljes szövegre
    #for regex,data in d_regex:
    #    value0=clean(regex,'',False)
    #    if regex.endswith('*'): regex=regex[:-1]
    #    else: regex=regex + r'\b'
    #    if regex.startswith('*'): regex=regex[1:]
    #    else: regex=r'\b' + regex
    #    regex=regex.replace('*','.*')

    #    entity,value=splitfirst(data,',')
    #    if value=='': value=value0

    #    re.sub(regex,entity,strIn)

    return


# ============================================================================ #
# # CLASS, OBJECTS
# ============================================================================ #

def set_attrs(obj,attrs):         # a setattr() python függvény felöltöztetése dict argumentummal
    '''
    Felhasználható az osztályok __init__ függvényében is  ( set_attrs(self,**arguments),  ahol az arguments a locals() függvénnyel kérhető be)
    Ha nincs még ilyen attribútum, akkor automatikus felvétel
    attrs:  dictionary (attrname to attrvalue)
    '''
    for key,value in attrs.items():
        setattr(obj,key,value)

def get_attrs(obj,attr_names):         # visszadja az objektum felsorolt attribútumait, dict formátumban
    '''
    obj:  osztálypéldány
    attr_names:  attribútum nevek listája vagy vesszős felsorolása
        - 'all' vagy 'üres':   az összes attribútum  (__dict__)
        - ha valamelyik attribútumnév érvénytelen, akkor nem kerül bele a dict-be (nincs hibaüzenet)
    '''
    if type(attr_names)==str:  attr_names = attr_names.split(',')
    result={}
    for attr_name in attr_names:
        try:
            result[attr_name]=getattr(obj,attr_name)
        except:
            pass
    return result

def Attr(obj,attr,default=None):             # a getattr python függvény exception nélküli változata
    try:
        return getattr(obj,attr)
    except:
        return default
    
def Attrs(obj,attr_names):          # tuple-ben adja vissza az attribútumokat
    if type(attr_names)==str:  attr_names = attr_names.split(',')
    result=[]
    for attr_name in attr_names:
        try:
            value=getattr(obj,attr_name)
            result.append(value)
        except:
            result.append(None)
    return tuple(result)


# print(Range('2020.01.01',add=5,count=5))
