# Partner- és címkezelési algortimusok
from ezhelper import *
from ezcharts import *
import pandas as pd
import numpy as np

import mysql.connector

def fn_mydb(database,host="localhost",user="root",password="SYSADM"):
    '''
    database:  pl. 'mkik20220214'
    '''
    return mysql.connector.connect(host=host,user=user,password=password,database=database)


def Codelist_fromdb(database,tblname,code='kod',felirat='felirat'):
    '''
    Beolvasás mysql adatbázisból

    code:  a kódmező neve   (lehet szám vagy string-mező is)
    felirat:  a felirat-mező neve   (string mező)
    '''
    
    mydb = fn_mydb(database=database)
    mycursor = mydb.cursor()
    mycursor.execute('select  ' + code + ', ' + felirat + 
        ' from ' + tblname)
    recs=mycursor.fetchall()
    mydb.close()
    return SerFromRecords(recs)


# print(Codelist_fromdb('mkik20220214','DICT_TEAOR_AGAZATOK','kod','leiras'))

