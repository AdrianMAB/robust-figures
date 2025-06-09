# pip install pandas numpy matplotlib scipy seaborn statsmodels scikit-learn
import math
import glob,os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import calendar
from calendar import monthrange
import matplotlib.pyplot as plt
import subprocess
import os
import shutil
from math import ceil
from sklearn.metrics import r2_score

# Distribuciones de probabilidad
from scipy import stats
import seaborn as sns
#from fitter import Fitter
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import statsmodels.distributions as smd
from sklearn.neighbors import KernelDensity
#from distfit import distfit

import time
import signal
import multiprocessing

def filtra_ts(ts,fechas):
    columna=fechas[3]
    i=datetime.strptime(fechas[0],fechas[2])
    f=datetime.strptime(fechas[1],fechas[2])
    ts[columna]=pd.to_datetime(ts[columna], format=fechas[2])
    ts=ts.set_index(columna)
    ts.index.name="Fecha"
    filas=((ts.index>=i)&(ts.index<f))#.values
    ts=ts.iloc[filas]
    return ts

def cuantiles_frecuencias(arreglo,rangos):
    cuantiles=[]
    for i in range(rangos):
        cuantiles.append(np.quantile(arreglo,(i+1)/(rangos)))
    print(cuantiles)
    clases=[]
    for i in range(rangos):
        cl_bool=(arreglo<cuantiles[i])
        #clase=p_obs_lluvia[]
        clases.append(sum(cl_bool))
    for i in range(rangos):
        if i<(rangos-1):
            clases[rangos-i-1]=clases[rangos-i-1]-clases[rangos-i-2]
    print(clases)

# Frecuencia de días sin lluvia (0) o con lluvia (1).
def FrDay(DF,columna):
    LL=np.zeros([366,2])
    for d in range(len(DF)):
        día=DF.index[d].timetuple().tm_yday-1
        if DF.iloc[d,columna]<0.1:
            LL[día][0]=LL[día][0]+1
        else:
            LL[día][1]=LL[día][1]+1
    LL=pd.DataFrame(LL)
    return LL

# Regresa el evento más frecuente: no lluvia(0), lluvia(1)
def FrMax(DF):
    BY=[]
    for d in range(len(DF)):
        if DF[0][d]<DF[1][d]:
            BY.append(1)
        else:
            BY.append(0)
    return pd.DataFrame(BY)

# Devuelve la precipitación máxima.
def TrMxDay(DF,columna):
    MX=np.zeros([366,1])
    for d in range(len(DF)):
        día=DF.index[d].timetuple().tm_yday-1
        if DF.iloc[d,columna]>MX[día][0]:
            MX[día][0]=DF.iloc[d,columna]
    MX=pd.DataFrame(MX)
    return MX

# Devuelve la precipitación mínima.
def TrMinDay(DF,columna):
    MX=np.zeros([366,1])+10**6
    for d in range(len(DF)):
        día=DF.index[d].timetuple().tm_yday-1
        if DF.iloc[d,columna]<MX[día][0]:
            MX[día][0]=DF.iloc[d,columna]
    MX=pd.DataFrame(MX)
    return MX

# Devuelve la precipitación acumulada.
def TrAcDay(DF,columna):
    MX=np.zeros([366,1])
    for d in range(len(DF)):
        día=DF.index[d].timetuple().tm_yday-1
        MX[día][0]=MX[día][0]+DF.iloc[d,columna]
    MX=pd.DataFrame(MX)
    return MX

# Devuelve la precipitación media diaria.
def TrMDay(DF,columna):
    MX=np.zeros([366,1])
    conteo=0
    for d in range(len(DF)):
        día=DF.index[d].timetuple().tm_yday-1
        MX[día][0]=MX[día][0]+DF.iloc[d,columna]
        if día==0:
            conteo=conteo+1
    MX=pd.DataFrame(MX)/conteo
    return MX

# Simplifica la serie temporal en n rangos iguales.
def encode_prec(TS,n):
    promedio=np.mean(TS)
    nmedia=np.rint(TS/promedio/n)
    nmedia=nmedia/np.max(nmedia)
    return nmedia

def año_aleatorio(probabilidad_lluvia_dia,parámetros,modo):
    np.random.seed()
    a_lluvia = stats.bernoulli.rvs(probabilidad_lluvia_dia).astype(float)
    for i in range(len(a_lluvia)):
        if a_lluvia[i]==1:
            np.random.seed()
            pr = stats.beta.rvs(parámetros[0],parámetros[1],parámetros[2],parámetros[3])
            a_lluvia[i] = pr
            #if modo=="loglluvia":
            #    a_lluvia[i]= pr
            if modo=="alpha5lluvia":
                a_lluvia[i]=np.power(pr,5)
    return a_lluvia

def TS_aleatoria(TS_obs,st,Prob1,parámetros):
    serie_aleatoria=pd.DataFrame(TS_obs[st]).copy()
    pr_aleatoria=año_aleatorio(Prob1,parámetros)
    for d in range(len(serie_aleatoria.index)):
        día_actual=serie_aleatoria.index[d].timetuple().tm_yday-1
        if día_actual==0:
            pr_aleatoria=año_aleatorio(Prob1,parámetros)
        serie_aleatoria[st][d]=pr_aleatoria[día_actual]
    return serie_aleatoria

def extract_year(pdb,st,año):
    DF_ST=pd.DataFrame(pdb[st]).copy()
    DF_ST=DF_ST[DF_ST.index>=datetime(año,1,1)]
    DF_ST=DF_ST[DF_ST.index<datetime(año+1,1,1)]
    return DF_ST

def extract_period(pdb,st,año_i,año_f):
    DF_ST=pd.DataFrame(pdb[st]).copy()
    DF_ST=DF_ST[DF_ST.index>=datetime(año_i,1,1)]
    DF_ST=DF_ST[DF_ST.index<datetime(año_f+1,1,1)]
    return DF_ST

def frecuencias_discreta(lluvia):
    discreta=[]
    for i in range(len(lluvia)):
        F=lluvia[i]
        T=np.full(shape=F,fill_value=i+1,dtype=int)
        discreta=np.concatenate([discreta,T]).astype(int)
    return discreta

def frecuencias_discreta_index(lluvia):
    discreta=[]
    for i in (lluvia.index):
        F=lluvia[i]
        T=np.full(shape=F,fill_value=i+1,dtype=int)
        discreta=np.concatenate([discreta,T]).astype(int)
    return discreta

def comparar_PDF_CDF(pdb,st,ai,af,delta_t,tolerancia_nan,dist,modo,path):
    pr_periodo=extract_period(pdb,st,ai,af).dropna()
    discreta_per=sel_tabla_frecuencias(pr_periodo,modo)
    sns.kdeplot(discreta_per,label="Periodo")
    plt.legend()
    #par_per=stats.dgamma.fit(discreta_per)
    par_per=dist[0].fit(discreta_per)
    for a in range(ai,af,delta_t):
        pr_década=extract_period(pdb,st,a,a+delta_t-1)#.dropna()
        #nan_data=np.sum(pr_década.isna()*1)
        nan_data=(pr_década.isna()*1).sum().values[0]
        if nan_data<tolerancia_nan:
            pr_década=pr_década.dropna()
            lluvia=sel_tabla_frecuencias(pr_década,modo)
            sns.kdeplot(lluvia,label=str(a))
            plt.legend()
            print("Año "+str(a)+", prueba KS:")
            print("p="+str(stats.kstest(lluvia, dist[1], args=par_per).pvalue))
    muestra_dgama=dist[0](*par_per).rvs(size=len(discreta_per))
    plt.xlim([0,366])
    sns.kdeplot(muestra_dgama,label="Periodo "+dist[1])
    plt.legend()
    plt.title("Curvas PDF empíricas para la estación "+str(st)+", "+str(delta_t)+" años")
    plt.ylabel("Densidad de probabilidad")
    plt.xlabel("Día")
    if modo=="lluvia":
        plt.xlabel("Precipitación (mm)")
        plt.xlim(0,np.max(lluvia)*1.1)
    plt.savefig(path+"PDF_"+modo+"_"+str(delta_t)+"_"+st+".pdf")
    plt.show()
    for a in range(ai,af,delta_t):
        pr_década=extract_period(pdb,st,a,a+delta_t-1)#.dropna()
        nan_data=(pr_década.isna()*1).sum().values[0]
        if nan_data<tolerancia_nan:
            pr_década=pr_década.dropna()
            lluvia=sel_tabla_frecuencias(pr_década,modo)
            sns.ecdfplot(lluvia,label=str(a))
            plt.legend()
    sns.ecdfplot(discreta_per,label="Periodo")
    sns.ecdfplot(muestra_dgama,label="Periodo "+dist[1])
    plt.xlim([0,366])
    plt.title("Curvas CDF empíricas para la estación "+str(st)+", "+str(delta_t)+" años")
    plt.ylabel("Probabilidad acumulada")
    plt.xlabel("Día")
    if modo=="lluvia":
        plt.xlabel("Precipitación (mm)")
        plt.xlim(0,np.max(lluvia)*1.1)
    plt.legend()
    plt.savefig(path+"CDF_"+modo+"_"+str(delta_t)+"_"+st+".pdf")
    plt.show()

def iterate_pr(ai,af,delta_t,pdb,st,tolerancia_nan,modo):
    for a in range(ai,af,delta_t):
        pr_década=extract_period(pdb,st,a,a+delta_t-1)#.dropna()
        nan_data=(pr_década.isna()*1).sum().values[0]
        if nan_data<tolerancia_nan:
            pr_década=pr_década.dropna()
            lluvia=sel_tabla_frecuencias(pr_década,modo)
            sns.kdeplot(lluvia,label=str(a))
            plt.legend()

def iterate_pr_slide(ai,af,delta_t,pdb,st,tolerancia_nan,modo,tcurva):
    for a in range(ai,af):
        pr_década=extract_period(pdb,st,a,a+delta_t-1)#.dropna()
        nan_data=(pr_década.isna()*1).sum().values[0]
        if nan_data<tolerancia_nan:
            pr_década=pr_década.dropna()
            lluvia=sel_tabla_frecuencias(pr_década,modo)
            if tcurva=="pdf":
                sns.kdeplot(lluvia,label=str(a))
            elif tcurva=="cdf":
                sns.ecdfplot(lluvia,label=str(a))
            plt.legend()

def plot_PDF_CDF(pdb,st,ai,af,delta_t,tolerancia_nan,dist,modo,path):
    pr_periodo=extract_period(pdb,st,ai,af).dropna()
    discreta_per=sel_tabla_frecuencias(pr_periodo,modo)
    sns.kdeplot(discreta_per,label="Periodo")
    plt.legend()
    par_per=dist[0].fit(discreta_per)
    #iterate_pr(ai,af,delta_t,pdb,st,tolerancia_nan,modo)
    iterate_pr_slide(ai,af,delta_t,pdb,st,tolerancia_nan,modo,"pdf")
    muestra_dgama=dist[0](*par_per).rvs(size=len(discreta_per))
    plt.xlim([0,366])
    sns.kdeplot(muestra_dgama,label="Periodo "+dist[1])
    plt.legend()
    plt.title("Curvas PDF empíricas para la estación "+str(st)+", "+str(delta_t)+" años")
    plt.ylabel("Densidad de probabilidad")
    plt.xlabel("Día")
    if modo=="lluvia":
        plt.xlabel("Precipitación (mm)")
        plt.xlim(0,np.max(pr_periodo.values.flatten()))
    if modo=="loglluvia":
        plt.xlabel("Logaritmo de la precipitación")
        plt.xlim(0,np.log(np.max(pr_periodo.values.flatten())))
    plt.savefig(path+"PDF_"+modo+"_"+str(delta_t)+"_"+st+".pdf")
    plt.clf()
    ###########
    # CDF
    ###########
    #iterate_pr(ai,af,delta_t,pdb,st,tolerancia_nan,modo)
    iterate_pr_slide(ai,af,delta_t,pdb,st,tolerancia_nan,modo,"cdf")
    sns.ecdfplot(discreta_per,label="Periodo")
    sns.ecdfplot(muestra_dgama,label="Periodo "+dist[1])
    plt.xlim([0,366])
    plt.title("Curvas CDF empíricas para la estación "+str(st)+", "+str(delta_t)+" años")
    plt.ylabel("Probabilidad acumulada")
    plt.xlabel("Día")
    if modo=="lluvia":
        plt.xlabel("Precipitación (mm)")
        plt.xlim(0,np.max(pr_periodo.values.flatten()))
    if modo=="loglluvia":
        plt.xlabel("Logaritmo de la precipitación")
        plt.xlim(0,np.log(np.max(pr_periodo.values.flatten())))
    plt.legend()
    plt.savefig(path+"CDF_"+modo+"_"+str(delta_t)+"_"+st+".pdf")
    plt.clf()


def sel_tabla_frecuencias(DF,modo):
    if modo=="lluvia_disc":
        return frecuencias_discreta(FrDay(DF,0)[1].astype(int))
    elif modo=="lluvia_disc_norm":
        return frecuencias_discreta(FrDay(DF,0)[1].astype(int))/365
    elif modo=="no_lluvia_disc":
        return frecuencias_discreta(FrDay(DF,0)[0].astype(int))
    elif modo=="acumulada_disc":
        return frecuencias_discreta(TrAcDay(DF,0)[0].astype(int))
    elif modo=="media_disc":
        return frecuencias_discreta(TrMDay(DF,0)[0].astype(int))
    elif modo=="max_disc":
        return frecuencias_discreta(TrMxDay(DF,0)[0].astype(int))
    elif modo=="lluvia":
        return DF[DF>0].dropna().values.flatten()
    elif modo=="loglluvia":
        lluvia=DF[DF>0].dropna().values.flatten()
        return np.log(lluvia)
    elif modo=="alpha5lluvia":
        lluvia=DF[DF>0].dropna().values.flatten()
        return np.power(lluvia,1/5)
    elif modo=="logacumulada_disc":
        acumulada=TrAcDay(DF,0)[0]
        acumulada[acumulada==0]
        acumulada=(np.log(acumulada[acumulada>=1])*100).astype(int)
        return frecuencias_discreta_index(acumulada)
    
def extract_dist_par(pdb,st,ai,af,delta_t,tolerancia_nan,dist,modo):
    P=[]
    for a in range(ai,af):
        pr_década=extract_period(pdb,st,a,a+delta_t-1)#.dropna()
        nan_data=(pr_década.isna()*1).sum().values[0]
        if nan_data<tolerancia_nan:
            pr_década=pr_década.dropna()
            lluvia=sel_tabla_frecuencias(pr_década,modo)
            if len(lluvia)>0:
                par_per=dist[0].fit(lluvia)
                P.append((a,)+par_per)
                #print(par_per)
    if len(P)>0:
        P=pd.DataFrame(P)
        P.columns=["Año","A","B","C","D"]
        P=P.set_index("Año")
    return P

def plot_hm(DF_coef,max,var_name,out_dir):
    plt.figure()
    plot=sns.heatmap(DF_coef,vmin=0,vmax=max)
    plot.set(title="Variation of "+var_name+" coefficient",xlabel="Station number", ylabel="Year")
    plt.savefig(out_dir+"heatmap"+var_name+".pdf",bbox_inches='tight')
    #plot.figure.clf()

def sort_by_axis(par_prob,coordenadas,columna):
    st_list=coordenadas.sort_values(by=columna)["Nombre"]
    st_list=[str(st_i) for st_i in st_list]
    return par_prob[list(set(st_list) & set(list(par_prob)))]

def filtra_st_coordenadas(parámetros,coord,eje):
    estaciones=[]
    st_list=coord.sort_values(by=eje)["Nombre"]
    st_list=list(map(str,st_list))
    st_par=list(parámetros)
    for st in st_list:
        if st in st_par:
            estaciones.append(st)
    return parámetros[estaciones]

def calculate_map(coordenadas,eje,pdb,ai,af,delta,tolerancia,dist,var,param_max,out_dir):
    TA=pd.DataFrame()
    TB=pd.DataFrame()
    TC=pd.DataFrame()
    TD=pd.DataFrame()
    for st in list(pdb):
        parametros_slide=extract_dist_par(pdb,st,ai,af,delta,tolerancia,dist,var)
        if len(parametros_slide)>0:
            TA[st]=parametros_slide.A
            TB[st]=parametros_slide.B
            TC[st]=parametros_slide.C
            TD[st]=parametros_slide.D
    TA=filtra_st_coordenadas(TA,coordenadas,eje)
    TB=filtra_st_coordenadas(TB,coordenadas,eje)
    TC=filtra_st_coordenadas(TC,coordenadas,eje)
    TD=filtra_st_coordenadas(TD,coordenadas,eje)
    print(list(TA))
    print(list(TB))
    print(list(TC))
    print(list(TD))
    pltA=plot_hm(TA,param_max,"A",out_dir)
    pltB=plot_hm(TB,param_max,"B",out_dir)
    pltC=plot_hm(TC,param_max,"C",out_dir)
    pltD=plot_hm(TD,param_max,"D",out_dir)

def tabla_par_st(pdb,ai,af,delta,tolerancia,dist,modo):
    TA=pd.DataFrame()
    TB=pd.DataFrame()
    TC=pd.DataFrame()
    TD=pd.DataFrame()
    for st in list(pdb):
        parametros_slide=extract_dist_par(pdb,st,ai,af,delta,tolerancia,dist,modo)
        if len(parametros_slide)>0:
            TA[st]=parametros_slide.A
            TB[st]=parametros_slide.B
            TC[st]=parametros_slide.C
            TD[st]=parametros_slide.D
    return [TA,TB,TC,TD]

def tab_mmm(DF,lim_min,lim_max):
    tmp=DF.copy()
    min_max=pd.DataFrame()
    tmp[tmp<lim_min]=np.nan
    tmp[tmp>lim_max]=np.nan
    min_max["Mínimo"]=DF.min()
    min_max["Promedio"]=DF.mean()
    min_max["Máximo"]=DF.max()
    return min_max

def tabla_par_max_min_st(pdb,ai,af,delta,tolerancia,dist,modo,lim_min,lim_max):
    DF_par=tabla_par_st(pdb,ai,af,delta,tolerancia,dist,modo)
    print(DF_par)
    par_min_max=[tab_mmm(DF_par[0],lim_min,lim_max),
             tab_mmm(DF_par[1],lim_min,lim_max),
             tab_mmm(DF_par[2],lim_min,lim_max),
             tab_mmm(DF_par[3],lim_min,lim_max)]
    return par_min_max

def tabla_prob(pdb,ai,af,delta,tolerancia,dist,modo,lim_min,lim_max):
    TA=pd.DataFrame()
    TB=pd.DataFrame()
    TC=pd.DataFrame()
    TD=pd.DataFrame()
    for st in list(pdb):
        parametros_slide=extract_dist_par(pdb,st,ai,af,delta,tolerancia,dist,modo)
        if len(parametros_slide)>0:
            TA[st]=parametros_slide.A
            TB[st]=parametros_slide.B
            TC[st]=parametros_slide.C
            TD[st]=parametros_slide.D
    MA=tab_mmm(TA,lim_min,lim_max)
    MB=tab_mmm(TB,lim_min,lim_max)
    MC=tab_mmm(TC,lim_min,lim_max)
    MD=tab_mmm(TD,lim_min,lim_max)
    par_min_max=[MA,MB,MC,MD]
    tablas_par=[TA,TB,TC,TD]
    return [par_min_max,tablas_par]

def dist_discreta_fr_day(pdb,st):
    analizar_st=pd.DataFrame(pdb[st])
    año_inicial=analizar_st.index[0].year
    año_final=analizar_st.index[len(analizar_st)-1].year
    lluvia_total=pd.DataFrame(np.zeros([366,2])).astype(int)
    for st in list(pdb):
        for a in range(año_inicial,año_final+1):
            año_a=extract_year(pdb,st,a)
            if año_a.isna().sum().iloc[0]==0:
                frecuencias=FrDay(año_a,0).astype(int)
                lluvia_total=frecuencias+lluvia_total
    return lluvia_total

def dist_discreta_fr_day_by_st(pdb,st):
    analizar_st=pd.DataFrame(pdb[st])
    año_inicial=analizar_st.index[0].year
    año_final=analizar_st.index[len(analizar_st)-1].year
    lluvia_total=pd.DataFrame(np.zeros([366,2])).astype(int)
    for a in range(año_inicial,año_final+1):
        año_a=extract_year(pdb,st,a)
        if año_a.isna().sum().iloc[0]==0:
            frecuencias=FrDay(año_a,0).astype(int)
            lluvia_total=frecuencias+lluvia_total
    return lluvia_total

def ser_al(ai,af,Prob1,parámetros,modo):
    serie_aleatoria=pd.date_range(start=str(ai)+'-01-01', end=str(af)+'-12-31').to_frame(index=True)
    serie_aleatoria[0]=np.zeros(len(serie_aleatoria))
    pr_aleatoria=año_aleatorio(Prob1,parámetros,modo)
    for d in range(len(serie_aleatoria.index)):
        día=serie_aleatoria.index[d]
        día_actual=día.timetuple().tm_yday-1
        if día_actual==0:
            pr_aleatoria=año_aleatorio(Prob1,parámetros,modo)
        serie_aleatoria.iloc[d,0]=pr_aleatoria[día_actual].round(1)
    return serie_aleatoria

def input_random(ai,af,estaciones,par_min_max,param_tipo,prob_lluvia_total,modo):
    random=pd.DataFrame()
    for st in estaciones:
        if (st in par_min_max[0].index):
            params=[par_min_max[0][param_tipo][st],
                par_min_max[1][param_tipo][st],
                par_min_max[2][param_tipo][st],
                par_min_max[3][param_tipo][st]]
            # Descarta distribuciones con valores fuera del rango.
            if np.isnan(params).any()==False:
                serie_sintetica=ser_al(ai,af,prob_lluvia_total,params,modo)
                serie_sintetica.columns=[st]
                random[st]=serie_sintetica
                #print(st)
    return random

def lluvias_acumuladas(i,pr_rnd):
    tabla_mx=pr_rnd.max()
    LLMX=tabla_mx.max()
    # 2 Estación con la lluvia máxima en un día. STMX
    STMX=tabla_mx.index[list(tabla_mx).index(LLMX)]
    # 3 Precipitación acumulada mínima anual. PAMIN
    pr_acumulada_anual=pr_rnd.resample("YE").sum()
    PAMIN=pr_acumulada_anual.min()
    # 4 Precipitación acumulada media anual. PAMED
    PAMED=pr_acumulada_anual.mean()
    # 5 Precipitación acumulada máxima anual. PAMAX
    PAMAX=pr_acumulada_anual.max()
    PAM=np.concatenate([PAMIN.values,PAMED.round(1).values,PAMAX.values])
    # 6 Precipitación acumulada media anual, todas las estaciones, por año.
    PRAC=pr_acumulada_anual.T.mean().values.round(1)
    STATS=list([i,LLMX,STMX])+list(PAM)+list(PRAC)
    return STATS

def resumen_series_aleatorias(inputs,nombreCSV):
    S=list()
    pr_rnd=pd.read_csv(inputs[0],index_col=0,parse_dates=True)
    sts=pr_rnd.columns
    columnas=list(["INPUT","LLMX","STMX"])
    columnas=columnas+list(sts+"ac_min")+list(sts+"ac_med")+list(sts+"ac_max")
    columnas=columnas+list(pr_rnd.resample("YE").sum().index.year)
    for i in inputs:
        pr_rnd=pd.read_csv(i,index_col=0,parse_dates=True)
        S.append(lluvias_acumuladas(i,pr_rnd))
    df=pd.DataFrame(S[1:],columns=columnas)
    df.to_csv(nombreCSV)
    return df

def plot_correl(x,y,xtitle,ytitle,salida_pdf,ticks_fix=[]):
  # r Pearson
  #r=np.corrcoef(x,y)
  # r2 Coeficiente de determinación
  r2=r2_score(x,y)
  redondealen=5
  maximo=np.max([np.max(x),np.max(y)])
  if len(ticks_fix)==0:
      lensq=ceil(maximo/redondealen)*redondealen
  else:
      lensq=np.max(ticks_fix)
  plt.figure(figsize=(5, 5))
  #plt.axis('square')
  plt.xlim([0,lensq])
  plt.ylim([0,lensq])
  plt.gca().set_xbound(0, lensq)
  plt.gca().set_ybound(0, lensq)
  plt.gca().set_aspect('equal')
  if len(ticks_fix)>0:
    plt.xticks(ticks_fix)
    plt.yticks(ticks_fix)
  #plt.gca().set_box_aspect(1)
  box_style=dict(boxstyle='round', facecolor='white', alpha=1.0)
  plt.rc('font', size=14)
  sepx=0.65
  sepy=0.1
  plt.text(sepx * lensq, sepy * lensq, "r$^2$="+str(round(r2,4)),{'weight':'heavy','size':14},bbox=box_style)
  plt.grid(color='gray', linestyle='--', linewidth=0.5)
  plt.ylabel(ytitle)
  plt.xlabel(xtitle)
  # Línea de tendencia
  z = np.polyfit(x, y, 1)
  p = np.poly1d(z)
  plt.plot(x,p(x),"r--")
  # Gráficar los puntos
  plt.scatter(x,y,color="blue")
  plt.savefig(salida_pdf,bbox_inches='tight')
  plt.clf()

def plot_dist_par(pdb,st,ai,af,delta_t,tolerancia_nan,distrib,modo,path_output,extension_file,size_fix=0):
    for a in range(ai,af):
        pr_década=extract_period(pdb,st,a,a+delta_t-1)#.dropna()
        nan_data=(pr_década.isna()*1).sum().values[0]
        if nan_data<tolerancia_nan:
            pr_década=pr_década.dropna()
            lluvia=sel_tabla_frecuencias(pr_década,modo)
            if len(lluvia)>0:
                # Ajusta los datos
                par_per=distrib[0].fit(lluvia)
                # Excluye los periodos que no se ajustaron correctamente.
                if len(par_per)>0:
                    sort_lluvia=sorted(lluvia)
                    #percentiles = np.linspace(0, 1, len(sort_lluvia))
                    incremento=0.0001
                    #percentiles = np.linspace(0+incremento, 1-incremento, len(sort_lluvia))
                    # Se añade un elemento para evitar que sea 1.
                    percentiles = np.linspace(0+incremento, 1, len(sort_lluvia)+1)
                    # Extraer los datos ajustados
                    # Se elimina el último elemento de los percentiles.
                    lluviadist=distrib[0].ppf(percentiles[:-1],*par_per)
                    # Revierte la transformación
                    if modo=="alpha5lluvia":
                        for i in range(len(sort_lluvia)):
                            sort_lluvia[i]=np.power(sort_lluvia[i],5)
                        for i in range(len(lluviadist)):
                            lluviadist[i]=np.power(lluviadist[i],5)
                    plot_correl(lluviadist,
                            sort_lluvia,
                            "Fitted precipitation (mm)",
                            "Observed precipitation (mm)",
                            path_output+st+"_"+distrib[1]+"_"+str(a)+"cor."+extension_file,
                            size_fix
                    )
def calcula_ticks(valor_máximo,espacios,redondeo):
  tick=ceil(ceil(valor_máximo/espacios)/redondeo)*redondeo
  return np.linspace(0,tick*espacios,espacios+1)

def r_dist_par(pdb,st,ai,af,delta_t,tolerancia_nan,distrib,modo):
    highest_r2 = float('-inf')
    lowest_r2 = float('inf')
    a_max=0
    a_min=0
    for a in range(ai,af):
        pr_década=extract_period(pdb,st,a,a+delta_t-1)#.dropna()
        nan_data=(pr_década.isna()*1).sum().values[0]
        if nan_data<tolerancia_nan:
            pr_década=pr_década.dropna()
            lluvia=sel_tabla_frecuencias(pr_década,modo)
            if len(lluvia)>0:
                # Ajusta los datos
                par_per=distrib[0].fit(lluvia)
                # Excluye los periodos que no se ajustaron correctamente.
                if len(par_per)>0:
                    sort_lluvia=sorted(lluvia)
                    #percentiles = np.linspace(0, 1, len(sort_lluvia))
                    incremento=0.0001
                    #percentiles = np.linspace(0+incremento, 1-incremento, len(sort_lluvia))
                    # Se añade un elemento para evitar que sea 1.
                    percentiles = np.linspace(0+incremento, 1, len(sort_lluvia)+1)
                    # Extraer los datos ajustados
                    # Se elimina el último elemento de los percentiles.
                    lluviadist=distrib[0].ppf(percentiles[:-1],*par_per)
                    # Revierte la transformación
                    if modo=="alpha5lluvia":
                        for i in range(len(sort_lluvia)):
                            sort_lluvia[i]=np.power(sort_lluvia[i],5)
                        for i in range(len(lluviadist)):
                            lluviadist[i]=np.power(lluviadist[i],5)
                    r2=r2_score(lluviadist,sort_lluvia)
                    new_max=max(highest_r2, r2)
                    new_min=min(lowest_r2, r2)
                    if new_max>highest_r2:
                        highest_r2 = new_max
                        a_max=a
                    if new_min<lowest_r2:
                        lowest_r2 = new_min
                        a_min=a
    return st,a_min,lowest_r2,a_max,highest_r2
