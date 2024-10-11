#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 10:45:00 2023

@author: sigridtragenap
"""


import matplotlib as mpl

#%% plotting settings
def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

params = {
   'axes.labelsize': 7,
   'axes.spines.top'  : False ,
   'axes.spines.right'  : False ,
   'axes.linewidth' : 0.5,
   'axes.unicode_minus' : False,
   'font.size': 7,
   'legend.fontsize': 7,
   "legend.title_fontsize" : 7,
   'xtick.major.top'      : False,
   'ytick.major.right'      : False,
    'xtick.labelsize'      : 7 ,
    'ytick.labelsize'     : 7 ,
   'figure.figsize': [2,2],
   'lines.linewidth' : 0.5,
   #'errorbar.capsize' : 10,
'mathtext.fontset' : 'cm',
"figure.subplot.left"    : 0.35 , # the left side of the subplots of the figure
"figure.subplot.right"   : 0.9   , # the right side of the subplots of the figure
"figure.subplot.bottom"   : 0.25  ,  # the bottom of the subplots of the figure
"figure.subplot.top"     : 0.97 ,
    "figure.dpi" : 200 ,
   'xtick.major.pad' : 1,
   'ytick.major.pad' : 1,
    'xtick.major.width':   0.5,
   'xtick.major.size':   1.5,
    'ytick.major.width':   0.5,
   'ytick.major.size':   1.5,
    'pdf.fonttype': 42,
    'pdf.use14corefonts'  : True
}
#Matplotlib.rcParams[‘pdf.- fonttype’]=42
mpl.rcParams.update(params)

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica']
mpl.rcParams['figure.dpi'] = 200

color_r="C2"
color_al='b'
color_im='C2'
