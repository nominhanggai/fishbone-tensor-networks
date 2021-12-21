# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from matplotlib import pyplot as plt
import matplotlib as mpl
import itertools as it
import numpy as np
from os import path

mpl.style.use('default')
# prop_cycle = plt.rcParams['axes.prop_cycle']
# colors = prop_cycle.by_key()['color']
colors = ['#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E', '#8EBA42', '#FFB5B8']
colors_mma_scien = [
    (0.9, 0.36, 0.054),
    (0.365248, 0.427802, 0.758297),
    (0.945109, 0.593901, 0.0),
    (0.645957, 0.253192, 0.685109),
    (0.285821, 0.56, 0.450773),
    (0.7, 0.336, 0.0),
    (0.491486, 0.345109, 0.8),
    (0.71788, 0.568653, 0.0),
    (0.70743, 0.224, 0.542415),
    (0.287228, 0.490217, 0.664674),
    (0.982289285128704, 0.5771321368979874, 0.011542503255145636),
    (0.5876740325800278, 0.2877284499870081, 0.7500695697462922),
    (0.4262088601796793, 0.5581552810007578, 0.2777996730417023),
    (0.9431487543762861, 0.414555896337833, 0.07140829055870854)
    ]

colors_mma_detai = [(0.368417, 0.506779, 0.709798), (0.880722, 0.611041, 0.142051), (0.560181, 0.691569, 0.194885), (0.922526, 0.385626, 0.209179), (0.528488, 0.470624, 0.701351), (0.772079, 0.431554, 0.102387), (0.363898, 0.618501, 0.782349), (0.647624, 0.37816, 0.614037), (0.571589, 0.586483, 0.0), (0.915, 0.3325, 0.2125), (0.9728288904374106, 0.621644452187053, 0.07336199581899142), (0.736782672705901, 0.358, 0.5030266573755369)]

font = {'family' : 'Latin Modern Sans',
        'size'   : 10}
mpl.rc('font', **font)
mpl.rcParams['text.usetex'] = True
    
wid = 3.375
fig_pop = plt.figure(dpi=800,figsize=(wid,wid*0.7))
gs = fig_pop.add_gridspec(1, 1, hspace=0.001, wspace=0.001)
grid = gs.subplots(sharex='col', sharey='row',)    
b = np.fromfile('./dk.dat', np.float32).reshape(101,200).T
b = b[:, ::20]
print(b.shape)
grid.set_ylim([0,150])
grid.set_xlim([80,200])
colors_pop = colors_mma_detai*100
grid.set_prop_cycle(color = colors_pop)
grid.text(0.5,-0.2, f'Bath Mode Index',
                       transform=grid.transAxes,ha='center',usetex=False,fontsize=10)
grid.text(-0.17,0.1, 'Interaction Stregnth $|d_n(t)|$',
                       transform=grid.transAxes,ha='left',usetex=False,fontsize=10,rotation=90)
grid.plot(b,lw=1.8,solid_capstyle='round',dash_capstyle='round',dash_joinstyle='bevel')
grid.tick_params(axis="y",direction="in", pad=4)
grid.tick_params(axis="x",direction="in", pad=4)
fig_pop.subplots_adjust(left=0.17,right=0.97,bottom=0.2,top=0.98)

legends = [r'$t\Delta/\pi=0$', r'$t\Delta/\pi=1$', r'$t\Delta/\pi=2$', r'$t\Delta/\pi=3$', r'$t\Delta/\pi=4$', r'$t\Delta/\pi=5$']
grid.legend(list(legends),loc=(0.03, 0.3),ncol=1,frameon=False,
              handlelength=1.2,columnspacing=0.3,handletextpad=0.2,
              fontsize=10,borderaxespad=0)
fig_pop.savefig(f'dk.pdf')
