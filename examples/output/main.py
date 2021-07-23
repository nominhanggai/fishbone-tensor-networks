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
def pop_grid(freq, coup=None,thres='0001',phys_d=[10,40,60],geom = ['ic','c','s'],ylim=[0.3,1], direc = 'whole', tmp=None,):   
    
   

    wid = 3.375
    fig_pop = plt.figure(dpi=800,figsize=(wid,wid*0.8))
    tl = len(tmp)
    cl = len(coup)
    gs = fig_pop.add_gridspec(cl, tl, hspace=0.001, wspace=0.001)
    grid = gs.subplots(sharex='col', sharey='row')
    y_ticks = [[0.3,0.5,0.7,0.9],[0.3,0.5,0.7,0.9,1.0],[0.3,0.5,0.7,0.9],[0.3,0.5,0.7,0.9]]
    x_ticks = [[0, 2.5],[0,2.5,5]]
    line_geom = {'ic': 'solid', 'c': (0,(0.001,1.5)), 's': (0,(1.2,1.2)), 'is': (0, (2.5, 2, 2.5, 1))}
    for m,n in it.product(range(cl),range(tl)):

        pop=[]
        legends = []
        linestyle_str = []
        for i, pd in it.product(geom, phys_d[cl*m+n]):
            file = f'./{direc}/pop_{i}_{freq}_{coup[m]}_{tmp[n]}_{pd}_{thres}.dat'
            if path.exists(file):
                b = np.fromfile(file, np.float32)
                b = np.insert(b, 0, 1.0)
                b = np.array([x/2+.5 for x in b])
                b = ([i*0.05 for i in range(101)],b)
                pop.append((b))
                print(i.upper(), pd, 'right')
                legends.append(i.upper() + f'{pd}')
                linestyle_str.append(line_geom[i])
            else:
                print(i.upper(), pd, 'not exist',file)
                
        linestyle_str = linestyle_str*len(colors_mma_detai)
        colors_pop = colors_mma_detai*len(pop)
        
        grid[m,n].tick_params(axis="y",direction="in", pad=4)
        grid[m,n].tick_params(axis="x",direction="in", pad=4)
        grid[m,n].set_prop_cycle(color = colors_pop,linestyle=linestyle_str)
        grid[m,n].set_xlim([0,5])
        grid[m,n].set_ylim(ylim)
        grid[m,n].text(0.98,0.06, f'$\eta_0={coup[m]}$, $\omega_0={freq}$, $T_0={tmp[n]}$',
                       transform=grid[m,n].transAxes,ha='right',usetex=True,fontsize=6)
        # grid[m,n].grid(False)
        # grid[m,n].text(.1,.8, geom[m].upper()+str(phys_d[n]),
        # transform=grid[m,n].transAxes)
        grid[m,n].set_yticks(y_ticks[cl*m+n])
        grid[m,n].set_xticks(x_ticks[n])
        # grid[m,n].set_yticks(ytick[m])
        for cur in pop: 
            grid[m,n].plot(*cur,lw=1.8,solid_capstyle='round',dash_capstyle='round',dash_joinstyle='bevel')
        
        # [grid[m,n].set_linewidth(2) for i in grid[m,n].spines.values()]
        grid[m,n].legend(list(legends),loc="upper right",ncol=4,frameon=False,
              handlelength=1.2,columnspacing=0.3,handletextpad=0.2,
              fontsize=5.5,borderaxespad=0)
    fig_pop.text(0,0.5, r'$\rho_{\uparrow}$', ha='left',usetex=True,fontsize=10)
    fig_pop.text(0.55,0, r'$t\Delta/\pi$', ha='center',usetex=True,fontsize=10)
    fig_pop.subplots_adjust(left=0.12,right=0.97,bottom=0.12,top=0.97)
        
    
    # ax.set_aspect(aspect=70,adjustable='box')

    # ax.set_xlabel('$t\Delta/\pi$',usetex=True)
    # [i.set_linewidth(2) for i in ax.spines.values()]
    # ax.set_xlim([0,5])
    # ax.set_ylim(ylim)
    # ax.set_yticks(np.arange(0,1.0,0.1))

    # ax.text(0.98,0.06, f'$\eta_0={coup}$, $\omega_0={freq_}$, $T={tmp_}$',
                       # transform=ax.transAxes,ha='right',usetex=True,fontsize=10)
    # ax.set_xlabel('$\Delta t$',usetex=True)
    
    # fig_pop.tight_layout(w_pad=0,pad=0,h_pad=0,rect=[0,0,1,1])
    # fig_pop.subplots_adjust(left=0,right=1,bottom=0,top=1)
    # ax.set_axis_off()
    
    plt.savefig(f'whole_{freq}_{thres}.pdf')#,bbox_inches='tight',pad_inches=0)
# [0.25, 1.0, 4.0]
# for i, freq, tmp in it.product([0.5], [1.0], ['1e-08', 2.0, 4.0, 6.0, 8.0]):
#     pop(i,ylim=[-0.6,1],thres='0001',direc='et',freq='_0.25', tmp=f'_{tmp}', phys_d=[10,20,40,60])
#     bond(i,ylim=[-0.6,1],thres='0001',direc='et',freq='_0.25', tmp=f'_{tmp}', head='num', phys_d=[10,20,40,60])

# for i, freq, tmp in it.product([2.0], [0.25], [4.0]):
#     phys_d = [10,20,40,60,80,100]
#     # phys_d = [10,80,100]
#     geom = ['ic', 'c', 's']
#     pop(i,ylim=[0.0,1],thres='0001',direc='whole',freq=f'_{freq}', tmp=f'_{tmp}', phys_d=phys_d,geom=geom)
#     pop(i,ylim=[0.0,1],thres='00001',direc='whole',freq=f'_{freq}', tmp=f'_{tmp}', phys_d=phys_d)
#     # occu(i,thres='0001',direc='whole',freq=f'_{freq}', tmp=f'_{tmp}', phys_d=phys_d)
#     # occu(i,thres='00001',direc='whole',freq=f'_{freq}', tmp=f'_{tmp}', phys_d=phys_d)
#     # bond(i,ylim=[0.4,1],thres='0001',direc='whole',freq=f'_{freq}', tmp=f'_{tmp}', head='hm', phys_d=phys_d,geom=geom)
#     # bond(i,ylim=[-0.6,1],thres='00001',direc='whole',freq=f'_{freq}', tmp=f'_{tmp}', head='hm', phys_d=phys_d)




# import matplotlib.font_manager

# print(
# sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist]
# ))
