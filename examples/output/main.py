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

    
def occu(coup,thres='0001',phys_d=[10,20,40,60,80],geom = ['ic','c','s','is'],ylim=None,
        freq='', direc = 'data1', tmp=''):
    occu = []
    legends = []
    linestyle_str = []
    line_geom = {'ic': 'solid', 'c': 'dotted', 's': 'dashed', 'is': (0, (2.5, 2, 2.5, 1))}
    # legends = it.product(['IC','C','S','IS'], phys_d[0:4])
    # legends = [a + f'{b}' for a, b in legends]
    for i, pd in it.product(geom, phys_d):
        file = f'./{direc}/num_{i}{freq}_{coup}{tmp}_{pd}_{thres}.dat'
        if path.exists(file):
            b = np.fromfile(file, np.float32)
            b = np.sum(b.reshape(100,200), axis=1)
            print(len(b))
            occu.append(b)
            print(i.upper(), pd, 'right')
            legends.append(i.upper() + f'{pd}')
            linestyle_str.append(line_geom[i])
        else:
            print(i.upper(), pd, 'not exist',file)


    
    # linestyle_str = ['solid']*k + ['dotted']*k + ['dashed']*k + [(0, (2.5, 2, 2.5, 1))]*k+[(1, (3, 10, 1, 1))]*k
    linestyle_str = linestyle_str*len(colors_mma_detai)
    colors_pop = colors_mma_detai*len(occu)
    
    fig_pop = plt.figure(dpi=800,figsize=(3.41667,3.41667*0.7))
    ax = fig_pop.add_subplot(111)
    
    ax.set_prop_cycle(color = colors_pop,linestyle=linestyle_str)
    ax.plot(np.array(occu).T,lw=1)
    # ax.set_aspect(aspect=70,adjustable='box')
    ax.tick_params(axis="y",direction="in", pad=4,width=2)
    ax.tick_params(axis="x",direction="in", pad=4,width=2)
    
    [i.set_linewidth(2) for i in ax.spines.values()]
    ax.set_xlim([0,100])
    ax.set_ylim(ylim)
    # ax.set_yticks(np.arange(0,1.0,0.1))
    freq_ = freq.replace('_','')
    tmp_ = tmp.replace('_','')
    ax.text(.5,1.05, f'$\eta_0={coup}$ $\omega_0={freq_}$ $T={tmp_}$',
                       transform=ax.transAxes,ha='center',usetex=True)
    # ax.set_xlabel('$\Delta t$',usetex=True)
    ax.legend(list(legends),loc="upper left",ncol=4,frameon=False,
              handlelength=1.2,columnspacing=0.3,handletextpad=0.2,
              fontsize=5,borderaxespad=0)
    fig_pop.tight_layout(w_pad=0,pad=0,h_pad=0,rect=[0,0,1,1])
    # fig_pop.subplots_adjust(left=0,right=1,bottom=0,top=1)
    # ax.set_axis_off()
    plt.savefig(f'occu{freq}_{coup}_{tmp_}_{thres}.pdf')



def bond(coup,thres='0001',phys_d=[10,20,40,60,80],geom = ['ic','c','s'],ylim=[0,1],
        freq='', tmp='', direc = 'data1',head='pop',spacing=5):
    a = []
    gl = len(geom)
    pl = len(phys_d)
    x_ticks = [[0,100],[0,100],[0,100, 200]]*3
    y_ticks = [[50,100,150,200,250],[50, 150, 500, 1000],[0,500, 1000]]*3
    for i, pd in it.product(geom, phys_d):
        file = f'./{direc}/{head}_{i}_{freq}_{coup}_{tmp}_{pd}_{thres}.dat'
        if path.exists(file):
            b = np.fromfile(file, np.float32)
            a.append(b.reshape(100,201).T)
            print(i, pd, 'right')
        else:
            print(i.upper(), pd, 'not exist',file)
            a.append(np.full([100,201],None).T)
    wid = 3.375
    fig = plt.figure(dpi=800,figsize=(wid,wid))
    # fig.text(0.5, 0.04, 'common X', ha='center')
    fig.text(-0.0, 0.52, 'Bond Dimension', va='center', rotation='vertical')
    fig.text(0.5, 0.0, 'Bond Index', ha='center')
    
    gs = fig.add_gridspec(gl, pl, hspace=0., wspace=0.)
    grid = gs.subplots(sharex='col', sharey='row')
    fig.suptitle(f'$\eta_0={coup}$, $\omega_0={freq}$, $T_0={tmp}$  ',usetex=True)
    
    
    lw = [0.8]*20
    # ytick = [range(0,51,10),range(0,299,70),range(0,801,200),range(0,801,200)]
    for m,n in it.product(range(gl),range(pl)):
        print(geom[m], phys_d[n])
        print(pl*m+n)
        grid[m,n].text(.1,.8, geom[m].upper()+str(phys_d[n]),
                       transform=grid[m,n].transAxes)
        # grid[m,n].set_yticks(range(0,1000,300))
        grid[m,n].set_xticks(x_ticks[n])
        grid[m,n].set_xlim([0,200])
        # grid[m,n].set_yticks(y_ticks[m])
        # grid[m,n].set_ylim(y_lim[m])
        grid[m,n].tick_params(axis="y",direction="in", pad=5)
        grid[m,n].tick_params(axis="x",direction="in", pad=5)
        grid[m,n].set_prop_cycle(color = colors)
        # grid[m,n].grid(False)
        grid[m,n].plot(a[pl*m+n][:,::spacing], lw=lw[m])
    fig.subplots_adjust(left=0.15,right=None,bottom=None,top=None)
    fig.savefig(f'hm{freq}_{coup}_{tmp}_{thres}.pdf')


def pop(coup,thres='0001',phys_d=[10,20,40,60,80],geom = ['ic','c','s','is'],ylim=[0,1],
        freq='', direc = 'data1', tmp=''):
    pop = []
    legends = []
    linestyle_str = []
    line_geom = {'ic': 'solid', 'c': 'dotted', 's': 'dashed', 'is': (0, (2.5, 2, 2.5, 1))}
    # legends = it.product(['IC','C','S','IS'], phys_d[0:4])
    # legends = [a + f'{b}' for a, b in legends]
    for i, pd in it.product(geom, phys_d):
        file = f'./{direc}/pop_{i}{freq}_{coup}{tmp}_{pd}_{thres}.dat'
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
            # pop.append([None]*101)

    
    
    # linestyle_str = ['solid']*k + ['dotted']*k + ['dashed']*k + [(0, (2.5, 2, 2.5, 1))]*k+[(1, (3, 10, 1, 1))]*k
    linestyle_str = linestyle_str*len(colors_mma_detai)
    colors_pop = colors_mma_detai*len(pop)
    wid = 3.375
    fig_pop = plt.figure(dpi=800,figsize=(wid,wid*0.7))
    ax = fig_pop.add_subplot(111)
    
    ax.set_prop_cycle(color = colors_pop,linestyle=linestyle_str)
    
    for cur in pop: 
        ln = ax.plot(*cur,lw=1.5,solid_capstyle='round',dash_capstyle='round')
    # ax.set_aspect(aspect=70,adjustable='box')
    ax.tick_params(axis="y",direction="in", pad=4,width=None)
    ax.tick_params(axis="x",direction="in", pad=4,width=None)
    # ax.set_xlabel('$t\Delta/\pi$',usetex=True)
    # [i.set_linewidth(2) for i in ax.spines.values()]
    ax.set_xlim([0,5])
    ax.set_ylim(ylim)
    # ax.set_yticks(np.arange(0,1.0,0.1))
    freq_ = freq.replace('_','')
    tmp_ = tmp.replace('_','')
    ax.text(0.98,0.06, f'$\eta_0={coup}$, $\omega_0={freq_}$, $T={tmp_}$',
                       transform=ax.transAxes,ha='right',usetex=True,fontsize=10)
    # ax.set_xlabel('$\Delta t$',usetex=True)
    ax.legend(list(legends),loc="upper right",ncol=5,frameon=False,
              handlelength=1.2,columnspacing=0.3,handletextpad=0.2,
              fontsize=10,borderaxespad=0)
    # fig_pop.tight_layout(w_pad=0,pad=0,h_pad=0,rect=[0,0,1,1])
    # fig_pop.subplots_adjust(left=0.5,right=1,bottom=0,top=1)
    # ax.set_axis_off()

    plt.savefig(f'pop{freq}_{coup}_{tmp_}_{thres}.pdf')
    
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
phys_d =[[10,40,60],[10,20,40,60,80,100],[10,20,40,60,80],[10,20,40,60,80,100]]

# phys_d =[[10,20, 40]]*4

pop_grid(coup=[1.0,4.0],tmp=[1.0,4.0],freq='4.0',phys_d=phys_d)

bond(4.0,ylim=None,thres='0001',direc='whole',freq=f'0.25', tmp=f'4.0', head='hm', geom=['ic','c','s'], phys_d=[10,20,40])







# import matplotlib.font_manager

# print(
# sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist]
# ))
