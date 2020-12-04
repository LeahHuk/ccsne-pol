# dochigridPol,['dsk','tor'],[5,1,2],[0,1],[1,2],[10,20,50],indgen(12)+1,1
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
#import scipy as sp
import xarray as xr
import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from matplotlib import gridspec as gridspec
from matplotlib import ticker
from scipy import stats
from scipy.cluster.hierarchy import fcluster, set_link_color_palette
from scipy.cluster.hierarchy import dendrogram, linkage

from collections import OrderedDict
import seaborn as sns
import SciFunctions
import FileFunctions
import time

plt.rcParams["patch.force_edgecolor"] = True
#sns.set_context("paper",rc={"axes.labelsize":11})
#sns.axes_style('whitegrid',{'axes.edgecolor':'darkgrey','axes.linewidth':1.0, 'grid.color':'gainsboro'})
sns.set(style="ticks",  font_scale=1.0, font='sans-serif',
        rc={'axes.edgecolor':'dimgrey','axes.linewidth':1.0, 'grid.color':'gainsboro', 'axes.labelsize':11,
        'legend.markerscale':0.5, 'legend.frameon':True, 'legend.fancybox':True, 'legend.borderpad':0.15, 'legend.framealpha':0.7,
        'legend.facecolor':'white', 'legend.handletextpad':0.15, 'legend.edgecolor':'dimgrey', 'legend.loc':"upper left",
        'ytick.direction': u'in', 'ytick.major.size': 4, 'ytick.minor.size': 2,
        'xtick.direction': u'in', 'xtick.major.size': 4, 'xtick.minor.size': 2})
today = datetime.date.today()

#SN = input('Which SN would you like to fit?: ')       # '1997eg'
#epoch = input('Which epoch of data would you like to fit?: ')
#resolution = input('What resolution is your spectrum?: ')
#params = input('How many free parameters are there?: ')
#write = input('Do you want to write the fit into to text?: ')
#plot = input('Do you want to create plots of models and fit info?: ')
#removepoints = input('Do you want to remove points from the fit?: ')
#shift = input('Do you want to normalize the polarization at the bluest point?: ')
#fineshift = input('Do  you want to allow fine grid shifting?: ')

#geometry = np.array(raw_input('Enter geometries to fit: '))   # ['dsk','ell','tor']
#taus = np.array(raw_input('Enter optical depths to fit: '))     # ['5','1','2']
#ncsm = np.array(raw_input('Enter ncsm ratios to fit: '))        # ['0','1']
#nsh = np.array(raw_input('Enter nsh ratios to fit: '))          # ['1','2','100']
#temps = np.array(raw_input('Enter the temperatures to fit: '))  # ['10','20','50']

SN = 'SN 1997eg'
regime = 'dist'
eps = [1,2,3]
params = 1
write = 0
cluster = 0
contours = 0
plot = 0
removepoints = 1
shift = 1
fineshift = 1

tofit = 'P' #'I','P'
resolution = np.array([5,40])
geometry = np.array(['dsk','tor'])
tau = np.array(['05','1','2'])
ncsm = np.array(['0','1'])
temp = np.array(['10','20','50'])
bins = np.arange(1,13,1)
thetas = np.ones(len(bins))*(0.99767,0.97908,0.94226,0.88789,0.81697,0.73084,0.63109,0.51958,0.39840,0.26980,0.13617,0.00000)
deg  = np.round( (np.arccos(thetas) * (180./np.pi)), 0)

for z in range(0,len(eps)):
#for z in range(0,1):
    epoch = eps[z]
    print('epoch = ' +str(epoch))
    if regime == 'central':
        nsh = np.array(['01','2'])
        path = 'C://Users//Leah//Desktop//CENTRAL RUNS//'
        prefix = 'C://Users//Leah//Desktop//CENTRAL RUNS//'+ SN + '//Epoch' +str(epoch) + '//'
        if SN == 'SN 1997eg':
            cutoffs = [7.5, 25, 16]
        if SN == 'SN 2010jl':
            cutoffs = [4, 1.5, 1.5] #[4.5, 3.75, 5, 2, 1.9] #?
    if regime == 'dist':
        path = 'C://Users//Leah//Desktop//DISTRIBUTED RUNS//'
        prefix = 'C://Users//Leah//Desktop//DISTRIBUTED RUNS//' + SN + '//Epoch' + str(epoch) + '//'
        nsh =  np.array(['100'])
        if SN == 'SN 1997eg':
            cutoffs = [3.3, 9.8, 2.3]
        if SN == 'SN 2010jl':
            cutoffs = [2.1, 4.1, 2] #[1.4, 1.1, 2.1, 2.1, 2.5] #

    if (shift == 0 and fineshift == 0):
        gridtype = 'no-shift'
    if (shift == 1 and fineshift == 0):
        gridtype = 'coarse-shift'
    if (shift == 0 and fineshift == 1):
        gridtype = 'fine-shift'
    if (shift == 1 and fineshift == 1):
        gridtype = 'full-shift'

    n_geom = len(geometry)
    n_tau = len(tau)
    n_ncsm = len(ncsm)
    n_nsh = len(nsh)
    n_temp = len(temp)
    n_thet = len(bins)
    n_models = n_geom * n_tau * n_ncsm * n_nsh * n_temp * n_thet

    #locations
    savefile = 'C://Users//Leah//Desktop//'
    suffix = tofit+'_'+gridtype+'_Geom'+str(geometry)+'Temp'+str(temp)+ '_' + str(today)
    print('reading in, building arrays...')
    start_time = time.clock()
    for r in range(0,len(resolution)):

        if resolution[r] == 5:
            n_lin = 217
            npoints = 121
    #        eps = [163,183,183]

        if resolution[r] == 20:
            n_lin = 53
            npoints = 31

        if resolution[r] == 40:
            n_lin = 27
            npoints = 16
#            eps = [21,21,21]

        n_lines = (np.arange(0,npoints)*resolution[r])+6150
        n_lam = (np.arange(0,n_lin)*resolution[r])+5950
        n_total = n_lin * n_thet

        ### Read in observed files, insert into arrays ###
        obs_columns= ['lambda','I','Ie','Q','Qe','U','Ue','P','Pe','v']
        obsfile = FileFunctions.findobsfile(SN,epoch,resolution[r])

        obsdata = FileFunctions.readinObs(obsfile)
        obspolcol = SciFunctions.computeP(obsdata[:,3],obsdata[:,4],obsdata[:,5],obsdata[:,6])
        vocol = SciFunctions.vspace(obsdata[:,0],6563.)
        obssheet = np.column_stack((obsdata,obspolcol[0],obspolcol[1],vocol))

        modelsheet = np.zeros((n_total,12))
        modelcube = np.zeros((n_geom,n_tau,n_ncsm,n_nsh,n_temp,n_thet,n_lin,12))
        shifts = np.zeros((n_geom,n_tau,n_ncsm,n_nsh,n_temp,n_thet))

        if resolution[r] == 5:
            obsframe = xr.DataArray(obssheet,
                                coords=[n_lines,obs_columns],
                                dims=['wave','spectra'])
        if resolution[r] == 40:
            obsframe_40 = xr.DataArray(obssheet,
                                coords=[n_lines,obs_columns],
                                dims=['wave','spectra'])

        #obsframe.where( (obsframe.wave >= 6150.) & (obsframe.wave <= 6750.) , drop=True)

        ### Read in all model files, insert into arrays ###
        ### print geometry[i], tau[j], ncsm[k], nsh[l], temp[m], bins[n]
        i,j,k,l,m,n = 0,0,0,0,0,0

        for i in range (0,n_geom):
            for j in range (0,n_tau):
                for k in range (0,n_ncsm):
                    for l in range (0,n_nsh):
                        for m in range (0,n_temp):

                            # read in column data for models, stacked by theta viewing angle
                            # 324 rows x [cost, wave, I, Ie, Q, Qe, U, Ue, P, Pe, v, deg]
                            # 27 rows x [cost, wave, I, Q, U, Ie, Qe, Ue, P, Pe, v, deg]

                            root = FileFunctions.findmodelfile(geometry[i], tau[j], ncsm[k], nsh[l], temp[m])
                            if resolution[r] == 5:
                                model = root
                            else:
                                model = root + '_'+str(resolution[r])

                            sheet = FileFunctions.readinModel(path,model,n_total)
                            polcol = SciFunctions.computeP(sheet[:,4], sheet[:,5], sheet[:,6], sheet[:,7])
                            vcol = SciFunctions.vspace(sheet[:,1],6563.)
                            angle  = np.round( np.arccos(sheet[:,0]) * (180./np.pi), 0)
                            modelsheet = np.column_stack((sheet, polcol[0], polcol[1], vcol, angle))

                            # unstack angle bins from each other, place in 'n' dimension
                            for n in range (0, n_thet):
                                modelname = root + '-' + str(bins[n])
                                specsheet = np.zeros((n_lin, modelsheet.shape[1]))
                                specsheet = FileFunctions.binchop(modelsheet, n, n_total, thetas)

                                # insert individual bin spectra into cube of data
                                modelcube[i,j,k,l,m,n,:,:] = specsheet[:,:]

        columns = ['cost','lambda','I','Ie','Q','Qe','U','Ue','P','Pe','v','angle']
        if resolution[r] == 5:
    #        addframe = xr.DataArray(addcube,coords=[addon[0,0],addon[0,1],addon[:,2],addon[:,3],addon[:,4],bins,n_lam,columns],
    #                                dims=['geometry','tau','ncsm','nsh','temp','incbin','wave','spectra'])
            modelframe = xr.DataArray(modelcube,
                                      coords=[geometry,tau,ncsm,nsh,temp,bins,n_lam,columns],
                                      dims=['geometry','tau','ncsm','nsh','temp','incbin','wave','spectra'])
        if resolution[r] == 40:
    #        addframe40 = xr.DataArray(addcube,coords=[addon[0,0],addon[0,1],addon[:,2],addon[:,3],addon[:,4],bins,n_lam,columns],
    #                                dims=['geometry','tau','ncsm','nsh','temp','incbin','wave','spectra'])
            modelframe_40 = xr.DataArray(modelcube,
                                      coords=[geometry,tau,ncsm,nsh,temp,bins,n_lam,columns],
                                      dims=['geometry','tau','ncsm','nsh','temp','incbin','wave','spectra'])

    #     add in single ellipsoids
        if regime == 'central':
            addon = np.array( (  ('ell', '05', '1', '2', '50'),('ell', '05', '0', '01', '10')  ) )
            for p in range(0,1):#len(addon)):
                addcube = np.zeros((1,1,1,1,1,n_thet,n_lin,12))
                root = FileFunctions.findmodelfile(addon[p,0], addon[p,1], addon[p,2], addon[p,3], addon[p,4])
                if resolution[r] == 5:
                    model = root
                else:
                    model = root + '_'+str(resolution[r])
                sheet = FileFunctions.readinModel(path,model,n_total)
                polcol = SciFunctions.computeP(sheet[:,4], sheet[:,5], sheet[:,6], sheet[:,7])
                vcol = SciFunctions.vspace(sheet[:,1],6563.)
                angle  = np.round( np.arccos(sheet[:,0]) * (180./np.pi), 0)
                modelsheet = np.column_stack((sheet, polcol[0], polcol[1], vcol, angle))
                # unstack angle bins from each other, place in 'n' dimension
                for n in range (0, n_thet):
                    modelname = root + '-' + str(bins[n])
                    specsheet = np.zeros((n_lin, modelsheet.shape[1]))
                    specsheet = FileFunctions.binchop(modelsheet, n, n_total, thetas)
                    addcube[0,0,0,0,0,n,:,:] = specsheet[:,:]
                addframe = xr.DataArray(addcube,
                                        coords=[ [addon[p,0]], [addon[p,1]], [addon[p,2]], [addon[p,3]], [addon[p,4]], bins, n_lam, columns],
        #                                coords=[ [list(set(addon[:,0]))[p]],[list(set(addon[:,1]))[p]],[list(set(addon[:,2]))[p]],[list(set(addon[:,3]))[p]],[list(set(addon[:,4]))[p]],bins,n_lam,columns],
                                        dims=['geometry','tau','ncsm','nsh','temp','incbin','wave','spectra'])
                if resolution[r] == 40:
                    modelframe_40 = modelframe_40.combine_first(addframe)
                if resolution[r] == 5:
                    modelframe = modelframe.combine_first(addframe)


    ###=======================================================================================================###

    ## stack models into single dimension of model name (i,j,k,l,m,n) for ease of processing over dimensions
    modelstack = modelframe.stack(Nmodel=('geometry','tau','ncsm','nsh','temp','incbin'))
    modelstack_40 = modelframe_40.stack(Nmodel=('geometry','tau','ncsm','nsh','temp','incbin'))

    fac = (obsframe.sel(spectra='I').isel(wave=0) / modelstack.sel(spectra='I').isel(wave=0)).drop('spectra')
    modelstack.loc[dict(spectra='I')] = modelstack.sel(spectra='I') * fac
    modelstack.loc[dict(spectra='Ie')]= modelstack.sel(spectra='Ie') * fac

    ## trim all models to same wavelength range as observed data - 16 data points
    modelstack = modelstack.where( (modelstack.wave >= 6150.) & (modelstack.wave <= 6750.) , drop=True)
    modelstack_40 = modelstack_40.where( (modelstack_40.wave >= 6150.) & (modelstack_40.wave <= 6750.) , drop=True)
    print(time.clock() - start_time, "seconds")
    start_time = time.clock()
    print('shifting...')
    ## set models to the bluest point of the observed data by shifting them in P
    if tofit == 'P':
        if shift == 1:
            specshift = obsframe.sel(spectra=tofit).isel(wave=0) - modelstack.sel(spectra=tofit).isel(wave=0)
            modelstack.loc[dict(spectra=tofit)] = modelstack.sel(spectra=tofit) + specshift
            sigshift = np.sqrt( (obsframe.sel(spectra='Pe').isel(wave=0))**2 + (modelstack.sel(spectra='Pe').isel(wave=0))**2)
            specshift40 = obsframe_40.sel(spectra=tofit).isel(wave=0) - modelstack_40.sel(spectra=tofit).isel(wave=0)
            modelstack_40.loc[dict(spectra=tofit)] = modelstack_40.sel(spectra=tofit) + specshift40
            sigshift40 = np.sqrt( (obsframe_40.sel(spectra='Pe').isel(wave=0))**2 + (modelstack_40.sel(spectra='Pe').isel(wave=0))**2)
        if shift == 0:
            specshift = modelstack.sel(spectra=tofit).isel(wave=0)  * 0
            specshift40 = modelstack_40.sel(spectra=tofit).isel(wave=0)  * 0
            sigshift = modelstack.sel(spectra=tofit).isel(wave=0)  * 0
            sigshift40 = modelstack_40.sel(spectra=tofit).isel(wave=0)  * 0
    if tofit == 'I':
         if shift == 1:
             specshift =  modelstack.sel(spectra=tofit).isel(wave=0) / obsframe.sel(spectra=tofit).isel(wave=0)
    #         modelstack.loc[dict(spectra=tofit)] = modelstack.sel(spectra=tofit) * specshift
             specshift40 =  modelstack_40.sel(spectra=tofit).isel(wave=0) / obsframe_40.sel(spectra=tofit).isel(wave=0)

         if shift == 0:
             specshift = modelstack.sel(spectra=tofit).isel(wave=0) * 0
             specshift40 = modelstack_40.sel(spectra=tofit).isel(wave=0) * 0

    specshift = specshift.drop('wave').drop('spectra')
    specshift40 = specshift40.drop('wave').drop('spectra')
    sigshift = sigshift.drop('wave').drop('spectra')
    sigshift40 = sigshift40.drop('wave').drop('spectra')

    ## remove wavelength points we don't wish to include in the fit
    if removepoints == 1:
        print('trimming...')
        # intermediate width line from +/- 1000 to 3000 km/s
        BN_modelstack = modelstack.where(
                        (abs(modelstack.sel(spectra='v').isel(Nmodel=0)) >= 3000.)|
                        (abs(modelstack.sel(spectra='v').isel(Nmodel=0)) <= 1000.),
                        drop=True)
        BN_obsframe   = obsframe.where(
                        (abs(obsframe.sel(spectra='v')) >= 3000.)|
                        (abs(obsframe.sel(spectra='v')) <= 1000.),
                        drop=True)
        BN_modelstack_40 = modelstack_40.where(
                        (abs(modelstack_40.sel(spectra='v').isel(Nmodel=0)) >= 3000.)|
                        (abs(modelstack_40.sel(spectra='v').isel(Nmodel=0)) <= 1000.),
                        drop=True)
        BN_obsframe_40 = obsframe_40.where(
                        (abs(obsframe_40.sel(spectra='v')) >= 3000.)|
                        (abs(obsframe_40.sel(spectra='v')) <= 1000.),
                        drop=True)

        # remove enchanced blue scattering wing region only for SN 1997eg
         # Iron line in 5A models only
        if SN == 'SN 1997eg':
            BN_modelstack = BN_modelstack.where(
                            (BN_modelstack.sel(spectra='v').isel(Nmodel=0) > -8587.00)|
                            (BN_modelstack.sel(spectra='v').isel(Nmodel=0) < -8817.00),
                            drop=True)
            BN_obsframe = BN_obsframe.where(
                          (BN_obsframe.sel(spectra='v') > -8587.00)|
                          (BN_obsframe.sel(spectra='v') < -8817.00),
                           drop=True)
            BN_obsframe   = BN_obsframe.where(
                        (BN_obsframe.sel(spectra='lambda') > 6430)|
                        (BN_obsframe.sel(spectra='lambda') < 6350),
                        drop=True)
            BN_modelstack = BN_modelstack.where(
                        (BN_modelstack.sel(spectra='lambda').isel(Nmodel=0) > 6430)|
                        (BN_modelstack.sel(spectra='lambda').isel(Nmodel=0) < 6350),
                        drop=True)
            BN_modelstack_40 = BN_modelstack_40.where(
                        (BN_modelstack_40.sel(spectra='lambda').isel(Nmodel=0) > 6430)|
                        (BN_modelstack_40.sel(spectra='lambda').isel(Nmodel=0) < 6350),
                        drop=True)
            BN_obsframe_40   = BN_obsframe_40.where(
                        (BN_obsframe_40.sel(spectra='lambda') > 6430)|
                        (BN_obsframe_40.sel(spectra='lambda') < 6350),
                        drop=True)

        dof5 = len(BN_modelstack.wave) - params - 1
        dof40 = len(BN_modelstack_40.wave) - params - 1

    if removepoints == 0:
         BN_modelstack = modelstack
         BN_modelstack_40 = modelstack_40
         BN_obsframe = obsframe
         BN_obsframe_40 = obsframe_40

    ###=================================================###
    # arrays to fit using chi square. Either flux or polarization
    print(time.clock() - start_time, "seconds")
    start_time = time.clock()
    print('fitting...')
    if tofit == 'P':
        dof = len(BN_modelstack_40.wave) - params - 1
        allF = BN_modelstack_40.sel(spectra=tofit)
        allFcoords = allF.coords
        allF = allF.drop('spectra')
        allFe = BN_modelstack_40.sel(spectra=(tofit+'e')).isel(wave=0)
        allFe = allFe.drop('spectra')

        Fo = BN_obsframe_40.sel(spectra=tofit)
        Fo = Fo.drop('spectra')
        Foe = BN_obsframe_40.sel(spectra=(tofit+'e'))
        Foe = Foe.drop('spectra')

        if fineshift == 1:

    #    allP = BN_modelstack.sel(spectra='P')
    #    allPcoords = allP.coords
    #    allP = allP.drop('spectra')
    #
    #    Po = BN_obsframe.sel(spectra='P')
    #    Po = Po.drop('spectra')
    #    Poe = BN_obsframe.sel(spectra='Pe')
    #    Poe = Poe.drop('spectra')

            nshift = 200.
            shiftrange =  np.max(Fo) - np.min(Fo)
            eshift = np.sqrt( (np.max(Foe)**2) + (np.min(Foe)**2) )
            shiftgrid = (np.arange(-nshift,nshift,1.0) / nshift)
            ngrid = xr.DataArray( shiftgrid, dims='shifts', coords=[np.arange(0,len(shiftgrid),1)] ) *shiftrange
#            negrid = xr.DataArray(shiftgrid, dims='shifts', coords=[np.arange(0,len(shiftgrid),1)] ) *eshift
            resids  = Fo - (allF + ngrid)
            eresids = Foe

            R = resids/eresids
            c = (resids/eresids)**2
            grid = c.sum(dim='wave')

            minloc = grid.argmin('shifts')
            chi2grid = grid.min(dim='shifts')     # lowest shifted chi2
            rchi2grid = chi2grid / dof40            # lowest rchi2

            norm = np.zeros( (len(minloc)) )
            for n in range ( 0, len(minloc) ):
                norm[n] = float( ngrid[ int(minloc[n]) ] )
            norms = xr.DataArray(norm, coords=chi2grid.coords)

            #R = R.transpose('wave','spectra','Nmodel')
            Residuals = np.zeros( (len(R.wave), len(minloc) ) )
            ad = np.zeros(( len(minloc) ))
            acrit = np.zeros((len(minloc),5))
            asig = np.zeros((len(minloc),5))
            for n in range ( 0, len(minloc) ):
                oneresid = R.sel(shifts=minloc[n]).isel(Nmodel=n).drop('shifts')
                Residuals[:,n] = oneresid[:]
                ad[n] = (stats.anderson(Residuals[:,n], 'norm')[0])
                acrit[n,:] = (stats.anderson(Residuals[:,n], 'norm')[1])
                asig[n,:] = (stats.anderson(Residuals[:,n], 'norm')[2])
            residuals = xr.DataArray(Residuals, coords=allFcoords)
            residuals = xr.concat([residuals], dim='spectra')
            A2 = xr.DataArray(ad, coords=chi2grid.coords)
            Acrit = acrit[0]
            Asig = asig[0]
        #    residuals.coords['spectra'] = 'residuals'
        #    residuals = residuals.assign_coords(spectra='residuals')
        #    residuals = residuals.transpose('wave','spectra','Nmodel')

        if fineshift == 0:

            resids  = -(allF) + Fo
            eresids = Foe
            R = resids/eresids
            c = (R)**2
            chi2grid = c.sum(dim='wave')
            rchi2grid = chi2grid / dof40
            ad = (stats.kstest(R, 'norm'))[0]

            norms = xr.DataArray(np.zeros((len(resids.Nmodel))), coords = chi2grid.coords)
            A2 = xr.DataArray(ad, coords=chi2grid.coords)
            residuals = xr.DataArray(R, coords=allFcoords)
            residuals = xr.concat([residuals],dim='spectra')
            residuals.assign_coords(spectra='residuals')

        #shift_stack = specshift.stack(Nmodel=('geometry','tau','ncsm','nsh','temp','incbin'))
        totalshift = norms + specshift

        BNmodelstack_40 = xr.concat( [BN_modelstack_40, residuals], dim='spectra')
        BNmodelstack_40['spectra'] = (['cost','lambda','I','Ie','Q','Qe','U','Ue','P','Pe','v','angle','residuals'])


    #if tofit =='I':
    #    dof = len(BN_modelstack.wave) - params - 1
    #    allF = BN_modelstack.sel(spectra=tofit)
    #    allFcoords = allF.coords
    #    allF = allF.drop('spectra')
    #
    #    Fo = BN_obsframe.sel(spectra=tofit)
    #    Fo = Fo.drop('spectra')
    #    Foe = BN_obsframe.sel(spectra=(tofit+'e'))
    #    Foe = Foe.drop('spectra')
    #
    #    if fineshift == 1:
    #
    #        nshift = 500.
    #        start = 0.0
    #        stop = 2.0
    #        step = (stop - start) / nshift
    #        shiftgrid = np.arange(start+step,stop+step,step)
    #        ngrid = xr.DataArray(shiftgrid , dims='shifts', coords=[np.arange(0,len(shiftgrid),1)] ) * specshift
    #
    #        resids  = (allF) - (Fo*ngrid)
    #        eresids = (Foe*ngrid)
    #
    #        R = resids/eresids
    #        c = (resids/eresids)**2
    #        grid = c.sum(dim='wave')
    #
    #        minloc = grid.argmin('shifts')
    #        chi2grid = grid.min(dim='shifts')     # lowest shifted chi2
    #        rchi2grid = chi2grid / dof5            # lowest rchi2
    #
    #        norm = np.zeros( (len(minloc)) )
    #        for n in range ( 0, len(minloc) ):
    #            norm[n] = ngrid.isel(shifts=int(minloc[n])).isel(Nmodel=n).values
    #        norms = xr.DataArray(norm, coords=chi2grid.coords)
    #
    #        Residuals = np.zeros( (len(R.wave), len(minloc) ) )
    #        for n in range ( 0, len(minloc) ):
    #            oneresid = R.sel(shifts=minloc[n]).isel(Nmodel=n).drop('shifts')
    #            Residuals[:,n] = oneresid[:]
    #        residuals = xr.DataArray(Residuals, coords=allFcoords)
    #        residuals = xr.concat([residuals], dim='spectra')
    #
    #    if fineshift == 0:
    #
    #        resids  = (allF) - (Fo*specshift)
    #        eresids = (Foe*specshift)
    #        R = resids/eresids
    #        c = (R)**2
    #        chi2grid = c.sum(dim='wave')
    #        rchi2grid = chi2grid / dof5
    #        norms = xr.DataArray(specshift, coords = chi2grid.coords)
    #        residuals = xr.DataArray(R, coords=allFcoords)
    #        residuals = xr.concat([residuals],dim='spectra')
    #
    #    totalshift = norms
    #    BNmodelstack = xr.concat( [BN_modelstack,residuals], dim='spectra')
    #    BNmodelstack['spectra'] = (['cost','lambda','I','Ie','Q','Qe','U','Ue','P','Pe','v','angle','residuals'])


    ###=================================================###
    allI = BN_modelstack.sel(spectra='I')
    allI = allI.drop('spectra')
    allIe = BN_modelstack.sel(spectra='Ie')
    allIe = allIe.drop('spectra')

    Io = BN_obsframe.sel(spectra='I')
    Io = Io.drop('spectra')
    Ioe = BN_obsframe.sel(spectra=('Ie'))
    Ioe = Ioe.drop('spectra')

    # line area excess estimation
    # 6560 + 6565
    # 6500 - 6555, 6570 - 6625
    Aex = ((allI.loc[6560] - BN_obsframe.sel(spectra='I').loc[6560])
           + (allI.loc[6565] - BN_obsframe.sel(spectra='I').loc[6565] )).drop('spectra')
    sigAex = np.sqrt(  (allIe.loc[6560])**2 + (BN_obsframe.sel(spectra='Ie').loc[6560])**2 + (allIe.loc[6565])**2 + (BN_obsframe.sel(spectra='Ie').loc[6565])**2 )
    a = (BN_obsframe.sel(spectra='I').loc[6500:6555] - allI.loc[6500:6555] )
    siga = np.sqrt(  (BN_obsframe.sel(spectra='Ie').loc[6500:6555]**2).sum(dim='wave')  + (allIe.loc[6500:6555]**2).sum(dim='wave')  )
    b = (BN_obsframe.sel(spectra='I').loc[6570:6625] - allI.loc[6570:6625] )
    sigb = np.sqrt(  (BN_obsframe.sel(spectra='Ie').loc[6570:6625]**2).sum(dim='wave')  + (allIe.loc[6570:6625]**2).sum(dim='wave')  )

    Aint = (a.sum('wave') + b.sum('wave')).drop('spectra')
    sigAint = np.sqrt( siga**2 + sigb**2  )
    Adiff = Aex-Aint
    sigAdiff = np.sqrt( sigAex**2 + sigAint**2 ).drop('spectra').drop('wave')
#    Adiff[np.where( (Adiff>0) & (Adiff<1))]

    ## averaging first 10 points of contiuum
    #avgI = allI.isel(wave=slice(0,10)).mean(dim='wave')
    #avgIe = np.sqrt( ((allIe.isel(wave=slice(0,10)))**2 ).sum(dim='wave') )
    #avgIo = Io.isel(wave=slice(0,10)).mean(dim='wave')
    #avgIoe = np.sqrt( ((Ioe.isel(wave=slice(0,10)))**2 ).sum(dim='wave') )
    #
    #Iratio = allI.sel(wave=6565.) / avgI
    #Iratio = Iratio.drop('wave')
    #Ieratio = np.abs(Iratio) * np.sqrt( (avgIe/avgI)**2 + (allIe.sel(wave=6565.)/allI.sel(wave=6565.))**2 )
    #Ieratio = Ieratio.drop('wave')
    #
    #Ioratio = Io.sel(wave=6565.) / avgIo
    #Ioratio = Ioratio.drop('wave')
    #Ioeratio = np.abs(Ioratio) * np.sqrt( (avgIoe/avgIo)**2 + (Ioe.sel(wave=6565.)/Io.sel(wave=6565.))**2 )
    #Ioeratio = Ioeratio.drop('wave')
    #
    #ratiofit = ((Ioratio - Iratio)**2) / (Ioeratio)**2

    ## using all 10 first points individually
    Iratio = allI.sel(wave=6565.) / allI.isel(wave=slice(0,10))
    Ieratio = np.abs(Iratio) * np.sqrt( (allIe.isel(wave=slice(0,10))/allI.isel(wave=slice(0,10)))**2 + (allIe.sel(wave=6565.)/allI.sel(wave=6565.))**2 )
    Ioratio = Io.sel(wave=6565.) / Io.isel(wave=slice(0,10))
    Ioeratio = np.abs(Ioratio) * np.sqrt( (Ioe.isel(wave=slice(0,10))/Io.isel(wave=slice(0,10)))**2 + (Ioe.sel(wave=6565.)/Io.sel(wave=6565.))**2 )

    ratiofit =  ( ((Ioratio - Iratio)**2) / (Ioeratio)**2 ).sum(dim='wave')

    Iratio = Iratio.mean(dim='wave')
    Ieratio = (1.0/10.0)*np.sqrt( (Ieratio**2).sum(dim='wave'))
    Ioratio = Ioratio.mean(dim='wave')
    Ioeratio = (1.0/10.0)*np.sqrt( (Ioeratio**2).sum(dim='wave'))


    best = np.argsort(ratiofit.sel('Nmodel'))
    ratiofit_sort = ratiofit.isel(Nmodel=best)
    ratio_sort = Iratio.isel(Nmodel=best)

    Ioplus = Ioratio+Ioeratio+1
    Iominus = Ioratio-Ioeratio-1
    print(Ioratio, Ioeratio)
    # cutoff using chi2 significance for ~10 points
    #sig = np.where(ratiofit_sort < 20.00)
    # cutoff using doofus human estimate of +/- 1 from observed ratio
    #sig = np.where( np.logical_and(ratio_sort<Ioplus+1, ratio_sort>Iominus-1) ==True)

    sig = np.where( np.logical_and(Iratio<Ioplus, Iratio>Iominus) ==True)

    sigratio = ratiofit[sig]
    #for d in range(0,len(sigratio)):
    #    print (str(sigratio.Nmodel[d].coords)[33:]).replace(' ',''), sigratio[d].values

    ###=================================================###

    fitgrid = xr.concat( [chi2grid, rchi2grid, specshift, norms, totalshift, sigshift, Iratio, Ieratio, A2, Aex, Aint, Adiff, sigAdiff], dim='fitinfo' )
    fitgrid['fitinfo'] = (['chi2','rchi2','norm','shift','total','sigtotal','ratio','sigratio','A2', 'Aex', 'Aint', 'Adiff','sigAdiff'])
    fitgrid2 = fitgrid.where(fitgrid.sel(fitinfo = 'chi2') != 0.0, drop=True)

    models     = modelstack.where(fitgrid2.sel(fitinfo = 'chi2') != 0.0, drop=True).drop('fitinfo')
    models40   = modelstack_40.where(fitgrid2.sel(fitinfo = 'chi2') != 0.0, drop=True).drop('fitinfo')
    modelsBN   = BN_modelstack.where(fitgrid2.sel(fitinfo = 'chi2') != 0.0, drop=True).drop('fitinfo')
    models40BN = BNmodelstack_40.where(fitgrid2.sel(fitinfo = 'chi2') != 0.0, drop=True).drop('fitinfo')

    minimize = fitgrid2.sel(fitinfo = 'rchi2').argsort()

    fitgrid_sort = fitgrid2.isel(Nmodel = minimize)
    models = models.isel(Nmodel = minimize)
    models40 = models40.isel(Nmodel = minimize)
    modelsBN = modelsBN.isel(Nmodel = minimize)
    models40BN = models40BN.isel(Nmodel = minimize)

#    fac = (obsframe.sel(spectra='I').isel(wave=0) / models.sel(spectra='I').isel(wave=0)).drop('spectra').drop('wave')
#    modelI = models.sel(spectra='I').drop('spectra') *fac
#
#    plt.plot(obsframe.wave, obsframe.sel(spectra='I'), drawstyle='steps-mid' )
#    plt.plot(modelstack.wave, modelI.isel(Nmodel=0), drawstyle='steps-mid' )
#    plt.xlim([6500, 6600])
#
#    A1 = (modelI.isel(Nmodel=0).loc[6560] - obsframe.sel(spectra='I').loc[6560]) + (modelI.isel(Nmodel=0).loc[6565] - obsframe.sel(spectra='I').loc[6565] )
#    a2 = (obsframe.sel(spectra='I').loc[6520:6555] - modelI.isel(Nmodel=50).loc[6520:6555] )
#    b2 = (obsframe.sel(spectra='I').loc[6570:6600] - modelI.isel(Nmodel=50).loc[6570:6600] )
#    A2 = a2.sum() + b2.sum()

    ###=================================================###
    ## POPULATIONS OF MODELS BASED ON RCHI2, GAUSSIAN MIXTURE, hierarchical clustering
    print(time.clock() - start_time, "seconds")
    start_time = time.clock()
    print('clustering...')
    ## first find true outliers that are too high to consider for clustering
    def MADs(y,thresh):
    # warning: this function does not check for NAs nor does it address issues when
    # more than 50% of your data have identical values
        m = np.median(y)
        ## 2 sided
    #    abs_dev = np.abs(y - m)
    #    left_mad = np.median(abs_dev[y <= m])
    #    right_mad = np.median(abs_dev[y >= m])
    #    y_mad = left_mad * np.ones(len(y))
    #    y_mad[y > m] = right_mad
        ## 1 sided values above median
        abs_dev = 1e-10* np.ones(len(y))
        abs_dev[y > m] = np.abs(y[np.where(y > m)] - m)
        right_mad = np.median(abs_dev[y >= m])
        y_mad = np.ones(len(y))
        y_mad[y > m] = right_mad

        modified_z_score = 0.6745 * abs_dev / y_mad
        modified_z_score[y == m] = 0
        return modified_z_score > thresh


    chi2s = fitgrid_sort.sel(fitinfo='rchi2').values
    if (SN == 'SN 2010jl') & (regime == 'central'):
        if (epoch == 2) or (epoch == 3):
            outlier = MADs(chi2s,0.5)
        else:
            outlier = MADs(chi2s,3.5)
    else:
        outlier = MADs(chi2s,3.5)
    outliers = chi2s[outlier]
    inliers = chi2s[np.where(outlier==False)]
    cut = (outliers.min() + inliers.max())/2
#    print '   '
#    print 'in    ', np.max(np.where(chi2s < cut)),'    ',chi2s[np.max(np.where(chi2s < cut))]
#    print 'out    ', np.min(np.where(chi2s > cut)),'    ', chi2s[np.min(np.where(chi2s > cut))]
#    print cut
#    continue

    def fancy_dendrogram(*args, **kwargs):
        max_d = kwargs.pop('max_d', None)
        if max_d and 'color_threshold' not in kwargs:
            kwargs['color_threshold'] = max_d
        annotate_above = kwargs.pop('annotate_above', 0)

        ddata = dendrogram(*args, **kwargs)

        if not kwargs.get('no_plot', False):
            plt.title('Hierarchical Clustering Dendrogram (truncated)')
            plt.xlabel('sample index or (cluster size)')
            plt.ylabel('distance')
            for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
                x = 0.5 * sum(i[1:3])
                y = d[1]
                if y > annotate_above:
                    plt.plot(x, y, 'o', c=c)
                    plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                                 textcoords='offset points',
                                 va='top', ha='center')
            if max_d:
                plt.axhline(y=max_d, c='k')
        return ddata

#    with sns.axes_style('whitegrid',{'axes.edgecolor':'darkgrey','axes.linewidth':1.0, 'grid.color':'gainsboro'}):
    palette = sns.xkcd_palette(['cornflower','dark pastel green','salmon','golden yellow','liliac','pale orange','light teal','pale brown'])
    palette = (palette)+(palette)
    if cluster == 1:
        ## whole distribution, outliers marked
        fig = plt.figure(figsize=(8.5, 5)) #
        gs1=gridspec.GridSpec(1,1)
        ax0 = plt.subplot(gs1[0])
        ax0 = sns.distplot(chi2s, rug=True, norm_hist=False, kde=False, hist_kws=dict(linewidth=0.5, edgecolor='b'))
        ax0.set_xlim(1,np.max(chi2s)+5)
        ax0.plot(outliers, np.zeros_like(outliers), 'ro', clip_on=False)
        ax0.axvline(cut, color='dimgrey', linestyle='--', linewidth=1.5)
        ax0.text(0.15,0.95,'N Models: '+str(len(chi2s)), transform=ax0.transAxes,color='black',fontsize=11)
        ax0.set_title('entire '+r'$\chi_{\nu}^2$'+' distribution',fontsize=11)
        ax0.set_xlabel(r'$\chi_{\nu}^2$',fontsize=11)
        ax0.set_ylabel('# of models (N)',fontsize=11)
        fig.tight_layout()
        fig.savefig(prefix+'rchi2 distribution1.png', dpi=800)
        plt.show()
        plt.close()

        ## distribution with outliers removed
        Nbin = 40
        fig = plt.figure(figsize=(8.5, 5)) #
        gs1=gridspec.GridSpec(1,1)
        ax1 = plt.subplot(gs1[0])
        ax1 = sns.distplot(chi2s[np.where(chi2s<cut)], kde=False, rug=True, bins=Nbin,
                           hist_kws=dict(linewidth=0.5, edgecolor='b'), ax=ax1)
        ax1.text(0.02,0.95,'N Models: '+str(len(inliers)), transform=ax1.transAxes,color='black',fontsize=11)
        ax1.set_xlim(1,int(inliers.max())+1)
        ax1.set_title(r'$\chi_{\nu}^2$'+' less than '+str(cut),fontsize=11)
        ax1.set_xlabel(r'$\chi_{\nu}^2$',fontsize=11)
        ax1.set_ylabel('# of models (N)',fontsize=11)
        fig.tight_layout()
        fig.savefig(prefix+'rchi2 distribution2.png', dpi=800)
        plt.show()
        plt.close()

    set_link_color_palette(map(rgb2hex,  palette[1:])) # palette[2:None:-1])) #
    inliers = np.reshape(inliers,(-1,1))
    Z1 = linkage(inliers, 'centroid')

    fig = plt.figure(figsize=(8.5, 5)) #
    gs=gridspec.GridSpec(1,1)
    ax9 = plt.subplot(gs[0])
    fancy_dendrogram(Z1, truncate_mode='lastp', p=40, leaf_rotation=45.,
                     show_contracted=True, annotate_above=2)
    plt.show()
#        cutoff = input('choose cutoff: ')
    cutoff = cutoffs[epoch-1]
    print('cutoff = ' +str(cutoff))
    plt.close()

    ## determine the number of clusters, locations in group belonging to each, and their min/max/avg
    ## if either of the edge clusters have only one or two models, merge them with their closest neighbor
    ## plot the clusters on top of the original histogram distribution
    clusters = fcluster(Z1, cutoff, criterion='distance')
    n = list(OrderedDict((element, None) for element in clusters))
    n = np.array((n))
    clusters2 = np.array(clusters)
    if len(clusters2[np.where(clusters2 == n[0])]) <= 4:
        if len(clusters2[np.where(clusters2 == n[1])]) <= 4:
            clusters2[np.where(clusters2 == n[1])] = n[2]
            clusters2[np.where(clusters2 == n[0])] = n[2]
            N = np.array(n[2:])
        else:
            clusters2[np.where(clusters2 == n[0])] = n[1]
            N = np.array(n[1:])
    elif len(clusters2[np.where(clusters2 == n[len(n)-1])]) <= 3:
        clusters2[np.where(clusters2 == n[len(n)-1])] = n[len(n)-2]
        N = np.array(n[:(len(n)-2)])
    else:
        N = np.array(n)
    Nmod = np.zeros((len(N)))
    C = np.zeros((len(inliers),len(N)))
    means = np.zeros((len(N)))
    minmax = np.zeros((len(N),2))

    if cluster == 1:
        fig = plt.figure(figsize=(8.5, 5))
        gs1=gridspec.GridSpec(1,1)
        ax = plt.subplot(gs1[0])
        fancy_dendrogram(Z1, truncate_mode='lastp', p=20, leaf_rotation=45.,
                         show_contracted=True, annotate_above=10, color_threshold=cutoff)
        ax.set_title('Hierarchical Clustering Dendrogram (truncated) for inliers', fontsize=11)
        ax.set_xlabel('sample index (cluster size)', fontsize=11)
        ax.set_ylabel('distance', fontsize=11)
        ax.axhline(cutoff, color='grey', linestyle='--')
        fig.tight_layout()
        fig.savefig(prefix+'rchi2 distribution3.png', dpi=800)
        plt.show()
        plt.close()

    for i in range(len(N)):
        C[:,i] = (clusters2 == N[i])
        Nmod[i] = len(clusters2[np.where(clusters2 == N[i])])
        means[i] = np.mean(inliers[np.where(clusters2==N[i])])
        minmax[i,:] = np.min(inliers[np.where(clusters2==N[i])]), np.max(inliers[np.where(clusters2==N[i])])
    lowest = int( (np.where(minmax[:,0] == minmax[:,0].min()))[0])
    lowcut = minmax[lowest,1]
    lowchi2s = inliers[np.where(inliers <= lowcut)]

    if cluster == 1:
        fig = plt.figure(figsize=(8.5, 5)) #
        gs1=gridspec.GridSpec(1,1)
        ax2 = plt.subplot(gs1[0])
        ax2 = sns.distplot(inliers, kde=False, rug=False, ax=ax2, bins=Nbin)
        for i in range(len(N)):
            nbin = int(  (( minmax[i,:].max() - minmax[i,:].min() )/cut) *Nbin )
            if nbin == 0:
                nbin = 1
            ax2 = sns.distplot(inliers[np.where(C[:,i]==True)], kde=False, rug=False,# bins=nbin,
                               hist_kws=dict(linewidth=0.5, edgecolor=palette[i+1]), ax=ax2, color=palette[i+1])
        ax2.text(0.02,0.95,'Group max: '+str(minmax[lowest,1])[0:4], transform=ax2.transAxes, color='black', fontsize=11)
        ax2.text(0.02,0.90,'Group mean: '+str(means[lowest])[0:4],transform=ax2.transAxes, color='black', fontsize=11)
        ax2.text(0.02,0.85,'N Models: '+str(len(inliers[np.where(C[:,lowest]==True)])), transform=ax2.transAxes, color='black', fontsize=11)
        ax2.set_title('Model Clusters in '+r'$\chi_{\nu}^2$', fontsize=11)
        ax2.set_xlabel(r'$\chi_{\nu}^2$', fontsize=11)
        ax2.set_ylabel('# of models (N)', fontsize=11)
        fig.tight_layout()
        fig.savefig(prefix+'rchi2 distribution4.png', dpi=800)
        plt.show()
        plt.close()

    ###=================================================###
    #badindex  = allF.where( np.isnan(allF), drop=True)
    #badmodels = badindex.Nmodel
    #badmodels = fitarray.where( (fitarray.sel(fitinfo='chi2') == 0), drop=True)

    critchi = stats.chi2.isf(q=0.05,df=dof)
    critrchi2 = critchi/dof
    #critrchi2_range = [critrchi2-1, 1-(critrchi2-1)]

    IR = fitgrid_sort.sel(fitinfo='ratio')

    sig = np.logical_and(Iominus <= Iratio.sel('Nmodel'), Iratio.sel('Nmodel') <=Ioplus)

    condR = (IR <= (Ioplus+1)) & ((Iominus-1) <= IR)
    condC = fitgrid_sort.sel(fitinfo='rchi2') <= lowcut
    condA = fitgrid_sort.sel(fitinfo='A2') <= Acrit[2]

    triplepass = models40.Nmodel[condR & condC & condA]
    tripleval = fitgrid_sort.sel(Nmodel=triplepass)

    significant = fitgrid_sort.where( (fitgrid_sort.sel(fitinfo='chi2') <= critchi), drop=True)

    ###=================================================###
    ## POPULATIONS OF MODELS BASED ON RCHI2, GAUSSIAN MIXTURE
    if write == 1:
        print(time.clock() - start_time, "seconds")
        start_time = time.clock()
        print('writing to output...')
        f = open(prefix+'Best fit Models_'+suffix+'.txt',"w")
        f.write(' '+ '\n')
        f.write( SN + '\n')
        f.write('resolution: '+str(resolution)+'A' + '\n')
        f.write('epoch: '+str(epoch) + '\n')
        f.write('date run: '+ str(today) + '\n' )
        f.write('shift type: '+str(gridtype) + '\n' )
        f.write('' + '\n')
        f.write('model parameters: ' + '\n')
        f.write('geometries: '+ str(geometry) + '\n' )
        f.write('optical depths: '+ str(tau) + '\n' )
        f.write('CSM fractions: '+ str(ncsm) + '\n' )
        f.write('Shock fractions: '+ str(nsh) + '\n' )
        f.write('Temperatures: '+ str(temp) + '\n' )
        f.write('' + '\n')
        f.write('degrees of freedom: '+str(dof) + '\n' )
        f.write('models in grid space: '+str(n_models) + '\n' )
        f.write('' + '\n')
        f.write('low cluster cutoff: '+str(minmax[lowest,1])+ '\n')
        f.write('A2 crit: '+str(Acrit)+'\n')
        f.write('A2 sig:  '+str(Asig)+'\n')
    #    f.write('bad models: ' + '\n')
    #    for n in range(0, len(badmodels)):
    #        f.write( str(badmodels[n].coords)[33:] + '\n' )
        f.write('' + '\n')
    #    f.write('models not present in grid: '+index(bad) )
        f.write('_________________________________________________________________________________   ' + '\n')
        f.write('Emission Ratio statistic fitting' + '\n' )
        f.write('observed ratio: '+str(Ioratio.values)[0:5]+'+/-'+str(Ioeratio.values)[0:6] +'\n')
        f.write('sig ratios between: '+str(Ioplus.values)[0:5]+ ' and ' + str(Iominus.values)[0:5]+'\n')
        f.write('Significant chisquare values, alpha = 0.001, below: '+ str(20) + '\n' )
        for d in range(0,len(sigratio)):
            f.write('{:31s}{:11.7f}'.format(
                    (str(sigratio.Nmodel[d].coords)[33:]).replace(' ',''),
                    float(sigratio[d].values)) + '\n')

        f.write('_________________________________________________________________________________   ' + '\n')
        f.write('Chi Square statistic fitting' + '\n' )
    #    f.write('mean chi2 value: '+str(meanchi2) )
    #    f.write('min chi2 value: '+str(minchi2) )
        f.write('Significant chisquare values, alpha = 0.05, below: '+ str(critchi) + '\n' )
        f.write('   ' + '\n')
        f.write('{:31s}{:>9s}{:>8s}{:>10s}{:>10s}{:>10s}{:>10s}{:>10s}'.format(
                'Model Name','chi2 ','rchi2 ','norm','shift','total','ratio','sigratio','A2')
                + '\n')
        f.write('---------------------------------------------------------------------------------  ' + '\n')
        for n in range(0, len(fitgrid_sort.Nmodel)):
           f.write('{:30s}{:9.3f}{:8.3f}{:11.7f}{:11.7f}{:11.7f}{:13.7f}{:13.7f}{:13.7f}'.format(
                    (str(fitgrid_sort.Nmodel[n].coords)[33:]).replace(' ', ''),
                    float(fitgrid_sort[0,n]),
                    float(fitgrid_sort[1,n]),
                    float(fitgrid_sort[2,n]),
                    float(fitgrid_sort[3,n]),
                    float(fitgrid_sort[4,n]),
                    float(fitgrid_sort[5,n]),
                    float(fitgrid_sort[6,n]),
                    float(fitgrid_sort[7,n]))
                    + '\n')
        f.close()

        e = open(prefix+'all data.txt',"w")
        e.write('{:5s}  {:3s}  {:4s} {:4s}   {:4s} {:3s}  {:3s}   {:9s}  {:8s}  {:11s} {:11s} {:11s} {:10s} {:10s} {:10s} {:12s} {:13s} {:10s} {:10s} {:10s} {:10s} {:10s}'.format(
            'geom','tau','ncsm','nsh','temp','inc','deg','chi2','rchi2','norm','shift','total','sigtotal','ratio','sigratio','A2','rho','ne','Aex','Aint','Adiff','sigAdiff')+'\n')
        for n in range(0,len(fitgrid_sort.Nmodel)):
            s = str(fitgrid_sort.Nmodel[n].coords)[34:-1].replace(' ','').replace('\'','')
            s = s.split(',')
            if len(s[1]) == 2:
                s1 = float('0.'+ s[1][1])
            else:
                s1 = float(s[1][0] + '.0')
            s2 = float('0.'+s[2])
            if s[3] == '100':
                s3 = 1.0
            elif s[3] == '01':
                s3 = 0.01
            else:
                s3 = 0.2
            s5 = deg[int(s[5])-1]
            if s[0] == 'dsk':
                if s1 == 0.5:
                    rho = np.array((2.388000E-12, 2.398438E-12, 2.398437E-12))
                    ne = np.array((1.413468E+12, 1.419986E+12, 1.419986E+12))
                if s1 == 1.0:
                    rho = np.array((4.788534E-12, 4.794484E-12, 4.794481E-12))
                    ne = np.array((2.833170E+12, 2.838554E+12, 2.838554E+12))
                if s1 == 2.0:
                    rho = np.array((9.594399E-12, 9.548290E-12, 9.548278E-12))
                    ne = np.array((5.672858E+12, 5.653022E+12, 5.653022E+12))
            if s[0] == 'tor':
                if s1 == 0.5:
                    rho = np.array((1.777616E-11, 1.781700E-11, 1.781696E-11))
                    ne = np.array((9.954000E+12, 9.999899E+12, 9.999899E+12))
                if s1 == 1.0:
                    rho = np.array((3.571321E-11, 3.561630E-11, 3.561615E-11))
                    ne = np.array((1.995190E+13, 1.998982E+13, 1.998982E+13))
                if s1 == 2.0:
                    rho = np.array((7.183896E-11, 7.093069E-11, 7.093007E-11))
                    ne = np.array((3.994970E+13, 3.981001E+13, 3.981001E+13))
            if int(s[4]) == 10:
                rho1 = rho[0]
                ne1 = ne[0]
            if int(s[4]) == 20:
                rho1 = rho[1]
                ne1 = ne[1]
            if int(s[4]) == 50:
                rho1 = rho[2]
                ne1 = ne[2]
            e.write('{:5s}  {:3.1f}  {:3.1f}  {:4.2f}  {:3d}  {:3d}  {:3.0f}  {:9.3f}  {:8.3f}  {:11.7f} {:11.7f} {:11.7f} {:11.7f} {:11.7f} {:11.7f} {:11.7f} {:8.6e} {:8.6e} {:10.6f} {:10.6f} {:10.6f} {:10.6f}'.format(
                     s[0], s1, s2, s3, int(s[4]), int(s[5]), s5,
                    float(fitgrid_sort[0,n]),
                    float(fitgrid_sort[1,n]),
                    float(fitgrid_sort[2,n]),
                    float(fitgrid_sort[3,n]),
                    float(fitgrid_sort[4,n]),
                    float(fitgrid_sort[5,n]),
                    float(fitgrid_sort[6,n]),
                    float(fitgrid_sort[7,n]),
                    float(fitgrid_sort[8,n]),
                    float(rho1), float(ne1) ,
                    float(fitgrid_sort[9,n]), float(fitgrid_sort[10,n]), float(fitgrid_sort[11,n]), float(fitgrid_sort[12,n]))
                    + '\n')
        e.close()

#        h = open(prefix+'special-interest.txt',"w")
#        h.write('{:5s}  {:3s}  {:4s} {:4s}   {:4s} {:3s}  {:3s}   {:9s}  {:8s}  {:11s} {:11s} {:11s} {:10s} {:10s} {:10s} {:13s} {:8s}'.format(
#                'geom','tau','ncsm','nsh','temp','inc','deg','chi2','rchi2','norm','shift','total','ratio','sigratio','A2','rho','ne')+'\n')
#        for n in range(0,len(triplepass.Nmodel)):
#            s = str(tripleval.Nmodel[n].coords)[34:-1].replace(' ','').replace('\'','')
#            s = s.split(',')
#            if len(s[1]) == 2:
#                s1 = float('0.'+ s[1][1])
#            else:
#                s1 = float(s[1][0] + '.0')
#            s2 = float('0.'+s[2])
#            if s[3] == '100':
#                s3 = 1.0
#            elif s[3] == '01':
#                s3 = 0.01
#            else:
#                s3 = 0.2
#            s5 = deg[int(s[5])-1]
#            if s[0] == 'dsk':
#                if s1 == 0.5:
#                    rho = np.array((2.388000E-12, 2.398438E-12, 2.398437E-12))
#                    ne = np.array((1.413468E+12, 1.419986E+12, 1.419986E+12))
#                if s1 == 1.0:
#                    rho = np.array((4.788534E-12, 4.794484E-12, 4.794481E-12))
#                    ne = np.array((2.833170E+12, 2.838554E+12, 2.838554E+12))
#                if s1 == 2.0:
#                    rho = np.array((9.594399E-12, 9.548290E-12, 9.548278E-12))
#                    ne = np.array((5.672858E+12, 5.653022E+12, 5.653022E+12))
#            if s[0] == 'tor':
#                if s1 == 0.5:
#                    rho = np.array((1.777616E-11, 1.781700E-11, 1.781696E-11))
#                    ne = np.array((9.954000E+12, 9.999899E+12, 9.999899E+12))
#                if s1 == 1.0:
#                    rho = np.array((3.571321E-11, 3.561630E-11, 3.561615E-11))
#                    ne = np.array((1.995190E+13, 1.998982E+13, 1.998982E+13))
#                if s1 == 2.0:
#                    rho = np.array((7.183896E-11, 7.093069E-11, 7.093007E-11))
#                    ne = np.array((3.994970E+13, 3.981001E+13, 3.981001E+13))
#            if int(s[4]) == 10:
#                rho1 = rho[0]
#                ne1 = ne[0]
#            if int(s[4]) == 20:
#                rho1 = rho[1]
#                ne1 = ne[1]
#            if int(s[4]) == 50:
#                rho1 = rho[2]
#                ne1 = ne[2]
#            h.write('{:5s}  {:3.1f}  {:3.1f}  {:4.2f}  {:3d}  {:3d}  {:3.0f}  {:9.3f}  {:8.3f}  {:11.7f} {:11.7f} {:11.7f} {:11.7f} {:11.7f} {:11.7f}  {:8.6e}  {:8.6e}'.format(
#                     s[0], s1, s2, s3, int(s[4]), int(s[5]), s5,
#                    float(tripleval[0,n]),
#                    float(tripleval[1,n]),
#                    float(tripleval[2,n]),
#                    float(tripleval[3,n]),
#                    float(tripleval[4,n]),
#                    float(tripleval[5,n]),
#                    float(tripleval[6,n]),
#                    float(tripleval[7,n]),
#                    float(rho1), float(ne1) )
#                    + '\n')
#        h.close()
        print(time.clock() - start_time, "seconds")
    ###=================================================###
    #with sns.axes_style('darkgrid',{'axes.edgecolor':'darkgrey','axes.linewidth':1.0,'axes.facecolor': 'gainsboro', 'grid.color':'snow', 'legend.frameon':True}):
#    sns.set_context(rc={'lines.markeredgewidth':1.0})
#    sns.set(font_scale = 1)
    if plot == 1:
        if contours == 1:
            print('plotting contours...')
            start_time = time.clock()
#            with sns.axes_style({'axes.edgecolor':'black', 'axes.linewidth':0.75, 'axes.linewidth':0.5,'legend.frameon':True}):
    ## PlOT CONTOURS OF RCHI2
    ## Contours of tau vs inc, one figure for each geom/temp iteration
            m1 = int(fitgrid_sort.sel(fitinfo='rchi2').min())
            M1 = int(inliers.max()) #m1+cut
            levels = np.linspace(m1, M1, 15)
            cmap = sns.color_palette('gist_earth',len(levels))
            fits = fitgrid_sort.unstack('Nmodel')
            for a in range(0,len(geometry)):
                for b in range(0,len(temp)):
                    rc = fits.sel(fitinfo='rchi2').isel(geometry=a).isel(temp=b)
                    if any(t == '100' for t in nsh) == True:
                        fig = plt.figure(figsize=(9,8))
                        gs1 = gridspec.GridSpec(2,1)
                    else:
                        fig = plt.figure(figsize=(9,16))
                        gs1 = gridspec.GridSpec(4,1)
    #                fig.text(0.51, 0.02, 'Inclination', ha='center', fontsize=14)
    #                fig.text(0.095, 0.5, 'Optical Depth', va='center', fontsize=14, rotation='vertical')
                    ax0 = fig.add_subplot(gs1[0,0])
                    cs = ax0.contourf(deg,[0.5,1.0,2.0],rc.isel(ncsm=0).isel(nsh=0).values,
                                      colors=cmap, levels=levels, extend='max')
                    cl = ax0.contour(deg,[0.5,1.0,2.0],rc.isel(ncsm=0).isel(nsh=0).values,
                                    colors ='black', levels=levels)
    #                    plt.setp(ax0.get_yticklabels(), visible=False)
                    plt.setp(ax0.get_xticklabels(), visible=False)
                    for c in range(0,len(nsh)):
                        for d in range(0,len(ncsm)):
                            rcj = rc.isel(ncsm=d).isel(nsh=c).values
    #                        if any(t == '100' for t in nsh) == True:
    #                            ax = fig.add_subplot(gs1[(2*c)+d,0], sharex=ax0, sharey=ax0)
    #                        else:
                            ax = fig.add_subplot(gs1[(2*c)+d,0], sharex=ax0, sharey=ax0)
                            cs = ax.contourf(deg,[0.5,1.0,2.0],rcj,
                                             colors=cmap, levels=levels, extend='max')
                            cl = ax.contour(deg,[0.5,1.0,2.0],rcj, colors ='black', levels=levels)
                            if any(t == '100' for t in nsh) == True:
                                ax.text(0.025,0.85,'ncsm: 0.'+str(ncsm[d]), weight='roman', transform=ax.transAxes,
                                        bbox=dict(boxstyle='round', fc='grey', alpha=0.8, ec='black', lw=1.5))
    #                            if ((2*c)+d) < 1:
    #                                plt.setp(ax.get_xticklabels(), visible=False)
                            else:
                                ax.text(0.025,0.85,'ncsm: 0.'+str(ncsm[d])+'\n'+'nsh: 0.'+str(nsh[c]), weight='roman',
                                        transform=ax.transAxes, bbox=dict(boxstyle='round', fc='grey', alpha=0.8, ec='black', lw=1.5))
    #                            if ((2*c)+d) < 3:
    #                                plt.setp(ax.get_xticklabels(), visible=False)
                            ax.set_xlabel('Inclination', fontsize=12)
                            ax.set_ylabel('Optical Depth', fontsize=12)
                            ax.yaxis.set_major_locator(ticker.FixedLocator([0.5,1.0,2.0]))
                            for tick in ax.xaxis.get_majorticklabels():
                                tick.set_horizontalalignment("right")
    ##                        if d==1:
    ##                             plt.setp(ax.get_yticklabels(), visible=False)
    ##                        if c==0:
    #                        if ((2*c)+d) < 3:
    ##                            if any(t == '100' for t in nsh) == False:
    #                        if ((2*c)+d) < 1:
    #                            plt.setp(ax.get_xticklabels(), visible=False)

    #                plt.subplots_adjust(hspace=0.04)#, wspace=0.02)
    #                if any(t == '100' for t in nsh) == True:
    #                    cax = fig.add_axes([0.125, 0.9, 0.78, 0.06])
    #                else:
                    cax = fig.add_axes([0.125, 0.89, 0.78, 0.015])
                    cb = fig.colorbar(cs, cax=cax, orientation='horizontal',
                                 format=ticker.FormatStrFormatter('%.0f'))
                    cb.ax.xaxis.set_ticks_position('top')
                    fig.text(0.09, 0.90, r'$\chi_{\nu}^2$', transform=cax.transAxes, fontsize=14)
            #        cb.ax.yaxis.set_label_position('left')
    #                fig.tight_layout()
                    plt.savefig(prefix+'contours//TauVsTheta_'+tofit+'_'+gridtype+'_Geom['+str(geometry[a])+']_Temp['+str(temp[b])+']_'+str(today)+'.png',
                                dpi=600, pad_inches = 0.05, bbox_inches='tight') #, bbox_extra_artists=[])
                    plt.close()

            ## Contours of temp vs inc, one figure for each geom/temp iteration
            for a in range(0,len(geometry)):
                for b in range(0,len(tau)):
                    rc = fits.sel(fitinfo='rchi2').isel(geometry=a).isel(tau=b)
                    if any(t == '100' for t in nsh) == True:
                        fig = plt.figure(figsize=(9,8))
                        gs1 = gridspec.GridSpec(2,1)
                    else:
                        fig = plt.figure(figsize=(9,16))
                        gs1 = gridspec.GridSpec(4,1)
                    ax0 = fig.add_subplot(gs1[0,0])
    #                fig.text(0.51, 0.02, 'Inclination', ha='center', fontsize=14)
    #                fig.text(0.095, 0.5, 'Temperature (K)', va='center', fontsize=14, rotation='vertical')
                    cs = ax0.contourf(deg,[10,20,50],rc.isel(ncsm=0).isel(nsh=0).values,
                                      colors=cmap, levels=levels, extend='max')
                    cl = ax0.contour(deg,[10,20,50],rc.isel(ncsm=0).isel(nsh=0).values,
                                    colors ='black', levels=levels)
    #                    plt.setp(ax0.get_yticklabels(), visible=False)
                    plt.setp(ax0.get_xticklabels(), visible=False)
                    for c in range(0,len(nsh)):
                        for d in range(0,len(ncsm)):
                            rcj = rc.isel(ncsm=d).isel(nsh=c).values
    #                        if any(t == '100' for t in nsh) == True:
    #                            ax = fig.add_subplot(gs1[c,d], sharex=ax0, sharey=ax0)
    #                        else:
                            ax = fig.add_subplot(gs1[(2*c)+d,0], sharex=ax0, sharey=ax0)
                            cs = ax.contourf(deg,[10,20,50],rcj,
                                             colors=cmap, levels=levels, extend='max')
                            cl = ax.contour(deg,[10,20,50],rcj, colors ='black',
                                            levels=levels)
                            if any(t == '100' for t in nsh) == True:
                                ax.text(0.025,0.85,'ncsm: 0.'+str(ncsm[d]), weight='roman', transform=ax.transAxes,
                                        bbox=dict(boxstyle='round', fc='grey', alpha=0.8, ec='black', lw=1.5))
    #                            if ((2*c)+d) < 1:
    #                                plt.setp(ax.get_xticklabels(), visible=False)
                            else:
                                ax.text(0.025,0.85,'ncsm: 0.'+str(ncsm[d])+'\n'+'nsh: 0.'+str(nsh[c]), weight='roman',
                                        transform=ax.transAxes, bbox=dict(boxstyle='round', fc='grey', alpha=0.8, ec='black', lw=1.5))
    #                            if ((2*c)+d) < 3:
    #                                plt.setp(ax.get_xticklabels(), visible=False)
                            ax.set_xlabel('Inclination', fontsize=12)
                            ax.set_ylabel('Temperature (K)', fontsize=12)
                            ax.yaxis.set_major_locator(ticker.FixedLocator([10,20,50]))
                            for tick in ax.xaxis.get_majorticklabels():
                                tick.set_horizontalalignment("right")
    #                plt.subplots_adjust(hspace=0.04)#, wspace=0.02)
    #                if any(t == '100' for t in nsh) == True:
    #                    cax = fig.add_axes([0.125, 0.91, 0.78, 0.06])
    #                else:
                    cax = fig.add_axes([0.125, 0.89, 0.78, 0.015])
                    cb = fig.colorbar(cs, cax=cax, orientation='horizontal', format=ticker.FormatStrFormatter('%.0f'))
                    cb.ax.xaxis.set_ticks_position('top')
                    fig.text(0.09, 0.90, r'$\chi_{\nu}^2$', transform=cax.transAxes, fontsize=14)
    #                fig.tight_layout()
                    plt.savefig(prefix+'contours//TempVsTheta_'+tofit+'_'+gridtype+'_Geom['+str(geometry[a])+']_Tau['+str(tau[b])+']_'+str(today)+'.png',
                                dpi=600, pad_inches = 0.05, bbox_inches='tight') #, bbox_extra_artists=[])
                    plt.close()


    ## PLOT 20 BEST FITTING MODELS
        print(time.clock() - start_time, "seconds")
        print('plotting spectra...')
        modelP = models40.sel(spectra='P').drop('spectra')
        modelPBN = models40BN.sel(spectra='P').drop('spectra')

        obsI = obsframe.sel(spectra='I').drop('spectra')
        obsIe = obsframe.sel(spectra='Ie').drop('spectra')
#        fac = (obsframe.sel(spectra='I').isel(wave=0) / models.sel(spectra='I').isel(wave=0)).drop('spectra').drop('wave')
        modelI = models.sel(spectra='I').drop('spectra')
        modelIBN = modelsBN.sel(spectra='Ie').drop('spectra')

#        obsI = obsI / obsframe.sel(spectra='I').sel(wave=6565.)
#        modelI = modelI / obsframe.sel(spectra='I').sel(wave=6565.)

        obsP = obsframe_40.sel(spectra='P').drop('spectra')
        obsPe = obsframe_40.sel(spectra='Pe').drop('spectra')
        obsPBN = BN_obsframe_40.sel(spectra='P').drop('spectra')
        obsPeBN = BN_obsframe_40.sel(spectra='Pe').drop('spectra')

        wave = obsframe.wave
        waveBN = BN_obsframe.wave
        wave40 = obsP.wave
        wave40BN = obsPBN.wave
        v40 = obsframe_40.sel(spectra='v')
        rs = models40BN.sel(spectra='residuals')
        t1 = [0,1,2,3]
        t2 = [4,5,6,7]
    #        plt.figure(figsize=(15, 60))
    #        gs0=gridspec.GridSpec(24,3, width_ratios=[1.5,2,1])
    #    with sns.axes_style('darkgrid',{'axes.edgecolor':'darkgrey','axes.linewidth':1.0,'axes.facecolor': 'gainsboro', 'grid.color':'snow', 'legend.frameon':True}):
    #        sns.set_context(rc={'lines.markeredgewidth':1.0})
    #        sns.set(font_scale = 1)
        fig = plt.figure(figsize=(9,10))
        gs0=gridspec.GridSpec(4,2, width_ratios=[2,1])
        for t in range(0,4):
            t0 = t2[t]
            PS = fitgrid_sort.sel(fitinfo='total').isel_points(Nmodel=[t0]).values[0]*100
            if regime == 'central':
                G = 'Central'
            if regime =='dist':
                G = 'Distributed'
            s = (str(models40.Nmodel[t0].coords)[34:-1]).replace(' ','').replace('\'','')
            s = s.split(',')
            if s[0] == 'dsk':
                s0 = 'Disk'
            if s[0] == 'tor':
                s0 = 'Toroid'
            if s[0] == 'ell':
                s0 = 'Ellipsoid'
            if len(s[1]) == 2:
                s1 = float('0.'+ s[1][1])
            else:
                s1 = float(s[1][0] + '.0')
            s2 = float('0.'+s[2])
            if s[3] == '100':
                s3 = 1.0
            elif s[3] == '01':
                s3 = 0.01
            else:
                s3 = 0.2
            s4 = s[4]+'000 K'
            s5 = '{:2.0f}'.format(deg[int(s[5])-1])

#            plt.figure(figsize=(9,3))
#            gs0=gridspec.GridSpec(1,2, width_ratios=[2,1])
#            with sns.axes_style('whitegrid',{'axes.edgecolor':'darkgrey','axes.linewidth':1.0, 'grid.color':'gainsboro'}):
            ax0 = plt.subplot(gs0[t,1])
            ax1 = plt.subplot(gs0[t,0])

            ax0.plot(wave, obsI, color='black', drawstyle='steps-mid', linewidth=1.5)
            ax0.plot(wave, modelI.isel(Nmodel=t0),color='red', drawstyle='steps-mid', linewidth=1.15)
            ax0.set_xlabel('wavelength '+r'$(\AA)$', fontsize=10)
            ax0.set_ylabel('Normalized flux '+r'(erg s$^{-1}$ cm$^{-2}$)', fontsize=10)
            ax0.text(0.02,0.90,'Model flux ratio: '+str(fitgrid_sort.sel(fitinfo='ratio').isel_points(Nmodel=[t0]).values)[2:6],transform=ax0.transAxes, fontsize=9, color='red', weight='roman')
            ax0.text(0.02,0.82,'Obs flux ratio: '+str(Ioratio.values)[0:4],transform=ax0.transAxes, fontsize=9, color='black', weight='roman')
            ax0.set_xlim(6350, 6750)
            if any(t == '100' for t in nsh) == True:
                ax0.set_ylim(0.2, 13.2)
                ax0.yaxis.set_major_locator(ticker.MultipleLocator(base=2))
            else:
                ax0.set_ylim(0.2, 42)
                ax0.yaxis.set_major_locator(ticker.MultipleLocator(base=10))
            ax0.xaxis.set_major_locator(ticker.MultipleLocator(base=100))
            ax0.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
            ax0.yaxis.tick_right()
            ax0.yaxis.set_label_position("right")

            ax1.plot(wave40, obsP*100, color='black', drawstyle='steps-mid', linewidth=1)
            ax1.errorbar(wave40, obsP*100, yerr=obsPe*100,color = 'darkgrey', capsize=3, capthick=1,linestyle='none', linewidth=1 )
            ax1.errorbar(wave40BN, obsPBN*100, yerr=obsPeBN*100, color = 'black', capsize=3, capthick=1, linestyle='none', linewidth=1)
            ax1.plot(wave40, modelP.isel(Nmodel=t0)*100, color='red', drawstyle='steps-mid', linewidth=1.15)
            ax1.plot(wave40BN, modelPBN.isel(Nmodel=t0)*100, linestyle='',mfc='red', marker='*', ms=11, mew=1.0, mec='black')
            if SN == 'SN 1997eg':
                ax1.set_ylim(0.8, 4.2)
            if SN == 'SN 2010jl':
                ax1.set_ylim(-0.02, 3.2)
            ax1.set_xlim(6145, 6755)
            ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
            ax1.yaxis.set_major_locator(ticker.MultipleLocator(base=1))
            ax1.set_xlabel('wavelength '+r'$(\AA)$', fontsize=10)
            ax1.set_ylabel('Polarization (%)', fontsize=10)
            ax1.text(0.03,0.90,s0+'  '+r'$\tau$: '+str(s1)+'  T: '+s4, transform=ax1.transAxes, fontsize=10, weight='roman', color='black')
            ax1.text(0.03,0.82,'Lcsm: '+str(s2)+'  Lsh: '+str(s3)+'  '+r'$\theta$:'+' '+s5+r'$\degree$', transform=ax1.transAxes, fontsize=10, weight='roman', color='black')
            ax1.text(0.03,0.74,'shift: '+str('{:5.2f}'.format(PS)),transform=ax1.transAxes, fontsize=10, weight='roman', color='black')
#            ax1.text(0.5,0.9,G,transform=ax1.transAxes, fontsize=10, weight='roman', color='black')
            ax1.text(0.88,0.90,r'$\chi_{\nu}^2$: '+str(fitgrid_sort.sel(fitinfo='rchi2').isel_points(Nmodel=[t]).values)[2:6],transform=ax1.transAxes, fontsize=10, weight='roman', color='black')

#            plt.tight_layout()
#            if regime == 'central':
#                plt.savefig(prefix+SN+'_'+str(epoch)+'_model'+str(t)+'_C.png', dpi=600, bbox_inches='tight',format='png')
#            if regime == 'dist':
#                plt.savefig(prefix+SN+'_'+str(epoch)+'_model'+str(t)+'_D.png', dpi=600, bbox_inches='tight',format='png')
#            plt.close()
            ax0.yaxis.label.set_visible(False)
            ax1.yaxis.label.set_visible(False)
            if t < 3:
#                plt.setp(ax.get_yticklabels(), visible=False)
#                ax.yaxis.label.set_visible(False)
#                plt.setp(ax1.get_yticklabels(), visible=False)
#                ax1.yaxis.label.set_visible(False)
                plt.setp(ax0.get_xticklabels(), visible=False)
                ax0.xaxis.label.set_visible(False)
                plt.setp(ax1.get_xticklabels(), visible=False)
                ax1.xaxis.label.set_visible(False)
        fig.text(0.94, 0.5, 'Normalized flux '+r'(erg s$^{-1}$ cm$^{-2}$)', fontsize=12, rotation='vertical', va='center')
        fig.text(0.08, 0.5, 'Polarization (%)', fontsize=12, rotation='vertical', va='center')
#        plt.tight_layout()
        plt.subplots_adjust(hspace=0.06)
        plt.subplots_adjust(wspace=0.03)
        if regime == 'central':
#            plt.savefig('C://Users//Leah//Desktop//'+SN+'_'+str(epoch)+'_Best4_1C.png', dpi=600, bbox_inches='tight',format='png')
            plt.savefig('C://Users//Leah//Desktop//'+SN+'_'+str(epoch)+'_Best4_2C.png', dpi=600, bbox_inches='tight',format='png')
        if regime == 'dist':
#            plt.savefig('C://Users//Leah//Desktop//'+SN+'_'+str(epoch)+'_Best4_1D.png', dpi=600, bbox_inches='tight',format='png')
            plt.savefig('C://Users//Leah//Desktop//'+SN+'_'+str(epoch)+'_Best4_2D.png', dpi=600, bbox_inches='tight',format='png')
        plt.close()

#            ax2 = plt.subplot(gs0[t,2])
#            kurts = stats.kurtosistest(rs.isel(Nmodel=t))
#            kurtstat = kurts.statistic
#            kurtsp = kurts.pvalue
#            skews = stats.skewtest(rs.isel(Nmodel=t))
#            skewstat = skews.statistic
##            norms = stats.normaltest(rs.isel(Nmodel=t))
##            normstat = norms.statistic
#            ax2 = sns.distplot(rs.isel(Nmodel=t), rug=True, fit=stats.norm)
#
#            ax2.text(0.03,0.9,'kurt:'+str(kurtstat)[0:4],
#                     transform=ax2.transAxes, fontsize=9, weight='roman')
#            ax2.text(0.03,0.8,'skew:'+str(skewstat)[0:4],
#                     transform=ax2.transAxes, fontsize=9, weight='roman')
#            ax2.set_xlim(-6, 6)
#            plt.subplots_adjust(wspace=0.05)
##---------------------------------------------------------------------------------------

#
#    geom_set = ['dsk','tor']
#    tau_set = ['05','1','2']
#    ncsm_set = '0'
#    nsh_set = '2'
#    bin_set = 5
#
#    wave = obsframe.wave
#    Io = obsframe.sel(spectra = 'I')
#    Ioe = obsframe.sel(spectra = 'Ie')
#    Ioe2 = Ioe**2
#
#    fitarray = fitgrid.unstack('Nmodel')
#
#    modelarrayBN = modelstack.unstack('Nmodel')
#    modelarrayBN = modelarrayBN.transpose('geometry','tau','ncsm','nsh','temp','incbin','wave','spectra')
#    modelarrayBN = modelarrayBN.sel(spectra='P').sel(temp='10')
#    modelarray = trim_modelframe.sel(spectra='I').sel(temp='10').sel(incbin=slice(2,12)).sel(geometry='dsk')
#
#    modelarrayI = modelarray.stack(Nmodel=('geometry','tau','ncsm','nsh','incbin'))
#    modelP = modelarray.sel(geometry=geom_set).sel(tau=tau_set).sel(ncsm=ncsm_set).sel(nsh=nsh_set).isel(incbin=bin_set)
#    modelI = modelarray.values
# -----------------------------------------------------------------------
#    RSDS = modelstack.sel(spectra ='residuals').unstack('Nmodel')
#    cmap = sns.husl_palette(12,h=3.0,l=0.6,s=0.8)
#
#    fig = plt.figure(figsize=(20,10))
##    fig.set_rasterized(True)
#    gs1=gridspec.GridSpec(3,4)
#
#    for j in range(0,len(geom_set)):
#        for i in range(0,len(tau_set)):
#                RS = RSDS.sel(geometry=geom_set[j]).sel(tau=tau_set[i]).sel(ncsm=ncsm_set).sel(nsh=nsh_set)
#                RS = np.reshape(RS.values,np.product(RS.shape))
#
#                ax = plt.subplot(gs1[i,j])
#                ax = sns.distplot(RS, bins=10, rug=True, fit=stats.norm)
##                ax.set_rasterized(True)
#    plt.tight_layout()
#    plt.savefig(savefile+'residual distribution.pdf',format='pdf', dpi=1000)
#    plt.close()
#
#    fig = plt.figure(figsize=(20,10))
#    gs1=gridspec.GridSpec(3,4)
#
#    for j in range(0,len(geom_set)):
#        for i in range(0,len(tau_set)):
#                RS = RSDS.sel(geometry=geom_set[j]).sel(tau=tau_set[i]).sel(ncsm=ncsm_set).sel(nsh=nsh_set)
#                RS = np.reshape(RS.values,np.product(RS.shape))
#
#                ax = plt.subplot(gs1[i,j])
##                stats.probplot(RS,dist='norm',fit=True,plot=ax)
#                ax = sns.distplot(RS,hist_kws=dict(cumulative=True),kde_kws=dict(cumulative=True), bins=10 )
#
#    plt.tight_layout()
#    plt.savefig(savefile+'CDF.pdf',format='pdf', dpi=3000)
#    plt.close()

#-------------------- PLOT Poe^2 vs Model P ------------------------------
#    modelI = np.reshape(modelI, np.product(modelI.shape),order='F')
#    Ioe2 = np.tile(Ioe2,(264,1))
#    Ioe2 = Ioe2.T
#    Ioe2 = np.reshape(Ioe2, np.product(Ioe2.shape),order='F')
#
#    shift = fitarray.sel(fitinfo='total')
#    cmap = sns.cubehelix_palette(as_cmap=True)
##        cmap = sns.husl_palette(16,h=3.0,l=0.6,s=0.8)
#    onemod = 192
#    ax = sns.jointplot(Ioe2,modelI,
#                         color='b',
#                         kind="reg")#, ylim=(-0.06,0.06))
##                             joint_kws=dict(bins=500, rug=True))
#    ax.set_axis_labels("Poe2", "ModelP")
#    ax.ax_joint.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0e'))
#    plt.tight_layout()
#    plt.savefig(savefile+'error distribution.pdf',format='pdf', dpi=3000)
#    plt.close()


#    plot.plot_joint(sns.distplot(disks.values, bins=20, kde=False))
#    model = 'dsk_1_1_01_10k'

##====================================================================================
# PLOT POLARIZATION RANGES BY INCLINATION ANGLE FOR UNSHIFTED MODELS
#colors = ['maroon','red','DarkOrange','OrangeRed','Gold','LimeGreen','SeaGreen','DarkTurquoise','DodgerBlue','Blue','DarkOrchid','Indigo' ]
##colors = colors.reverse()
#plt.figure(figsize=(18, 22))
#gs1=gridspec.GridSpec(3,1)
#
##ax1 = plt.subplot(gs1[1])
##    models = models.sel(geometry=geom_set).sel(tau=tau_set).sel(ncsm=ncsm_set).sel(nsh=nsh_set)
##    models = modelarray.sel(geometry='dsk').sel(tau='5').sel(ncsm='0').sel(nsh='01')
#colors = sns.hls_palette(12,l=0.4,s=0.75)    # rainbow palette, good for individual angles
##    for j in range(len(models.incbin)):
#
#grid = modelstack_40.unstack('Nmodel')
#grid = grid.sel(spectra='P')
#grid = grid.stack(N=('geometry','tau','ncsm','nsh'))
#fits = models40.unstack('Nmodel')
#fits = fits.sel(spectra='P')
#fits = fits.stack(N=('geometry','tau','ncsm','nsh'))
#
##ax1.plot(obsframe_40.wave, obsframe_40.sel(spectra='P'), color = sns.xkcd_rgb['black'],
##        drawstyle='steps-mid', linewidth=1)
##ax1.errorbar(obsframe_40.wave, obsframe_40.sel(spectra='P'), yerr=obsframe_40.sel(spectra='Pe'),
##            color = sns.xkcd_rgb['black'], capsize=3, capthick=1, linestyle='none', linewidth=1 )
#
#for t in range(0,len(temp)):
#    for i in range(0,len(thetas)):
##for i in reversed(range(0,len(thetas))):
#        for j in range(1,len(grid.N)):
#            ax = plt.subplot(gs1[t])
#
#            ax.plot(obsframe_40.wave, grid.isel(incbin=i).isel(temp=t).isel(N=j),
#                    color=colors[i],linewidth=1.25)
#            ax.plot(obsframe_40.wave, obsframe_40.sel(spectra='P'), color = sns.xkcd_rgb['black'],
#                     drawstyle='steps-mid', linewidth=3, zorder=10)
#            ax.errorbar(obsframe_40.wave, obsframe_40.sel(spectra='P'), yerr=obsframe_40.sel(spectra='Pe'),
#                        color = sns.xkcd_rgb['black'], capsize=3, capthick=1, linestyle='none',
#                        linewidth=3, zorder=10)
#            ax.set_xlim(6150,6750)
#            ax.set_ylim(0.01, 0.04)
##    y = grid.isel(incbin=i)
##    y1 = y.max(dim='N')
##    y2 = y.min(dim='N')
##    ax0.fill_between(obsframe_40.wave, y1, y2, facecolor=colors[i],alpha=0.3)
#
#
##ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
#plt.suptitle('SLIP Polarization vs Inclination: Disk Geometry')
#
##plt.tight_layout()
#plt.savefig(savefile+'SLIP-PolVsInc-Tor2.png',format='png', dpi=1000)
#plt.close()

#------------------------------------------------------------------------------------------------------------------
#  PLOT INCLINATION VS PHOTON FRACTION for different geometry and tau combinations
#        models = modelarray.sel(geometry=geom_set).sel(tau=tau_set)
#        #colors = sns.cubehelix_palette(n_colors=12, start=3.0, gamma=0.8, hue=1.5,rot=-1.0)
#        colors = sns.husl_palette(14,h=3.0,l=0.6,s=0.8)
#        fig = plt.figure(figsize=(25,20))
#        gs1 = gridspec.GridSpec(3,2)
#        coord = [[0.0,0.1],[0.01,0.2,100],deg]
#
#        for k in range(len(models.nsh)):
#            for m in range(len(models.ncsm)):
#
#                ax = plt.subplot(gs1[k,m])
#
#                ax.set_title('ncsm: '+str(coord[0][m])+' nsh: '+str(coord[1][k]), fontsize=20)
#                for j in range(len(models.incbin)):
#                     ax.plot(models.wave, models.isel(ncsm=m).isel(incbin=j).isel(nsh=k),
#                              color=colors[j], drawstyle='steps-mid', linewidth=1.25,
#                              label='{:2.0f}'.format(coord[2][j]))
#
#                plt.legend(loc="lower left", ncol=3, bbox_to_anchor=[0.01,0.01], fancybox=True, edgecolor='black', shadow=True)
#                ax.plot(models.wave, Po, color = sns.xkcd_rgb['black'], drawstyle='steps-mid', linewidth=1.25)
#                ax.errorbar(models.wave, Po, yerr=Poe, color = sns.xkcd_rgb['black'], capsize=3, capthick=1,
#                                  linestyle='none', linewidth=1.25 )
#                ax.yaxis.set_major_locator(ticker.LinearLocator(4))
#                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
#                plt.axis([6150, 6750, 0.01, 0.04])
#
#        gs1.tight_layout(fig, rect=[0.04, 0.04, 0.97, 0.95])   # rect = (left, bottom, right, top)
#        fig.text(0.5,0.03,'Wavelength (Angstroms)',horizontalalignment='center',fontsize=25)
#        fig.text(0.02,0.5,'Fractional Polarization',rotation='vertical',verticalalignment='center',fontsize=25 )
#        plt.suptitle('Disks, Tau:0.5, Temp:10k: Polarization vs Inclination for varying photon fractions',x=0.5,y=0.97, fontsize=30)
#
#        plt.savefig(savefile+geom_set+'_'+tau_set+'_10k_VaryPhotons-allBins.eps',format='eps', dpi=5000)
#        plt.close()

#------------------------------------------------------------------------------------------------------------------
#  PLOT INCLINATION VS TAU, GEOM for different photon fractions
#        models = modelarray.sel(ncsm=ncsm_set).sel(nsh=nsh_set)
#        colors = sns.husl_palette(14,h=3.0,l=0.6,s=0.8)
#        fig = plt.figure(figsize=(25,20))
#        gs1 = gridspec.GridSpec(3,2)
#        coord = [['dsk','tor'],[0.5,1.0,2.0],deg]
#        for k in range(len(models.tau)):
#            for m in range(len(models.geometry)):
#
#                ax = plt.subplot(gs1[k,m])
#                ax.set_title('geom: '+ coord[0][m] + '  tau: '+str(coord[1][k]), fontsize=20)
#                for j in range(len(models.incbin)):
#                     ax.plot(models.wave, models.isel(geometry=m).isel(incbin=j).isel(tau=k),
#                              color=colors[j], drawstyle='steps-mid', linewidth=1.25,
#                              label='{:2.0f}'.format(coord[2][j]) )
#
#                plt.legend(loc="lower left", ncol=3, bbox_to_anchor=[0.01,0.01], fancybox=True, edgecolor='black', shadow=True)
#                ax.plot(models.wave, Po, color = sns.xkcd_rgb['black'], drawstyle='steps-mid', linewidth=1.25)
#                ax.errorbar(models.wave, Po, yerr=Poe, color = sns.xkcd_rgb['black'], capsize=3, capthick=1,
#                                  linestyle='none', linewidth=1.25 )
#                ax.axis([6150, 6750, 0.0, 0.1])
#
#        gs1.tight_layout(fig, rect=[0.04, 0.04, 0.97, 0.95])   # rect = (left, bottom, right, top)
#        fig.text(0.5,0.03,'Wavelength (Angstroms)',horizontalalignment='center',fontsize=25)
#        fig.text(0.02,0.5,'Fractional Polarization',rotation='vertical',verticalalignment='center',fontsize=25 )
#        plt.suptitle('NCSM:'+str(ncsm_set)+', NSH:'+str(nsh_set)+' Temp:10k: Polarization vs Inclination for varying geometry and optical depth',x=0.5,y=0.97, fontsize=30)
#
#        plt.savefig(savefile+'NCSM'+ncsm_set+'_NSH'+nsh_set+'_10k_VaryGeomTau-allBins.eps',format='eps', dpi=5000)
#        plt.close()

#------------------------------------------------------------------------------------------------------------------
#   PLOT all models(geom, tau, photon fractions) at a single viewing angle
#        models = modelarray.sel(incbin=bin_set)
#        colors = sns.color_palette('Paired')
#        fig = plt.figure(figsize=(25,20))
#        gs1 = gridspec.GridSpec(3,2)
#        coord = [['dsk','tor'],[0.5,1.0,2.0],['0.0','0.1'],['0.01','0.2','100']]
#        for k in range(len(models.tau)):
#            for m in range(len(models.geometry)):
#                ax = plt.subplot(gs1[k,m])
#                ax.set_title('geom: '+ coord[0][m] + '  tau: '+ str(coord[1][k]), fontsize=20)
#
#                for j in range(len(models.nsh)):
#                    for n in range(len(models.ncsm)):
#                        ax.plot(models.wave, models.isel(nsh=j).isel(ncsm=n).isel(geometry=m).isel(tau=k),
#                                color=colors[(2*j)+n], drawstyle='steps-mid', linewidth=1.25,               #[j+(3*n)] for j dominant
#                                label=coord[2][n]+':'+coord[3][j])
#                plt.legend(loc="lower left", ncol=3, bbox_to_anchor=[0.01,0.01], fancybox=True, edgecolor='black', shadow=True)
#
#                ax.plot(models.wave, Po, color = sns.xkcd_rgb['black'], drawstyle='steps-mid', linewidth=1.25)
#                ax.errorbar(models.wave, Po, yerr=Poe, color = sns.xkcd_rgb['black'], capsize=3, capthick=1,
#                            linestyle='none', linewidth=1.25 )
#                ax.axis([6150, 6750, 0.0, 0.04])
#
#        gs1.tight_layout(fig, rect=[0.04, 0.04, 0.97, 0.95])   # rect = (left, bottom, right, top)
#        fig.text(0.5,0.03,'Wavelength (Angstroms)',horizontalalignment='center',fontsize=25)
#        fig.text(0.02,0.5,'Fractional Polarization',rotation='vertical',verticalalignment='center',fontsize=25 )
#        plt.suptitle( 'Inclination Bin:'+str(bin_set)+' Temp:10k',x=0.5,y=0.97, fontsize=30)
#
#        plt.savefig(savefile+'Bin'+str(bin_set)+'_10k_.eps',format='eps', dpi=5000)
#        plt.close()

#------------------------------------------------------------------------------------------------------------------

#        plt.figure(figsize=(20, 12))
#        #gs1=gridspec.GridSpec(12,1)
##        modellist = np.array((['tor','05','0','01','10',12],['tor','1','0','01','10',12],['dsk','1','1','01','10',11],['dsk','1','0','2','10',12]))
#
##        colors = sns.hls_palette(4,l=0.4,s=0.75)    # rainbow palette, good for individual angles
#        for j in range(len(modellist)):
#             plt.plot( modelarray.wave,
#                       #modelarray.sel_points(geometry=modellist[j][0],)
#                       modelarray.sel_points(geometry=[modellist[j,0]], tau=[modellist[j,1]], ncsm=[modellist[j,2]], nsh=[modellist[j,3]], temp=[modellist[j,4]], incbin=[modellist[j,5]] ),
#                       color=colors[j], drawstyle='steps-mid', linewidth=1)
#
#        plt.plot(modelarray.wave, Po, color = sns.xkcd_rgb['black'], drawstyle='steps-mid', linewidth=1)
#        plt.errorbar(modelarray.wave, Po, yerr=Poe, color = sns.xkcd_rgb['black'], capsize=3, capthick=1, linestyle='none', linewidth=1 )
#        plt.title('dsk_05_0_01_10k - Polarization vs Inclination')
#        plt.axis([6150,6750, 0.0, 0.08])
#        plt.tight_layout()
#        plt.savefig(savefile+'InterestingModels-Normalized.eps',format='eps', dpi=3000)
#        plt.close()


    #plt.title(model+'_'+str(theta)+'_'+str(res)+'A  Polarized Spectrum')
    ##plt.plot,wave,P,color='red'
    #for ii in range(len(wave)-1):
    #    plt.plot([wave[ii], wave[ii+1]], [P[ii], P[ii]], '-b')
    #    plt.plot([wave[ii+1], wave[ii+1]], [P[ii], P[ii+1]], '-b')
    #plt.axis([5950, 7030, ymin, ymax])
    #plt.xlabel('wavelength (A)')
    #plt.ylabel('Percent polarization')
    #plt.tight_layout()
    #plt.savefig(savefile+model+'_'+str(theta)+' PolarizedSpectrum.jpg',format='jpg',dpi=2000)
    #plt.close()
