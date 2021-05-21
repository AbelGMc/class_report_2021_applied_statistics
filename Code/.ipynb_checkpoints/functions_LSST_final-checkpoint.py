#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stat
def curve_plot(curve,pb_wavelength):
    plt.figure()
    if(curve.shape[1]==3):
        for band in range(6):
            index_band = curve[:,1] == pb_wavelength[band]
            if sum(index_band)==0:
                continue
            x = curve[index_band,0]
            y = curve[index_band,2]
            plt.plot(x, y, '.', label='band %d'%band)
            plt.xlabel('mjd')
            plt.ylabel('flux')
            plt.legend(loc='upper right')
            plt.title("Flux data for curves")
        plt.show()
    else:
        for band in range(6):
            curve_band = curve[curve[:,2]==band,:]
            x = curve_band[:,1]
            y = curve_band[:,3]
            plt.plot(x, y, '.-', label='band %d'%band)
        plt.xlabel('mjd')
        plt.ylabel('flux')
        plt.legend(loc='upper right')
        plt.title("Flux data for curves")
        
def get_argumented_curves(curve_data,gp_regressor_list,n_added_timepoints,class_to_numbers,train_meta_data,pb_wavelength):
    """Get features from augumented training features and targets.
    There is a one-one relation between curve_data,gp_regressor_list,train_meta_data
    
    Parameters
    --------------
    curve_data: :class: `list`
        original list of curve data, ordered by train_meta_data. 
    gp_regressor_list: :class: `list` 
        each element is a fitted gaussian process regressor, ordered by train_meta_data. 
    n_timepoints: :class: `int`
        how many points to be placed evenly between min and max of mjd
    class_to_numbers: :class: `dictionary`
        key: class name(int) value: numbers to sample(int)
    train_meta_data: :class: `pandas data.frame`
        Our meta data
    pb_wavelength: `numpy array`
        wave lengthes for 0~6 passband
    
    Returns
    --------------
    [arg_curves, arg_meta]: :class: `list`
        arg_curves: list of np ndarray with colums for mjd, wavelength, flux
        arg_meta: list of np ndarray with colums for ra,dec,host_photoz,host_photoz_err,target,object_id,length_param
    """
    n_train = len(gp_regressor_list)

    arg_curves = []
    arg_meta = []
    
    for i in range(n_train):
        time_data = curve_data[i][curve_data[i][:,2]==1][:,1]
        n_timepoints = n_added_timepoints + len(time_data)
        time_min = time_data.min()
        time_max = time_data.max()
        
        x_sampling = np.concatenate( (np.linspace(time_min,time_max,n_added_timepoints),np.array(time_data)) )
        x_sampling = (np.tile(x_sampling,3)).reshape(-1,1) # np.tile: repeat entir arrary
        wave_sampling = (np.repeat(pb_wavelength[[1,3,5]],n_timepoints)).reshape(-1,1) # np.repeat: repeat each. i.e. first n_timepoints values are for passband 0 and so on.
        X_sampling = np.concatenate((x_sampling,wave_sampling),axis = 1)
        ## 
        try:
            y_sampling = gp_regressor_list[i].sample_y(X_sampling, n_samples=int(class_to_numbers[train_meta_data.iloc[i,11]]), random_state=125)  # with shape ( 3*n_timepoints, n_sampling )
            
            for j in range(int(class_to_numbers[train_meta_data.iloc[i,11]])):
                arg_curves.append(np.concatenate((X_sampling,y_sampling[:,j].reshape(-1,1) ),axis=1))
                arg_meta.append( train_meta_data.iloc[i,[1,2,6,8,11,0]] )

        except:
            y_sampling = gp_regressor_list[i].predict(X_sampling).reshape(-1,1)
            arg_curves.append(np.concatenate((X_sampling,y_sampling ),axis=1))
            temp = np.array(train_meta_data.iloc[i,[1,2,6,8,11,0]] )
            temp = np.concatenate((temp,np.array([gp_regressor_list[1].kernel_.get_params()["k1__k1__constant_value"]])))
            arg_meta.append(temp)
            print("error in curve %d with kernel"%i)
            print(gp_regressor_list[i].kernel_)

        for k in range(10):
            if(i==(k+1)*int(n_train/10)):
                print("%d%% done..."%(10*(k+1)))
    
    return([ arg_curves, arg_meta ])
    
def curves_to_training(arg_curves,arg_meta,pb_wavelength):
    """Get features from augumented training features and targets.
    
    Parameters
    --------------
    curve_data: :class: `list`
        list of np ndarray with colums for mjd, wavelength, flux
    arg_meta: :class: `list` 
        list of np ndarray with colums for ra,dec,host_photoz,host_photoz_err,target,object_id,length_param
    pb_wavelength: `numpy array`
        wave lengthes for 0~6 passband
    
    Returns
    --------------
    [X,y]
    """
    n_train = len(arg_curves)
    colnames = {0:"ra",1:"decl",2:"host_photoz",3:"host_photoz_err",
            4:"max_flux_ratio_blue",5:"min_flux_ratio _blue",6:"max_flux_ratio_red",7:"min_flux_ratio_red",
            8:"max_mag",
            9:"time_bwd_max_0.2",10:"time_bwd_max_0.5",
            11:"pos_flux_ratio",12:"max_dt",
            13:"skewness",14:"Kurtosis",
            15:"length_param",
            16:"skewness_g_i",17:"Kurtosis_g_i",18:"skewness_i_y",19:"Kurtosis_i_y",
           }
    p_feature = len(colnames)
    
    y = np.array([meta_data[4] for meta_data in arg_meta])
    X = np.zeros((n_train,p_feature))
    
    for i in range(4):
        X[:,i] = np.array([meta_data[i] for meta_data in arg_meta])
    
    for i in range(n_train):
        curve = arg_curves[i]
        time_max = np.max(curve[:,0])
        time_min = np.min(curve[:,0])
        flux_curve_g = curve[curve[:,1]==pb_wavelength[1],2]
        time_curve_g = curve[curve[:,1]==pb_wavelength[1],0]

        flux_curve_i = curve[curve[:,1]==pb_wavelength[3],2]
        time_curve_i = curve[curve[:,1]==pb_wavelength[3],0]

        flux_curve_y = curve[curve[:,1]==pb_wavelength[5],2]
        time_curve_y = curve[curve[:,1]==pb_wavelength[5],0]
        ###-----------------------------------
        # 4:"max_flux_ratio_blue",5:"min_flux_ratio _blue",6:"max_flux_ratio_red",7:"min_flux_ratio_red"
        # [max,min]_flux_ratio_[blue,red], Blue for g-i and red for i-y
        ##----
        relative_g_i = flux_curve_g - flux_curve_i / (np.abs(flux_curve_g)+np.abs(flux_curve_i))
        relative_g_i[np.where(np.isfinite(relative_g_i)==False)] = 0
        relative_i_y = flux_curve_i - flux_curve_y / (np.abs(flux_curve_i)+np.abs(flux_curve_y))
        relative_i_y[np.where(np.isfinite(relative_i_y)==False)] = 0
        
        X[i,4] = np.max(relative_g_i)
        X[i,5] = np.min(relative_g_i)
        X[i,6] = np.max(relative_i_y)
        X[i,7] = np.min(relative_i_y)

        ###----------------------------------
        # 8:"max_mag"
        X[i,8] = np.max(flux_curve_i)

        ###----------------------------------
        # 9:"time_max_0.2",10"time_max_0.5"
        temp = (time_curve_i[np.where(flux_curve_i<0.8*np.max(flux_curve_i))] - time_curve_i[np.argmax(flux_curve_i)]).reshape(-1,1)
        if(temp.shape[0]==0):
            X[i,9] = 2*(time_max - time_min)
        else:
            X[i,9] = np.min(np.abs(temp))
        temp = (time_curve_i[np.where(flux_curve_i<0.5*np.max(flux_curve_i))] - time_curve_i[np.argmax(flux_curve_i)]).reshape(-1,1)
        if(temp.shape[0]==0):
            X[i,10] = 2*(time_max - time_min)
        else:
            X[i,10] = np.min(np.abs(temp))
        ###----------------------------------
        # 11:"pos_flux_ratio",12:"max_dt"
        X[i,11]= np.max(flux_curve_i)/(np.max(flux_curve_i)-np.min(flux_curve_i))
        X[i,12] = time_curve_y[np.argmax(flux_curve_y)] - time_curve_g[np.argmax(flux_curve_g)]        
        ###----------------------------------
        # 13:"skewness",14:"Kurtosis"
        X[i,13] = stat.skew(flux_curve_i)
        X[i,14] = stat.kurtosis(flux_curve_i)

        X[i,15] = arg_meta[i][-1]

        ###----------------------------------
        # 16:"skewness_g_i",17:"Kurtosis_g_i",18:"skewness_i_y",19:"Kurtosis_i_y",
        X[i,16] = stat.skew(relative_g_i)
        X[i,17] = stat.kurtosis(relative_g_i)

        X[i,18] = stat.skew(relative_i_y )
        X[i,19] = stat.kurtosis(relative_i_y)
        
        for k in range(10):
            if(i==(k+1)*int(n_train/10)):
                print("%d%% done..."%(10*(k+1)))
    return([X,y])
    
####==============================================================


def get_curves_from_raw(curve_data,gp_regressor_list,n_added_timepoints,train_meta_data,pb_wavelength):
    """Get features from augumented training features and targets.
    There is a one-one relation between curve_data,gp_regressor_list,train_meta_data
    
    Parameters
    --------------
    curve_data: :class: `list`
        original list of curve data
    gp_regressor_list: :class: `list` 
        each element is a fitted gaussian process regressor, ordered by train_meta_data. 
    n_timepoints: :class: `int`
        how many points to be placed evenly between min and max of mjd
    train_meta_data: :class: `pandas data.frame`
        Our meta data
    pb_wavelength: `numpy array`
        wave lengthes for 0~6 passband
    
    Returns
    --------------
    [arg_curves, arg_meta]: :class: `list`
        arg_curves: list of np ndarray with colums for mjd, wavelength, flux
        arg_meta: list of np ndarray with colums for ra,dec,host_photoz,host_photoz_err,target,object_id
    """
    n_train = len(gp_regressor_list)
    n_arg_train = n_train*1

    arg_curves = []
    arg_meta = []
    
    for i in range(n_train):
        time_data = curve_data[i][curve_data[i][:,2]==1][:,1]
        n_timepoints = n_added_timepoints + len(time_data)
        time_min = time_data.min()
        time_max = time_data.max()

        x_sampling = np.concatenate( (np.linspace(time_min,time_max,n_added_timepoints),np.array(time_data)) )
        x_sampling = (np.tile(x_sampling,3)).reshape(-1,1) # np.tile: repeat entir arrary
        wave_sampling = (np.repeat(pb_wavelength[[1,3,5]],n_timepoints)).reshape(-1,1) # np.repeat: repeat each. i.e. first n_timepoints values are for passband 0 and so on.
        X_sampling = np.concatenate((x_sampling,wave_sampling),axis = 1)

        y_sampling = gp_regressor_list[i].predict(X_sampling)  

        arg_curves.append(np.concatenate((X_sampling,y_sampling.reshape(-1,1) ),axis=1))
        arg_meta.append( train_meta_data.iloc[i,[1,2,6,8,11,0]] )

        for k in range(10):
            if(i==(k+1)*int(n_train/10)):
                print("%d%% done..."%(10*(k+1)))
    
    return([ arg_curves, arg_meta ])
