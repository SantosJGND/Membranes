
import scipy
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import estimate_bandwidth
from sklearn.cluster import MeanShift, estimate_bandwidth

import pandas as pd
import itertools as it
import os

import plotly
import plotly.plotly as py
import plotly.graph_objs as go

def read_image(img,data_dir= 'data/',flt_sep= '.',sort_abc= True):
    '''
    read ImageJ image transect output.
    '''
    image_dir= data_dir + img + '/' 

    trenches= []
    abcissa= []
    file_list= [name for name in os.listdir(image_dir)]
    file_flt= {}
    
    for transeto in file_list:

        abcis= transeto.strip('x').split(flt_sep)[:-1]
        abcis= flt_sep.join(abcis)
        abcis= float(abcis)
        
        file_flt[transeto]= abcis
    
    trans_sort= sorted(file_flt,key= file_flt.get)
    
    for transeto in trans_sort:
        abcis= file_flt[transeto]
        filename= image_dir + transeto

        with open(filename,'r') as fp:
            lines= fp.readlines()
        
        values= [x.strip().split()[1] for x in lines]

        abcissa.append(abcis)
        trenches.append(values)
    
    trenches= np.array(trenches,dtype= float)
    
    return trenches, abcissa


def get_spectros(data_dir):
    '''
    get image transect data across instances.
    '''
    available= [name for name in os.listdir(data_dir)]
    image_dict= {}

    for img in available:
        
        trenches, abcissa= read_image(img,data_dir= data_dir)

        image_dict[img]= {
            'I': trenches,
            'surface': abcissa
        }
        
    return image_dict


from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
from scipy.stats import norm

def frame_peaks(array_spec,
                spec_fs, 
                spec_ts,
                frame= 0,
                Sample_N= 500,
                p_threshold= 0.0004,
                amp_cap= 1e6,
                peak_cap= .3,
                peak_iso= 200,
                band_qtl= 0.02,
                frame_plot= False,
               label= 'title',
               extremes= True,
               center= True):
    '''
    get peaks from single frame of strictly positive values.
    frame to be found as column of array array_spec. 
    - spec_fs = range of variation.
    - spec_ts = frame factor list.
    - frame= index in array_spec 2nd dimension (columns).
    '''
    ## get probs from
    probs= list(array_spec[:,frame]) 
    probs= np.array(probs)

    probs[probs > amp_cap]= amp_cap

    prob_sum= np.sum(probs)
    probs= probs / prob_sum
        
    
    # #############################################################################
    # Compute clustering with MeanShift
    # The following bandwidth can be automatically detected using
    new_freqs= np.random.choice(list(spec_fs),Sample_N,replace= True,p= probs)

    new_freqs= new_freqs.reshape(-1,1)

    bandwidth = estimate_bandwidth(new_freqs, quantile=band_qtl, n_samples=Sample_N)
    
    if bandwidth== 0:
        bandwidth= peak_iso

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=False,cluster_all= False).fit(new_freqs)

    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    cluster_centers= list(it.chain(*cluster_centers))

    ## trim_clusters:
    ### interpolation makes it easier to chose between neibhour centroids
    ### that are unlikely to exist as obs. frequency values.
    from scipy.interpolate import interp1d
    f2= interp1d(spec_fs, array_spec[:,frame], kind='cubic')
    
    cluster_centers= cluster_threshold(cluster_centers,f2,t= peak_iso)
    #### get amplitudes of peaks and store them
    cluster_center_amps= {}
    peak_cent= []
    amps_centres= []

    shapes= []
    for cent in sorted(cluster_centers):
            
        closest= abs(spec_fs - cent)
        #
        closet= np.argmin(closest)

        amp_sel= array_spec[closet,frame]
        cluster_center_amps[cent]= amp_sel

    cluster_amps= list(cluster_center_amps.values())

    for cent in sorted(cluster_centers):
        amp_sel= cluster_center_amps[cent]        
        amp_sel_pval= norm.cdf(amp_sel,loc= np.mean(cluster_amps),scale= np.std(cluster_amps))

        #print(cent, amp_sel,amp_sel_pval)
        #print(amp_sel,np.mean(cluster_amps))
        if amp_sel < np.mean(cluster_amps):
            continue
        if amp_sel >= peak_cap:
            peak_cent.append(cent)
            amps_centres.append(amp_sel)
    

    cluster_centers= peak_cent
    
    # remove extremes if necessary
    ext= [min(cluster_centers),max(cluster_centers)]
    if extremes:
        ext_idx= [x for x in range(len(cluster_centers)) if cluster_centers[x] in ext]
        
        cluster_centers= [cluster_centers[x] for x in ext_idx]
        amps_centres= [amps_centres[x] for x in ext_idx]
        
    #### Center if requested.
    if center:
        mean_c= np.mean(list(set(cluster_centers)))
        cluster_centers= [x - mean_c for x in cluster_centers]
        
        spec_fs= spec_fs - mean_c
    
    peak_cent= cluster_centers
    
    ## get time stamps for each of the peaks.
    time_spec= [spec_ts[frame]]* len(amps_centres)

    if frame_plot:
        
        kde= KernelDensity(kernel='gaussian', bandwidth= bandwidth).fit(new_freqs)
        X_plot = np.linspace(0, max(spec_fs) + 100, 1000)[:, np.newaxis]
        log_dens = kde.score_samples(X_plot)
        
        fig= [go.Scatter(
            x= spec_fs,
            y= array_spec[:,frame],
            mode= 'lines'
        )]
        
        shapes= []

        for center in peak_cent:

            shapes.append({
                'type': 'line',
                'x0': center,
                'y0': 0,
                'x1': center,
                'y1': max(array_spec[:,frame]),
                'line': {
                    'color': 'red',
                    'width': 4,
                    'dash': 'solid'
                },
            })
        
        layout= go.Layout(
            title= 'ID: {}; frame inx: {}'.format(label, frame),
            shapes= shapes,
            xaxis= dict(title= 'frequency'),
            yaxis= dict(title= 'amplitude')
        )
        
        figure_frame= go.Figure(data= fig,layout= layout)
        
        return peak_cent, time_spec, amps_centres, figure_frame

    else:
        return peak_cent, time_spec, amps_centres


    
def cluster_threshold(center_list,scor_func,t= 200):
    trim_cl= {}

    for cml in center_list:
        if not trim_cl:
            trim_cl[0]= np.array([cml])
            continue

        d= 0

        for clamp in trim_cl.keys(): 
            dists= abs(trim_cl[clamp] - cml)

            if min(dists) < t:
                trim_cl[clamp]= np.array([*list(trim_cl[clamp]),cml])
                d += 1

        if d == 0:
            trim_cl[len(trim_cl)]= np.array([cml])
    
    pval_trim= [[scor_func(x) for x in trim_cl[z]] for z in trim_cl.keys()]
    #pval_trim= [[np.exp(sklearn_scor.score_samples(x.reshape(-1,1))) for x in trim_cl[z]] for z in trim_cl.keys()]
    trim_cl= [trim_cl[x][np.argmax(pval_trim[x])] for x in trim_cl.keys()]

    return trim_cl

