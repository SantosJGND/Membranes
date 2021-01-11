
import scipy
import numpy as np

from scipy.signal import find_peaks, find_peaks_cwt

import pandas as pd
import itertools as it
import os

import plotly
import plotly.graph_objs as go
from scipy.signal import find_peaks, find_peaks_cwt

##
import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



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


class frame_obj:
    def __init__(self, amps ,frame= "0"):
        self.amps= amps
        self.frame= frame
    
    def peaks(self, spec_fs, peak_cap= .35):
        
        if peak_cap >= 1: 
            peak_min= peak_cap 
        else:
            peak_min= max(self.amps) * peak_cap
            
        clust_find, _= find_peaks(self.amps,
                                  height= peak_min)
        #
        self.clust_idx= clust_find
        self.cluster_centers= spec_fs[clust_find]
        self.amps_centres= self.amps[clust_find]

        self.neighs= {
            idx: {
                "c": [spec_fs[idx-1], spec_fs[idx +1]],
                "a": [self.amps[idx-1], self.amps[idx+1]]
            } for idx in clust_find
        }
        

    def maxima(self):
        if len(self.cluster_centers) != 1:

            clust_idx= {
                self.cluster_centers[x]: self.clust_idx[x] for x in range(len(self.cluster_centers))
            }
            clust_dict= {
                self.cluster_centers[x]: self.amps_centres[x] for x in range(len(self.cluster_centers))
            }

            amps_sort=  sorted(clust_dict,key= clust_dict.get, reverse= True)
            self.cluster_centers= amps_sort[:2]
            self.clust_idx= [clust_idx[x] for x in self.cluster_centers]
            self.amps_centres= [clust_dict[x] for x in self.cluster_centers]
    
    def extremes(self):
        ext= [min(self.cluster_centers),max(self.cluster_centers)]
        
        ext_idx= [x for x in range(len(self.cluster_centers)) if self.cluster_centers[x] in ext]
        
        self.clust_idx= [self.clust_idx[x] for x in ext_idx]
        self.cluster_centers= [self.cluster_centers[x] for x in ext_idx]
        self.amps_centres= [self.amps_centres[x] for x in ext_idx]
        
    def centre(self):
        mean_c= np.mean(list(set(self.cluster_centers)))
        self.orimean= mean_c
        self.cluster_centers= [x - mean_c for x in self.cluster_centers]
        for idx in self.neighs.keys():
            self.neighs[idx]["c"]= [x - mean_c for x in self.neighs[idx]["c"]]
        
#spec_ts= surface

class peak_finder:
    
    def __init__(self, pxl_dist= 4e-4, peak_cap= .35):
        self.pxl_dist= pxl_dist
        self.peak_cap= peak_cap
        self.centered= False
        
    def reset(self):
        self.centered= False
        self.spec_ts= []
    
    def peaks(self, array_spec, spec_ts= []):
        
        transect_length= array_spec.shape[0] * self.pxl_dist
        spec_fs= np.linspace(0,transect_length,array_spec.shape[0])
        
        frames= []
        for idx in range(array_spec.shape[1]):
            frame= frame_obj(array_spec[:,idx],frame= idx)
            frame.peaks(spec_fs, peak_cap= self.peak_cap)
            frames.append(frame)
        
        self.spec_fs= spec_fs
        self.frames= frames
        
        if spec_ts:
            self.spec_ts= spec_ts
        else:
            self.spec_ts= list(range(array_spec.shape[0]))

    def maxima(self):
        for frame in self.frames:
            frame.maxima()
    
    def extremes(self):
        for frame in self.frames:
            frame.extremes()
    
    def centre(self):
        for frame in self.frames:
            frame.centre()
        
        self.centered= True

    def plot(self, idx= 0):
        frame= self.frames[idx]
        surface= np.array(self.spec_fs)
        
        if self.centered:
            surface= surface - np.mean(frame.orimean)
        fig= [go.Scatter(
            x= surface,
            y= frame.amps,
            mode= 'lines'
        )]
        
        shapes= []

        for center in frame.cluster_centers:

            shapes.append({
                'type': 'line',
                'x0': center,
                'y0': 0,
                'x1': center,
                'y1': max(frame.amps),
                'line': {
                    'color': 'red',
                    'width': 4,
                    'dash': 'solid'
                },
            })
        
        layout= go.Layout(
            title= 'ID: {}; frame inx: {}'.format(str(frame.frame), idx),
            shapes= shapes,
            xaxis= dict(title= 'frequency'),
            yaxis= dict(title= 'amplitude')
        )
        
        figure_frame= go.Figure(data= fig,layout= layout)
        iplot(figure_frame)
        
    def package(self):
        ts_list= []
        peaks= []
        amps= []
        
        new_array= []

        for frame in self.frames:
            #
            for idx in range(len(frame.cluster_centers)):
                clidx= frame.clust_idx[idx]
                nline= [self.spec_ts[frame.frame]]
                nline.extend([frame.neighs[clidx]["c"][0], frame.cluster_centers[idx], frame.neighs[clidx]["c"][1]])
                nline.extend([frame.neighs[clidx]["a"][0], frame.amps_centres[idx], frame.neighs[clidx]["a"][1]])
                new_array.append(nline)
                peaks.extend(frame.cluster_centers)
                ts_list.extend([self.spec_ts[frame.frame]]* len(frame.cluster_centers))
                amps.extend(frame.amps_centres)
        
        self.neigh_tracks= np.array(new_array)
        self.tracks= np.array([
            ts_list,
            peaks,
            amps
        ]).T
    
    def write(self, outdir= './', filename= 'out.txt'):
        table= pd.DataFrame(self.neigh_tracks,columns= ['frame_pos','peakXm1','peakX','peakXp1','peak.amp.m1','peak.amp','peak.amp.p1'])
        filename= outdir + filename
        table.to_csv(filename, sep= '\t', index= False)
        
        
    def get(self):
        return self.tracks
    

