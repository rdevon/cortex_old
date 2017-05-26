'''
Module for AOD analysis functions

'''

import igraph
import numpy as np
import os
from os import path
from scipy.stats import (describe, kendalltau, linregress, mannwhitneyu, ttest_1samp,
                         ttest_ind, ttest_rel)
import statsmodels.api as sm
from tabulate import tabulate
import theano
import yaml

from .analyzer import Analyzer

def rename_rois(labels):
    for i in xrange(len(labels)):
        label = labels[i]
        label = label.replace('Left', 'L')
        label = label.replace('Right', 'R')
        label = label.replace('Inferior', 'Inf')
        label = label.replace('Middle', 'Mid')
        label = label.replace('Anterior', 'Ant')
        label = label.replace('Gyrus', 'Gy')
        label = label.replace('Superior', 'Sup')
        label = label.replace('Medial', 'Med')
        label = label.replace('Orbitalis', 'Orb')
        labels[i] = label[:35]
    return labels

def get_betas(tcs, stim, idx):
    betas = []
    stim = sm.add_constant(stim)
    for s in idx:
        model = sm.OLS(tcs[:, s], stim)
        results = model.fit()
        betas.append(results.params[1:])
    betas = np.array(betas)
    return betas.transpose(1, 2, 0)

def get_task_relatedness(tcs, stim, idx=None):
    if idx is None: idx = range(tcs.shape[1])
    betas = get_betas(tcs, stim, idx)
    
    if betas.ndim == 2:
        t, p = ttest_1samp(betas[:, idx], 0, axis=1)
    elif betas.ndim == 3:
        ts = []
        ps = []
        for betas_ in betas:
            t, p = ttest_1samp(betas_[:, idx], 0, axis=1)
            ts.append(t)
            ps.append(p)
        t = np.array(ts)
        p = np.array(ps)
    return t, p

def get_task_difference(tcs, stim, idx1, idx2, use_mw=True):
    idx = range(tcs.shape[1])
    betas = get_betas(tcs, stim, idx)

    if use_mw:
        utests = np.array([mannwhitneyu(np.array(betas)[i, idx1],
                                        np.array(betas)[i, idx2],
                                        alternative='two-sided')
                           for i in range(len(betas))])
        t = utests[:, 0]
        p = utests[:, 1]
    else:
        if betas.ndim == 2:
            t, p = ttest_ind(betas[:, idx1], betas[:, idx2], axis=1)
        elif betas.ndim == 3:
            ts = []
            ps = []
            for betas_ in betas:
                t, p = ttest_ind(betas_[:, idx1], betas_[:, idx2], axis=1)
                ts.append(t)
                ps.append(p)
            t = np.array(ts)
            p = np.array(ps)
    return t, p

def do_fdr_correct(p, has_dependence=False, sig=0.01):
    if p.ndim > 1:
        shape = p.shape
        p = p.reshape((reduce(lambda x, y: x * y, shape),))
    else:
        shape = None
    m = p.shape[0]
    p_sorted = sorted(p)
    p_argsort = np.argsort(p)
    if has_dependence:
        c = np.array([1 / float(i + 1) for i in xrange(m,)]).sum()
    else:
        c = 1.
    fdr = np.array([sig * (i + 1) / (m * c) for i in xrange(m)])
    w = np.where(p_sorted < fdr)[0]
    try:
        k = w.max() + 1
    except ValueError:
        k = 0
    mask = np.zeros_like(p)
    mask[p_argsort[:k]] = 1.
    if shape is not None: mask = mask.reshape(shape)
    return k, mask

def group(mat, thr=.2, do_abs=False, sort=False):
    if do_abs: mat = abs(mat)
    max_weight = abs(mat).max()
    #thr *= max_weight
    idx = range(mat.shape[0])
    wheres = np.where(mat > thr)

    edgelist = []
    weights = []

    for x, y in zip(wheres[0], wheres[1]):
        if x < y:
            edgelist.append((x, y))
            weights.append((mat[x, y]))

    if len(weights) > 0:
        weights /= np.std(weights)
    else:
        return range(mat.shape[0]), [[i] for i in idx]

    g = igraph.Graph(edgelist, directed=False)
    g.vs['label'] = idx
    cls = g.community_multilevel(return_levels=True, weights=weights)
    cl = list(cls[0])
    if sort:
        cl = sorted(cl, key=len)[::-1]

    clusters = []
    n_clusters = len(cl)
    for i in idx:
        found = False
        for j, cluster in enumerate(cl):
            if i in cluster:
                clusters.append(j)
                found = True
                break

        if not found:
            clusters.append(n_clusters)
            n_clusters += 1

    clusters = np.array(clusters).astype('int64')
    return clusters, cl
        

class AODAnalyzer(Analyzer):
    def __init__(self, spatial_map_key, time_course_key, run=False,
                 label_file=None, sign_flip=False, **kwargs):
        self.spatial_map_key = spatial_map_key
        self.time_course_key = time_course_key
        self.roi_dict = None
        self.images = None
        self.sign_flip = sign_flip
        self.stats = {}
        super(AODAnalyzer, self).__init__(**kwargs)
        if run:
            self.run(label_file=label_file)
        
    def build(self):
        super(AODAnalyzer, self).build()
        nifti_out_path = path.join(self.session.manager.out_path, 'niftis')
        if not path.isdir(nifti_out_path):
            os.mkdir(nifti_out_path)
        self.visualizer.add('data.make_images',
                            self.spatial_map_key,
                            set_global_norm=True,
                            out_path=nifti_out_path)
        
        self.visualizer.add('data.viz', maps='ica_viz.maps',
                            time_courses='ica_viz.tcs',
                            time_course_scales='ica_viz.tc_scales', t_limit=50,
                            y=7, title='')
        
        self.f_tcs = theano.function(self.session.inputs,
                                     self.tensors[self.time_course_key],
                                     updates=self.session.updates,
                                     on_unused_input='ignore')
        self.targets = self.session.manager.datasets['data'][self.mode].extras['targets']
        self.novels = self.session.manager.datasets['data'][self.mode].extras['novels']
        self.Y = self.session.manager.datasets['data'][self.mode].Y[:, 0]
        
    def make_roi_dict(self, inputs=None):
        self.logger.info('Making ROI dictionary')
        if inputs is None: inputs = self.get_data()
        images, niftis, roi_dict = self.visualizer.run(0, inputs=inputs, data_mode=self.mode)
        self.roi_dict = roi_dict
        self.images = images
        self.niftis = niftis
        
    def get_feature_info(self, label_file=None):
        if label_file is not None:
            with open(label_file, 'r') as f:
                f_info = yaml.load(f)
        else:
            labels = [self.roi_dict[k]['top_clust']['rois']
                      if len(self.roi_dict[k]['top_clust']['rois']) > 0 else ['?']
                      for k in self.roi_dict.keys()]
            labels = [l[0] if len(l) == 1 else l for l in labels]
            f_info = dict((k, dict(labels=l, is_on=True)) for k, l in enumerate(labels))
                    
        return f_info
                
    def set_features(self, label_file=None):
        self.logger.info('Setting features')
        f_info = self.get_feature_info(label_file=label_file)
        for k, v in f_info.items():
            labels = v['labels']
            if isinstance(labels, list):
                name = labels[0]
            else:
                name = labels
                labels = [labels]
            if self.sign_flip is not None:
                for i, (l, sf) in enumerate(zip(labels, self.sign_flip)):
                    if sf == -1:
                        l_ = l.replace('+', '%tmp')
                        l_ = l_.replace('-', '+')
                        l_ = l_.replace('%tmp', '-')
            self.features[k] = dict(name=name, labels=labels,
                                    is_on=v.get('is_on', True), stats={})
        
    def run(self, label_file=None):
        self.logger.info('Running analysis')
        inputs = self.get_data()
        self.make_roi_dict(inputs)
        
        if self.sign_flip:
            a = np.array([self.images[i].get_data()
                          for i in range(len(self.images))]).reshape((60, -1))
            self.sign_flip = (2 * (describe((a - a.mean(axis=0)) / a.std(), axis=1).skewness > 0) - 1)
            self.visualizer.add('data.viz', maps='ica_viz.maps',
                                time_courses='ica_viz.tcs',
                                time_course_scales='ica_viz.tc_scales', t_limit=50,
                                y=7, title='', sign_flip=self.sign_flip)
            
        else:
            self.sign_flip = None
            
        self.set_features(label_file=label_file)
        
        self.logger.info('Getting time courses')
        tcs = self.f_tcs(*inputs)
        
        if isinstance(tcs, np.ndarray):
            tcs = {self.time_course_key: tcs}
        elif isinstance(tcs, list):
            tcs = dict((k, v) for k, v in zip(self.time_course_key, tcs))
            
        if self.sign_flip is not None:
            for k, v in tcs.items():
                if k == 'Scale':
                    continue
                tcs[k] = v * np.array(self.sign_flip)[None, None, :]
            
        self.tcs = tcs
        
        hc = np.where(self.Y[:, 0] == 1)[0].tolist()
        sz = np.where(self.Y[:, 1] == 1)[0].tolist()
        
        self.logger.info('T-tests')
        for k, v in tcs.items():
            self.stats[k] = {}
            stim = np.concatenate([self.targets[:, None], self.novels[:, None]],
                axis=1)
            t, p = get_task_relatedness(v, stim)
            u, pu = get_task_difference(v, stim, hc, sz, use_mw=False)
            
            for sname, t_, p_, u_, pu_ in zip(['targets', 'novels'], t, p, u, pu):
                self.stats[k][sname] = dict(p=p_, t=t_, u=u_, pu=pu_)
                    
                for j in xrange(t_.shape[0]):
                    self.features[j]['stats'].update(
                        **{'{}_{}_t'.format(k, sname): t_[j],
                            '{}_{}_p'.format(k, sname): p_[j],
                            '{}_{}_u'.format(k, sname): u_[j],
                            '{}_{}_pu'.format(k, sname): pu_[j]})
                    
    def make_fnc(self, tc, idx=None, sig=0.05, thr=0., has_dependence=False,
                 use_average=False, clusters=None, omit_off=True, **kwargs):
        tc = self.tcs[tc]
        
        if idx is None: idx = range(tc.shape[1])
        if omit_off:
            c_idx = [k for k in self.features.keys() if self.features[k]['is_on']]
            tc = tc[:, :, c_idx]
        cc = np.array([np.corrcoef(tc[:, i].T) - np.eye(tc.shape[2]) for i in idx])
        cc_av = cc.mean(0)
        tt, tp = ttest_1samp(cc, 0, axis=0)
        k, mask = do_fdr_correct(tp, has_dependence=has_dependence, sig=sig)
        tt_sig = tt * mask
    
        if use_average:
            return cc_av
        else:
            return tt_sig
        
    def save_map(self, idx=None, order=None, clusters=None, omit_off=True, map_idx=-1):
        inputs = self.get_data()
        if clusters is None and order is None:
            order = [k for k in self.features.keys() if self.features[k]['is_on']]
        labels = [self.features[k]['labels'] for k in self.features.keys()]
        self.visualizer.run(
            map_idx, inputs=inputs, data_mode=self.mode,
            name='maps', order=order, clusters=clusters, labels=labels)
        
    def make_table(self, task_sig=0.01, min_p=10e-7, tablefmt='plain',
                   group_diffs=False, omit_empty_columns=True):
        fdr_dict = dict()
        diff_dict = dict()
        
        for tc_name in self.stats.keys():
            stats = self.stats[tc_name]
            keys = stats.keys()
            fdr = dict()
            diff = dict()
            
            for sname in keys:
                p = stats[sname]['p']
                K, mask = do_fdr_correct(p, sig=task_sig)
                asp = np.argsort(p)[:K]
                fdr[sname] = asp
                
                if group_diffs:
                    pu = stats[sname]['pu']
                    K, mask = do_fdr_correct(pu, sig=task_sig)
                    asp = np.argsort(pu)[:K]
                    diff[sname] = asp
                
            fdr_dict[tc_name] = fdr
            diff_dict[tc_name] = diff
    
        td = ['', '']        
        if group_diffs:
            for tc_name in self.stats.keys():
                td += [tc_name, '', 'diff_{}'.format(tc_name), '']
            table = [td, ['ID', 'Label'] + ['Targets', 'Novels'] * (2 * len(self.stats))]
        else:
            for tc_name in self.stats.keys():
                td += [tc_name, '']
            table = [td, ['ID', 'Label'] + ['Targets', 'Novels'] * len(self.stats)]
        
        for i, feature in self.features.items():
            if feature['is_on']:
                td = [i, feature['name']]
                for tc_name in self.stats.keys():
                    stats = self.stats[tc_name]
                    
                    for sname in stats.keys():
                        asp = fdr_dict[tc_name][sname]
                        stat = stats[sname]['p'][i]
                        td.append('%.1e' % stat if (i in asp and stat < min_p)
                                  else '')
                        
                    if group_diffs:
                        for sname in stats.keys():
                            asp = diff_dict[tc_name][sname]
                            stat = stats[sname]['pu'][i]
                            td.append('%.1e' % stat if (i in asp and stat < min_p)
                                      else '')
                    
                if not all([t == '' for t in td[2:]]):
                    table.append(td)
                    
        if omit_empty_columns:
            for j in range(len(table[0]))[::-1]:
                column = [table[i][j] for i in range(len(table))[2:]]
                if all([c == '' for c in column]):
                    for i in range(len(table)):
                        table[i].pop(j)
        
        print tabulate(table, headers='firstrow', tablefmt=tablefmt)
        
    def save_niftis(self):
        pass
        
    def save_maps(self, task_sig=0.01, min_p=10e-7):
        self.logger.info('Saving maps')
        inputs = self.get_data()
        
        for tc_name in self.stats.keys():
            stats = self.stats[tc_name]
            keys = stats.keys()
            asp = set()
            
            for sname in keys:
                p = stats[sname]['p']
                K_ = next((i for i, p_ in enumerate(np.sort(p)[::-1]) if p_ <= min_p), None)
                if K_ is None:
                    K_ = 0
                else:
                    K_ = p.shape[0] - K_
                K, mask = do_fdr_correct(p, sig=task_sig)
                K = min(K, K_)
                idx = np.argsort(p)[:K]
                asp.update(set(idx))
            
            if len(asp) > 0:
                order = [i for i in np.argsort(stats['targets']['p']) if i in asp]
                self.visualizer.run(-1, inputs=inputs, data_mode=self.mode,
                                    name='{}_map'.format(tc_name), order=order,
                                    stats=dict(p_t=stats['targets']['p'],
                                               p_n=stats['novels']['p']))
            
            asp = set()
            for sname in keys:
                p = stats[sname]['pu']
                K_ = next((i for i, p_ in enumerate(np.sort(p)[::-1]) if p_ <= min_p), None)
                if K_ is None:
                    K_ = 0
                else:
                    K_ = p.shape[0] - K_
                K, mask = do_fdr_correct(p, sig=task_sig)
                K = min(K, K_)
                idx = np.argsort(p)[:K]
                asp.update(set(idx))
                
            if len(asp) > 0:
                order = [i for i in np.argsort(stats['targets']['pu']) if i in asp]
                self.visualizer.run(-1, inputs=inputs, data_mode=self.mode,
                                    name='{}_map_diff'.format(tc_name), order=order,
                                    stats=dict(p_t=stats['targets']['pu'],
                                               p_n=stats['novels']['pu']))
            
            
            
    