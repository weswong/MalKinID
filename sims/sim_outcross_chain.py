#!/usr/bin/env python
# coding: utf-8


from pandas import DataFrame, read_csv
import pandas as pd
import random
import numpy as np
import json
from collections import defaultdict, Counter
import math
import sys
import itertools
import time
import scipy.stats
import sys
import matplotlib.pyplot as plt
import itertools
import random
sys.path.insert(0, '/Users/weswong/Documents/GitHUb/Pedigree/Cotransmission/genome/')
import genome
from genome import Genome
from genome import utils
genome.initialize_from('sequence')
from sklearn.neighbors import KernelDensity


#Meiosis -------------------------------------------------------


chr_lengths = {1:643292,
               2:947102,
               3:1060087,
               4:1204112,
               5:1343552,
               6:1418244,
               7:1501717,
               8:1419563,
               9:1541723,
               10: 1687655,
               11:2038337,
               12:2271478,
               13:2895605,
               14:3291871}


#equation relating v to the scale in obligate chiasma version
obligate_co_scale_fn = np.poly1d(np.array([  2.86063522e-05,  -1.28111927e-03,   2.42373279e-02,
                                            -2.52092360e-01,   1.57111461e+00,  -5.99256708e+00,
                                             1.36678013e+01,  -1.72133175e+01,   9.61531678e+00]))

def oc_get_crossover_points(v, chrom_length, bp_er_cM=11300):
    '''obligate chiasma version
    Generate the first obligate chiasma by drawing from a Uniform Distribution
    Expand outwards from that point until you reach both ends of the chromosome'''
    xpoints=[]
    obligate_chiasma_pos = int(math.ceil(np.random.uniform(low=0., high= float(chrom_length))))
    xpoints.append(obligate_chiasma_pos)
    bp_per_cM = bp_er_cM
    scale = obligate_co_scale_fn(v)

    #move to the right
    interarrival_time = np.random.gamma(shape=v, scale =scale)
    d = int(math.ceil(interarrival_time * bp_per_cM* 100))
    right_point = d + obligate_chiasma_pos
    while right_point < chrom_length:
        xpoints.append(right_point)
        interarrival_time = np.random.gamma(shape=v, scale =scale)
        d = int(math.ceil(interarrival_time * bp_per_cM* 100))
        right_point += d
    
    #move to the left
    interarrival_time = np.random.gamma(shape=v, scale =scale)
    d = int(math.ceil(interarrival_time * bp_per_cM* 100))
    left_point = obligate_chiasma_pos - d
    while left_point > 0:
        xpoints.append(left_point)
        interarrival_time = np.random.gamma(shape=v, scale =scale)
        d = int(math.ceil(interarrival_time * bp_per_cM* 100))
        left_point -= d
        
    return xpoints

def crossover(g1,g2,xpoints):
    #S phase, DNA duplication time
    
    #sister chromatids on homologous chromosome #1
    c1 = np.copy(g1)
    c2 = np.copy(g1)
    
    #sister chromatids on homologous chromosome #2
    c3 = np.copy(g2)
    c4 = np.copy(g2)
    if not xpoints:
        return c1,c2, c3,c4
    
    for breakpoint in xpoints:
        probability = np.random.random()
        if probability < 0.25: # c1 and c3
            t = np.copy(c1[breakpoint:])
            c1[breakpoint:] = c3[breakpoint:]
            c3[breakpoint:] = t
        elif probability >= 0.25 and probability < 0.5: #c1 and c4
            t = np.copy(c1[breakpoint:])
            c1[breakpoint:] = c4[breakpoint:]
            c4[breakpoint:] = t
        elif probability >= 0.5 and probability < 0.75: #c2 and c3
            t = np.copy(c2[breakpoint:])
            c2[breakpoint:] = c3[breakpoint:]
            c3[breakpoint:] = t
        elif probability >=0.75: #c2 and c4
            t = np.copy(c2[breakpoint:])
            c2[breakpoint:] = c4[breakpoint:]
            c4[breakpoint:] = t
    return c1, c2, c3, c4

def meiosis(in1,in2,N=4,v=2, oc=True, bp_per_cM = 11300):
    '''v defines the shape of the gamma distribution, it is required to have a non-zero shape parameter
    if v = 0, we assume user means no crossover model
    v =1 corresponds to no interference
    obligate crossover means use the obligate crossover version'''
    if N > 4:
        raise IndexError('Maximum of four distinct meiotic products to sample.')
    genomes=[genome.reference_genome() for _ in range(4)]
    
    for idx,(start,end) in enumerate(utils.pairwise(Genome.chrom_breaks)):
        c1,c2=in1.genome[start:end],in2.genome[start:end]
        xpoints = oc_get_crossover_points(v, len(c1), bp_per_cM)

        #log.debug('Chr %d, xpoints=%s',chrom_names[idx],xpoints)
        c1, c2, c3, c4=crossover(c1,c2,xpoints)
        
        #independent assortment
        outputs=sorted([c1,c2,c3,c4], key=lambda *args: random.random())       
        for j in range(4):
            genomes[j][start:end]=outputs[j]
    return [Genome(genomes[j]) for j in range(N)]


def sample_n_oocysts_function(alpha=2.5,beta=1.0):
    # Fit A.Ouedraogo membrane feeding data from Burkina Faso
    # Private communication, in preparation (2015)
    return 1 + int(random.weibullvariate(alpha, beta))
oocyst_distribution = np.asarray([sample_n_oocysts_function(2.5, 1.0) for _ in range(5000)])
oocyst_count_distribution = Counter(oocyst_distribution)
total = np.sum(list(oocyst_count_distribution.values()))
oocyst_count_prop = {}
for k,v in oocyst_count_distribution.items():
    oocyst_count_prop[k] = v / total
cdf = 0
oocyst_cdf_array = []
for x in sorted(oocyst_count_prop.keys()):
    cdf += oocyst_count_prop[x]
    oocyst_cdf_array.append(cdf)
    
    
truncated_oocyst_cdf_array = oocyst_cdf_array[:10]
oocyst_counts  = [x for x in sorted(oocyst_count_prop.keys())][:10]

def prob_meiotic_sibling():
    random_X = np.random.uniform(0,1)
    bin_idx = np.digitize(random_X, truncated_oocyst_cdf_array)
    sampled_oocyst_count = oocyst_counts[bin_idx - 1*(bin_idx == 10)]
    prob_meiotic = (1/sampled_oocyst_count)
    return round(prob_meiotic,2)




chrom_dict_lengths= {1: 643000,
                         2: 947000,
                         3: 1100000,
                         4: 1200000,
                         5: 1350000,
                         6: 1420000,
                         7: 1450000,
                         8: 1500000,
                         9: 1550000,
                         10: 1700000,
                         11: 2049999,
                         12: 2300000,
                         13: 2950000,
                         14: 3300000}
    
position_maps_dict = {}
for chrom_n in range(1,15):
    position_maps_dict[chrom_n] = np.arange(1, chrom_dict_lengths[chrom_n]+1)
    
class Simulation:
    position_maps_dict = position_maps_dict
    
    def __init__(self, v=2, kbp_cM=11.3, meioticsibling=False):
        self.v = v
        self.kbp_cM = kbp_cM
        self.bp_per_cM = self.kbp_cM * 1000
        
        if meioticsibling == True:
            self.pedigree_dict = self.simulate_meiotic_sibling_pedigree()
        else:
            self.pedigree_dict = self.simulate_pedigree()
        self.relationships = self.name_relationships()
        self.calculate_stats()
        self.count_ibd_segments()    
        
    def simulate_pedigree(self, pedigree_dict=None):
        if not pedigree_dict:
            pedigree_dict = {}
            for _, node_id in enumerate(['P1', 'P2', 'A1', 'A2', 'A3']):
                genome = Genome.from_reference()
                genome.genome = genome.genome + _
                genome.id = node_id
                pedigree_dict[node_id] = genome
        
    
        pedigree_dict['F1'] = meiosis(pedigree_dict['P1'], pedigree_dict['P2'], N=1, v=self.v, bp_per_cM = self.bp_per_cM)[0]
        pedigree_dict['F12'] = meiosis(pedigree_dict['P1'], pedigree_dict['P2'], N=1, v=self.v, bp_per_cM = self.bp_per_cM)[0]
        
        pedigree_dict['F2'] = meiosis(pedigree_dict['A1'], pedigree_dict['F1'], N=1, v=self.v, bp_per_cM = self.bp_per_cM)[0]
        pedigree_dict['F22'] = meiosis(pedigree_dict['A1'], pedigree_dict['F1'], N=1, v=self.v, bp_per_cM = self.bp_per_cM)[0]

        pedigree_dict['F3'] = meiosis(pedigree_dict['A2'], pedigree_dict['F2'], N=1, v=self.v, bp_per_cM = self.bp_per_cM)[0]
        pedigree_dict['F32'] = meiosis(pedigree_dict['A2'], pedigree_dict['F2'], N=1, v=self.v, bp_per_cM = self.bp_per_cM)[0]
        
        pedigree_dict['F4'] = meiosis(pedigree_dict['A3'], pedigree_dict['F3'], N=1, v=self.v, bp_per_cM = self.bp_per_cM)[0]
        pedigree_dict['F42'] = meiosis(pedigree_dict['A3'], pedigree_dict['F3'], N=1, v=self.v, bp_per_cM = self.bp_per_cM)[0]


        return pedigree_dict

    
    def name_relationships(self):
        relationships = {}
        for combo in itertools.combinations(self.pedigree_dict.keys(), 2):
            n1,n2 = combo
            relationships[n1 + '.' + n2] = (n1, n2)
        return relationships
    
    def calculate_ibd_segment_boundaries(self, ibd_map, chrom_n): # position map is entire chromosome
        ibd_segment_boundaries = []
        q = ibd_map[1:]
        p = ibd_map[:-1]
        comparison = q != p
        flip_points = np.where(comparison)[0]
        position_map = Simulation.position_maps_dict[chrom_n]
        
        if len(position_map) == 0:
            position_map = np.arange(1, len(ibd_map) + 1)
            #np.asarray([x + 1 for x in range(len(ibd_map))])

        if len(flip_points) == 0: # no flip points
            if ibd_map[0] == 1:
                ibd_segment_boundaries.append((position_map[0], position_map[-1]))
            
        elif len(flip_points) == 1: # 1 flip point
            flip_point = flip_points[0]
            identity = p[flip_point]
            #print('identity', identity)
            if identity == 1: # changes from 1 -> 0 (ie starts off in IBD)
                ibd_segment_boundaries.append((position_map[0], position_map[flip_point]))
            else:
                ibd_segment_boundaries.append((position_map[flip_point], position_map[-1]))

        else: # len(flip_points) > 0 multiple flip points; indices on IBD map where transitions 0-1 or 1-0
            #print('test', flip_points)
            for idx,flip_point in enumerate(flip_points):
                identity = p[flip_point] #if 1, changes from 1 -> 0, if 0, changes from 0-> 1

                if idx == 0: # for first flip point
                    if identity == 1: # IBD
                        ibd_segment_boundary = (1, position_map[flip_point])
                        ibd_segment_boundaries.append(ibd_segment_boundary)

                else: # for all flip point segments
                    prev_flip_point = flip_points[idx-1] 
                    ibd_segment_boundary = (position_map[prev_flip_point], position_map[flip_point]) # left start

                    if identity == 1: 
                        ibd_segment_boundaries.append(ibd_segment_boundary)

                if idx == len(flip_points) -1: #for the last flip_point
                    if ibd_map[-1] == 1: 
                        ibd_segment_boundary = (position_map[flip_point], position_map[-1])
                        ibd_segment_boundaries.append(ibd_segment_boundary)
        return ibd_segment_boundaries
        
    def calculate_ibd_block_length(self, ibd_segment_boundaries): 
        ibd_block_lengths = [] # if no IBD segment, return an empty list
        for segment_boundary in ibd_segment_boundaries: #segment_boundaries = (start, stop)
            ibd_block_length = segment_boundary[1] - segment_boundary[0] + 1
            ibd_block_lengths.append(ibd_block_length) 
        return ibd_block_lengths

    def count_ibd_segments(self):
        ibd_block_numbers = {}
        for relationship in self.ibd_segment_lengths:
            ibd_block_numbers[relationship] = {}
            #ibd_block_numbers[relationship] = self.ibd_block_dict[relationship]
            for chrom_n in range(1,15):
                #print(self.ibd_block_dict)
                #print(relationship, chrom_n, self.ibd_block_dict[relationship][chrom_n])
                ibd_block_number = len([x for x in self.ibd_segment_lengths[relationship][chrom_n] if float(x) != 0.0])
                ibd_block_numbers[relationship][chrom_n] = ibd_block_number

        self.ibd_block_numbers = ibd_block_numbers
        
    
    
    def calculate_stats(self):
        ibd_segment_lengths = {}
        dbd_segment_lengths = {}
        r_totals = {}
        ibd_maps = {}
        all_ibd_segment_boundaries = {}
        all_dbd_segment_boundaries = {}
        max_ibd_segment_lengths = {}
        r_chrom = {}
        
        for relationship in self.relationships:
            ibd_segment_lengths[relationship] = {}
            dbd_segment_lengths[relationship] = {}
            all_ibd_segment_boundaries[relationship] = {}
            all_dbd_segment_boundaries[relationship] = {}
            ibd_maps[relationship] = {}
            max_ibd_segment_lengths[relationship] = {}
            r_chrom[relationship] = {}
            
            
            n1, n2 = self.relationships[relationship]
            x1, x2 = self.pedigree_dict[n1], self.pedigree_dict[n2]
            ibd_map = x1.genome == x2.genome
            r_total = np.mean(ibd_map)
            r_totals[relationship] = r_total

            for chrom_n in range(1,15): 
                ibd_map = x1.return_chromosome(chrom_n) == x2.return_chromosome(chrom_n)
                r_chrom[relationship][chrom_n] = np.mean(ibd_map)
                ibd_segment_boundaries = self.calculate_ibd_segment_boundaries(ibd_map,chrom_n)
                all_ibd_segment_boundaries[relationship][chrom_n] = ibd_segment_boundaries
                
                ibd_segment_lengths[relationship][chrom_n] = self.calculate_ibd_block_length(ibd_segment_boundaries)
                dbd_segment_lengths[relationship][chrom_n] = self.calculate_dbd_block_length(ibd_segment_boundaries, chrom_n)
                if len(ibd_segment_lengths[relationship][chrom_n]) == 0:
                    ibd_segment_lengths[relationship][chrom_n] = [0]
                max_ibd_segment_lengths[relationship][chrom_n] = max(ibd_segment_lengths[relationship][chrom_n])
                ibd_maps[relationship][chrom_n] = ibd_map
        
        self.all_ibd_segment_boundaries = all_ibd_segment_boundaries   
        self.dbd_segment_lengths = dbd_segment_lengths
        self.ibd_segment_lengths = ibd_segment_lengths
        self.r_totals = r_totals
        #self.ibd_maps = ibd_maps
        self.max_ibd_segment_lengths = max_ibd_segment_lengths
        self.r_chrom = r_chrom

    #def plot_ibd_segments(self, relationship):
    #    plt.figure(figsize = (15, 10))
    #    for chrom_n in range(1,15):
    #        ibd_map = self.ibd_maps[relationship][chrom_n]
    #        plt.subplot(3,5,chrom_n)
    #        plt.suptitle(relationship)
    #        plt.title(chrom_n)
    #        plt.plot(range(len(ibd_map)), ibd_map)
    #        plt.ylim(-0.05,1.05)
            
# Checking and unit testing ------------------------------------
    def calculate_dbd_block_length(self, ibd_segment_boundaries, chrom):
        dbd_block_lengths = []
        end_chrom = chrom_dict_lengths[chrom]
        #print(chrom, ibd_segment_boundaries, end_chrom)
    
        if len(ibd_segment_boundaries)>1:
            if ibd_segment_boundaries[0][0] != 1: # first IBD block isn't from the start of the chromosome
                dbd_block_lengths = [ibd_segment_boundaries[0][0] - 1] + dbd_block_lengths
            if ibd_segment_boundaries[-1][1] != end_chrom:
                dbd_block_length = end_chrom - ibd_segment_boundaries[-1][1]
                dbd_block_lengths.append(dbd_block_length)
                
            for idx, value in enumerate(ibd_segment_boundaries):
                if idx == 0: # shift 
                    continue
                else:
                    previous_segment, current_segment = (ibd_segment_boundaries[idx-1], ibd_segment_boundaries[idx])
                    dbd_block_length = current_segment[0] - previous_segment[1] - 1
                    dbd_block_lengths.append(dbd_block_length)
        
        elif len(ibd_segment_boundaries) == 1:
            if (ibd_segment_boundaries[0][0]==1) & (ibd_segment_boundaries[0][1] == end_chrom):
                dbd_block_lengths.append(0)
            elif (ibd_segment_boundaries[0][0] == 1):
                if ibd_segment_boundaries[0][1] != end_chrom:
                    dbd_block_lengths.append(end_chrom - ibd_segment_boundaries[0][1] - 1)
            elif (ibd_segment_boundaries[0][0] != 1):
                dbd_block_lengths.append(ibd_segment_boundaries[0][0] - 1)
                if ibd_segment_boundaries[0][1] != end_chrom:
                    dbd_block_lengths.append(end_chrom - ibd_segment_boundaries[0][1] - 1)
        else:
            dbd_block_lengths.append(end_chrom)
        
        #print(chrom, dbd_block_lengths)
        return dbd_block_lengths

    def add_ibd_dbd(self, chrom_n, verbose=False):
        all_segments = self.ibd_length_dict[chrom_n] + self.dbd_length_dict[chrom_n]
        if not verbose:
            return sum(all_segments) == self.chrom_dict_lengths[chrom_n]
        else:
            return sum(all_segments), self.chrom_dict_lengths[chrom_n]

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)




if __name__ == '__main__':
    iteration = sys.argv[1]


    # running and storing sim results
    simulation_r_totals = defaultdict(list)
    simulation_ibd_segment_lengths = defaultdict(lambda: defaultdict(list))
    simulation_ibd_segment_numbers = defaultdict(lambda: defaultdict(list))
    simulation_ibd_segment_boundaries = defaultdict(lambda: defaultdict(list))
    simulation_ibd_segment_max = defaultdict(lambda:defaultdict(list))
    simulation_r_chrom = defaultdict(lambda:defaultdict(list))

    for number in range(500):
        print(number)
        S = Simulation(v=2, kbp_cM=11.3)
        for relationship in S.relationships:
            simulation_r_totals[relationship].append(S.r_totals[relationship])
            for chrom_n in S.ibd_segment_lengths[relationship]:
                simulation_ibd_segment_lengths[relationship][chrom_n] += S.ibd_segment_lengths[relationship][chrom_n]
                simulation_ibd_segment_numbers[relationship][chrom_n].append(S.ibd_block_numbers[relationship][chrom_n])
                simulation_ibd_segment_boundaries[relationship][chrom_n].append(S.all_ibd_segment_boundaries[relationship][chrom_n])
                simulation_ibd_segment_max[relationship][chrom_n].append(S.max_ibd_segment_lengths[relationship][chrom_n])
                simulation_r_chrom[relationship][chrom_n].append(S.r_chrom[relationship][chrom_n])

    file_basename = '/Users/weswong/Documents/GitHUb/Pedigree/sims/outcross_tx_chain/outcross_tx_chain_{i}'.format(i=iteration)
    json.dump(simulation_ibd_segment_lengths, open(file_basename + '_ibd_segment_lengths.json', 'w'), cls = NumpyEncoder)
    json.dump(simulation_r_totals, open(file_basename + '_r_totals.json', 'w'), cls = NumpyEncoder)
    json.dump(simulation_ibd_segment_numbers, open(file_basename +'_ibd_segment_numbers.json', 'w'), cls = NumpyEncoder)
    json.dump(simulation_ibd_segment_max, open(file_basename + '_ibd_segment_max.json', 'w'), cls = NumpyEncoder)
    #json.dump(simulation_r_chrom, open(file_basename+ '_r_chrom.json','w'),cls = NumpyEncoder)
    
    


# In[ ]:




