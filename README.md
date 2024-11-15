# MalKinID
 Malaria Kinship Identifier

MalKinID (Malaria Kinship Identifier) is a  likelihood-based classification model designed to identify genealogical relationships among malaria parasites based on genome-wide IBD proportions and IBD segment distributions. MalKinID was calibrated to the genomic data from three laboratory-based genetic crosses (yielding 440 parent-child and 9060 full-sibling comparisons)


By default, MalKinID infers the genealogical relationship between two parasite strains based on the total relatedness (total proportion of genome that is IBD), and the per-chromosome IBD segment block and segment count distributions. 

An example input file is provided in the example_input.txt file. 
The first column is the comparison examined (in this case s1-s7), the second column the total relatedness. The next 14 columns are the IBD segment count (n_ibD) for chromosomes 1-14 (coded as chromosome_number:n_IBD) and the following 14 is the size of the largest IBD segment (max_ibd) on the chromosome, proportional to the size of the chromsome (coded as chromosome_number:max_IBD)

the ll_parameters file contains the likelihood parameters used in the model.

The code in the ipynb notebook shows the basic utilization of MalKinID. For each comparison, a Sim object is created that stores the relevant information for each sample comparison. The following attributes can be retrieved from the Sim object: comparison, s1 (sample 1),2 (sample 2), r_total (total proportion of genome IBD), max_ibd_segment (a flat list with the total proportion IBD for each chromosome, ordered from 1-14), n_segment_count (a flat list with the number of IBD segments observed for each chromosome, ordered from 1-14), and max_ll_categorization (the inferred max likelihood categorization). 

Log-likelihood estimation is achieved by the calc_all_likelihoods() function. This function can be modified to include or exclude certain genealogical hypotheses.
        


Pedigree Tree Simulation
These are template code that wil be streamlined in a later public release. Address questions to weswong@hsph.harvard.edu
These code are only relevant when wanting to re-adapt MalKinID to infer genealogical relationships on non-outcrossing/inbred trees.

Template code to run the meiosis model based on a specified pedigree is provided /sim_serial_cotx_chain.py. To specify the pedigree, modify the simulate_pedigree function (lines 211-249). The pedigree tree is specified as a flat dictionary where the keys are the name of the node and the value the examined genome. Simulated parasites are generated using the the meiosis function (line 226). 

Simulated progeny are generated using the meiosis function (meiosis(in1,in2,N=4,v=2, oc=True, bp_per_cM = 11300)), where the first two arguments are the genomes generated from the Genome subpackage (make sure that the Genome subpackage location is specified at line 26). These genomes are numpy arrays where the elements are integers indicating parental origin. For example, parent1 can be represented by a vector of zeros and parent2 a vector of ones. Recombinant progeny will of parent1 and parent2 contain a mix zeros and ones. sim_serial_cotx_chain.py takes in one argument that specifies the simulation iteration -- this is purely for annotation purposes to allow one to run multiple simulations in parallel and keep track of the outputs associated with each iteration.

The second parameter specifies the number of unique meiotic progeny that are generated. In general, most applications will be to generate a single meiotic progeny, and thus N should be set to one. By default, the meiosis function returns a list of unique progenies and when N =1 it will return a list where the first element is a sampled progeny from the meiotic event.
For meiotic siblings (lines 233-237), N should be set to 2. The output will be a list containing two elements, one for each sampled progeny. These can then be assigned as different nodes of the pedigree tree.

This will generate several outputs that contain the simulated distributions for each pairwise combination in the tree as a json file. For the r_total distributions, the json is a one level dictionary where the keys of the first layer describes the examined comparisons (which by default simply combine the names of the node. For the other features (simulation_ibd_segment_max and simulation_ibd_n_segment), the json is a two layer dictionary where the first key describes the comparison, and the second layer the chromosome.

These raw simulation outputs can be fit to the components of the pseudolikelihood by adapting the first few code cells of [http://localhost:8888/notebooks/notebooks/Likelihood_functions_simulation.ipynb](https://github.com/weswong/MalKinID/blob/main/sims/distribution_fitting_template.ipynb). This code assumes that all the simulation results are in a common directory. The task is to consolidate the simulation results for each feature into a single dictionary that aggregates the results from all simulation iterations.
Once aggregated, the code will fit the components of the pseudolikelihood model to the raw data using the following functions: fit_beta (r_total), create_pdfs (max IBD segment block), and calc_seg_count_pmf(n_segment_distribution).
These can be dumped into a dill file to be loaded later.

These results are then integrated into the Sim class as components of the likelihood function to begin performing classification. The most likely classification is defined as the one that has the highest pseudolikelihood.



