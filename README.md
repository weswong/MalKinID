# MalKinID
 Malaria Kinship Identifier

MalKinID (Malaria Kinship Identifier) is a  likelihood-based classification model designed to identify genealogical relationships among malaria parasites based on genome-wide IBD proportions and IBD segment distributions. MalKinID was calibrated to the genomic data from three laboratory-based genetic crosses (yielding 440 parent-child and 9060 full-sibling comparisons)


By default, MalKinID infers the genealogical relationship between two parasite strains based on the total relatedness (total proportion of genome that is IBD), and the per-chromosome IBD segment block and segment count distributions. 

An example input file is provided in the example_input.txt file. 
The first column is the comparison examined (in this case s1-s7), the second column the total relatedness. The next 14 columns are the IBD segment count (n_ibD) for chromosomes 1-14 (coded as chromosome_number:n_IBD) and the following 14 is the size of the largest IBD segment (max_ibd) on the chromosome, proportional to the size of the chromsome (coded as chromosome_number:max_IBD)

the ll_parameters file contains the likelihood parameters used in the model.

The code in the ipynb notebook shows the basic utilization of MalKinID. For each comparison, a Sim object is created that stores the relevant information for each sample comparison. The following attributes can be retrieved from the Sim object: comparison, s1 (sample 1), s2 (sample 2), r_total (total proportion of genome IBD), max_ibd_segment (a flat list with the total proportion IBD for each chromosome, ordered from 1-14), n_segment_count (a flat list with the number of IBD segments observed for each chromosome, ordered from 1-14), and max_ll_categorization (the inferred max likelihood categorization). 

Log-likelihood estimation is achieved by the calc_all_likelihoods() function. This function can be modified to include or exclude certain genealogical hypotheses.
        
