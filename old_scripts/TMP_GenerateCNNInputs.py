#! /usr/bin/env python3
# Robert Patton, rpatton@fredhutch.org
# v1.1, 04/27/2022

# Modified from GenerateProfileInputs.py v1.4 in order to manually conduct GC correction (instead of taking Griffin
# runs) from BAM files and incorporate additional bp resolution metrics. Current output structure:
# ----------------------------------------------------------------------------------------------------------------------
# 1: GC-corrected depth
# 2: (nucleosome) phased profile
# 3: Shannon entropy
# 4: Region fragment profile normalized Shannon entropy
# 5: Fragment heterogeneity (unique fragment lengths / total fragments)
# 6: Short:long ratio (x <= 120 / 140 <= x <= 250)
# 7: Adenine - one hot encoded
# 8: Cytosine - one hot encoded
# 9: Guanine - one hot encoded
# 10: Tyrosine - one hot encoded
# ----------------------------------------------------------------------------------------------------------------------
# Inputs are BAM of interest (sample), associated GC_bias, a BED file with equal-sized regions of interest,
# reference sequence, and optionally a list of regions (corresponding to the names used in the BED file) for which
# the output array will be printed for visualization. Outputs are numpy arrays for each sample-region combination,
# labeled as sample_region.npy (numpy format, for matching with truth values downstream). Non one-hot encoded outputs
# are all standardized so that max = 1.0, min = 0.0 (for deep learning), and 5' -> 3' directionality is enforced.

import sys
import pysam
import random
import argparse
import numpy as np
import pandas as pd
from functools import partial
from multiprocessing import Pool
from matplotlib import pyplot as plt
from scipy.fft import rfft, rfftfreq, irfft

mapping = dict(zip("NACGT", range(5)))
freq_max = 0.0068493  # theoretical nucleosome period is 146 bp -> f = 0.0068493


def one_hot_encode(seq):
    seq2 = [mapping[nt] for nt in seq]
    return np.eye(5)[seq2]


def get_gc_bias_dict(bias_path):
    if bias_path is not None:
        bias = pd.read_csv(bias_path, sep='\t')
        bias['smoothed_GC_bias'] = np.where(bias['smoothed_GC_bias'] < 0.05, np.nan, bias['smoothed_GC_bias'])
        bias = bias[['length', 'num_GC', 'smoothed_GC_bias']]
        bias = bias.set_index(['num_GC', 'length']).unstack()
        bias = bias.to_dict()
        bias2 = {}
        for key in bias.keys():
            length = key[1]
            bias2[length] = {}
            for num_GC in range(0, length + 1):
                temp_bias = bias[key][num_GC]
                bias2[length][num_GC] = temp_bias
        bias = bias2
        del bias2
        return bias
    else:
        return None


def frag_ratio(frag_lengths):  # compute the ratio of short to long fragments
    short_frags = len([x for x in frag_lengths if x <= 120])
    long_frags = len([x for x in frag_lengths if 140 <= x <= 250])
    if short_frags > 0 and long_frags > 0:
        ratio = short_frags / long_frags
        return ratio
    else:
        return 0


def shannon_entropy(values, bins, binned=False):
    if not binned:
        histogram = np.histogram(values, bins=bins)[0]
    else:
        histogram = values
    if sum(histogram) < 2:
        return 0
    pdf = histogram / sum(histogram)
    pdf = [p for p in pdf if p != 0.0]
    moments = [p * np.log(p) for p in pdf]
    return - sum(moments)


def mean_entropy(pdf_matrix, counts, bins):
    entropies = []
    for row in pdf_matrix:
        draws = np.random.multinomial(counts, row)
        entropies.append(shannon_entropy(draws, bins, binned=True))
    return np.mean(entropies)


def point_entropy(total_lengths, point_lengths):
    min_size, max_size = min(total_lengths), max(total_lengths)
    bin_range = list(range(min_size, max_size, 1))
    hist = np.histogram(total_lengths, bins=bin_range)[0]
    hist = [h for h in hist if h != 0.0]
    dist_pdfs = np.random.dirichlet(hist, 1000)
    frag_nums = list(set([len(frags) for frags in point_lengths]))
    norm_dict = {i: mean_entropy(dist_pdfs, i, bin_range) for i in frag_nums}
    point_shannon_entropies = [shannon_entropy(frags, bin_range) for frags in point_lengths]
    point_norm_entropies = [(val / norm_dict[len(frags)]) if len(frags) > 2 else 0 for val, frags
                            in zip(point_shannon_entropies, point_lengths)]
    return point_shannon_entropies, point_norm_entropies


def normalize_data(data):
    if np.max(data) - np.min(data) == 0:
        return data
    else:
        return(data - np.min(data)) / (np.max(data) - np.min(data))


def generate_profile(bed_region, params):
    bam_path, sample, out_direct, frag_range, gc_bias, ref_seq_path, to_plot, map_q = params
    bam = pysam.AlignmentFile(bam_path, 'rb')
    ref_seq = pysam.FastaFile(ref_seq_path)
    bed_tokens = bed_region.strip().split('\t')
    chrom, start_pos, end_pos, site, strand = \
        bed_tokens[0], int(bed_tokens[1]), int(bed_tokens[2]), str(bed_tokens[3]), str(bed_tokens[4])
    roi_length = end_pos - start_pos
    depth = [0] * roi_length
    roi_sequence = ref_seq.fetch(chrom, start_pos, end_pos).upper()
    # process all fragments falling inside the ROI and establish GC-corrected depth ####################################
    roi_reads = bam.fetch(chrom, start_pos, end_pos)
    roi_fragment_lengths = []
    fragment_length_profile = [[] for _ in range(roi_length)]
    for read in roi_reads:
        fragment_length = read.template_length
        if frag_range[0] <= np.abs(fragment_length) <= frag_range[1] and read.is_paired and read. \
                mapping_quality >= map_q and not read.is_duplicate and not read.is_qcfail:
            roi_fragment_lengths.append(abs(fragment_length))
            read_start = read.reference_start - start_pos
            if read.is_reverse and fragment_length < 0:
                read_length = read.reference_length
                fragment_start = read_start + read_length + fragment_length
                fragment_end = read_start + read_length
            elif not read.is_reverse and fragment_length > 0:
                fragment_start = read_start
                fragment_end = read_start + fragment_length
            else:
                continue
            # now do GC bias correction:
            if gc_bias is not None:
                fragment_seq = (ref_seq.fetch(bed_tokens[0], fragment_start + start_pos, fragment_end + start_pos)). \
                    upper()
                fragment_seq = list(fragment_seq.replace('T', '0').replace('A', '0').replace('C', '1').
                                    replace('G', '1').replace('N', str(np.random.randint(0, 2))))
                fragment_gc_content = sum([int(m) for m in fragment_seq])
                fragment_bias = gc_bias[np.abs(fragment_length)][fragment_gc_content]
            else:
                fragment_bias = 1
            fragment_cov = np.array(range(fragment_start, fragment_end + 1))
            for index in [val for val in fragment_cov if 0 <= val < roi_length]:
                fragment_length_profile[index].append(abs(fragment_length))
                if 0.05 < fragment_bias < 10:
                    depth[index] += 1 / fragment_bias
    if strand == "-":
        roi_sequence = roi_sequence[::-1]
        depth = depth[::-1]
        fragment_length_profile = fragment_length_profile[::-1]
    # get phased profile ###############################################################################################
    mean_depth = np.mean(depth)
    if mean_depth < 1.0:
        return (sample + '_' + site), np.nan  # skip genes with less than 1X coverage
    fourier = rfft(np.asarray(depth))
    freqs = rfftfreq(roi_length)
    test_freqs = [idx for idx, val in enumerate(freqs) if 0 < val <= freq_max]  # frequencies in filter
    clean_fft = [0] * len(fourier)
    try:
        clean_fft[test_freqs[0]:test_freqs[-1]] = fourier[test_freqs[0]:test_freqs[-1]]
        inverse_signal = irfft(clean_fft)  # reconstruct signal
        phased_signal = inverse_signal + len(inverse_signal) * [mean_depth]  # add in base component
    except IndexError:  # not enough data to construct a reasonable Fourier
        phased_signal = [0] * roi_length
    # get fragment profiles ############################################################################################
    shan_profile, norm_profile = point_entropy(roi_fragment_lengths, fragment_length_profile)
    ratio_profile = [frag_ratio(point_frags) for point_frags in fragment_length_profile]
    hetero_profile = [len(set(point_frags)) / len(point_frags) if len(point_frags) is not 0 else 0 for point_frags in
                      fragment_length_profile]
    # get one-hot sequence #############################################################################################
    seq_profile = np.delete(one_hot_encode(roi_sequence), 0, 1)
    # normalize profiles ###############################################################################################
    depth = normalize_data(depth)
    phased_signal = normalize_data(phased_signal)
    shan_profile = normalize_data(shan_profile)
    norm_profile = normalize_data(norm_profile)
    hetero_profile = normalize_data(hetero_profile)
    ratio_profile = normalize_data(ratio_profile)
    # plot if selected region ##########################################################################################
    if site in to_plot:  # plot selected regions
        # axes are depth+phasing, shannon, norm-shannon, heterogeneity, and short:long ratio
        x_vals = list(range(-int(roi_length / 2), int(roi_length / 2)))  # currently assume centered regions
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, figsize=(20, 11))
        fig.suptitle(sample + ' ' + site + ' CNN Input Profile', x=0.17, y=1.1, fontsize=24)
        ax1.set_title(roi_sequence[:63] + ' . . .                                          . . . ' + roi_sequence[-63:])
        # 1: depth+phasing
        ax1.hlines(y=1, xmin=x_vals[0], xmax=x_vals[-1], color='gray', linestyle='dashed')
        ax1.fill_between(x_vals, 0, depth, label='GC-Corrected Coverage', color='gray')
        ax1.plot(x_vals, phased_signal, color='#8927D6FA', label='Phased Component', lw=3, alpha=0.8)
        ax1.set_xlim(x_vals[0], x_vals[-1])
        ax1.set_ylim(0, 1.1)
        ax1.set_ylabel('Scaled Depth', fontsize=12)
        # 2: shannon entropy
        ax2.hlines(y=1, xmin=x_vals[0], xmax=x_vals[-1], color='gray', linestyle='dashed')
        ax2.plot(x_vals, shan_profile, color='red', label='Shannon Entropy')
        ax2.fill_between(x_vals, 0, shan_profile, color='red', alpha=0.3)
        ax2.set_xlim(x_vals[0], x_vals[-1])
        ax2.set_ylim(0, 1.1)
        ax2.set_ylabel('Scaled Entropy', fontsize=12)
        # 3: normalized shannon entropy
        ax3.hlines(y=1, xmin=x_vals[0], xmax=x_vals[-1], color='gray', linestyle='dashed')
        ax3.plot(x_vals, norm_profile, color='blue', label='Region-Fragments-Normalized Shannon Entropy')
        ax3.fill_between(x_vals, 0, norm_profile, color='blue', alpha=0.3)
        ax3.set_xlim(x_vals[0], x_vals[-1])
        ax3.set_ylim(0, 1.1)
        ax3.set_ylabel('Scaled Entropy', fontsize=12)
        # 4: fragment heterogeneity
        ax4.hlines(y=1, xmin=x_vals[0], xmax=x_vals[-1], color='gray', linestyle='dashed')
        ax4.plot(x_vals, hetero_profile, color='green', label='Fragment Heterogeneity (unique lengths / total frags)')
        ax4.fill_between(x_vals, 0, hetero_profile, color='green', alpha=0.3)
        ax4.set_xlim(x_vals[0], x_vals[-1])
        ax4.set_ylim(0, 1.1)
        ax4.set_ylabel('Scaled Ratio', fontsize=12)
        # 5: short:long ratio
        ax5.hlines(y=1, xmin=x_vals[0], xmax=x_vals[-1], color='gray', linestyle='dashed')
        ax5.plot(x_vals, ratio_profile, color='orange', label='Short:Long Ratio')
        ax5.fill_between(x_vals, 0, ratio_profile, color='orange', alpha=0.3)
        ax5.set_xlim(x_vals[0], x_vals[-1])
        ax5.set_ylim(0, 1.1)
        ax5.set_ylabel('Scaled Ratio', fontsize=12)
        lines, labels = [], []
        for ax in fig.axes:
            ax_line, ax_label = ax.get_legend_handles_labels()
            lines.extend(ax_line)
            labels.extend(ax_label)
        fig.legend(lines, labels, title_fontsize=18, loc='lower center', bbox_to_anchor=(0.79, 0.99))
        plt.tight_layout()
        fig.savefig(out_direct + '/' + sample + '_' + site + '.pdf', bbox_inches="tight")
        plt.close()
    # combine and save profiles ########################################################################################
    signal_array = np.column_stack((depth, phased_signal, shan_profile, norm_profile, hetero_profile, ratio_profile))
    out_array = np.concatenate((signal_array, seq_profile), axis=1)
    return (sample + '_' + site), out_array  # identifier, numpy matrix


def main():
    # parse command line arguments:
    parser = argparse.ArgumentParser(description='\n### GenerateCNNInputs.py ### Generate formatted profiles for CNNs'
                                                 'regions of interest for a bam file - see code for features.')
    parser.add_argument('-n', '--sample_name', help='sample identifier', required=True)
    parser.add_argument('-i', '--input', help='bam file', required=True)
    parser.add_argument('-b', '--bias', help='GC bias file (from Griffin)', default=None)
    parser.add_argument('-a', '--annotation', help='bed file with regions of interest', required=True)
    parser.add_argument('-g', '--reference_genome', help='reference genome (.fa)', required=True)
    parser.add_argument('-r', '--results_dir', help='directory for results', required=True)
    parser.add_argument('-p', '--plot_list', help='list of genes/regions to generate plots for', required=True)
    parser.add_argument('-q', '--map_quality', help='minimum mapping quality (default=20)', type=int, default=20)
    parser.add_argument('-f', '--size_range', help='fragment size range (bp; default=[15, 500])', nargs=2, type=int,
                        default=(15, 500))
    parser.add_argument('-c', '--cpus', help='cpus available for parallel processing', type=int, default=1)
    args = parser.parse_args()

    print('Loading input files . . .')

    sample_name = args.sample_name
    bam_path = args.input
    bias_path = args.bias
    bed_path = args.annotation
    ref_seq_path = args.reference_genome
    results_dir = args.results_dir
    plot_list = args.plot_list
    map_q = args.map_quality
    size_range = args.size_range
    cpus = args.cpus

    print('\narguments provided:')
    print('\tsample_name = "' + sample_name + '"')
    print('\tbam_path = "' + bam_path + '"')
    print('\tGC_bias_path = "' + bias_path + '"')
    print('\tref_seq_path = "' + ref_seq_path + '"')
    print('\tsites_bed = "' + bed_path + '"')
    print('\tplot_list = "' + plot_list + '"')
    print('\tresults_dir = "' + results_dir + '"')
    print('\tsize_range =', size_range)
    print('\tmap_q =', map_q)
    print('\tCPUs =', cpus)
    print('\n')
    sys.stdout.flush()

    gc_bias = get_gc_bias_dict(bias_path)
    to_plot = pd.read_table(plot_list, header=None)[0].tolist()
    sites = [region for region in open(bed_path, 'r')]
    random.shuffle(sites)
    params = [bam_path, sample_name, results_dir, size_range, gc_bias, ref_seq_path, to_plot, map_q]

    print('Running generate_profile on ' + str(len(sites)) + ' regions . . .')

    with Pool(cpus) as pool:
        results = list(pool.imap_unordered(partial(generate_profile, params=params), sites, len(sites) // cpus))

    print('Merging and saving results . . .')

    results_dict = {}
    for result in results:
        results_dict[result[0]] = result[1]
    np.savez_compressed(results_dir + '/' + sample_name, **results_dict)

    print('Finished')


if __name__ == "__main__":
    main()
