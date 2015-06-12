#!/usr/bin/env python


from __future__ import print_function
import sys
from collections import defaultdict
import random

import numpy as np
from sklearn import preprocessing, svm


def read_maf(filename):
    mutations = defaultdict(set)
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Hugo"):
                continue

            fields = line.split("\t")
            gene_id, tumor_barcode = fields[1], fields[15]

            sample_id = tumor_barcode.rsplit("-", 3)[0][:-1]
            if gene_id != "0":
                mutations[sample_id].add(int(gene_id))

    return mutations


def read_network(filename):
    edges = defaultdict(list)
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            fields = line.split("\t")
            node_1, node_2 = fields[0:2]
            edges[int(node_1)].append(int(node_2))
            edges[int(node_2)].append(int(node_1))

    return edges


def get_features(mutations, sample_labels, network):
    gene_ids = {}
    cur_id = 0
    for genes in mutations.itervalues():
        for gene in genes:
            if gene not in gene_ids:
                gene_ids[gene] = cur_id
                cur_id += 1
    for gene in network:
        if gene not in gene_ids:
            gene_ids[gene] = cur_id
            cur_id += 1

    adj_matrix = np.zeros((cur_id, cur_id))
    for node, neighbors in network.iteritems():
        for neighbor in neighbors:
            adj_matrix[gene_ids[node]][gene_ids[neighbor]] = 1

    print("Propagating")
    features = np.zeros((len(mutations), cur_id))
    labels = np.zeros(len(mutations))
    for sample_id, sample in enumerate(mutations):
        #print(sample_id, "/", len(mutations))
        genes = set(map(gene_ids.get, mutations[sample]))
        weights = propagate(genes, adj_matrix)

        for gene_id, weight in enumerate(weights):
            features[sample_id][gene_id] = weight
            labels[sample_id] = sample_labels[sample]
            #labels[sample_id] = random.randint(0, 1)

    return features, labels, gene_ids


def split_training_set(features, labels, rate):
    first_features, first_labels = [], []
    second_fetures, second_labels = [], []
    for i in xrange(features.shape[0]):
        if random.random() > rate:
            first_features.append(features[i])
            first_labels.append(labels[i])
        else:
            second_fetures.append(features[i])
            second_labels.append(labels[i])
    return (np.array(first_features), np.array(first_labels),
            np.array(second_fetures), np.array(second_labels))


def propagate(start_nodes, adj_matrix):
    ALPHA = 0.6
    ITERS = 2

    values = np.zeros((1, adj_matrix.shape[0]))
    for node in start_nodes:
        values[0][node] = 1

    for it_num in xrange(ITERS):
        values = ALPHA * values.dot(adj_matrix)
        for node in start_nodes:
            values[0][node] += 1 - ALPHA

    return sum(values)


def train_svm(features, labels):
    print("Training SVM")
    norm_features = preprocessing.scale(features)
    train_features, train_labels, test_features, test_labels = \
                                split_training_set(norm_features, labels, 0.4)
    clf = svm.SVC()
    clf.fit(train_features, train_labels)

    svm_features = np.zeros(features.shape[1])
    for i in xrange(len(clf.support_)):
        for j in xrange(features.shape[1]):
            svm_features[j] += clf.dual_coef_[0][i] * clf.support_vectors_[i][j]

    print("Score:", clf.score(test_features, test_labels))
    return svm_features


def print_features(svm_features, gene_ids):
    id_2_gene = {}
    for gene, g_id in gene_ids.items():
        id_2_gene[g_id] = gene

    print("Main features:")
    svm_main = sorted(enumerate(svm_features),
                      key=lambda x: abs(x[1]), reverse=True)
    for f_id, f_val in svm_main[:20]:
        print("{0}\t{1:5.2f}".format(id_2_gene[f_id], f_val))


def main():
    network = read_network(sys.argv[1])
    labels = {}
    mutations = {}
    for maf_no, maf_file in enumerate(sys.argv[2:]):
        sample_mutations = read_maf(maf_file)
        mutations.update(sample_mutations)
        labels.update({l : maf_no for l in sample_mutations})
    features, labels, gene_ids = get_features(mutations, labels, network)
    svm_features = train_svm(features, labels)
    print_features(svm_features, gene_ids)


if __name__ == "__main__":
    main()
