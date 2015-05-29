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
    all_genes = set()
    sample_weights = {}
    print("Genrating features")
    for sample, genes in mutations.iteritems():
        weights = propagate(genes, network)
        sample_weights[sample] = weights
        for gene in weights:
            all_genes.add(gene)

    features = np.zeros((len(mutations), len(all_genes)))
    labels = np.zeros(len(mutations))
    for i, mutation in enumerate(sample_weights):
        for j, gene in enumerate(all_genes):
            features[i][j] = sample_weights[mutation].get(gene, 0)
            labels[i] = sample_labels[mutation]

    return features, labels


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


def propagate(nodes, network):
    values = {n : 1 for n in nodes}
    new_values = defaultdict(int)
    alpha = 0.05
    THRESHOLD = 0.1
    for it_num in xrange(10):
        for node, value in values.iteritems():
            for neighbor in network[node]:
                new_values[neighbor] += alpha * value
        for node in nodes:
            new_values[node] += (1 - alpha)

        values = {node : new_values[node] for node in new_values
                  if new_values[node] > THRESHOLD}
        new_values = defaultdict(int)

    return values


def train_svm(features, labels):
    print("Training SVM")
    norm_features = preprocessing.scale(features)
    train_features, train_labels, test_features, test_labels = \
                                split_training_set(norm_features, labels, 0.4)
    clf = svm.SVC()
    clf.fit(train_features, train_labels)
    print(clf.score(test_features, test_labels))


def main():
    network = read_network(sys.argv[1])
    labels = {}
    mutations = {}
    for maf_no, maf_file in enumerate(sys.argv[2:]):
        sample_mutations = read_maf(maf_file)
        mutations.update(sample_mutations)
        labels.update({l : maf_no for l in sample_mutations})
    features, labels = get_features(mutations, labels, network)
    train_svm(features, labels)


if __name__ == "__main__":
    main()
