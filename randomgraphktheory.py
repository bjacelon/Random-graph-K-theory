#!/usr/bin/env python3

# Copyright (c) 2023 Bhishan Jacelon, Igor Khavkine
#
# Distributed under the "MIT License":
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import csv
import scipy.stats as stats
from sympy import Matrix
from sympy.ntheory import multiplicity
from cypari import pari
from math import floor, log, sqrt

# Allocate 1GB of memory right away, to avoid errors from Pari.
# Only relevant for larger graphs. Increase if necessary
pari.allocatemem(1024*1000*1000)

def randompairing(n):
    """ Returns a random pairing of the integers 0 to n-1 """
    lst = list(range(n))
    random.shuffle(lst)
    return [(lst[i], lst[i+1]) for i in range(0, n, 2)]

def random_regular_multigraph(n, r):
    """ Computes r random pairings and returns their concatenation """
    edges = []
    for _ in range(r):
        edges += randompairing(n)
    return edges

def random_bernoulli_graph(n, w):
    """Creates a random Bernoulli graph"""
    edges = []
    if w<=-1:
        edge_prob = 1 / (1-w)
    else:
        edge_prob = ((1+w) * log(n)) / n
    for i in range(n):
        for j in range(n):
            if random.random() < edge_prob:
                edges.append((i, j))
    return edges

def random_erdos_graph(n, w):
    """Creates a random Erdos-Renyi graph"""
    edges = []
    if w<=-1:
        edge_prob = 1 / (1-w)
    else:
        edge_prob = ((1+w) * log(n)) / n
    for i in range(n):
        for j in range(n):
            if random.random() < edge_prob and not j<i:
                edges.append((i, j))
    return edges

def pSylow(l, p):
    """Computes the Sylow p-subgroup associated to le"""
    return [0 if e == 0 else multiplicity(p,e) for e in l]

def mainoutput(n, r, P, m, alpha, nodisplay, graph_type, k=0):
    """Computes the K-theory of random graph C*-algebras"""

    if graph_type == 1:

        output_filename = f'bernoulli_n={n}_w={r}_P={P}_m={m}_alpha={alpha}.csv'

    elif graph_type == 2:

        output_filename = f'erdosrenyi_n={n}_w={r}_P={P}_m={m}_alpha={alpha}.csv'

    elif graph_type == 3:

        output_filename = f'regular_n={n}_r={r}_P={P}_m={m}_alpha={alpha}.csv'

    with open(output_filename, mode='w', newline='') as output_file:

        # Create column headers for csv file depending on input primes P
        headers = ['Graph', 'Connected', 'Tor(K_0)', 'Rank(K_1)', 'All cyclic']

        power = {p: floor(log(m/5, p)) for p in P}
        
        for p in P:
            
            headers += [f"Sylow {p}-subgroup", f"{p}-cyclic", f"{p}-Sylow = 0"]
            
            for N in range(1,power[p]+1):
                
                if N==1:
                    headers += [f"{p}-Sylow = Z/{p}Z"]
                else:
                    headers += [f"{p}-Sylow = Z/({p}^{N}Z)", f"{p}-Sylow = (Z/{p}Z)^{N}"]
                    
        csv_writer = csv.DictWriter(output_file, fieldnames=headers)
        csv_writer.writeheader()

        # Keep track of column totals
        column_totals = {header: 0 for header in headers if not header.startswith('Sylow') and header != 'Graph'}

        for i in range(m):

            if graph_type == 1:

                g = random_bernoulli_graph(n, r)
                G = nx.DiGraph(g)
                # G.add_edges_from(g)

                # Check if g is connected and plot the first graph
                if not nx.is_strongly_connected(G):
                    
                    print(f"Graph {i+1} is disconnected")
                    connected = 0
                    
                else:
                    
                    print(f"Graph {i+1} is connected")
                    connected = 1

                if i == k and not nodisplay:

                    # Create the plot figure
                    fig = plt.figure()

                    # Draw the directed graph using networkx
                    pos = nx.spring_layout(G)  # Define the layout of the nodes
                    nx.draw(G, pos, with_labels=True, arrows=True)

                    # Set the plot title and show the plot
                    plt.title(f'Bernoulli digraph (n={n}, w={r})')
                    plt.show()

            elif graph_type == 2:

                g = random_erdos_graph(n, r)
                G = nx.Graph(g)
                # G.add_edges_from(g)

                # Check if g is connected and plot the first graph
                if not nx.is_connected(G):
                    
                    print(f"Graph {i+1} is disconnected")
                    connected = 0
                    
                else:
                    
                    print(f"Graph {i+1} is connected")
                    connected = 1

                if i == k and not nodisplay:

                    # Create the plot figure
                    fig = plt.figure()

                    # Draw the graph using networkx
                    pos = nx.spring_layout(G)  # Define the layout of the nodes
                    nx.draw(G, pos, with_labels=True)

                    # Set the plot title and show the plot
                    plt.title(f'Erdos-Renyi graph (n={n}, w={r})')
                    plt.show()

            elif graph_type == 3:

                g = random_regular_multigraph(n, r)
                G = nx.Graph(g)

                # Check if g is connected and plot the first graph
                if not nx.is_connected(G):
                    
                    print(f"Graph {i+1} is disconnected")
                    connected = 0
                    
                else:
                    
                    print(f"Graph {i+1} is connected")
                    connected = 1

                if i == k and not nodisplay:

                    # Define the layout of the nodes
                    pos = nx.circular_layout(G)

                    # Add edge labels
                    edge_labels = {}

                    adj_mat = np.zeros((n, n), dtype=int)
                
                    for (a,b) in g:
                        adj_mat[a,b] += 1
                        adj_mat[b,a] += 1

                    for a in range(n):

                      for b in range(a+1,n):
                          
                        count = adj_mat[a,b]
                        
                        if count > 1:
                            
                            edge_labels[a,b] = str(count)

                    # Create the plot figure
                    plt.figure()

                    # Set the plot title
                    plt.title(f'Random regular graph (n={n}, r={r})')

                    # Draw the graph using networkx
                    nx.draw(G, pos, with_labels=True)

                    # Draw edge labels
                    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

                    # Show the plot
                    plt.show()

            column_totals['Connected'] += connected

            # Compute K_0 and K_1

            pv_mat = -np.eye(n, dtype=int)
            
            for (a,b) in g:

                pv_mat[a,b] += 1
                
                if a != b and graph_type != 1:
                    
                    pv_mat[b, a] += 1

            print(pv_mat)

            # Convert pv_mat to pari input, then compute its smith normal form
            mat = pari.matrix(n,n, pv_mat.flatten())
            
            # Compute the matrix determinant.
            dt = pari.matdet(mat)
            if dt != 0:
                # If the matrix is non-degenerate, this method
                # seems to be faster for larger denser graphs.
                snf = pari.matsnf(pari.mathnfmod(mat,dt))
            else:
                # If the matrix is non-degenerate, compute snf directly.
                snf = pari.matsnf(mat)

            le = [abs(e) for e in snf]

            # K_1 is the torsion free part of K_0
            k1 = sum(1 for e in le if e==0)

            # Remove 0s (which are remembered by k1) and 1s (which do not contribute to the cokernel)
            le = [e for e in le if e!=1 and e!=0]

            # Check whether K_0 is cyclic and tally the results
            column_totals['Tor(K_0)'] += int(len(le)<=1)
            
            # Check whether K_1 is nonzero and tally the results
            column_totals['Rank(K_1)'] += int(k1!=0)

            print("Tor(K_0): ", le)
            print("Rank(K_1): ", k1)

            row = {'Graph': i + 1, 'Connected': connected, 'Tor(K_0)': le, 'Rank(K_1)': k1}

            cyclic = {p: 0 for p in P}

            for p in P:

                # Compute the p-Sylow subgroup of K_0
                Syl = pSylow(le,p)
                
                print(f"Sylow {p}-subgroup: ", Syl)
                row[f"Sylow {p}-subgroup"] = Syl

                # Check whether the p-Sylow subgroup is cyclic and tally the results
                cyclic[p] = int(sum(1 for x in Syl if x != 0) <= 1 or len(Syl) == 0)
                row[f"{p}-cyclic"] = cyclic[p]
                column_totals[f"{p}-cyclic"] += cyclic[p]

                # Check whether the p-Sylow subgroup is trivial and tally the results
                triv = int(not Syl or all(x == 0 for x in Syl))     
                row[f"{p}-Sylow = 0"] = triv
                column_totals[f"{p}-Sylow = 0"] += triv

                if triv==1:

                    print(f"{p}-Sylow = 0: ", triv)
                    
                # Check whether the p-Sylow subgroup is of the form Z/({p}^{N})Z or Z/({p}Z)^{N} and tally the results
                for N in range(1,power[p]+1):

                    one_Ntimes = int(sum(1 for x in Syl if x != 0) == N and all(x == 1 for x in Syl if x != 0))
                    N_once = int(sum(x != 0 for x in Syl) == 1 and Syl.count(N) == 1)

                    if N==1:

                        row[f"{p}-Sylow = Z/{p}Z"] = N_once
                        column_totals[f"{p}-Sylow = Z/{p}Z"] += N_once

                        if N_once == 1:
                            print(f"{p}-Sylow = Z/{p}Z: ", N_once)

                    else:

                        row[f"{p}-Sylow = Z/({p}^{N}Z)"] = N_once
                        column_totals[f"{p}-Sylow = Z/({p}^{N}Z)"] += N_once
                        row[f"{p}-Sylow = (Z/{p}Z)^{N}"] = one_Ntimes
                        column_totals[f"{p}-Sylow = (Z/{p}Z)^{N}"] += one_Ntimes

                        if N_once == 1:
                            print(f"{p}-Sylow = Z/({p}^{N})Z: ", N_once)
                        if one_Ntimes == 1:
                            print(f"{p}-Sylow = Z/({p}Z)^{N}: ", one_Ntimes)
                    
            print()

            # Check whether all specified Sylow subgroups are cyclic and tally the results
            row['All cyclic'] = int(all(cyclic[p] ==1 for p in P))
            column_totals['All cyclic'] += row['All cyclic']

            csv_writer.writerow(row)
            
        csv_writer.writerow(column_totals)

        # Compute 100(1-alpha)% confidence intervals using the normal distribution
        confidence_intervals = {header: 0 for header in headers if not header.startswith('Sylow') and header != 'Graph'}
        for header, value in column_totals.items():
            p = value / m
            var = p * (1 - p) / m
            z = stats.norm.ppf(1 - alpha / 2)
            margin = z * sqrt(var)
            lower_bound = p - margin
            upper_bound = p + margin
            interval = f"[{lower_bound:.4f}, {upper_bound:.4f}]"
            confidence_intervals[header] = interval

        csv_writer.writerow(confidence_intervals)


def generate_graph(option):
    
    if option == 1 or option == 2:

        n = int(input("Enter the number of vertices (n): "))
        
        r = float(input("Edge probability is ((1+w)*log(n))/n if w>-1 and 1/(1-w) if w<=-1. Enter the weight (w): "))

    elif option == 3:
        
        n = int(input("Enter the number of vertices (n even): "))
        
        r = int(input("Enter the regularity degree (r>2): "))
    
    else:
        
        print("Invalid option. Please choose a valid graph generation option.")
        return None, None, None, None, None, None  # Return None values for all variables
    
    m = int(input("Enter the number of graphs to be generated (m): "))

    P = input("Enter the list of primes (space-separated) whose Sylow subgroups are to be determined: ").split()

    P = [int(prime) for prime in P]

    alpha = float(input("Enter the significance level (alpha): "))

    nodisplay = input("Skip displaying a sample graph? (y/n): ")

    nodisplay = nodisplay.lower() in ['y', 'yes']

    return n, r, m, P, alpha, nodisplay

if __name__ == "__main__":
    
    import argparse

    while True:

        print("1 Bernoulli digraph")
        print("2 Erdos-Renyi graph")
        print("3 r-regular multigraph")
        
        graph_type_input = input("Enter the type of random graph to be generated (1, 2 or 3) or 'quit' to exit: ")

        if graph_type_input == 'quit':
            
            break

        else:

            graph_type_input = graph_type_input.strip()  # Remove leading/trailing whitespace

            graph_type = int(graph_type_input)  # Convert the input to an integer

            # Call input variables for the graph type
            graph_input = generate_graph(graph_type)

            # Unpack the returned values from generate_graph
            n, r, m, P, alpha, nodisplay = graph_input

            if n is not None:

                # Call the main function with the user-provided values
                mainoutput(n, r, P, m, alpha, nodisplay, graph_type, k=0)

                # Continue to the graph_input prompt unless the user enters 'quit'
                continue

            else:
                
                print("Please try again with a valid option.")
