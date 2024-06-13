#!/usr/bin/env python3

# Copyright (c) 2024 Bhishan Jacelon, Igor Khavkine
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

def pSylow(l, p):
    """Computes the Sylow p-subgroup associated to le"""
    return [0 if e == 0 else multiplicity(p,e) for e in l]

def print_orbit(p, kmaxp, uorbit):
        print("p =", p, ", kmax =", kmaxp)
        for k in range (1,kmaxp+1):
            if k in uorbit:
                for j in range(uorbit[k]): print('*',end='')
                for j in range(uorbit[k],k): print('0',end='')
            print()

def unit_orbit(l, u):
    """Compute the automorphism orbit of u in the cokernel of the
    Smith normal form with diagonal l (assume no 0s in l).
    Returns double dictionary p_uorbit, so that p_uorbit[p][k], for each
    prime p gives the position of the least significant non-zero base-p digit
    of u in each Z/(p^k) block. Position 1 is the most significant digit,
    position 2 is the next most significant digit, etc."""
    # Get the prime factors and multiplicities for the elements of l.
    # All the distinct primes should already be present in l[0],
    # and the multiplicities should decrease in l[i] with increasing i.
    pk = pari.factor(l[0])
    ps = [int(p) for p in pk[0]]
    kmax = [int(k) for k in pk[1]]
    kmax = {ps[i]: kmax[i] for i in range(len(ps))}
    # For each prime p and each u[i] and multiplicity k[i], record whether 
    # a given base-p digit of u[i] (mod p^k[i]) is zero or non-zero
    p_udigits = {}
    for p in ps:
        p_udigits[p] = {}
        udigits = p_udigits[p]
        ks = [multiplicity(p,e) for e in l]
        for k in ks:
            if k != 0 and not (k in udigits):
                # Initialize digit data, assuming all 0s.
                udigits[k] = [False]*k
        for i in range(len(l)):
            # Compute p-adic digits, padd / truncate digits to right length.
            ds = [int(d) for d in pari.digits(u[i], p)]
            k = ks[i]
            if len(ds) < k:
                ds = [0]*(k-len(ds)) + ds
            else:
                ds = ds[-k:]
            # Find least-significant non-zero digit position, -1 if none.
            zds = [j for j in range(k-1,-1, -1) if ds[j] != 0]
            if zds:
                j = zds[0]
            else:
                j = -1
            # Set all digits it dominates in partial order to True:
            #  (kk,jj)<=(k,j) if (kk>=k and jj<=j) or (kk<k and jj-kk<=j-k)
            for kk in udigits:
                if kk >= k:
                    for jj in range(j+1): udigits[kk][jj] = True
                else:
                    for jj in range(j-(k-kk)+1): udigits[kk][jj] = True
    # Compute orbit signatures.
    p_uorbit = {}
    for p in ps:
        uorbit = {}
        udigits = p_udigits[p]
        # For each k, record the number of leading True's in udigits[k].
        for k in range(1,kmax[p]+1):
            if k in udigits:
                try:
                    uorbit[k] = udigits[k].index(False)
                except:
                    uorbit[k] = k
        # Print orbit
        print_orbit(p,kmax[p],uorbit)
        p_uorbit[p] = uorbit
    return p_uorbit

def mainoutput(n, r, m=1000, alpha=0.01, k=0, nodisplay=False):
    """Computes the K-theory of random regular graph C*-algebras"""

    det_values = []  # List to store 'Det(I-A)' values

    with open(f'regular_n={n}_r={r}_m={m}_alpha={alpha}.csv', mode='w', newline='') as output_file:

        # Create column headers for csv file
        headers = ['Graph', 'Connected', 'Rank(K_1)', 'Tor(K_0)', 'Det(I-A)', 'Full shift', 'Cuntz polygon', 'Cuntz algebra']
                    
        csv_writer = csv.DictWriter(output_file, fieldnames=headers)
        csv_writer.writeheader()

        # Keep track of column totals
        column_totals = {header: 0 for header in headers if header != 'Graph'}

        for i in range(m):
            
            g = random_regular_multigraph(n, r)

            # Check if g is connected and plot the first graph
            if not nx.is_connected(nx.Graph(g)):
                print(f"Graph {i+1} is Disconnected")
                connected = 0
            else:
                print(f"Graph {i+1} is Connected")
                connected = 1

            column_totals['Connected'] += connected

            if i == k and not nodisplay: # Optionally skip graph display.
                G = nx.Graph(g)
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

                plt.figure()
                plt.title(f"Random Regular Graph (n={n}, r={r})")
                nx.draw(G, pos, with_labels=True)
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
                plt.show()

            # Compute K_0 and K_1

            pv_mat = -np.eye(n, dtype=int)
            for (a,b) in g:
                pv_mat[a,b] += 1
                pv_mat[b,a] += 1

            # Convert pv_mat to pari input, then compute its smith normal form
            mat = pari.matrix(n,n, pv_mat.flatten())
            # Compute the matrix determinant.
            dt = pari.matdet(mat)

            if dt != 0:
                # If the matrix is non-degenerate, this method
                # seems to be faster for larger denser graphs.
                #full_snf = pari.matsnf(pari.mathnfmod(mat,dt), 1)
                full_snf = pari.matsnf(pari.mathnf(mat), 1)
            else:
                # If the matrix is non-degenerate, compute snf directly.
                full_snf = pari.matsnf(mat, 1)
            # We computed the SNF form UMV = D. Now store the matrices D and U.
            snf = full_snf[2]
            snfu = full_snf[0]

            le1 = [abs(snf[i, i]) for i in range(n)]

            # K_1 is the torsion free part of K_0
            k1 = sum(1 for e in le1 if e==0)

            # Remove 0s (which are remembered by k1) and 1s (which do not contribute to the cokernel)
            le = [e for e in le1 if e!=1 and e!=0]

            # Check whether K_0 is cyclic and tally the results

            cyclic = int(len(le)<=1)
            
            column_totals['Tor(K_0)'] += cyclic

            column_totals['Rank(K_1)'] += k1

            # Check whether det(I-A)=(-1)^n*det(A-I) is positive and tally the results
            det = dt * ((-1) ** n)
            shift = int(det<0 and len(le)<=1)
            column_totals['Det(I-A)'] += int((det < 0))

            column_totals['Full shift'] += shift

            # Add det value to the list
            det_values.append(float(det))

            # Tally instances of exact isomorphism to a Cuntz polygon, via the class of the unit.

            exact = 0

            if dt != 0:
                if len(le) == 0:
                    exact += 1
                else:
                    # Internally Pari stores matrices as lists of columns, so to
                    # get to the first row, we need to transpose the matrix first.
                    utranspose = snfu.mattranspose()
                    vector1 = pari.vector(n,[1] * n)
                    unit = vector1 * utranspose
                    
                    print("Class of unit: ", unit)
                    print("Orbit of unit:")
                    orbit_unit = unit_orbit(le,unit)
                    
                    polygon_unit = [1] * len(le)
                    print("Orbit of polygon_unit:")
                    orbit_polygon_unit = unit_orbit(le,polygon_unit)

                    exact += int((orbit_unit == orbit_polygon_unit))

            column_totals['Cuntz polygon'] += exact

            # Tally instances of exact isomorphism to a Cuntz algebra, via the class of the unit.

            cuntz = exact & cyclic

            column_totals['Cuntz algebra'] += cuntz

            print("Tor(K_0): ", le)
            print("Rank(K_1): ", k1)

            row = {'Graph': i + 1, 'Connected': connected, 'Rank(K_1)': k1, 'Tor(K_0)': le, 'Det(I-A)': det, 'Full shift': shift, 'Cuntz polygon': exact, 'Cuntz algebra': cuntz}

            print()

            csv_writer.writerow(row)
            
        csv_writer.writerow(column_totals)

        # Compute 100(1-alpha)% confidence intervals using the normal distribution
        confidence_intervals = {header: 0 for header in headers if header != 'Graph'}
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

    # Create and save histogram as a PDF
    plt.figure()
    plt.hist(det_values, bins=30, edgecolor='black')
    plt.title(f'Det(I-Adjacency(G_n={n}_r={r}))')
    plt.xlabel('Det(I-A)')
    plt.ylabel(f'Frequency (m={m})')

    # Calculate summary statistics
    median = np.median(det_values)
    mean = np.mean(det_values)
    std_dev = np.std(det_values)

    # Add legend with summary statistics
    summary_stats = f'Median {median:.2e}\nMean {mean:.2e}\nStd. Dev. {std_dev:.2e}'
    plt.text(0.95, 0.9,
        summary_stats,
        horizontalalignment='right',
        verticalalignment='center',
        transform=plt.gca().transAxes,
        bbox=dict(facecolor='white', alpha=0.5))
    
    plt.savefig(f'det_histogram_G_n={n}_r={r}.pdf')
    plt.close()

##    # Create box-and-whisker plot
##    plt.boxplot(det_values)
##    plt.xlabel('Values')
##    plt.ylabel('Det Values')
##    plt.title('Box-and-Whisker Plot of Det Values')

# Command line invocation
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute the K-theory of random regular graph C*-algebras")
    parser.add_argument('--nodisplay', action="store_true", help="Skip displaying a sample graph")
    parser.add_argument("n", type=int, help="Number of graph vertices (even number larger than 2)")
    parser.add_argument("r", type=int, help="Common degree of all vertices (number between 3 and n)")
    parser.add_argument("-m", type=int, default=1000, help="Number of times to repeat the calculation")
    parser.add_argument("-alpha", type=float, default=0.01, help="Significance level")
    args = parser.parse_args()
    mainoutput(args.n, args.r, args.m, args.alpha, nodisplay=args.nodisplay)

