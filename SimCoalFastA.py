import argparse
import numpy as np
import random

def mutate(seq, m):
    jukes_cantor = {'A':['T','C','G'], 'T':['A','C','G'], 'C':['T','A','G'], 'G':['T','C','A']}
    pos = random.sample(range(len(seq)), m)
    for i in pos: seq = seq[:i] + random.choice(jukes_cantor[seq[i]]) + seq[(i+1):]
    return seq

def print_seqs(seqs):
    for i in seqs.keys():
        print(">indiv",i,sep="")
        print(seqs[i])

def getvarsites(d, seqlen):
    var = []
    pos = []
    for p in range(seqlen):
        site = [ d[x][p:p+1] for x in d.keys() ]
        #if not all_same(site):
        if len(set(site)) > 1: # all var sites
            var.append(site)
            pos.append(p)
    return pos,var

def getsfs(var):
    N = len(var[0])
    sfs = [ 0 for i in range(int(N/2)) ]
    for v in var:
        a = list(set(v))
        a = filter(lambda x: 'A' in x or 'C' in x or 'T' in x or 'G' in x, a)
        cm = sorted(list(map(lambda x: v.count(x), a)), reverse=True)
        sfs[cm[1]-1] += 1
    return sfs

def make_newick(mat, branch_lengths, ntips):
    nodes = sorted(list(set(mat[:,0])))
    rootnode = max(nodes)
    tree_builder = {}
    for node in nodes:
        children = mat[mat[:,0] == node,1]
        br = branch_lengths[mat[:,0] == node]
        if all(children <= ntips):
            tips = [ "indiv"+str(i) for i in children ]
            brln = [ ":"+str(i) for i in br ]
            tree_builder[node] = "("+tips[0]+brln[0]+","+tips[1]+brln[1]+")"
        elif all(children > ntips):
            edges = [ tree_builder[i]+":"+str(j) for i,j in zip(children,br) ]
            tree_builder[node] = "("+edges[0]+","+edges[1]+")"
        else:
            for i in range(len(children)):
                if children[i] <= ntips:
                    edge = ",indiv"+str(children[i])+":"+str(br[i])
                else:
                    clade = tree_builder[children[i]]+":"+str(br[i])
            tree_builder[node] = "(" + clade + edge + ")"
    return tree_builder[rootnode] + ";"

# main simulation function
# N = haplotypes
# bp = number of base pairs
# theta in per site units
def coalescent(N, bp, theta, allseqs=False, tree=False):
    # init tips and nodes
    tips = list(range(1, N+1))
    currnode = max(tips)
    nodes = tips
    # coalesce first pair
    lineages = random.sample(nodes, 2)
    currnode += 1
    mat = np.array([[currnode, lineages[0]],[currnode, lineages[1]]])
    nodes = [ node for node in nodes if node not in lineages ]
    nodes.append(currnode)
    # iterate over remaining nodes
    while len(nodes) > 1:
        lineages = random.sample(nodes, 2)
        currnode += 1
        mat_add = np.array([[currnode, lineages[0]],[currnode, lineages[1]]])
        mat = np.vstack((mat, np.array([[currnode, lineages[0]],[currnode, lineages[1]]])))
        nodes = [ node for node in nodes if node not in lineages ]
        nodes.append(currnode)
    # coalescent times from exp distribution
    # make them sorted and descending
    coaltimes = sorted(np.random.exponential(theta, N-1))[::-1]
    # add time 0 for tips
    coaltimes = coaltimes + list(np.zeros(N))
    # list of nodes
    # also sorted and descending
    nodes = sorted(list(set(mat[:,0])))[::-1]
    # add tips
    all_nodes = nodes + list(range(1,N+1))
    # make dictionary with branching times
    bts = dict(list(zip(all_nodes, coaltimes)))
    # make branch length array
    branch_lengths = []
    # loop through edges:
    for i in range(mat.shape[0]):
        branch_lengths.append(bts[mat[i,0]] - bts[mat[i,1]])
    base_differences = np.array([ list(np.random.poisson(br * bp * 0.5, 1))[0] for br in branch_lengths ])
    # ancestral sequence
    anc_seq = "".join([ random.choice("ATGC") for i in range(bp) ])
    seqs = {max(nodes):anc_seq}
    # loop through nodes
    # add mutations throughout genealogy
    for node in nodes:
        children = mat[mat[:,0] == node,1]
        bp_diffs = base_differences[mat[:,0] == node]
        for child in range(len(children)):
            seqs[children[child]] = mutate(seqs[node], bp_diffs[child])
    if tree:
        return make_newick(mat, np.array(branch_lengths), N)
    elif allseqs:
        return seqs
    else: 
        return { i:seqs[i] for i in range(1, N+1) }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="simPoly.py",
    #version="0.3",
    formatter_class=argparse.RawTextHelpFormatter,
    description="""
    Simulates neutral polymorphism sequence data.\n""",
    epilog="""
    Output is in FASTA format

    examples:
    python simPoly.py -N 10 -s 300 -t 0.01
    \n""")
    parser.add_argument(
    '--sample_size', '-N', metavar='INT', type=int, required=True,
    help='Number of sequences (individuals) sampled.')
    parser.add_argument(
    '--seq_len', '-s', metavar='INT', type=int, required=True,
    help='Sequence length in base pairs.')
    parser.add_argument(
    '--theta', '-t', metavar='FLOAT', type=float, default=4,
    help='The value of theta (4Neu) in number of sites.')
    parser.add_argument(
    '--segsites', '-S', action="store_true", default=False,
    help="Print only segregating sites.")
    parser.add_argument(
    '--sfs', '-F', action="store_true", default=False,
    help="Print only site frequency spectrum (SFS).")
    parser.add_argument(
    '--newick', '-nw', action="store_true", default=False,
    help="Print coalescent tree in Newick format.")

    args = parser.parse_args()

    seqs = coalescent(args.sample_size, args.seq_len, args.theta)

    if args.segsites:
        _,var = getvarsites(seqs, args.seq_len)
        var = np.array(var)
        for i in range(var.shape[1]):
            print(">indiv"+str((i+1)),sep="")
            print("".join(list(var[:,i])))
    elif args.sfs:
        _,var = getvarsites(seqs, args.seq_len)
        SFS = getsfs(var)
        print(",".join([ str(i) for i in SFS]))
    elif args.newick:
        print(coalescent(args.sample_size, args.seq_len, args.theta, tree=args.newick))
    else:
        print_seqs(seqs)    

    # if args.segsites or args.sfs:
    #     simulate_neutral_segsites(theta=args.theta, bp=args.seq_len, indiv=args.sample_size, sfs=args.sfs)
    # else:
    #     simulate_neutral_sequence(theta=args.theta, bp=args.seq_len, indiv=args.sample_size)

