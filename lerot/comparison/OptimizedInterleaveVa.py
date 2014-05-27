import argparse
import collections
import itertools

from .OptimizedInterleave import OptimizedInterleave
from ..utils import get_class, split_arg_str


def enumerate_allowed_dispositions(pos_limits, size_limits):
    """ Generator that yields position and block size pairs. """
    return itertools.product(
        range(pos_limits[0], pos_limits[1] + 1),
        range(size_limits[0], size_limits[1] + 1),
    )


def check_disposition(disposition, length):
    """ Receives an iterable of (size, pos) pairs and check if they are OK.

    The disposition is said to be OK if it is possible to place verticals
    of the size `size` at positions given by `pos` without intersection.
    """
    ranking = [0] * length
    for pos, size in disposition:
        end = pos + size
        if end > length:
            return False
        for rank in xrange(pos, end):
            if ranking[rank] != 0:
                return False
            ranking[rank] = 1
    return True


def fill_ranking(disposition, block_rankings, length):
    """ Fill the ranking with docs from block_rankings according to disposition. """
    ranking = [None] * length
    # Ignore the 'Web' ranking for now.
    web_ranking = block_rankings[-1]
    for disp, block_ranking in zip(disposition, block_rankings[:-1]):
        pos, size = disp
        for rank, doc in zip(xrange(pos, pos + size), block_ranking[:size]):
            ranking[rank] = doc
    # Fill the rest with 'Web' documents.
    idx = 0
    for rank in xrange(length):
        if ranking[rank] is None:
            ranking[rank] = web_ranking[idx]
            idx += 1
    return ranking


class OptimizedInterleaveVa(OptimizedInterleave):

    def __init__(self, arg_str=None):
        OptimizedInterleave.__init__(self, arg_str)
        parser = argparse.ArgumentParser()
        parser.add_argument('--allowed_leavings',
                            choices=['prefix_constraint',
                                     'prefix_constraint_va'],
                            default='prefix_constraint_va')
        parser.add_argument("--credit_va", action="store_true", default=False)
        parser.add_argument("--um_class", type=str,
                            default="environment.FederatedClickModel")
        parser.add_argument("--um_args", type=str,
                            default="0.2 0.1")
        args = vars(parser.parse_known_args(split_arg_str(arg_str))[0])
        self.allowed_leavings = getattr(self, args['allowed_leavings'])
        if args["credit_va"]:
            self.precompute_rank = self.precompute_rank_va
            self.um_class = get_class(args["um_class"])
            self.um = self.um_class(args["um_args"])
        
    def precompute_rank_va(self, R):
        sortR = R[:]
        examination = self.um.get_examination_prob(sortR)
        sortR.sort(key=dict(zip(sortR, examination)).get, reverse=True)
        return OptimizedInterleave.precompute_rank(self, sortR)

    def prefix_constraint_va(self, rankings, length):
        rankings_by_type = collections.defaultdict(lambda: [[] for r in rankings])
        for idx, ranking in enumerate(rankings):
            for doc in ranking:
                rankings_by_type[doc.get_type()][idx].append(doc)
        block_size_limits = {}
        block_pos_limits = {}
        for t, block_rankings in rankings_by_type.iteritems():
            # Eq. (11).
            block_size_limits[t] = (
                min(len(b) for b in block_rankings),
                max(len(b) for b in block_rankings)
            )
            if t == 'Web':
                continue
            # Find the ranks (0-based) of all the vertical blocks
            # in each ranking: rank(X_t, X).
            block_poss = []
            for ranking in rankings:
                for i, doc in enumerate(ranking):
                    if doc.get_type() == t:
                        block_poss.append(i)
                        break
                else:
                    block_poss.append(len(ranking))
            # Eq. (12)-(13).
            block_pos_limits[t] = (min(block_poss), max(block_poss))

        # print block_pos_limits
        # print block_size_limits

        nblocks = [len(set(d.get_type() for d in r if d.get_type() != 'Web')) for r in rankings]

        # Eq. (14)-(15).
        v_types = [t for t in rankings_by_type if t != 'Web']
        # Now we iterate over all possible combinations of selected verticals,
        # positions and the sizes of the vertical blocks. Only if
        # such disposition is possible, i.e., the blocks don't overlap,
        # we iterate over all the possible contents of the blocks and
        # the 'Web' documents surrounding the blocks.
        L = []
        for num_blocks in xrange(min(nblocks), max(nblocks) + 1):
            # All the combinations of num_blocks verticals
            for selected_verts in itertools.combinations(v_types, num_blocks):
                # All the possible pairs of position / size for each selected vertical.
                for disposition in itertools.product(*[enumerate_allowed_dispositions(
                        block_pos_limits[t], block_size_limits[t]) for t in selected_verts]):
                    if not check_disposition(disposition, length):
                        continue
                    # Eq. (9)-(10) -- the main prefix constraint (extended).
                    allowed_leavings_by_type = dict((
                        t, self.prefix_constraint(rankings_by_type[t], disp[1])
                    ) for (t, disp) in zip(selected_verts, disposition))
                    selected_verts_list = list(selected_verts)
                    num_insert_web_docs = length - sum(disp[1] for disp in disposition)
                    if num_insert_web_docs > 0:
                        allowed_leavings_by_type['Web'] = self.prefix_constraint(
                                rankings_by_type['Web'], num_insert_web_docs
                        )
                        selected_verts_list += ['Web']
                    # Now enumerate all the possible rankings for all vertical + Web
                    for block_rankings in itertools.product(
                            *[allowed_leavings_by_type[t] for t in selected_verts_list]):
                        # Note: it is essential that block_rankings[-1] contains web ranking.
                        ranking = fill_ranking(disposition, block_rankings, length)
                        L.append(ranking)
        return L


import unittest
from ranker import SyntheticDeterministicRankingFunction
from document import Document

class TestEvaluation(unittest.TestCase):

    def setUp(self):
        pass

    def makeSerps(self, templates):
        types = {'w': 'Web', 'i': 'Image', 'n': 'News'}
        serps = []
        mapping = {}
        for template in templates:
            counter = 0
            serp = []
            for d in template:
                while counter in mapping and mapping[counter] != d:
                    counter += 1
                mapping[counter] = d
                counter += 1
                serp.append(Document(counter, types[d]))

            serps.append(serp)
        return serps

    def testVA(self):
        #strA = 'wwwnnwwiiiwww'
        #strB = 'wnnnwiiwwwwww'

        #strA = 'wwwwwwwiiiwnw'
        #strB = 'wwwwwwwiiiwwn'

        strA = 'wni'
        strB = 'win'

        dA, dB = self.makeSerps([strA, strB])

        #Web, News = 'Web', 'News'
        #dA, dB = [Document(1, Web), Document(0, Web), Document(5, Web), Document(9, Web), Document(4, Web), Document(7, Web), Document(10, Web), Document(2, News), Document(3, News), Document(6, News)], [Document(0, Web), Document(1, Web), Document(4, Web), Document(5, Web), Document(9, Web), Document(7, Web), Document(2, News), Document(3, News), Document(6, News), Document(8, News)]

        comparison = OptimizedInterleaveVa('')
        A = SyntheticDeterministicRankingFunction(dA)
        B = SyntheticDeterministicRankingFunction(dB)
        dI, a = comparison.interleave(A, B, None, min(len(dA), len(dB)))
        print dA
        print dB
        print list(dI)


if __name__ == '__main__':
    # from pycallgraph import PyCallGraph
    # from pycallgraph.output import GraphvizOutput
    #
    # with PyCallGraph(output=GraphvizOutput()):
    unittest.main()
