from collections import namedtuple

Doc = namedtuple('Doc', ['url', 'vert'])


class VerticalAwareRanker:
    def getDocs(self, numdocs):
        """ This method should return the list of `numdocs` instances of class
        Doc """
        raise NotImplementedError("the derived class needs to implement \
        getDocs")


def checkVerticalBlock(doc_list, vertical_documents):
    """ Check that vertical documents are grouped """
    pos, size = getBlockPositionAndSize(doc_list, vertical_documents)
    if pos == -1:
        return True
    return all(d not in vertical_documents for d in doc_list[(pos + size):])


def getBlockPositionAndSize(doc_list, vertical_documents):
    """ Returns (position, size); position == -1 if no block found """
    blockStartPos = -1
    blockEndPos = len(doc_list)
    for (k, doc) in enumerate(doc_list):
        if doc in vertical_documents:
            if blockStartPos == -1:
                blockStartPos = k
        elif blockStartPos != -1:
            blockEndPos = k
            break
    return (blockStartPos, blockEndPos - blockStartPos)


def analyzeAllowedRankings(rankings, rA, rB, vertical_documents):
    """
        Analyze size and position of the vertical block
        in the allowed rankings compared to rA and rB
    """
    positionA, sizeA = getBlockPositionAndSize(rA, vertical_documents)
    positionB, sizeB = getBlockPositionAndSize(rB, vertical_documents)
    positions = []
    sizes = []
    for rk in rankings:
        if rk == rA or rk == rB:
            continue
        p, s = getBlockPositionAndSize(rk, vertical_documents)
        positions.append(p)
        sizes.append(s)

    biases = set([])
    if positions and all(p > max(positionA, positionB) for p in positions):
        msg = 'BIAS! Vertical block is too low'
        print msg
        biases.add(msg)

    if sizes and all(s > max(sizeA, sizeB) for s in sizes):
        msg = 'BIAS! Vertical block is too big'
        print msg
        biases.add(msg)

    if sizes and all(s < min(sizeA, sizeB) for s in sizes):
        msg = 'BIAS! Vertical block is too small'
        print msg
        biases.add(msg)
    # TODO: make more checks
    return biases
