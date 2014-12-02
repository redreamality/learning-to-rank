# This file is part of Lerot.
#
# Lerot is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Lerot is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Lerot.  If not, see <http://www.gnu.org/licenses/>.


"""
Interface to query data with functionality for reading queries from svmlight
format, both sequentially and in batch mode.
"""

import sys
import gc
import gzip
import numpy as np
import os.path
import requests
from .document import Document
import time
__all__ = ['Query', 'Queries', 'QueryStream', 'load_queries', 'write_queries']



class SimpleBufferedLineReader:
    """Read lines from a file, but keep a short buffer to allow rewinds"""

    __prev__ = None
    __next__ = None
    __fh__ = None

    def __init__(self, fh):
        self.__fh__ = fh

    def has_next(self):
        if self.__next__ == None:  # not set
            self.__next__ = self.__fh__.readline()
        if len(self.__next__) < 1:  # empty string - EOF
            return False
        return True

    def next(self):
        self.__prev__ = self.__next__
        self.__next__ = None
        return self.__prev__

    def rewind(self):
        if self.__prev__:
            self.__next__ = self.__prev__
            self.__prev__ = None


class Query:
    __qid__ = None
    __feature_vectors__ = None
    __labels__ = None
    __predictions__ = None
    __comments__ = None
    # document ids will be initialized as zero-based, os they can be used to
    # retrieve labels, predictions, and feature vectors
    __docids__ = None
    __ideal__ = None
   

    def __init__(self, qid, feature_vectors, labels=None, comments=None):
        self.__qid__ = qid
        self.__feature_vectors__ = feature_vectors
        self.__labels__ = labels
        self.__docids__ = [Document(x) for x in range(len(labels))]
        self.__comments__ = comments
        
        
    def has_ideal(self):
        return not self.__ideal__ is None

    def set_ideal(self, ideal):
        self.__ideal__ = ideal

    def get_ideal(self):
        return self.__ideal__

    def get_qid(self):
        return self.__qid__

    def get_docids(self):
        return self.__docids__

    def get_document_count(self):
        return len(self.__docids__)

    def get_feature_vectors(self):
        return self.__feature_vectors__

    def set_feature_vector(self, docid, feature_vector):
        self.__feature_vectors__[docid.get_id()] = feature_vector

    def get_feature_vector(self, docid):
        return self.__feature_vectors__[docid.get_id()]

    def get_labels(self):
        return self.__labels__

    def set_label(self, docid, label):
        self.__labels__[docid.get_id()] = label

    def get_label(self, docid):
        return self.__labels__[docid.get_id()]

    def set_labels(self, labels):
        self.__labels__ = labels

    def get_comments(self):
        return self.__comments__

    def get_comment(self, docid):
        if self.__comments__ is not None:
            return self.__comments__[docid.get_id()]
        return None

    def get_predictions(self):
        return self.__predictions__

    def get_prediction(self, docid):
        if self.__predictions__:
            return self.__predictions__[docid.get_id()]
        return None

    def set_predictions(self, predictions):
        self.__predictions__ = predictions

    def write_to(self, fh, sparse=False):
        for doc in self.__docids__:
            features = [':'.join((repr(pos + 1),
                repr(value))) for pos, value in enumerate(
                self.get_feature_vector(doc)) if not (value == 0 and sparse)]
            print >> fh, self.get_label(doc), ':'.join(("qid",
                self.get_qid())), ' '.join(features),
            comment = self.get_comment(doc)
            if comment:
                print >> fh, comment
            else:
                print >> fh, ""
    

class QueryStream:
    """iterate over a stream of queries, only keeping one query at a time"""
    __reader__ = None
    __numFeatures__ = 0

    def __init__(self, fh, num_features, preserve_comments=False):
        self.__reader__ = SimpleBufferedLineReader(fh)
        self.__num_features__ = num_features
        self.__preserve_comments__ = preserve_comments

    def __iter__(self):
        return self

    # with some inspiration from multiclass.py
    # http://svmlight.joachims.org/svm_struct.html
    # takes self and number of expected features
    # returns qid and features, one query at a time
    def next(self):
        prev = None
        instances = None
        targets = None
        comments = None
        initialized = False
        while self.__reader__.has_next():
            line = self.__reader__.next()
            line = line.rstrip("\n")
            # explicitly start a new query
            if line.startswith("# qid "):
                # if we have already read a query - break
                # don't need to rewind, we'll just skip the comment next time
                if initialized:
                    break
                # otherwise just start reading
                else:
                    continue
            # remove comments
            if line.startswith("#"):
                continue
            comment = ""
            if line.find("#") >= 0:
                comment_index = line.find("#")
                comment = line[comment_index:]
                line = line[:comment_index]
            # extract target and features
            tokens = line.split()
            if not tokens:
                continue
            target = int(tokens[0])
            qid = tokens[1].split(':')[1]
            if not qid:
                print >> sys.stderr, "Invalid line - skipping '", line, "'"
                continue
            if not prev:
                prev = qid
            if qid != prev:
                self.__reader__.rewind()
                break
            features = [tuple(t.split(":")) for t in tokens[2:]]
            tmp_array = np.zeros((1, len(features)))
            for k, v in features:
                tmp_array[0, int(k) - 1] = float(v)
            if not initialized:
                instances = tmp_array
                targets = np.array([target])
                if self.__preserve_comments__:
                    comments = [comment]
                initialized = True
            else:
                instances = np.vstack((instances, tmp_array))

                targets = np.hstack((targets, target))
                if self.__preserve_comments__:
                    comments.append(comment)

        if not initialized:
            raise StopIteration
        else:
            #(qid, [[featuresDoc1], [featuresDocN], targets, comments])
            return Query(prev, instances, targets, comments)

    # read all queries from a file at once
    def read_all(self):
        queries = {}
        for query in self:
            queries[query.get_qid()] = query
        return queries


class Queries:
    """a list of queries with some convenience functions"""
    __num_features__ = 0
    __queries__ = None

    # cache immutable query values
    __qids__ = None
    __feature_vectors__ = None
    __labels__ = None

    def __init__(self, fh, num_features, preserve_comments=False):
        self.__queries__ = QueryStream(fh, num_features,
            preserve_comments).read_all()

        self.__num_features__ = num_features

    def __iter__(self):
        return iter(self.__queries__.itervalues())

    def __getitem__(self, index):
        return self.get_query(index)

    def __len__(self):
        return len(self.__queries__)

    def keys(self):
        return self.__queries__.keys()

    def values(self):
        return self.__queries__.values()

    def get_query(self, index):
        return self.__queries__[index]

    def get_qids(self):
        if not self.__qids__:
            self.__qids__ = [query.get_qid() for query in self]
        return self.__qids__

    def get_labels(self):
        if not self.__labels__:
            self.__labels__ = [query.get_labels() for query in self]
        return self.__labels__

    def get_feature_vectors(self):
        if not self.__feature_vectors__:
            self.__feature_vectors__ = [query.get_feature_vectors()
                for query in self]
        return self.__featureVectors__

    def set_predictions(self):
        raise NotImplementedError("Not yet implemented")

    def get_predictions(self):
        if not self.__predictions__:
            self.__predictions__ = [query.get_predictions() for query in self]
        return self.__predictions__

    def get_size(self):
        return self.__len__()




class LivingLabsQueries(Queries):
    __KEY__ = ''
    __HOST__ = "http://living-labs.net:5000/api"
    __QUERYENDPOINT__ = "participant/query"
    __DOCENDPOINT__ = "participant/doc"
    __DOCLISTENDPOINT__ = "participant/doclist"
    __RUNENDPOINT__ = "participant/run"
    __FEEDBACKENDPOINT__ = "participant/feedback"
    __KEY__ = ''
    __HEADERS__ = {'content-type': 'application/json'}
    __doc_ids__ = {}#used for living-labs
    __doc_counter__ = 0
    __LL_queries__ = {}

    
    def __init__(self, KEY):
        self.__KEY__ = KEY
        self.__queries__ = {}
        self.__LL_queries__  = self.__get_queries__()
        self.__doc_ids__ = {}
        time.sleep(4)
        self.__num_features__ = self.__get_num_features__(self.__LL_queries__ )
        qlen = len(self.__LL_queries__ ['queries'])
        print qlen, 'Num of queries'
        counter = 0
        for query in self.__LL_queries__ ['queries']:
            counter += 1
            qid = query['qid']
            print qid,  counter, 'of', qlen
            doclist = self.__get_doclist__(qid)
            time.sleep(4)
            instances = self.__get_features__(qid, doclist, self.__num_features__)
            self.__queries__[qid] = Query(qid, instances, [0]*len(instances), "")#instances = self.get_features(qid, HOST, KEY) # Should by numpy array
            self.__set_doc_ids__(qid, doclist)
            time.sleep(4)


    def __set_doc_ids__(self, qid, doclist):
        """
        Updates global dictionary of document ids for a given query. 
        Uses document ID as used in lerot and maps this to document ID on living-labs
        dictionary[queryID][LerotDocID] = living-labs ID
        """
        self.__doc_ids__[qid] = {}
        for doc in range(len(doclist['doclist'])): 
            self.__doc_ids__[qid][doc] = doclist['doclist'][doc]['docid']
            


    def __get_queries__(self):
        """Returns a Dictionary of all Queries."""
        time.sleep(1)
        print "/".join([self.__HOST__, self.__QUERYENDPOINT__, self.__KEY__])
        try:
            r = requests.get("/".join([self.__HOST__, self.__QUERYENDPOINT__, self.__KEY__]), headers=self.__HEADERS__, timeout=3)
        except Exception, e:
            print e
        if r.status_code != requests.codes.ok:
            print r.text
            r.raise_for_status()
        return r.json()
    
    
    def __get_features__(self, qid, doclist, num_features):
        """
        Returns numpy array of numpy arrays (of features for each document)
        """
        feature_list = []
        for doc in range(len(doclist['doclist'])):
            if 'relevance_signals' in doclist['doclist'][doc] and len(doclist['doclist'][doc]['relevance_signals'])>0:# Some documents had empty feature lists, check to avoid crash
                doc_feature_list = np.zeros(num_features)
                for feature in xrange(len(doclist['doclist'][doc]['relevance_signals'])-1):
                    doc_feature_list[feature] = doclist['doclist'][doc]['relevance_signals'][feature][1]
                feature_list.append(doc_feature_list)
        feature_list = np.asarray(feature_list)
        return feature_list
    
    
    def __get_num_features__(self, queries):
        """
        Returns (integer): the number of features in the documents. Only checks one document.
        """
        feature_num = 0
        for query in queries['queries']:
            qid = query['qid']
            time.sleep(2)
            doclist = self.__get_doclist__(qid)
            for doc in range(len(doclist['doclist'])):
                if 'relevance_signals' in doclist['doclist'][doc]:
                    for feature in doclist['doclist'][doc]['relevance_signals']:
                        feature_num += 1
                    return feature_num


    def __get_doclist__(self, qid):
        """
        Return the document list for a given query.
        """
        time.sleep(1)
        print "/".join([self.__HOST__, self.__DOCLISTENDPOINT__, self.__KEY__, qid]), self.__HEADERS__
        r = requests.get("/".join([self.__HOST__, self.__DOCLISTENDPOINT__, self.__KEY__, qid]), headers=self.__HEADERS__, timeout=10)
        print 'Worked'
        if r.status_code != requests.codes.ok:
                print r.text
                r.raise_for_status()
        return r.json()
    


def load_livinglabs_queries(key):
    """Utility method for loading living-labs queries."""
    queries = LivingLabsQueries(key)
    return queries



def load_queries(filename, features, preserve_comments=False):
    """Utility method for loading queries from a file."""
    if filename.endswith(".gz"):
        fh = gzip.open(filename)
    else:
        fh = open(filename)
    gc.disable()
    queries = Queries(fh, features, preserve_comments)
    gc.enable()
    fh.close()
    return queries


def write_queries(filename, queries):
    """Utility method for writing queries to a file. Returns the number of
        queries written"""
    if os.path.exists(filename):
        raise ValueError("Target file already exists: %s" % filename)
    if filename.endswith(".gz"):
        fh = gzip.open(filename, "w")
    else:
        fh = open(filename, "w")
    query_count = 0
    for query in queries:
        query.write_to(fh)
        query_count += 1
    fh.close()
    return query_count
