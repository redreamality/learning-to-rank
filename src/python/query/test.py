import cStringIO
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath('..'))

import query as qu


class TestQuery(unittest.TestCase):

    def setUp(self):
        self.test_num_features = 6
        self.test_query = """
        4 qid:1 1:2.6 2:1 3:2.1 4:0 5:2 6:1.4 # highly relevant
        1 qid:1 1:1.2 2:1 3:2.9 4:0 5:2 6:1.9 # bad
        0 qid:1 1:0.5 2:1 3:2.3 4:0 5:2 6:5.6 # not relevant
        0 qid:1 1:0.5 2:1 3:2.3 4:0 5:2 6:5.6 # not relevant
        """

    def test_queries(self):
        query_fh = cStringIO.StringIO(self.test_query)
        queries = qu.Queries(query_fh, self.test_num_features)
        query_fh.close()
        self.assertEqual(1, queries.get_size())

    def test_query(self):
        query_fh = cStringIO.StringIO(self.test_query)
        queries = qu.Queries(query_fh, self.test_num_features)
        query = queries['1']
        query_fh.close()

        self.assertEqual(4, query.get_document_count())
        self.assertEqual(4, len(query.get_feature_vectors()))
        self.assertEqual([0, 1, 2, 3], query.get_docids())
        # TODO: do "labels" have to be np array? not a list?
        self.assertEqual([4, 1, 0, 0], query.get_labels().tolist())
        self.assertEqual(1, query.get_label(1))
        self.assertEqual(None, query.get_predictions())
        self.assertEqual(None, query.get_comments())
        self.assertEqual(None, query.get_comment(0))

    def test_query_with_comments(self):
        query_fh = cStringIO.StringIO(self.test_query)
        queries = qu.Queries(query_fh, self.test_num_features, True)
        query = queries['1']
        query_fh.close()

        self.assertEqual(4, query.get_document_count())
        self.assertEqual(4, len(query.get_feature_vectors()))
        self.assertEqual([0, 1, 2, 3], query.get_docids())
        # TODO: do "labels" have to be np array? not a list?
        self.assertEqual([4, 1, 0, 0], query.get_labels().tolist())
        self.assertEqual(1, query.get_label(1))
        self.assertEqual(None, query.get_predictions())
        self.assertEqual(["# highly relevant", "# bad", "# not relevant",
            "# not relevant"], query.get_comments())
        self.assertEqual("# highly relevant", query.get_comment(0))

if __name__ == '__main__':
    unittest.main()
