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

# KH 07/10/2012


from .AbstractUserModel import AbstractUserModel

import requests
import json
import time
from numpy import asarray
from time import strftime, strptime, localtime, sleep
import sys


class LivingLabsRealUser(AbstractUserModel):
            
    runs = {}
    __doc_ids__ = {}
    __reversed_docids__ = {}
    __HOST__ = "http://living-labs.net:5000/api"
    KEY = "" #3061C7A142E10020-SNHXGT52ORZRQ3SC
    
    __QUERYENDPOINT__ = "participant/query"
    __DOCENDPOINT__ = "participant/doc"
    __DOCLISTENDPOINT__ = "participant/doclist"
    __RUNENDPOINT__ = "participant/run"
    __FEEDBACKENDPOINT__ = "participant/feedback"
    
    __HEADERS__ = {'content-type': 'application/json'}

    
    
    def __init__(self, key, doc_ids):
        self.KEY = key
        self.__doc_ids__ = doc_ids
        self.__reversed_docids__ = self.__get_Inverse_docids__(self.__doc_ids__)
    
    
    
    def __get_feedback__(self, qid, runid):
        """
        Returns the feedback for a given query
        """
        sleep(0.1)
        while True:
            try:
                r = requests.get("/".join([self.__HOST__, self.__FEEDBACKENDPOINT__, self.KEY, qid, runid]), headers=self.__HEADERS__, timeout=20)
                break
            except requests.exceptions.Timeout as e:
                print e, 'Retrying....'
                r = self.__get_feedback__(qid, runid)
        if r.status_code != requests.codes.ok:
            print r.text
            r.raise_for_status()
        return r.json()
    

    def __get_Inverse_docids__(self, input_dict):
        """
        Input: dictionary[queryID][living-labsDocID] = LerotDocID or dictionary[queryID][LerotDocID] = Living-labsDocID
        Returns a dictionary[queryID][living-labsDocID] = LerotDocID or dictionary[queryID][LerotDocID] = Living-labsDocID
        """
        return_dict = {}
        for qid in input_dict:
            return_dict[qid] = {v: k for k, v in input_dict[qid].items()}
        return return_dict
    
    
    def __lerot2LL_docids__(self, query, lerot_list):
        """
        Returns: List of living-labs doc ids coinciding with the lerot list entered
        """
        return_list = []
        for doc in lerot_list:
            return_list.append({'docid' : self.__doc_ids__[query.get_qid()][doc.get_id()]})
        return return_list
    
    
    
    def __LL2lerot_docids__(self, query, LL_feedbacklist, lerot_list):
        string = '\nQid: '+ query.get_qid()+ ' Unknown Docids:'
        for doc1 in LL_feedbacklist['doclist']:
            if not doc1['docid'] in self.__doc_ids__[query.get_qid()].values():
                string.join(str(doc1['docid'])+' , ') 
        string.join(('FeedbackList: ', str(LL_feedbacklist)))
        with open('feedbacklog.txt', 'a') as f:
            f.write(string)
        """
        Returns list of clicks in lerot coinciding to lerot uploaded list e.g [0 0 0 1 0 0 0 0 0 0] 
        """
        print 'qid: ', query.get_qid()
        return_list = []
        for doc2 in lerot_list:
            common_doc = False#keep track if common document has been found
            for doc1 in LL_feedbacklist['doclist']:
                if doc2['docid'] == doc1['docid']:#If document ID is the same in both lists
                    if doc1['clicked'] == True:#if the document was clicked, append 1 and break to next document in feedback
                        return_list.append(1)
                        common_doc = True
                        break
                    if doc1['clicked'] == False:#if not clicked, append 0 and break to next document in feedback
                        common_doc = True
                        return_list.append(0)
                        break
            if common_doc == False:#if current docid is not within uploaded list, append 0 regardless
                return_list.append(0)
        return asarray(return_list)


    def get_win(self, query, feedback_list, lerot_ranked_list):
        """
        Used for seznam site which interleaves ranked list with it's own list
        Returns 'ranked list winner' with number of clicks of each ranker e.g. [0 2] where [lerot_list_score seznam_list_score]
        """
        ranker_winner = [0, 0]
        for doc1 in feedback_list['doclist']:
            common_doc = False#keep track if common document has been found
            for doc2 in lerot_ranked_list['doclist']:
                if doc2['docid'] == doc1['docid']:#If document ID is the same in both lists
                    if doc1['clicked'] == True:#if the document was clicked, append
                        common_doc = True
                        ranker_winner[0] += 1
                        break
                    if doc1['clicked'] == False:
                        common_doc = True
                        break
            if common_doc == False:
                if doc1['clicked'] == True:
                    ranker_winner[1] += 1
        return ranker_winner



    def upload_run(self, query, upload_list, runid):
        """
        Uploads a run to living-labs api. 
        """
        doc_list = self.__lerot2LL_docids__(query, upload_list)
        payload = {"runid": runid, "doclist": doc_list}
        r = requests.put("/".join([self.__HOST__, self.__RUNENDPOINT__, self.KEY, query.get_qid()]), data=json.dumps(payload), headers=self.__HEADERS__)
        sleep(0.1)
        if r.status_code != requests.codes.ok:
            print r.text
            r.raise_for_status()
        the_time = strftime("%a, %d %b %Y %H:%M:%S -0000", localtime())
        
        return payload, the_time    
        
    
    def get_clicks(self, result_list, labels, **kwargs):
        time.sleep(0.1)
        """
        Returns the list of clicked documents from an uploaded lerot ranking list, and the feedback
        """
        query = kwargs['query']
        upload_time = kwargs['upload_time']
        lerot_ranked_list = kwargs['ranker_list']['doclist']
        runid = kwargs['run_id']
        qid = query.__qid__
        feedbacks = self.__get_feedback__(qid, runid)
        for feedback in feedbacks['feedback']:
            for doc in feedback['doclist']:
                if doc['team'] == 'site':
                    print feedback
                    sys.exit()
            if strptime(feedback['modified_time'], "%a, %d %b %Y %H:%M:%S -0000") >= strptime(upload_time, "%a, %d %b %Y %H:%M:%S -0000"):
                print feedback
                return feedback, self.__LL2lerot_docids__(query, feedback, lerot_ranked_list)
        return None, None
                        
                    