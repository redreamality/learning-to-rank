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

import argparse
import re
import sys
from random import random
from numpy import zeros

from .AbstractUserModel import AbstractUserModel
from ..utils import split_arg_str

import requests
import json
import time
import random
import sys
from time import gmtime, strftime, strptime, localtime



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
        self.__reversed_docids__ = self.__getInv_doc_ids__(self.__doc_ids__)
    
    
    
    def __get_feedback__(self, qid):
        r = requests.get("/".join([self.__HOST__, self.__FEEDBACKENDPOINT__, self.KEY, qid]), headers=self.__HEADERS__)
        if r.status_code != requests.codes.ok:
            print r.text
            r.raise_for_status()
        return r.json()
    

    def __getInv_doc_ids__(self, input_dict):
        return_dict = {}
        """Inverts inner dictionary to 
        dictionary[queryID][living-labs ID] = LerotDocID"""
        for qid in input_dict:
            return_dict[qid] = {v: k for k, v in input_dict[qid].items()}
        return return_dict
    
    
    def __translate_docids__(self, query, qid, input_list, ranker_list=None):
        return_list = []
        '''If list is from lerot, translate to real docids'''
        if ranker_list==None:# and str(type(input_list[0])) == "<class 'lerot.document.Document'>":
            for doc in input_list:
                return_list.append({'docid' : self.__doc_ids__[qid][doc.get_id()]})
        else:#Else list is not from lerot, in which case translate to lerot ranked list (with click numbers) which was previously uploaded'''
            for doc1 in ranker_list:
                for doc2 in input_list['doclist']:
                    if doc1['docid'] == doc2['docid']:
                        if doc2['clicked'] == True:
                            return_list.append(1)
                        if doc2['clicked'] == False:
                            return_list.append(0)
            print ranker_list
            print input_list
        return return_list

    
    def upload_run(self, query, result_list, runid):
        qid = query.__qid__
        doc_list = self.__translate_docids__(query, qid, result_list)
        payload = {"runid": runid, "doclist": doc_list}
        r = requests.put("/".join([self.__HOST__, self.__RUNENDPOINT__, self.KEY, qid]), data=json.dumps(payload), headers=self.__HEADERS__)
        if r.status_code != requests.codes.ok:
            print r.text
            r.raise_for_status()
        the_time = strftime("%a, %d %b %Y %H:%M:%S -0000", localtime())
        
        return payload, the_time    

    
    def get_clicks(self, result_list, labels, **kwargs):
        query = kwargs['query']
        upload_time = kwargs['upload_time']
        ranker_list = kwargs['ranker_list']['doclist']
        qid = query.__qid__
        feedbacks = self.__get_feedback__(qid)
        for feedback in feedbacks['feedback']:
            if strptime(feedback['modified_time'], "%a, %d %b %Y %H:%M:%S -0000") > strptime(upload_time, "%a, %d %b %Y %H:%M:%S -0000"):
                print feedback['modified_time'], '>', upload_time
                return self.__translate_docids__(query, qid, feedback, ranker_list)
        #print feedback['modified_time'], '<', upload_time
            
                        
                    