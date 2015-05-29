import requests
import json
import time
import random

HOST = "http://living-labs.net:5000/api"
KEY = "3061C7A142E10020-SNHXGT52ORZRQ3SC"

QUERYENDPOINT = "participant/query"
DOCENDPOINT = "participant/doc"
DOCLISTENDPOINT = "participant/doclist"
RUNENDPOINT = "participant/run"
FEEDBACKENDPOINT = "participant/feedback"

HEADERS = {'content-type': 'application/json'}

def get_queries():
        print "/".join([HOST, QUERYENDPOINT, KEY]), HEADERS
        r = requests.get("/".join([HOST, QUERYENDPOINT, KEY]), headers=HEADERS)
        if r.status_code != requests.codes.ok:
                print r.text
                r.raise_for_status()
        return r.json()

queries = get_queries()



def get_doclist(qid):
        print "/".join([HOST, DOCLISTENDPOINT, KEY, qid])
        r = requests.get("/".join([HOST, DOCLISTENDPOINT, KEY, qid]), headers=HEADERS)
        if r.status_code != requests.codes.ok:
                print r.text
                r.raise_for_status()
        return r.json()

runs = {}
for query in queries["queries"]:
        qid = query["qid"]
        runs[qid] = get_doclist(qid)
        
        
        
def get_feedback(qid):
        print "/".join([HOST, FEEDBACKENDPOINT, KEY, qid])
        r = requests.get("/".join([HOST, FEEDBACKENDPOINT, KEY, qid]),
                                        headers=HEADERS)
        time.sleep(random.random())
        if r.status_code != requests.codes.ok:
                print r.text
                r.raise_for_status()
        return r.json()


while True:
        for query in queries["queries"]:
                qid = query["qid"]
                feedbacks = get_feedback(qid)
                clicks = dict([(doc['docid'], 0) for doc in runs[qid]['doclist']])
                for feedback in feedbacks['feedback']:
                        for doc in feedback["doclist"]:
                                if doc["clicked"] and doc["docid"] in clicks:
                                        clicks[doc["docid"]] += 1
                runs[qid]['doclist'] = [{'docid': docid}
                                        for docid, _ in
                                        sorted(clicks.items(),
                                                   key=lambda x: x[1],
                                                   reverse=True)]
                r = requests.put("/".join([HOST, RUNENDPOINT, KEY, qid]),
                                        data=json.dumps(runs[qid]), headers=HEADERS)

                if r.status_code != requests.codes.ok:
                        print r.text
                        r.raise_for_status()
                time.sleep(random.random())