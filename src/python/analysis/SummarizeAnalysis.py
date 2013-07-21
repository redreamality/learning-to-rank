import gzip
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import os
from AbstractAnalysis import AbstractAnalysis
from numpy import mean, std


class SummarizeAnalysis(AbstractAnalysis):
    def __init__(self, *parms):
        AbstractAnalysis.__init__(self, *parms)
        self.analyticsfilename = os.path.join(self.analyticsroot,
                                              "summary.yml")
        self.analyticsfilenametmp = os.path.join(self.analyticsroot,
                                              "summary.yml.tmp")

        self.summaries = {}
        self.discount_factor = 0.995

    def _update(self, um, data, fold, run, filename):
        if not um in self.summaries:
            self.summaries[um] = {}
        if not data in self.summaries[um]:
            self.summaries[um][data] = {"agg_online_ndcg": None,
                                        "agg_offline_ndcg": None}

        if filename.endswith(".gz"):
            fh = gzip.open(filename, "r")
        else:
            fh = open(filename, "r")
        yamldata = yaml.load(fh, Loader=Loader)
        fh.close()

        if not yamldata or \
                not "online_ndcg" in yamldata or \
                not "offline_ndcg" in yamldata:
            return False

        if not self.summaries[um][data]["agg_online_ndcg"]:
            count_queries = len(yamldata["online_ndcg"])
            self.summaries[um][data]["agg_online_ndcg"] =  \
                                    [[] for i in range(count_queries)]
            self.summaries[um][data]["agg_offline_ndcg"] = \
                                    [[] for i in range(count_queries)]

        for i, value in enumerate(yamldata["online_ndcg"]):
            prev = 0.0
            if i > 0:
                prev = self.summaries[um][data]["agg_online_ndcg"][i - 1][-1]
            self.summaries[um][data]["agg_online_ndcg"][i].append(prev +
                                            self.discount_factor ** i * value)
        for i, value in enumerate(yamldata["offline_ndcg"]):
            self.summaries[um][data]["agg_offline_ndcg"][i].append(value)

        return True

    def finish(self):
        dump = {}
        for um in self.summaries:
            if not um in dump:
                dump[um] = {}
            for data in self.summaries[um]:
                if not data in dump[um]:
                    dump[um][data] = []
                if not self.summaries[um][data]["agg_online_ndcg"]:
                    continue
                count_queries = len(self.summaries[um][data]
                                    ["agg_online_ndcg"])
                for i in range(count_queries):
                    dump[um][data].append([i,
                        float(mean(self.summaries[um][data]["agg_offline_ndcg"][i])),
                        float(std(self.summaries[um][data]["agg_offline_ndcg"][i])),
                        float(min(self.summaries[um][data]["agg_offline_ndcg"][i])),
                        float(max(self.summaries[um][data]["agg_offline_ndcg"][i])),
                        float(mean(self.summaries[um][data]["agg_online_ndcg"][i])),
                        float(std(self.summaries[um][data]["agg_online_ndcg"][i])),
                        float(min(self.summaries[um][data]["agg_online_ndcg"][i])),
                        float(max(self.summaries[um][data]["agg_online_ndcg"][i]))])
        fh = open(self.analyticsfilenametmp, 'w')
        yaml.dump(dump, fh, Dumper=Dumper)
        fh.close()
        os.rename(self.analyticsfilenametmp, self.analyticsfilename)
        return self.analyticsfilename


if __name__ == "__main__":
    import sys
    a = SummarizeAnalysis(sys.argv[1])
    a.update()
    print a.finish()
