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

import os
import argparse
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import gzip
import glob
import time
import logging

from collections import namedtuple
# Fallback settings in case celery is not installed.
celery = namedtuple('Dummy', ['task'])(task=lambda x: x)

try:
    from celery import Celery
    # TODO: move this to config
    celery = Celery('tasks',
                #broker='amqp://USER:PW@HOST/QUEUE',
                backend='amqp')
                #include=['environment',
                #         'retrieval_system',
                #         'comparison',
                #         'evaluation',
                #         'query',
                #         'ranker',
                #         'sampler',
                #         'analysis'])
    celery.conf.CELERYD_PREFETCH_MULTIPLIER = 1
    celery.conf.CELERYD_HIJACK_ROOT_LOGGER = False
    celery.conf.BROKER_POOL_LIMIT = 2
    celery.conf.CELERY_ACKS_LATE = True
    celery.conf.CELERYD_POOL_RESTARTS = True
except:
    import sys
    print >>sys.stderr, 'Celery support is disabled'

from utils import get_class
from experiment import GenericLearningExperiment


class MetaExperiment:
    def __init__(self):
        # parse arguments
        parser = argparse.ArgumentParser(description="""Meta experiment""")

        file_group = parser.add_argument_group("FILE")
        file_group.add_argument("-f", "--file", help="Filename of the config "
                                "file from which the experiment details"
                                " should be read.")
        # option 2: specify all experiment details as arguments
        detail_group = parser.add_argument_group("DETAILS")
        detail_group.add_argument("-p", "--platform", help="Specify "
                                  "'local' or 'celery'")
        detail_group.add_argument('--data', help="Data in the following"
                                  "format: trainfile,testfile,d,r such that "
                                  "a data file can be found in "
                                  "datadir/trainfile/Fold1/train.txt",
                            type=str, nargs="+")
        detail_group.add_argument('--um', nargs="+")
        detail_group.add_argument('--uma', help="",
                            type=str, nargs="+")
        detail_group.add_argument('--analysis', nargs="*")
        detail_group.add_argument('--data_dir')
        detail_group.add_argument('--output_base')
        detail_group.add_argument('--experiment_name')
        detail_group.add_argument("-r", "--rerun", action="store_true",
                                  help="Rerun last experiment.",
                                  default=False)
        detail_group.add_argument("--queue_name", type=str)

        args = parser.parse_known_args()[0]

        logging.basicConfig(format='%(asctime)s %(module)s: %(message)s',
                        level=logging.INFO)

        # determine whether to use config file or detailed args
        self.experiment_args = None
        if args.file:
            config_file = open(args.file)
            self.experiment_args = yaml.load(config_file, Loader=Loader)
            config_file.close()
            try:
                self.meta_args = vars(parser.parse_known_args(
                                    self.experiment_args["meta"].split())[0])
            except:
                parser.error("Please make sure there is a 'meta' section "
                             "present in the config file")
            # overwrite with command-line options if given
            for arg, value in vars(args).items():
                if value:
                    self.meta_args[arg] = value
        else:
            self.meta_args = vars(args)

        for k in self.meta_args.keys() + ["meta"]:
            if k in self.experiment_args:
                del self.experiment_args[k]


        if self.meta_args["platform"] == "local":
            self.run = self.run_local
        elif self.meta_args["platform"] == "celery":
            self.experiment_args["processes"] = 0
            self.run = self.run_celery
        elif self.meta_args["platform"] == "conf":
            self.run = self.run_conf
        else:
            parser.error("Please specify a valid platform.")

        usermodels = {}
        for umstr in self.meta_args["uma"]:
            parts = umstr.split(',')
            um, car = parts[:2]
            car = int(car)
            if len(parts) != car * 2 + 2:
                parser.error("Error in uma")
            p_click = ", ".join(parts[2:2 + car])
            p_stop = ", ".join(parts[2 + car:])
            if not um in usermodels:
                usermodels[um] = {}
            usermodels[um][car] = "--p_click %s --p_stop %s" % \
                                                    (p_click, p_stop)

        basedir = os.path.join(os.path.abspath(self.meta_args["output_base"]),
                               self.meta_args["experiment_name"])

        i = 0
        while os.path.exists(os.path.join(basedir, "v%03d" % i)):
            i += 1
        if i > 0 and self.meta_args["rerun"]:
            i -= 1
        logging.info("Running experiment v%03d" % i)
        basedir = os.path.join(basedir, "v%03d" % i)
        if not os.path.exists(basedir):
            os.makedirs(basedir)
        logging.info("Results appear in %s" % basedir)

        config_bk = os.path.join(basedir, "meta_config_bk.yml")
        config_bk_file = open(config_bk, "w")
        yaml.dump(self.meta_args,
                  config_bk_file,
                  default_flow_style=False,
                  Dumper=Dumper)
        config_bk_file.close()

        skip = 0
        self.configurations = []
#        for run_id in range(self.experiment_args["num_runs"]):
        for um in self.meta_args["um"]:
            for dstr in self.meta_args["data"]:
                data, d, r = dstr.split(',')
                d, r = int(d), int(r)
                user_model_args = usermodels[um][r]
                folds = glob.glob(os.path.join(
                            os.path.abspath(self.meta_args["data_dir"]),
                            data,
                            "Fold*"))
                for fold in folds:
                    args = self.experiment_args.copy()
                    args["data_dir"] = self.meta_args["data_dir"]
                    args["fold_dir"] = fold
        #            args["run_id"] = run_id
                    args["feature_count"] = d
                    args["user_model_args"] = user_model_args
                    args["output_dir"] = os.path.join(basedir,
                                'output',
                                um,
                                data,
                                os.path.basename(fold))
                    args["output_prefix"] = os.path.basename(fold)
                    if self.meta_args["rerun"]:
                        if not os.path.exists(os.path.join(
                                                args["output_dir"],
                                                "%s-%d.txt.gz" %
                                                (args["output_prefix"],
                                                 run_id))):
                            self.configurations.append(args)
                        else:
                            skip += 1
                    else:
                        self.configurations.append(args)
        logging.info("Created %d configurations (and %d skipped)" % (
                                                    len(self.configurations),
                                                    skip))
        self.analytics = []
        for analyse in self.meta_args["analysis"]:
            aclass = get_class(analyse)
            a = aclass(basedir)
            self.analytics.append(a)

    def update_analytics(self):
        logging.info("Updating analytics for all existing log files.")
        for a in self.analytics:
            a.update()

    def update_analytics_file(self, log_file):
        for a in self.analytics:
            a.update_file(log_file)

    def finish_analytics(self):
        for a in self.analytics:
            a.finish()

    def store(self, conf, r):
        if not os.path.exists(conf["output_dir"]):
            try:
                os.makedirs(conf["output_dir"])
            except:
                pass
        log_file = os.path.join(conf["output_dir"], "%s-%d.txt.gz" %
                                (conf["output_prefix"], conf["run_id"]))
        log_fh = gzip.open(log_file, "wb")
        yaml.dump(r, log_fh, default_flow_style=False, Dumper=Dumper)
        log_fh.close()
        return log_file

    def run_conf(self):
        if self.meta_args["rerun"]:
            self.update_analytics()

        logging.info("Creating log files %d tasks locally" % len(self.configurations))
        for conf in self.configurations:
            train = glob.glob(os.path.join(conf["fold_dir"], "*train.txt*"))[0]
            test = glob.glob(os.path.join(conf["fold_dir"], "*test.txt*"))[0]
            conf["test_queries"] = test
            conf["training_queries"] = train

            if not os.path.exists(conf["output_dir"]):
                try:
                    os.makedirs(conf["output_dir"])
                except:
                    pass
            config_bk = os.path.join(conf["output_dir"], "config_bk.yml")
            config_bk_file = open(config_bk, "w")
            yaml.dump(conf,
                      config_bk_file,
                      default_flow_style=False,
                      Dumper=Dumper)
            config_bk_file.close()
        logging.info("Done")

    def run_local(self):
        if self.meta_args["rerun"]:
            self.update_analytics()

        logging.info("Running %d tasks locally" % len(self.configurations))
        for conf in self.configurations:
            train = glob.glob(os.path.join(conf["fold_dir"], "*train.txt*"))[0]
            test = glob.glob(os.path.join(conf["fold_dir"], "*test.txt*"))[0]
            conf["test_queries"] = test
            conf["training_queries"] = train

            if not os.path.exists(conf["output_dir"]):
                try:
                    os.makedirs(conf["output_dir"])
                except:
                    pass
            config_bk = os.path.join(conf["output_dir"], "config_bk.yml")
            config_bk_file = open(config_bk, "w")
            yaml.dump(conf,
                      config_bk_file,
                      default_flow_style=False,
                      Dumper=Dumper)
            config_bk_file.close()
            e = GenericLearningExperiment("-f " + config_bk)
            r = e.run_experiment()
            log_file = self.store(conf, r)
            self.update_analytics_file(log_file)
            logging.info("Done with %s, run %d" %
                         (conf["output_dir"], conf["run_id"]))
        logging.info("Done")

    def apply(self, conf):
        return run_task.apply_async((conf, ), queue=self.queuename)

    def run_celery(self):
        self.queuename = self.meta_args["experiment_name"]
        if "queue_name" in self.meta_args and \
                    not self.meta_args["queue_name"] == None:
            self.queuename = self.meta_args["queue_name"]
        logging.info("Submitting %d tasks to queue %s " %
                     (len(self.configurations),
                      self.queuename))
        results = []
        for conf in self.configurations:
            results.append((conf, self.apply(conf)))

        if self.meta_args["rerun"]:
            self.update_analytics()

        while results:
            found = False
            for confresult in results[:]:
                conf, asyncresult = confresult
                if asyncresult.ready():
                    found = True
                    try:
                        log_file = self.store(conf, asyncresult.result)
                        self.update_analytics_file(log_file)
                        logging.info("Done with %s, run %d" %
                                     (conf["output_dir"], conf["run_id"]))
                    except Exception, e:
                        logging.info("Rerun %s, run %d, state %s,d error %s" %
                                     (conf["output_dir"],
                                      conf["run_id"],
                                      asyncresult.state,
                                      e))
                        if not str(asyncresult.state) == "SUCCES":
                            results.append((conf, self.apply(conf)))
                    results.remove(confresult)
            if not found:
                time.sleep(1)
            else:
                self.finish_analytics()
        logging.info("Done")


@celery.task
def run_task(conf):
    if not os.path.exists(conf["output_dir"]):
        try:
            os.makedirs(conf["output_dir"])
        except:
            pass

    logging.basicConfig(format='%(asctime)s %(module)s: %(message)s',
                        level=logging.INFO,
                        filename=os.path.join(conf["output_dir"],
                                              "output.%d.log" %
                                              conf["run_id"]))
    logging.info("Starting %s, run %d" % (conf["output_dir"], conf["run_id"]))

    if os.environ.get('TMPDIR'):
        fold = "/".join(conf["fold_dir"].strip("/").split("/")[-2:])
        train = glob.glob("/scratch/*/" + fold + "/*train.txt*")[0]
        test = glob.glob("/scratch/*/" + fold + "/*test.txt*")[0]
    else:
        train = glob.glob(os.path.join(conf["fold_dir"], "*train.txt*"))[0]
        test = glob.glob(os.path.join(conf["fold_dir"], "*test.txt*"))[0]
    conf["test_queries"] = test
    conf["training_queries"] = train

    if not os.path.exists(conf["output_dir"]):
        try:
            os.makedirs(conf["output_dir"])
        except:
            pass

    config_bk = os.path.join(conf["output_dir"], "config_bk.%d.yml" %
                             conf["run_id"])
    config_bk_file = open(config_bk, "w")
    yaml.dump(conf,
              config_bk_file,
              default_flow_style=False,
              Dumper=Dumper)
    config_bk_file.close()

    e = GenericLearningExperiment("-f " + config_bk)
    return e.run_experiment()
