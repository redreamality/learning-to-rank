Lerot: an Online Learning to Rank Framework
===========================================
This project is designed to run experiments on online learning to rank methods for information retrieval. Below is a short summary of its prerequisites, how to run an experiment, and possible extensions.

Prerequisites
-------------
- Python (2.7 or higher)
- PyYaml
- Numpy
- Scipy
- Celery (only for distributed runs)
- Gurobi (only for OptimizedInterleave)

All prerequisites (except for Celery and Gurobi) are included in the academic distribution of Enthought 
Python, e.g., version 7.1.

Installation
------------
Install the prerequisites plus Lerot as follows::

    $ pip install PyYAML numpy scipy celery
    $ git clone https://bitbucket.org/ilps/lerot.git
    $ cd lerot
    $ python setup.py install

Running experiments
-------------------
1) prepare data in svmlight format, e.g., download the *MQ2007* (see next section on `Data`_) ::

        $ mkdir data
        $ wget http://research.microsoft.com/en-us/um/beijing/projects/letor/LETOR4.0/Data/MQ2007.rar -O data/MQ2007.rar
        $ unrar x data/MQ2007.rar data/
        
2) prepare a configuration file in yml format, e.g., starting from the template below, store as ``config/experiment.yml`` (or simply use ``config/config.yml`` instead ) ::

        training_queries: data/MQ2007/Fold1/train.txt
        test_queries: data/MQ2007/Fold1/test.txt
        feature_count: 46
        num_runs: 1
        num_queries: 10
        query_sampling_method: random
        output_dir: outdir
        output_prefix: Fold1
        user_model: environment.CascadeUserModel
        user_model_args:
            --p_click 0:0.0,1:0.5,2:1.0
            --p_stop 0:0.0,1:0.0,2:0.0
        system: retrieval_system.ListwiseLearningSystem
        system_args:
            --init_weights random
            --sample_weights sample_unit_sphere
            --comparison comparison.ProbabilisticInterleave
            --delta 0.1
            --alpha 0.01
            --ranker ranker.ProbabilisticRankingFunction
            --ranker_arg 3
            --ranker_tie random
        evaluation:
            - evaluation.NdcgEval

3) run the experiment using python::
        
        $ python src/scripts/learning-experiment.py -f config/experiment.yml

4) summarize experiment outcomes::
   
        $ python src/scripts/summarize-learning-experiment.py --fold_dirs outdir
   
   Arbitrarily many folds can be listed per experiment. Results are aggregated  over runs and folds. The output format is a simple text file that can be  further processed using e.g., gnuplot. The columns are: mean_offline_perf stddev_offline_perf mean_online_perf stddev_online_perf

Data
----
Lerot acceptes data formatted in the SVMlight (see http://svmlight.joachims.org/) format.
You can download learning to rank data sets here:

- **GOV**: http://research.microsoft.com/en-us/um/beijing/projects/letor/LETOR3.0/Gov.rar (you'll need files in QueryLevelNorm)
- **OHSUMED**: http://research.microsoft.com/en-us/um/beijing/projects/letor/LETOR3.0/OHSUMED.zip
- **MQ2007**: http://research.microsoft.com/en-us/um/beijing/projects/letor/LETOR4.0/Data/MQ2007.rar (files for supervised learning)
- **MQ2008**: http://research.microsoft.com/en-us/um/beijing/projects/letor/LETOR4.0/Data/MQ2008.rar (files for supervised learning)
- **Yahoo!**: http://webscope.sandbox.yahoo.com/catalog.php?datatype=c
- **MSLR-WEB10K**: http://research.microsoft.com/en-us/um/beijing/projects/mslr/data/MSLR-WEB10K.zip
- **MSLR-WEB30K**: http://research.microsoft.com/en-us/um/beijing/projects/mslr/data/MSLR-WEB30K.zip
- **Yandex Internet Mathematics 2009**: http://imat2009.yandex.ru/academic/mathematic/2009/en/datasets (query identifier need to be parsed out of comment into qid feature)

Note that Lerot reads from both plain text and text.gz files.


Extensions
----------
The code doesn't need to be can easily be extended with new learning and/or feedback mechanisms for future experiments. The most obvious points for extension are:

1) comparison - extend ComparisonMethod to add new interleaving or inference methods; existing methods include balanced interleave, team draft, and  probabilistic interleave.
2) retrieval_system - extend OnlineLearningSystem to add a new mechanism for learning from click feedback. New implementations need to be able to provide a  ranked list for a given query, and ranking solutions should have the form of a vector.

License
-------
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.