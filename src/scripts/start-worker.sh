cd ../python
nice celery worker -A experiment.MetaExperiment:celery -Q $1 -E -c $2  -b amqp://USER:PW@HOST/QUEUE --autoreload
