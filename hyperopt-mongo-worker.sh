#!/usr/bin/env bash
export MONGO_DB_HOST=localhost
export MONGO_DB_PORT=27017
export EXPERIMENT_NAME=testing3
export PYTHONPATH=$PYTHONPATH:$(pwd)
hyperopt-mongo-worker --mongo=$MONGO_DB_HOST:$MONGO_DB_PORT/$EXPERIMENT_NAME --poll-interval=1.0 --exp-key=$EXPERIMENT_NAME
