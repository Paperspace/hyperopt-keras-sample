CURRENT_DIR=$(shell pwd)
HYPEROPT_DIR=$(CURRENT_DIR)/hyper_param

install_graph_viz:
	apt-get update && apt-get install -y graphviz

install_req: install_graph_viz
	pip install -r requirements.txt

run_hyperopt: install_req
	cd $(HYPEROPT_DIR) && python hyperopt_optimize.py

run_hyperopt_worker: install_req
	cd $(HYPEROPT_DIR) && export PYTHONPATH=$PYTHONPATH:$(pwd) && hyperopt-mongo-worker --mongo $(MONGO_DB_HOST):$(MONGO_DB_PORT)/$(EXPERIMENT_NAME) --poll-interval=2.0 --exp-key=$(EXPERIMENT_NAME)
