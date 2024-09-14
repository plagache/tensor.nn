### Variables
# SYSTEM_PYTHON = /usr/bin/python3.11
# PYTHON = ${BIN}/python3.11
# PIP = ${BIN}/pip
VENV = .venv
BIN = ${VENV}/bin
PYTHON = ${BIN}/python
ACTIVATE = ${BIN}/activate

EXAMPLES_DIR = examples/
EXAMPLE = tensor.py
# EXAMPLE = mnist.py
# EXAMPLE = mlp_train.py
PROGRAM = ${EXAMPLES_DIR}${EXAMPLE}

# DATASETS_DIR = datasets/
# DATASET = mlp.csv
ARGUMENTS = ${DATASETS_DIR}${DATASET}


### Setup
setup: venv pip_upgrade install

venv:
	# ${SYSTEM_PYTHON} -m venv ${VENV}
	uv venv --python 3.12
	ln -sf ${ACTIVATE} activate

pip_upgrade:
	uv pip install --upgrade pip

install: \
	module \
	requirements \

#
module: setup.py
	uv pip install -e '.' --upgrade

requirements: requirements.txt
	uv pip install -r requirements.txt --upgrade


### Info
list:
	uv pip list

size:
	du -hd 0
	du -hd 0 ${VENV}


### Run
run:
	${PYTHON} ${PROGRAM} \
	${ARGUMENTS}

#


### Test
test_module: setup setup.py
	uv pip install -e '.[testing]' --upgrade

test:
	${PYTHON} -m unittest discover test \
	# ${PYTHON} test/test_ops.py \
	# -v

#


### Clean
clean:

fclean: clean
	rm -rf ${VENV}
	rm -rf activate

re: fclean setup run

.SILENT:
.PHONY: setup venv pip_upgrade install module requirements list version run clean fclean re test
