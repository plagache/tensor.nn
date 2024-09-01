### Variables
SYSTEM_PYTHON = /usr/bin/python3.11
VENV = .venv
BIN = ${VENV}/bin
PYTHON = ${BIN}/python3.11
PIP = ${BIN}/pip
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
	${SYSTEM_PYTHON} -m venv ${VENV}
	ln -sf ${ACTIVATE} activate

pip_upgrade:
	${PIP} install --upgrade pip

install: \
	module \
	requirements \

#
module: setup.py
	${PIP} install -e '.' --upgrade

requirements: requirements.txt
	${PIP} install -r requirements.txt --upgrade


### Info
list:
	${PIP} list

version:
	${PYTHON} --version

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
	${PIP} install -e '.[testing]' --upgrade

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
