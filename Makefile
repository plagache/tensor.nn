SYSTEM_PYTHON = /usr/bin/python3.11
VENV = .venv
BIN = ${VENV}/bin
PYTHON = ${BIN}/python3.11
PIP = ${BIN}/pip
ACTIVATE = ${BIN}/activate

PROGRAMS = examples
SIMPLE_OPS = ${PROGRAMS}/simple_operation.py
MNIST = ${PROGRAMS}/mnist.py

# ARGUMENTS =

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
	${PIP} install -e . --upgrade

requirements: requirements.txt
	${PIP} install -r requirements.txt --upgrade

list:
	${PIP} list

version:
	${PYTHON} --version

size:
	du -hd 0
	du -hd 0 ${VENV}

run:
	${PYTHON} ${SIMPLE_OPS} \
	# ${ARGUMENTS}

#
mnist: static
	${PYTHON} ${MNIST} \
	# ${ARGUMENTS}

#
test:
	${PYTHON} test/test_ops.py \
	-v

#
function:
	${PYTHON} examples/simple_function.py \

static:
	mkdir -p static/Datasets

clean:
	rm -rf static/

fclean: clean
	rm -rf ${VENV}
	rm -rf activate

re: fclean setup run

.SILENT:
.PHONY: setup venv pip_upgrade install module requirements list version run clean fclean re test
