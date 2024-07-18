# Variables
SYSTEM_PYTHON = /usr/bin/python3.11
VENV = .venv
BIN = ${VENV}/bin
PYTHON = ${BIN}/python3.11
PIP = ${BIN}/pip
ACTIVATE = ${BIN}/activate

EXAMPLES = examples

# PROGRAMS = simple_operation.py
# PROGRAMS = simple_function.py
# PROGRAMS = mnist.py
PROGRAMS = mlp.py

# ARGUMENTS =


# Setup
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

static:
	mkdir -p static/datasets


# Info
list:
	${PIP} list

version:
	${PYTHON} --version

size:
	du -hd 0
	du -hd 0 ${VENV}

# Run
run: static
	${PYTHON} ${EXAMPLES}/${PROGRAMS} \
	# ${ARGUMENTS}

#


# Test
test_module: setup.py
	${PIP} install -e '.[testing]' --upgrade

test:
	${PYTHON} -m unittest discover test \
	# ${PYTHON} test/test_ops.py \
	# -v

#


# Clean
clean:
	rm -rf static/

fclean: clean
	rm -rf ${VENV}
	rm -rf activate

re: fclean setup run

.SILENT:
.PHONY: setup venv pip_upgrade install module requirements list version run clean fclean re test
