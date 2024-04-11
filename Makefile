SYSTEM_PYTHON = /usr/bin/python3
VENV = .venv
BIN = ${VENV}/bin
PYTHON = ${BIN}/python3
PIP = ${BIN}/pip
ACTIVATE = ${BIN}/activate

PROGRAMS = examples
PROGRAM = ${PROGRAMS}/simple_operation.py

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
	du -hd 0 ${VENV}

run:
	${PYTHON} ${PROGRAM} \
	# ${ARGUMENTS}

#
test:
	${PYTHON} test/test_ops.py \
	-v

#
clean:

fclean: clean
	rm -rf ${VENV}
	rm -rf activate

re: fclean setup run

.SILENT:
.PHONY: setup venv pip_upgrade install module requirements list version run clean fclean re test
