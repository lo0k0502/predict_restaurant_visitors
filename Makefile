PY_FILES := $(wildcard *.py)

%:
	python3 -c 'from $@ import $@; $@()'

all: $(basename $(PY_FILES))