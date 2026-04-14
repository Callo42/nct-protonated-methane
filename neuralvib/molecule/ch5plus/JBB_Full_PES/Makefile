# Build the JBB CH5+ PES Python extension via f2py.
#
# Requirements:
#   - a Fortran compiler (gfortran by default; override with FC=...)
#   - numpy available to the python interpreter invoked here
#
# Usage (recommended, under uv):
#   uv run make
#
# Usage (manual):
#   make PYTHON=python3 FC=gfortran

PYTHON ?= python
FC     ?= gfortran
SRC    := $(CURDIR)/src
PYF    := $(CURDIR)/JBBCH5ppotential.pyf
MODULE := JBBCH5ppotential
F_SOURCES := $(wildcard $(SRC)/*.f) $(wildcard $(SRC)/*.f90)

.PHONY: all clean

# Absolute paths are required because f2py invokes meson inside a temp build
# directory, so relative `-I` and source paths would not resolve.
all:
	$(PYTHON) -m numpy.f2py -c $(PYF) $(F_SOURCES) -I$(SRC) \
		--f77exec=$(FC) --f90exec=$(FC)

clean:
	rm -f $(MODULE)*.so
	rm -f *.mod $(SRC)/*.mod
	rm -rf __pycache__ $(SRC)/__pycache__
