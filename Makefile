.PHONY: compile install clean all help

compile:
	uv pip compile requirements.in -o requirements.txt

install:
	uv pip install --upgrade pip && \
	uv pip install -r requirements.txt; \
	clear

clean:
	rm -rf .pytest_cache .mypy_cache .coverage .pytest_cache __pycache__ *.egg-info dist build; clear

all: compile install clean

help:
	@echo "Makefile commands:"
	@echo "  compile  - Compile the requirements"
	@echo "  install  - Install the requirements"
	@echo "  clean    - Clean up temporary files"
	@echo "  all      - Run all of the above commands"
