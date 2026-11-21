# Rebuild the analysis from results/: rank every engine, evaluate against
# the reference, then draw figures and tables.
#   make figures SYSTEM=hcho_nh3 REFERENCE=dft COMPARE=xtb,mace,aimnet,ani1xnr

SYSTEM ?= hcho_nh3
REFERENCE ?= dft
COMPARE ?= xtb,mace,uma,aimnet,ani1xnr
comma := ,
ENGINES := $(REFERENCE) $(subst $(comma), ,$(COMPARE))

.PHONY: rank evaluate figures test lint

rank:
	for engine in $(ENGINES); do \
		uv run nrnet rank --system $(SYSTEM) --engine $$engine || exit 1; \
	done

evaluate: rank
	uv run nrnet evaluate --system $(SYSTEM) --reference $(REFERENCE) \
		--compare $(COMPARE)

figures: evaluate
	uv run nrnet figures --system $(SYSTEM) --reference $(REFERENCE) \
		--compare $(COMPARE)

test:
	uv run pytest -q

lint:
	uv run ruff check --extend-select E501 src tests
	uv run ruff format --check src tests
