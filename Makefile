.PHONY: env pilot full-two-state full-nyha agents analyze all clean

PY := /tmp/rework_venv/bin/python
CODE := code

env:
	python3 -m venv /tmp/rework_venv
	/tmp/rework_venv/bin/pip install --quiet -r requirements.txt

pilot:
	cd $(CODE) && $(PY) run_pilot.py

full-two-state:
	cd $(CODE) && $(PY) run_full_experiments.py --two-state --n 3000 --n-sims-ts 300 --seeds 1 2 3 --skip-agent

full-nyha:
	cd $(CODE) && $(PY) run_full_experiments.py --nyha --n 5000 --n-sims-nyha 200 --seeds 1 2 3 --skip-agent

agents:
	cd $(CODE) && $(PY) run_agents.py two_state --n 3000 --n-sims-ts 300
	cd $(CODE) && $(PY) run_agents.py nyha --n-sims-nyha 200

analyze:
	cd $(CODE) && $(PY) analyze_results.py
	cd $(CODE) && $(PY) mechanism_figure.py
	cd $(CODE) && $(PY) supplementary_figures.py

all: env full-two-state full-nyha agents analyze

clean:
	rm -rf results/ figures/
