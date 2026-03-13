.PHONY: all analysis figures supplementary discovery clean check

# Reproduce all results from scratch
all: analysis supplementary discovery figures

# Primary analysis: literature-calibrated populations, bootstrap CIs, sensitivity grid, random search
# Produces results/revised_manuscript_data.json (~10 min on Apple Silicon)
analysis:
	python revised_analysis.py

# Generate manuscript figures from analysis results
# Requires results/revised_manuscript_data.json
figures:
	python figures.py

# Supplementary analyses: random search benchmarking and stability analyses
# Produces results/supplementary_analyses.json
supplementary:
	python supplementary_analyses.py

# Full pipeline: baseline, analytical bounds, multi-agent discovery
# Requires ANTHROPIC_API_KEY for the agent competition (~45 min including API latency)
discovery:
	python run_discovery.py

# Verify results match expected values from the manuscript
check:
	python -c "\
	import json, os; \
	d = json.load(open('results/revised_manuscript_data.json')); \
	p = d['primary_analysis']['primary']['scoring_comparison']; \
	delta = p['standard']['delta']; \
	c_stat = p['standard']['c_statistic']; \
	brier = p['standard']['brier']['brier_score']; \
	print(f'Standard score Delta: {delta:.3f} (expect 0.035)'); \
	print(f'C-statistic: {c_stat:.3f} (expect 0.965)'); \
	print(f'Brier score: {brier:.3f} (expect 0.142)'); \
	assert abs(delta - 0.035) < 0.005, f'Delta out of range: {delta}'; \
	assert abs(c_stat - 0.965) < 0.005, f'C-stat out of range: {c_stat}'; \
	assert abs(brier - 0.142) < 0.005, f'Brier out of range: {brier}'; \
	print('Primary analysis checks passed.'); \
	assert os.path.exists('results/supplementary_analyses.json'), 'supplementary_analyses.json not found'; \
	s = json.load(open('results/supplementary_analyses.json')); \
	rs_max = s['expanded_random_search']['best_delta']; \
	print(f'Random search max Delta: {rs_max:.3f} (expect ~0.270)'); \
	assert abs(rs_max - 0.270) < 0.020, f'Random search max out of range: {rs_max}'; \
	print('Supplementary checks passed.'); \
	print('All checks passed.')"

clean:
	rm -rf results/*.pdf results/*.png __pycache__
