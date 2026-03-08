.PHONY: all analysis figures discovery clean check

# Reproduce all results from scratch
all: analysis discovery figures

# Primary analysis: literature-calibrated populations, bootstrap CIs, sensitivity grid, random search
# Produces results/revised_manuscript_data.json (~10 min on Apple Silicon)
analysis:
	python revised_analysis.py

# Generate manuscript figures from analysis results
# Requires results/revised_manuscript_data.json
figures:
	python figures.py

# Full pipeline: baseline, analytical bounds, multi-agent discovery
# Requires ANTHROPIC_API_KEY for the agent competition (~45 min including API latency)
discovery:
	python run_discovery.py

# Verify results match expected values from the manuscript
check:
	python -c "\
	import json; \
	d = json.load(open('results/revised_manuscript_data.json')); \
	p = d['primary_analysis']['primary']['scoring_comparison']; \
	delta = p['standard']['delta']; \
	c_stat = p['standard']['c_statistic']; \
	nri = p['trajectory_aware']['nri_vs_standard']['nri_total']; \
	print(f'Standard score Delta: {delta:.3f} (expect 0.035)'); \
	print(f'C-statistic: {c_stat:.3f} (expect 0.965)'); \
	print(f'Trajectory-aware NRI: {nri:+.2f} (expect +0.73)'); \
	assert abs(delta - 0.035) < 0.005, f'Delta out of range: {delta}'; \
	assert abs(c_stat - 0.965) < 0.005, f'C-stat out of range: {c_stat}'; \
	assert abs(nri - 0.73) < 0.05, f'NRI out of range: {nri}'; \
	print('All checks passed.')"

clean:
	rm -rf results/*.pdf results/*.png __pycache__
