# ============================================================================
# Makefile — Coverage Simulations (Binomial calibration)
# ============================================================================
# Outputs:
#   - CSV sims  : outputs/simulation/coverage/
#   - Tables    : outputs/simulation/coverage/table/
#   - Plots     : outputs/simulation/coverage/table/plot/
# ============================================================================

.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.DEFAULT_GOAL := help


# ============================================================================
# 0) ENVIRONMENT
# ============================================================================
export OMP_NUM_THREADS ?= 1
export MKL_NUM_THREADS ?= 1
export OPENBLAS_NUM_THREADS ?= 1

PY ?= PYTHONPATH=src poetry run python


# ============================================================================
# 1) DIRECTORIES
# ============================================================================
OUT_ROOT     := outputs
COV_OUTDIR   := $(OUT_ROOT)/simulation/coverage
COV_TBL_DIR  := $(COV_OUTDIR)/table
COV_PLOT_DIR := $(COV_TBL_DIR)/plot

SRC_SIM_DIR := src/simulation
COV_SCRIPT_SIM   := $(SRC_SIM_DIR)/make_coverage.py
COV_SCRIPT_PLOT  := $(SRC_SIM_DIR)/plot_coverage.py
COV_SCRIPT_TABLE := $(SRC_SIM_DIR)/make_table_coverage.py


# ============================================================================
# 2) GLOBAL CONFIG — COVERAGE SIM
# ============================================================================
COV_ALPHA ?= 0.05
COV_P0    ?= 0.01
COV_N_MC  ?= 3000
COV_SEED  ?= 123

COV_E_A   ?= 0.5
COV_E_B   ?= 0.5

# Exp 1: vary p_true
COV_N_FIXED ?= 2000
COV_P_MIN   ?= 0.0
COV_P_MAX   ?= 0.2
COV_P_STEPS ?= 41

# Exp 2: vary n
COV_P_TRUE_FIXED ?= 0.01
COV_N_MIN        ?= 100
COV_N_MAX        ?= 100000
COV_N_STEPS      ?= 18


# ============================================================================
# 3) HELP
# ============================================================================
.PHONY: help
help:
	@echo "==============================================================="
	@echo "  COVERAGE SIMULATIONS — BINOMIAL CALIBRATION"
	@echo "==============================================================="
	@echo ""
	@echo "Targets:"
	@echo "  make coverage_sim      : run MC -> outputs/simulation/coverage/*.csv"
	@echo "  make coverage_tables   : build summary tables -> outputs/simulation/coverage/table/"
	@echo "  make coverage_plots    : build plots -> outputs/simulation/coverage/table/plot/"
	@echo "  make coverage_all      : sim + tables + plots"
	@echo ""
	@echo "Clean:"
	@echo "  make clean_coverage    : remove outputs/simulation/coverage/"
	@echo ""
	@echo "Overrides examples:"
	@echo "  make coverage_all COV_N_MC=20000 COV_SEED=7"
	@echo "  make coverage_sim COV_N_FIXED=5000 COV_P_MAX=0.05 COV_P_STEPS=51"
	@echo "  make coverage_sim COV_N_MIN=200 COV_N_MAX=200000 COV_N_STEPS=22"
	@echo "==============================================================="


# ============================================================================
# (C) SIMULATIONS — COVERAGE
# ============================================================================
$(COV_OUTDIR):
	@mkdir -p $(COV_OUTDIR)

$(COV_TBL_DIR):
	@mkdir -p $(COV_TBL_DIR)

$(COV_PLOT_DIR):
	@mkdir -p $(COV_PLOT_DIR)


.PHONY: coverage_sim
coverage_sim: $(COV_OUTDIR)
	@echo "\n[COV] Running Monte Carlo simulations..."
	@test -f "$(COV_SCRIPT_SIM)" || (echo "[ERR] Missing script: $(COV_SCRIPT_SIM)"; exit 1)
	$(PY) $(COV_SCRIPT_SIM) \
		--alpha $(COV_ALPHA) \
		--p0 $(COV_P0) \
		--n-mc $(COV_N_MC) \
		--seed $(COV_SEED) \
		--e-a $(COV_E_A) --e-b $(COV_E_B) \
		--n-fixed $(COV_N_FIXED) \
		--p-min $(COV_P_MIN) --p-max $(COV_P_MAX) --p-steps $(COV_P_STEPS) \
		--p-true-fixed $(COV_P_TRUE_FIXED) \
		--n-min $(COV_N_MIN) --n-max $(COV_N_MAX) --n-steps $(COV_N_STEPS) \
		--outdir $(COV_OUTDIR)
	@echo "✔ CSV written to: $(COV_OUTDIR)"
	@ls -lh $(COV_OUTDIR)/*.csv 2>/dev/null || true


.PHONY: coverage_tables
coverage_tables: $(COV_TBL_DIR)
	@echo "\n[COV] Building tables..."
	@test -f "$(COV_SCRIPT_TABLE)" || (echo "[ERR] Missing script: $(COV_SCRIPT_TABLE)"; exit 1)
	@test -f "$(COV_OUTDIR)/coverage_vary_p.csv" || (echo "[ERR] Missing: $(COV_OUTDIR)/coverage_vary_p.csv (run make coverage_sim)"; exit 1)
	@test -f "$(COV_OUTDIR)/coverage_vary_n.csv" || (echo "[ERR] Missing: $(COV_OUTDIR)/coverage_vary_n.csv (run make coverage_sim)"; exit 1)
	$(PY) $(COV_SCRIPT_TABLE) \
		--indir $(COV_OUTDIR) \
		--outdir $(COV_TBL_DIR)
	@echo "✔ Tables written to: $(COV_TBL_DIR)"
	@ls -lh $(COV_TBL_DIR)/*.csv 2>/dev/null || true


.PHONY: coverage_plots
coverage_plots: $(COV_PLOT_DIR)
	@echo "\n[COV] Building plots..."
	@test -f "$(COV_SCRIPT_PLOT)" || (echo "[ERR] Missing script: $(COV_SCRIPT_PLOT)"; exit 1)
	@test -f "$(COV_OUTDIR)/coverage_vary_p.csv" || (echo "[ERR] Missing: $(COV_OUTDIR)/coverage_vary_p.csv (run make coverage_sim)"; exit 1)
	@test -f "$(COV_OUTDIR)/coverage_vary_n.csv" || (echo "[ERR] Missing: $(COV_OUTDIR)/coverage_vary_n.csv (run make coverage_sim)"; exit 1)
	$(PY) $(COV_SCRIPT_PLOT) \
		--indir $(COV_OUTDIR) \
		--outdir $(COV_PLOT_DIR)
	@echo "✔ Plots written to: $(COV_PLOT_DIR)"
	@ls -lh $(COV_PLOT_DIR)/*.png 2>/dev/null || true


.PHONY: coverage_all
coverage_all: coverage_sim coverage_tables coverage_plots
	@echo "\n✔ COVERAGE PIPELINE DONE (sim + tables + plots)."


# ============================================================================
# CLEAN
# ============================================================================
.PHONY: clean_coverage
clean_coverage:
	@echo "\n[COV] Removing $(COV_OUTDIR)..."
	@rm -rf $(COV_OUTDIR)
	@echo "✔ Coverage outputs cleaned."