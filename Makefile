# CONCH Inference Makefile for TCGA Kidney Classification
# TCGA Kidney: KIRC, KICH, KIRP (3 tumor types)

PYTHON := python
H5_PATH_KIRC := /project/hnguyen2/mvu9/processing_datasets/processing_tcga_256/kirc/conch_features_fp/patch_256x256_20x/h5_files/*.h5
H5_PATH_KICH := /project/hnguyen2/mvu9/processing_datasets/processing_tcga_256/kich/conch_features_fp/patch_256x256_20x/h5_files/*.h5
H5_PATH_KIRP := /project/hnguyen2/mvu9/processing_datasets/processing_tcga_256/kirp/conch_features_fp/patch_256x256_20x/h5_files/*.h5
OUTPUT_DIR := results
CONFIG := config.yaml
DEVICE := cuda

# Default target
.PHONY: help
help:
	@echo "CONCH Inference for TCGA Kidney Classification"
	@echo ""
	@echo "Available targets:"
	@echo "  make install          - Install dependencies"
	@echo "  make run-all         - Run inference on all 3 tumor types (KIRC, KICH, KIRP)"
	@echo "  make run-kirc        - Run inference on KIRC only"
	@echo "  make run-kich        - Run inference on KICH only"
	@echo "  make run-kirp        - Run inference on KIRP only"
	@echo "  make run-cpu         - Run inference on CPU"
	@echo "  make plot-heatmap   - Generate heatmaps from results"
	@echo "  make plot-gif       - Generate similarity score GIF animation"
	@echo "  make clean           - Clean results directory"
	@echo "  make setup           - Create necessary directories"
	@echo ""
	@echo "Configuration:"
	@echo "  KIRC: $(H5_PATH_KIRC)"
	@echo "  KICH: $(H5_PATH_KICH)"
	@echo "  KIRP: $(H5_PATH_KIRP)"
	@echo "  OUTPUT_DIR: $(OUTPUT_DIR)"
	@echo "  DEVICE: $(DEVICE)"

# Install dependencies
.PHONY: install
install:
	@echo "Installing dependencies..."
	$(PYTHON) -m pip install -r requirements.txt

# Create necessary directories
.PHONY: setup
setup:
	@echo "Creating directories..."
	@if not exist data mkdir data
	@if not exist models mkdir models
	@if not exist results mkdir results

# Run inference on all tumor types
.PHONY: run-all
run-all:
	@echo "Running CONCH inference on all TCGA kidney tumor types..."
	@echo "  KIRC: $(H5_PATH_KIRC)"
	@echo "  KICH: $(H5_PATH_KICH)"
	@echo "  KIRP: $(H5_PATH_KIRP)"
	@echo "  Device: $(DEVICE)"
	$(PYTHON) run_inference.py --config $(CONFIG) --output_dir $(OUTPUT_DIR) --device $(DEVICE)

# Run inference on KIRC only
.PHONY: run-kirc
run-kirc:
	@echo "Running CONCH inference on KIRC..."
	$(PYTHON) run_inference.py --config $(CONFIG) --h5_path "$(H5_PATH_KIRC)" --output_dir $(OUTPUT_DIR)_kirc --device $(DEVICE)

# Run inference on KICH only
.PHONY: run-kich
run-kich:
	@echo "Running CONCH inference on KICH..."
	$(PYTHON) run_inference.py --config $(CONFIG) --h5_path "$(H5_PATH_KICH)" --output_dir $(OUTPUT_DIR)_kich --device $(DEVICE)

# Run inference on KIRP only
.PHONY: run-kirp
run-kirp:
	@echo "Running CONCH inference on KIRP..."
	$(PYTHON) run_inference.py --config $(CONFIG) --h5_path "$(H5_PATH_KIRP)" --output_dir $(OUTPUT_DIR)_kirp --device $(DEVICE)

# Run inference on CPU
.PHONY: run-cpu
run-cpu:
	@echo "Running CONCH inference on CPU..."
	$(PYTHON) run_inference.py --config $(CONFIG) --output_dir $(OUTPUT_DIR) --device cpu

# Generate heatmaps from inference results
.PHONY: plot-heatmap
plot-heatmap:
	@echo "Generating heatmaps from results..."
	@test -f results/predictions.csv || (echo "Error: results/predictions.csv not found. Run inference first!" && exit 1)
	$(PYTHON) plot_heatmap.py --results results/predictions.csv --output_dir results/heatmaps --config $(CONFIG)

# Generate similarity score GIF animation
.PHONY: plot-gif
plot-gif:
	@echo "Generating similarity score GIF..."
	@test -f results/predictions.csv || (echo "Error: results/predictions.csv not found. Run inference first!" && exit 1)
	$(PYTHON) create_similarity_gif.py --results results/predictions.csv --output results/similarity_animation.gif --config $(CONFIG)

# Clean results
.PHONY: clean
clean:
	@echo "Cleaning results directory..."
	@if exist results\*.csv del /q results\*.csv
	@echo "Done."

# Clean all generated files
.PHONY: distclean
distclean:
	@echo "Cleaning all generated files..."
	@if exist results rmdir /s /q results
	@if exist __pycache__ rmdir /s /q __pycache__
	@for /r %%i in (*__pycache__) do @rmdir /s /q "%%i"
	@echo "Done."

# Download CONCH model (placeholder - requires HuggingFace account)
.PHONY: download-model
download-model:
	@echo "To download the CONCH model:"
	@echo "1. Register at https://huggingface.co/MahmoodLab/CONCH"
	@echo "2. Download conch_pytorch_model.pt"
	@echo "3. Place it in the models/ directory"

# Show current configuration
.PHONY: info
info:
	@echo "Current Configuration:"
	@echo "  H5_PATH_KIRC=$(H5_PATH_KIRC)"
	@echo "  H5_PATH_KICH=$(H5_PATH_KICH)"
	@echo "  H5_PATH_KIRP=$(H5_PATH_KIRP)"
	@echo "  OUTPUT_DIR=$(OUTPUT_DIR)"
	@echo "  CONFIG=$(CONFIG)"
	@echo "  DEVICE=$(DEVICE)"
