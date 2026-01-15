# FakeNewsStyle Copilot Instructions

## Project Overview
FakeNewsStyle is a research framework for fake news detection in Spanish, emphasizing multi-feature fusion architectures. It integrates semantic, stylistic, emotional, readability, and domain features for robust classification.

## Architecture
- **Modular Pipeline**: Feature extraction → fusion → classification (see `docs/architecture.md`)
- **Key Components**: Semantic (BETO/RoBERTa-es), Style (stylometry), Emotion (lexicons), Readability (indices), Domain (memory/clustering)
- **Data Flow**: Raw XLSX → Processed PKL → Feature extraction → Fusion → Prediction

## Development Workflow
- **Data Preparation**: Use `main.py --prepare_corpus 1` or scripts in `scripts/` to convert XLSX to PKL
- **Entry Point**: `main.py` controls corpus preparation; future experiments will extend this
- **Path Handling**: Always use `pathlib.Path` for cross-platform compatibility (e.g., `BASE_DIR = Path(__file__).resolve().parent`)

## Code Patterns
- **Imports**: Project setup with `sys.path.insert(0, str(BASE_DIR))` for relative imports
- **CLI**: `argparse` for command-line flags (e.g., `--prepare_corpus` as int flag)
- **Data Conversion**: `convert_folder_xlsx_to_pkl()` in `src/data/convert_xlsx_to_pkl.py` - converts XLSX to PKL using pandas
- **Error Handling**: Raise `FileNotFoundError` for missing files (e.g., no XLSX in raw dir)

## Dependencies
- Minimal setup: numpy, pandas, scipy, openpyxl (see `requirements.txt`)
- Future: ML libraries for feature extraction and classification

## Conventions
- **Language**: Spanish comments and docs, English code
- **Structure**: `src/` for modules, `scripts/` for utilities, `data/raw` and `data/processed` separation
- **Reproducibility**: Modular design for ablation studies and experiments

## Key Files
- `docs/architecture.md`: Detailed pipeline diagram and explanation
- `src/data/convert_xlsx_to_pkl.py`: Core data conversion utility
- `scripts/all_xlsx_to_pkl.py`: CLI wrapper for data prep