"""Generate SHAP explanation artifacts for the trained weather model.

This script produces global and local SHAP explanation outputs that can be
visualised in the Streamlit dashboard. Run it after training the model:

    python scripts/generate_explanations.py

Optional arguments:

    --model-path       Path to the trained sklearn Pipeline (default: models/best_model.joblib)
    --output-dir       Directory for explanation artifacts (default: reports/explanations)
    --background-size  Number of background samples for SHAP (default: 100)
    --max-display      Max features shown in plots (default: 20)
    --cleaned-data     Path to the cleaned dataset (default: data/processed/cleaned_weather_dataset.csv)
    --local-only       Only generate local explanation for a single sample
    --global-only      Only generate global explanations
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.explainability import (
    DEFAULT_BACKGROUND_SIZE,
    DEFAULT_EXPLANATIONS_DIR,
    DEFAULT_MAX_DISPLAY,
    DEFAULT_MODEL_PATH,
    generate_all_explanations,
    generate_global_explanations,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate SHAP explanation artifacts for the weather model.")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to the trained model pipeline")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_EXPLANATIONS_DIR), help="Directory for explanation outputs")
    parser.add_argument("--background-size", type=int, default=DEFAULT_BACKGROUND_SIZE, help="Number of background samples for SHAP")
    parser.add_argument("--max-display", type=int, default=DEFAULT_MAX_DISPLAY, help="Maximum features to display in plots")
    parser.add_argument("--cleaned-data", type=str, default="data/processed/cleaned_weather_dataset.csv", help="Path to cleaned dataset")
    parser.add_argument("--local-only", action="store_true", help="Only generate local explanations")
    parser.add_argument("--global-only", action="store_true", help="Only generate global explanations")
    return parser.parse_args()


def main() -> None:
    """Run the explanation generation workflow."""
    args = parse_args()

    if args.local_only:
        logger.info("Generating local explanations only")
        from src.explainability import explain_prediction
        import pandas as pd

        # Load a sample row from the cleaned dataset to explain
        cleaned_df = pd.read_csv(args.cleaned_data)
        sample_row = cleaned_df.iloc[[0]]
        result = explain_prediction(
            sample_row,
            model_path=args.model_path,
            output_dir=args.output_dir,
            background_size=args.background_size,
            max_display=args.max_display,
        )
        logger.info("Prediction: %s (probability=%.4f)", result["prediction_label"], result["probability"])
        logger.info("Artifacts saved to %s", args.output_dir)
        return

    if args.global_only:
        logger.info("Generating global explanations only")
        artifacts = generate_global_explanations(
            model_path=args.model_path,
            output_dir=args.output_dir,
            background_size=args.background_size,
            max_display=args.max_display,
            cleaned_data_path=args.cleaned_data,
        )
        for name, path in artifacts.items():
            logger.info("  %s: %s", name, path)
        return

    logger.info("Generating all explanations")
    result = generate_all_explanations(
        model_path=args.model_path,
        output_dir=args.output_dir,
        background_size=args.background_size,
        max_display=args.max_display,
        cleaned_data_path=args.cleaned_data,
    )
    logger.info("Global artifacts:")
    for name, path in result["global"].items():
        logger.info("  %s: %s", name, path)
    logger.info("Local artifacts:")
    for name, path in result["local"].items():
        logger.info("  %s: %s", name, path)


if __name__ == "__main__":
    main()