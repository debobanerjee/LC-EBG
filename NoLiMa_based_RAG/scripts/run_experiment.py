# # scripts/run_experiment.py
# import argparse
# import yaml
# from ragwithtopk.experiments.config import ExperimentConfig
# from ragwithtopk.experiments.runner import ExperimentRunner

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--config", required=True)
#     args = ap.parse_args()

#     with open(args.config, "r", encoding="utf-8") as f:
#         d = yaml.safe_load(f)

#     cfg = ExperimentConfig(**d)
#     ExperimentRunner(cfg).run_all()

# if __name__ == "__main__":
#     main()

import argparse
import os
import yaml
from ragwithtopk.experiments.config import ExperimentConfig
from ragwithtopk.experiments.runner import ExperimentRunner

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)

    # Pull OpenAI key from environment if not provided
    if not d.get("openai_api_key"):
        d["openai_api_key"] = os.environ["OPENAI_API_KEY"]

    cfg = ExperimentConfig(**d)
    ExperimentRunner(cfg).run_all()

if __name__ == "__main__":
    main()