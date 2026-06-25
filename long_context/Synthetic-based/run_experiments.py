#!/usr/bin/env python3
"""
Command-line script to run context length experiments.

Usage:
    python run_experiments.py --model gpt-4o --characters 10000,20000,30000 --trials 100
    python run_experiments.py --model claude-sonnet-4 --preset mid --trials 50
    python run_experiments.py --resume experiment_outputs/gpt-4o_20231220_143052.jsonl
    python run_experiments.py --list-models
"""

import argparse
import os
import sys
from dotenv import load_dotenv

# Import shared experiment code
from experiments import (
    MODELS,
    PRESETS,
    ContextLengthExceeded,
    setup_clients,
    load_random_facts,
    get_output_filename,
    load_existing_results,
    successful_records,
    run_experiments,
)

# Load environment variables
load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Run context length experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_experiments.py --model gpt-4o --characters 10000,20000,30000
    python run_experiments.py --model claude-sonnet-4 --preset mid --trials 50
    python run_experiments.py --model gemini-2.5-pro --preset short --parallel 5
    python run_experiments.py --resume experiment_outputs/gpt-4o_20231220_143052.jsonl
    python run_experiments.py --list-models
    python run_experiments.py --list-presets
        """
    )
    
    parser.add_argument("--model", "-m", type=str, help="Model to use (see --list-models)")
    parser.add_argument("--characters", "-c", type=str, help="Comma-separated list of context lengths")
    parser.add_argument("--preset", "-p", type=str, choices=list(PRESETS.keys()), 
                        help="Use a preset list of context lengths")
    parser.add_argument("--trials", "-n", type=int, default=100, help="Number of trials per context length (default: 100)")
    parser.add_argument("--parallel", type=int, default=3, help="Max parallel API calls (default: 3)")
    parser.add_argument("--facts-file", type=str, default="random_facts.txt", help="Path to random facts file")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--list-presets", action="store_true", help="List available presets")
    parser.add_argument("--no-save", action="store_true", help="Don't save outputs to disk")
    parser.add_argument("--resume", "-r", type=str, help="Resume from an existing results file (appends new results)")
    parser.add_argument("--answer-only", action="store_true",
                        help="Run the answer-only baseline (no citation requirement). "
                             "Outputs go to a file suffixed '_answer_only' so they don't "
                             "collide with the answer+evidence runs.")

    args = parser.parse_args()
    
    if args.list_models:
        print("Available models:")
        for name, model in MODELS.items():
            print(f"  {name:20} -> {model.api_name} ({model.provider.value})")
        return
    
    if args.list_presets:
        print("Available presets:")
        for name, lengths in PRESETS.items():
            print(f"  {name:10} -> {lengths}")
        return
    
    # Handle resume mode
    existing_results = {}
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"Error: Resume file not found: {args.resume}")
            sys.exit(1)
        
        print(f"Loading existing results from {args.resume}...")
        existing_results = load_existing_results(args.resume)
        
        # Extract model from existing results if not specified
        if not args.model:
            for records in existing_results.values():
                if records:
                    inferred_model_name = records[0].get("model")
                    if inferred_model_name:
                        for key, m in MODELS.items():
                            if m.api_name == inferred_model_name:
                                args.model = key
                                print(f"Inferred model from results: {args.model}")
                                break
                    break
        
        # Extract character lengths from existing results if not specified
        if not args.characters and not args.preset:
            args.characters = ",".join(str(k) for k in sorted(existing_results.keys()))
            print(f"Inferred character lengths from results: {args.characters}")
        
        total_lines = sum(len(records) for records in existing_results.values())
        total_success = sum(
            len(successful_records(records)) for records in existing_results.values()
        )
        print(
            f"Found {total_lines} lines in file ({total_success} successful API calls) "
            f"across {len(existing_results)} context lengths"
        )
        for length in sorted(existing_results.keys()):
            recs = existing_results[length]
            ns = len(successful_records(recs))
            print(f"  {length:,} chars: {ns} successful / {len(recs)} lines")
    
    if not args.model:
        parser.error("--model is required (use --list-models to see options)")
    
    if args.model not in MODELS:
        print(f"Error: Unknown model '{args.model}'")
        print("Use --list-models to see available models")
        sys.exit(1)
    
    model = MODELS[args.model]
    
    # Determine context lengths
    if args.characters:
        character_lengths = [int(x.strip()) for x in args.characters.split(",")]
    elif args.preset:
        character_lengths = PRESETS[args.preset]
    else:
        parser.error("Either --characters or --preset is required")
    
    # Load random facts
    print(f"Loading random facts from {args.facts_file}...")
    if not os.path.exists(args.facts_file):
        print(f"Error: Facts file not found: {args.facts_file}")
        sys.exit(1)
    random_facts = load_random_facts(args.facts_file)
    print(f"Loaded {len(random_facts)} random facts")
    
    # Setup API clients
    print("Setting up API clients...")
    clients = setup_clients()
    if model.provider.value not in clients:
        print(f"Error: {model.provider.value} client not available. Check your API key.")
        sys.exit(1)
    
    # Setup output file
    output_file = None
    if not args.no_save:
        if args.resume:
            output_file = args.resume
            print(f"Appending outputs to: {output_file}")
        else:
            base_name = model.name + ("_answer_only" if args.answer_only else "")
            output_file = get_output_filename(base_name)
            print(f"Saving outputs to: {output_file}")
    
    # Run experiments
    print()
    try:
        run_experiments(
            model=model,
            character_lengths=character_lengths,
            random_facts=random_facts,
            clients=clients,
            trials=args.trials,
            max_parallel=args.parallel,
            output_file=output_file,
            existing_results=existing_results,
            verbose=True,
            answer_only=args.answer_only,
        )
    except ContextLengthExceeded:
        if output_file:
            print(f"\nStopped early (context limit). Partial outputs: {output_file}")
        sys.exit(1)
    
    if output_file:
        print(f"\nExperiment outputs saved to: {output_file}")


if __name__ == "__main__":
    main()
