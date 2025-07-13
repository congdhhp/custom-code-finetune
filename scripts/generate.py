#!/usr/bin/env python3
"""
Code generation script for SWTBot Fine-tuning Pipeline.
Generates SWTBot test code using trained models.
"""

import argparse
import logging
import sys
import json
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.logging import setup_logging
from inference.generator import SWTBotCodeGenerator
from inference.evaluator import CodeEvaluator


def load_prompts_from_file(file_path: str) -> list:
    """Load prompts from a file."""
    prompts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith('.json'):
            data = json.load(f)
            if isinstance(data, list):
                prompts = data
            else:
                prompts = [data.get('prompt', str(data))]
        else:
            # Treat as text file with one prompt per line
            prompts = [line.strip() for line in f if line.strip()]
    
    return prompts


def save_results(results: list, output_file: str):
    """Save generation results to file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_file.endswith('.json'):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        # Save as text file
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, result in enumerate(results):
                f.write(f"=== Generation {i+1} ===\n")
                f.write(f"Prompt: {result['prompt']}\n")
                f.write(f"Score: {result.get('evaluation', {}).get('overall_score', 'N/A'):.2f}\n")
                f.write("Generated Code:\n")
                f.write(result['generated_code'])
                f.write("\n\n")


def main():
    """Main generation function."""
    parser = argparse.ArgumentParser(description="Generate SWTBot test code")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the trained model"
    )
    parser.add_argument(
        "--prompt",
        help="Single prompt for code generation"
    )
    parser.add_argument(
        "--prompts-file",
        help="File containing multiple prompts"
    )
    parser.add_argument(
        "--output",
        default="generated_code.txt",
        help="Output file for generated code"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to generate per prompt"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling parameter"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate generated code quality"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode for continuous generation"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting SWTBot code generation")
    
    try:
        # Load model
        logger.info(f"Loading model from {args.model_path}")
        
        generation_config = {
            "generation": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty,
            }
        }
        
        generator = SWTBotCodeGenerator(args.model_path, generation_config)
        
        # Setup evaluator if needed
        evaluator = None
        if args.evaluate:
            evaluator = CodeEvaluator()
            logger.info("Code evaluator initialized")
        
        # Get prompts
        prompts = []
        
        if args.interactive:
            # Interactive mode
            logger.info("Entering interactive mode. Type 'quit' to exit.")
            while True:
                try:
                    prompt = input("\nEnter prompt: ").strip()
                    if prompt.lower() in ['quit', 'exit', 'q']:
                        break
                    if not prompt:
                        continue
                    
                    prompts = [prompt]
                    
                    # Generate code
                    logger.info("Generating code...")
                    if args.num_samples > 1:
                        generated_codes = generator.generate_multiple(
                            prompt, 
                            args.num_samples,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_k=args.top_k,
                            top_p=args.top_p,
                            repetition_penalty=args.repetition_penalty
                        )
                    else:
                        generated_codes = [generator.generate_code(
                            prompt,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_k=args.top_k,
                            top_p=args.top_p,
                            repetition_penalty=args.repetition_penalty
                        )]
                    
                    # Display results
                    for i, code in enumerate(generated_codes):
                        print(f"\n--- Sample {i+1} ---")
                        print(code)
                        
                        if evaluator:
                            eval_result = evaluator.evaluate_code(code, prompt)
                            print(f"\nQuality Score: {eval_result['overall_score']:.2f}")
                            if eval_result['issues']:
                                print("Issues:", ", ".join(eval_result['issues']))
                
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Error in interactive mode: {e}")
            
            logger.info("Exiting interactive mode")
            return
        
        elif args.prompt:
            prompts = [args.prompt]
        elif args.prompts_file:
            prompts = load_prompts_from_file(args.prompts_file)
            logger.info(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
        else:
            logger.error("Either --prompt, --prompts-file, or --interactive must be specified")
            sys.exit(1)
        
        # Generate code for all prompts
        all_results = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            # Generate code
            if args.num_samples > 1:
                generated_codes = generator.generate_multiple(
                    prompt, 
                    args.num_samples,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty
                )
            else:
                generated_codes = [generator.generate_code(
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty
                )]
            
            # Process each generated sample
            for j, code in enumerate(generated_codes):
                result = {
                    "prompt_id": i,
                    "sample_id": j,
                    "prompt": prompt,
                    "generated_code": code,
                    "generation_params": {
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": args.temperature,
                        "top_k": args.top_k,
                        "top_p": args.top_p,
                        "repetition_penalty": args.repetition_penalty
                    }
                }
                
                # Evaluate if requested
                if evaluator:
                    evaluation = evaluator.evaluate_code(code, prompt)
                    result["evaluation"] = evaluation
                    logger.info(f"Generated code quality score: {evaluation['overall_score']:.2f}")
                
                all_results.append(result)
        
        # Save results
        logger.info(f"Saving {len(all_results)} results to {args.output}")
        save_results(all_results, args.output)
        
        # Print summary
        logger.info("Generation completed successfully!")
        logger.info(f"Total prompts processed: {len(prompts)}")
        logger.info(f"Total samples generated: {len(all_results)}")
        
        if evaluator:
            scores = [r.get("evaluation", {}).get("overall_score", 0) for r in all_results]
            if scores:
                avg_score = sum(scores) / len(scores)
                max_score = max(scores)
                min_score = min(scores)
                logger.info(f"Quality scores - Avg: {avg_score:.2f}, Max: {max_score:.2f}, Min: {min_score:.2f}")
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
