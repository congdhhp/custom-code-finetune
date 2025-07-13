#!/usr/bin/env python3
"""
Data collection script for SWTBot Fine-tuning Pipeline.
Collects and processes Java code from SWTBot repositories.
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.collector import DataCollector
from data.processor import JavaCodeProcessor


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/data_collection.log')
        ]
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    """Main data collection function."""
    parser = argparse.ArgumentParser(description="Collect SWTBot training data")
    parser.add_argument(
        "--config", 
        default="configs/data_config.yaml",
        help="Path to data configuration file"
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory for collected data"
    )
    parser.add_argument(
        "--skip-collection",
        action="store_true",
        help="Skip data collection, only process existing files"
    )
    parser.add_argument(
        "--skip-processing",
        action="store_true", 
        help="Skip processing, only collect raw data"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    Path("logs").mkdir(exist_ok=True)
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting SWTBot data collection pipeline")
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
        
        # Setup directories
        output_dir = Path(args.output_dir)
        raw_dir = output_dir / "raw"
        processed_dir = output_dir / "processed"
        
        # Step 1: Data Collection
        if not args.skip_collection:
            logger.info("Step 1: Collecting data from repositories")
            collector = DataCollector(config, str(raw_dir))
            collection_stats = collector.collect_all_repositories()
            
            logger.info("Collection completed:")
            logger.info(f"  Repositories processed: {collection_stats['repositories_processed']}")
            logger.info(f"  Files collected: {collection_stats['files_collected']}")
            logger.info(f"  Total size: {collection_stats['total_size_mb']:.2f} MB")
        else:
            logger.info("Skipping data collection")
            collector = DataCollector(config, str(raw_dir))
        
        # Step 2: Data Processing
        if not args.skip_processing:
            logger.info("Step 2: Processing collected Java files")
            
            # Get collected files
            collected_files = collector.get_collected_files()
            java_files = [f for f in collected_files if f.suffix.lower() == '.java']
            
            logger.info(f"Found {len(java_files)} Java files to process")
            
            # Process files
            processor = JavaCodeProcessor(config)
            processing_stats = processor.process_files(java_files, str(processed_dir))
            
            logger.info("Processing completed:")
            logger.info(f"  Files processed: {processing_stats['files_processed']}")
            logger.info(f"  Files accepted: {processing_stats['files_accepted']}")
            logger.info(f"  Files rejected: {processing_stats['files_rejected']}")
            logger.info(f"  Total lines: {processing_stats['total_lines']}")
            logger.info(f"  Total characters: {processing_stats['total_characters']}")
            
            # Show rejection reasons
            if processing_stats['rejection_reasons']:
                logger.info("Rejection reasons:")
                for reason, count in processing_stats['rejection_reasons'].items():
                    logger.info(f"  {reason}: {count}")
        else:
            logger.info("Skipping data processing")
        
        # Step 3: Generate summary report
        logger.info("Step 3: Generating summary report")
        generate_summary_report(output_dir)
        
        logger.info("Data collection pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in data collection pipeline: {e}")
        sys.exit(1)


def generate_summary_report(output_dir: Path):
    """Generate a summary report of the data collection process."""
    report = {
        "collection_summary": {},
        "processing_summary": {},
        "file_statistics": {}
    }
    
    # Load collection metadata
    collection_metadata_file = output_dir / "raw" / "collection_metadata.json"
    if collection_metadata_file.exists():
        with open(collection_metadata_file, 'r') as f:
            report["collection_summary"] = json.load(f)
    
    # Load processing statistics
    processing_stats_file = output_dir / "processed" / "processing_stats.json"
    if processing_stats_file.exists():
        with open(processing_stats_file, 'r') as f:
            report["processing_summary"] = json.load(f)
    
    # Calculate file statistics
    processed_files_file = output_dir / "processed" / "processed_java_files.jsonl"
    if processed_files_file.exists():
        file_stats = {
            "total_files": 0,
            "total_lines": 0,
            "total_characters": 0,
            "avg_lines_per_file": 0,
            "avg_chars_per_file": 0,
            "swtbot_keyword_distribution": {}
        }
        
        with open(processed_files_file, 'r') as f:
            for line in f:
                file_data = json.loads(line)
                file_stats["total_files"] += 1
                file_stats["total_lines"] += file_data["metadata"]["line_count"]
                file_stats["total_characters"] += file_data["metadata"]["char_count"]
                
                # Count SWTBot keywords
                for keyword in file_data["metadata"]["swtbot_keywords"]:
                    file_stats["swtbot_keyword_distribution"][keyword] = \
                        file_stats["swtbot_keyword_distribution"].get(keyword, 0) + 1
        
        if file_stats["total_files"] > 0:
            file_stats["avg_lines_per_file"] = file_stats["total_lines"] / file_stats["total_files"]
            file_stats["avg_chars_per_file"] = file_stats["total_characters"] / file_stats["total_files"]
        
        report["file_statistics"] = file_stats
    
    # Save report
    report_file = output_dir / "data_collection_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nSummary report saved to: {report_file}")
    print("\nData Collection Summary:")
    print("=" * 50)
    
    if "collection_summary" in report and report["collection_summary"]:
        cs = report["collection_summary"]
        print(f"Repositories processed: {cs.get('repositories_processed', 0)}")
        print(f"Files collected: {cs.get('files_collected', 0)}")
        print(f"Total size: {cs.get('total_size_mb', 0):.2f} MB")
    
    if "file_statistics" in report and report["file_statistics"]:
        fs = report["file_statistics"]
        print(f"Final processed files: {fs.get('total_files', 0)}")
        print(f"Total lines of code: {fs.get('total_lines', 0):,}")
        print(f"Average lines per file: {fs.get('avg_lines_per_file', 0):.1f}")
        
        # Show top SWTBot keywords
        keyword_dist = fs.get("swtbot_keyword_distribution", {})
        if keyword_dist:
            print("\nTop SWTBot keywords:")
            sorted_keywords = sorted(keyword_dist.items(), key=lambda x: x[1], reverse=True)
            for keyword, count in sorted_keywords[:10]:
                print(f"  {keyword}: {count}")


if __name__ == "__main__":
    main()
