#!/usr/bin/env python3
"""
LIWC Processing Script

This script automates the daily processing of new entries through LIWC-22 CLI:
1. Re-triggers collate_data.py to get latest collated data
2. Identifies new entries not yet processed by LIWC
3. Processes new entries using LIWC-22 CLI
4. Exports results to CSV with incremental updates

"""

import pandas as pd
import numpy as np
import subprocess
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import logging
from typing import Dict, List, Tuple, Optional
import argparse
import shutil
import time
import platform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import start_liwc_server  # type: ignore[import-not-found]  # noqa: E402  (from ../tools)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def project_process_all_journals(
    cli_path: str = "LIWC-22-cli",
    start_server: bool = False,
    server_cli: str = "LIWC-22-license-server",
    server_wait: float = 5.0,
) -> None:
    """Process all journals via LIWC-22 CLI and write output/liwc/liwc.csv.

    If start_server is True, a LIWC license server process will be started first
    and terminated after processing.
    """
    root = Path(__file__).resolve().parent.parent
    input_path = root / "input" / "journals.csv"
    output_dir = root / "output" / "liwc"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "liwc.csv"

    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Ensure CLI exists
    cli_exec = shutil.which(cli_path)
    if not cli_exec:
        print(f"LIWC CLI not found: {cli_path}. Ensure it's in PATH or pass --cli /path/to/LIWC-22-cli", file=sys.stderr)
        sys.exit(1)

    server_proc = None
    if start_server:
        server_exec = shutil.which(server_cli)
        if not server_exec:
            print(f"LIWC license server CLI not found: {server_cli}. Pass --server-cli with full path or add to PATH.", file=sys.stderr)
            sys.exit(1)
        # Start the license server in background
        logger.info("Starting LIWC license server: %s", server_exec)
        server_proc = subprocess.Popen([server_exec], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(max(0.5, server_wait))

    df = pd.read_csv(input_path)
    # Choose text column
    text_col = None
    if "JournalAnonymised" in df.columns:
        text_col = "JournalAnonymised"
    elif "JournalContent" in df.columns:
        text_col = "JournalContent"
    else:
        print("No text column found (expected JournalAnonymised or JournalContent).", file=sys.stderr)
        sys.exit(1)

    if "JournalUniqueID" not in df.columns:
        print("Missing JournalUniqueID column.", file=sys.stderr)
        sys.exit(1)

    # Prepare LIWC input with UniqueID and Content
    liwc_input = pd.DataFrame({
        "UniqueID": df["JournalUniqueID"].astype(str),
        "Content": df[text_col].fillna("").astype(str),
    })
    # Keep only non-empty content
    liwc_input = liwc_input[liwc_input["Content"].str.strip() != ""].copy()
    if liwc_input.empty:
        print("No textual content to process.")
        pd.DataFrame({"JournalUniqueID": []}).to_csv(out_csv, index=False)
        return

    # Write temporary input/output for CLI
    fd_in, tmp_in = tempfile.mkstemp(suffix=".csv", prefix="liwc_input_")
    os.close(fd_in)
    liwc_input.to_csv(tmp_in, index=False)
    fd_out, tmp_out = tempfile.mkstemp(suffix=".csv", prefix="liwc_output_")
    os.close(fd_out)

    try:
        # Prepare CLI command
        cmd = [
            cli_exec,
            "--mode", "wc",
            "--input", str(tmp_in),
            "--output", str(tmp_out),
        ]

        # Try up to 3 times; on first failure, optionally start GUI app on macOS and retry
        attempts = 3
        tried_gui = False
        for attempt in range(1, attempts + 1):
            logger.info("Running LIWC (attempt %d/%d): %s", attempt, attempts, " ".join(cmd))
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                break

            out_txt = res.stdout or ""
            err_txt = res.stderr or ""
            unable_msg = "Unable to connect to LIWC-22 license server"

            # If connection failure, try starting GUI once on macOS then retry
            if (unable_msg in out_txt or unable_msg in err_txt) and platform.system() == "Darwin" and not tried_gui:
                try:
                    logger.info("Attempting to start LIWC GUI via 'open -a LIWC-22' and retrying...")
                    subprocess.run(["open", "-a", "LIWC-22"], check=False)
                    tried_gui = True
                    time.sleep(3.0)
                    continue
                except Exception:
                    pass

            # Otherwise wait briefly and retry
            time.sleep(2.0)

            if attempt == attempts:
                # Final failure: print diagnostics and exit
                # If server process died, surface its output
                server_out = server_err = ""
                if server_proc is not None and server_proc.poll() is not None:
                    try:
                        so, se = server_proc.communicate(timeout=0.2)
                        server_out = so.decode("utf-8", errors="ignore") if isinstance(so, (bytes, bytearray)) else (so or "")
                        server_err = se.decode("utf-8", errors="ignore") if isinstance(se, (bytes, bytearray)) else (se or "")
                    except Exception:
                        pass

                print(
                    (
                        "LIWC failed (code {code}).\nStdout:\n{out}\nStderr:\n{err}\n"
                        "Input: {inp}\nExpected output: {outp}\n"
                        "Server stdout (if any):\n{sout}\nServer stderr (if any):\n{serr}\n"
                    ).format(
                        code=res.returncode,
                        out=out_txt,
                        err=err_txt,
                        inp=tmp_in,
                        outp=tmp_out,
                        sout=server_out,
                        serr=server_err,
                    ),
                    file=sys.stderr,
                )
                sys.exit(1)

        # Read LIWC output and emit with JournalUniqueID
        liwc_out = pd.read_csv(tmp_out)
        if "Filename" in liwc_out.columns:
            liwc_out = liwc_out.rename(columns={"Filename": "JournalUniqueID"})
        elif "UniqueID" in liwc_out.columns:
            liwc_out = liwc_out.rename(columns={"UniqueID": "JournalUniqueID"})
        else:
            # If neither, attach back by index merge
            liwc_out.insert(0, "JournalUniqueID", liwc_input["UniqueID"].values[: len(liwc_out)])

        liwc_out.to_csv(out_csv, index=False)
        print(f"\u2713 Wrote LIWC results to {out_csv}")
    finally:
        # clean up temp files and stop server
        try:
            if 'tmp_in' in locals() and os.path.exists(tmp_in):
                os.unlink(tmp_in)
        except Exception:
            pass
        try:
            if 'tmp_out' in locals() and os.path.exists(tmp_out):
                os.unlink(tmp_out)
        except Exception:
            pass
        if server_proc is not None:
            try:
                server_proc.terminate()
                try:
                    server_proc.wait(timeout=5)
                except Exception:
                    server_proc.kill()
            except Exception:
                pass

class LIWCProcessor:
    """Main class for processing Unity Scheduler data through LIWC-22 CLI."""
    
    def __init__(self, base_path: str = "/data/jkuc/unity_scheduler"):
        self.base_path = Path(base_path)
        self.data_path = self.base_path / "data" / "bot"
        self.liwc_path = self.base_path / "liwc"
        self.collated_data_path = self.data_path / "processed" / "collated_data.csv"
        self.liwc_results_path = self.liwc_path / "liwc_results.csv"
        self.liwc_processed_ids_path = self.liwc_path / "processed_ids.txt"
        
        # Ensure LIWC directory exists
        self.liwc_path.mkdir(exist_ok=True)
        
        # LIWC-22 CLI settings
        self.liwc_cli_command = "LIWC-22-cli"
        self.liwc_mode = "wc"  # word count mode
        
    def run_collate_data_script(self) -> bool:
        """
        Re-trigger the collate_data.py script to get latest data.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Running collate_data.py script to update collated data...")
        
        try:
            collate_script_path = self.base_path / "monitoring" / "collate_data.py"
            
            # Run the collate script
            result = subprocess.run(
                [sys.executable, str(collate_script_path)],
                cwd=str(self.base_path),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                logger.info("Successfully updated collated data")
                logger.debug(f"Collate script output: {result.stdout}")
                return True
            else:
                logger.error(f"Error running collate script: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Collate script timed out after 5 minutes")
            return False
        except Exception as e:
            logger.error(f"Unexpected error running collate script: {e}")
            return False
    
    def load_collated_data(self) -> Optional[pd.DataFrame]:
        """
        Load the latest collated data.
        
        Returns:
            pd.DataFrame or None: The collated data or None if error
        """
        try:
            if not self.collated_data_path.exists():
                logger.error(f"Collated data file not found: {self.collated_data_path}")
                return None
                
            # Load with appropriate dtypes for question columns
            df_header = pd.read_csv(self.collated_data_path, nrows=0)
            question_cols = [col for col in df_header.columns if col.startswith(('ISS_', 'Journal_'))]
            
            dtype_dict = {col: 'str' for col in question_cols}
            
            df = pd.read_csv(self.collated_data_path, dtype=dtype_dict)
            logger.info(f"Loaded {len(df)} entries from collated data")
            
            # Replace 'nan' strings with actual NaN
            for col in question_cols:
                df[col] = df[col].replace('nan', np.nan)
                df[col] = df[col].replace('', np.nan)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading collated data: {e}")
            return None
    
    def load_processed_ids(self) -> set:
        """
        Load the set of already processed UniqueIDs.
        
        Returns:
            set: Set of processed UniqueIDs
        """
        processed_ids = set()
        
        if self.liwc_processed_ids_path.exists():
            try:
                with open(self.liwc_processed_ids_path, 'r') as f:
                    processed_ids = set(line.strip() for line in f if line.strip())
                logger.info(f"Loaded {len(processed_ids)} previously processed IDs")
            except Exception as e:
                logger.warning(f"Error loading processed IDs: {e}")
        
        return processed_ids
    
    def save_processed_ids(self, processed_ids: set) -> None:
        """
        Save the set of processed UniqueIDs.
        
        Args:
            processed_ids (set): Set of processed UniqueIDs
        """
        try:
            with open(self.liwc_processed_ids_path, 'w') as f:
                for uid in sorted(processed_ids):
                    f.write(f"{uid}\n")
            logger.info(f"Saved {len(processed_ids)} processed IDs")
        except Exception as e:
            logger.error(f"Error saving processed IDs: {e}")
    
    def identify_new_entries(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify entries that haven't been processed by LIWC yet.
        
        Args:
            df (pd.DataFrame): Collated data
            
        Returns:
            pd.DataFrame: New entries to process
        """
        processed_ids = self.load_processed_ids()
        
        # Filter to entries with content and not already processed
        new_entries = df[
            (df['Content'].notna()) & 
            (df['Content'].str.strip() != '') &
            (df['WordCount'] > 0) &
            (~df['UniqueID'].isin(processed_ids))
        ].copy()
        
        logger.info(f"Identified {len(new_entries)} new entries to process")
        logger.info(f"Entry types: {new_entries['DataType'].value_counts().to_dict()}")
        
        return new_entries
    
    def prepare_liwc_input_file(self, df: pd.DataFrame) -> str:
        """
        Prepare a CSV file for LIWC-22 CLI processing.
        
        Args:
            df (pd.DataFrame): Entries to process
            
        Returns:
            str: Path to the temporary input file
        """
        # Create temporary file for LIWC input
        temp_fd, temp_path = tempfile.mkstemp(suffix='.csv', prefix='liwc_input_')
        
        try:
            # Prepare data for LIWC - we need UniqueID and Content columns
            liwc_input = df[['UniqueID', 'Content']].copy()
            
            # Clean content for LIWC processing
            liwc_input['Content'] = liwc_input['Content'].astype(str)
            liwc_input['Content'] = liwc_input['Content'].str.strip()
            
            # Save to temporary file
            liwc_input.to_csv(temp_path, index=False)
            
            logger.info(f"Prepared LIWC input file with {len(liwc_input)} entries: {temp_path}")
            
            return temp_path
            
        except Exception as e:
            # Clean up the file descriptor
            os.close(temp_fd)
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            logger.error(f"Error preparing LIWC input file: {e}")
            raise
        finally:
            # Close the file descriptor if it's still open
            try:
                os.close(temp_fd)
            except OSError:
                pass  # Already closed
    
    def run_liwc_analysis(self, input_path: str, output_path: str) -> bool:
        """
        Run LIWC-22 CLI analysis on the input file.
        
        Args:
            input_path (str): Path to input CSV file
            output_path (str): Path to output CSV file
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Running LIWC-22 CLI analysis...")
        
        try:
            # Construct LIWC-22 CLI command
            cmd = [
                self.liwc_cli_command,
                "--mode", self.liwc_mode,
                "--input", input_path,
                "--output", output_path
            ]
            
            logger.info(f"Executing LIWC command: {' '.join(cmd)}")
            
            # Run LIWC analysis
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout for large datasets
            )
            
            if result.returncode == 0:
                logger.info("LIWC analysis completed successfully")
                logger.debug(f"LIWC output: {result.stdout}")
                return True
            else:
                logger.error(f"LIWC analysis failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("LIWC analysis timed out after 30 minutes")
            return False
        except FileNotFoundError:
            logger.error(f"LIWC-22 CLI not found. Please ensure '{self.liwc_cli_command}' is installed and in PATH")
            return False
        except Exception as e:
            logger.error(f"Unexpected error running LIWC analysis: {e}")
            return False
    
    def merge_liwc_results(self, new_entries: pd.DataFrame, liwc_output_path: str) -> pd.DataFrame:
        """
        Merge LIWC results with the original entry data.
        
        Args:
            new_entries (pd.DataFrame): Original entries that were processed
            liwc_output_path (str): Path to LIWC output file
            
        Returns:
            pd.DataFrame: Merged results
        """
        try:
            # Load LIWC results
            liwc_results = pd.read_csv(liwc_output_path)
            logger.info(f"Loaded LIWC results for {len(liwc_results)} entries")
            
            # Merge with original entry data
            merged = pd.merge(
                new_entries,
                liwc_results,
                left_on='UniqueID',
                right_on='Filename',  # LIWC uses Filename column for the ID
                how='inner'
            )
            
            # Drop the duplicate Filename column
            if 'Filename' in merged.columns:
                merged = merged.drop(columns=['Filename'])
            
            logger.info(f"Successfully merged LIWC results with {len(merged)} entries")
            
            return merged
            
        except Exception as e:
            logger.error(f"Error merging LIWC results: {e}")
            raise
    
    def save_liwc_results(self, results_df: pd.DataFrame, is_new_file: bool = False) -> None:
        """
        Save or append LIWC results to the main results file.
        
        Args:
            results_df (pd.DataFrame): Results to save
            is_new_file (bool): Whether to create a new file or append
        """
        try:
            if is_new_file or not self.liwc_results_path.exists():
                # Create new file
                results_df.to_csv(self.liwc_results_path, index=False)
                logger.info(f"Created new LIWC results file with {len(results_df)} entries")
            else:
                # Load existing results and append new ones
                existing_results = pd.read_csv(self.liwc_results_path)
                
                # Combine results (avoiding duplicates based on UniqueID)
                combined_results = pd.concat([existing_results, results_df], ignore_index=True)
                combined_results = combined_results.drop_duplicates(subset=['UniqueID'], keep='last')
                
                # Save combined results
                combined_results.to_csv(self.liwc_results_path, index=False)
                logger.info(f"Appended {len(results_df)} new entries to LIWC results (total: {len(combined_results)})")
                
        except Exception as e:
            logger.error(f"Error saving LIWC results: {e}")
            raise
    
    def update_processed_ids(self, processed_unique_ids: List[str]) -> None:
        """
        Update the list of processed UniqueIDs.
        
        Args:
            processed_unique_ids (List[str]): List of newly processed UniqueIDs
        """
        current_processed_ids = self.load_processed_ids()
        current_processed_ids.update(processed_unique_ids)
        self.save_processed_ids(current_processed_ids)
    
    def process_new_entries(self, force_reprocess: bool = False) -> Dict[str, any]:
        """
        Main method to process new entries through LIWC-22 CLI.
        
        Args:
            force_reprocess (bool): Whether to force reprocessing of all entries
            
        Returns:
            Dict: Processing results summary
        """
        start_time = datetime.now()
        results = {
            'success': False,
            'processed_count': 0,
            'error_message': None,
            'processing_time': None
        }
        
        try:
            # Step 1: Update collated data
            logger.info("=== Starting LIWC Processing Pipeline ===")
            
            if not self.run_collate_data_script():
                results['error_message'] = "Failed to update collated data"
                return results
            
            # Step 2: Load collated data
            collated_data = self.load_collated_data()
            if collated_data is None:
                results['error_message'] = "Failed to load collated data"
                return results
            
            # Step 3: Identify new entries
            if force_reprocess:
                logger.info("Force reprocessing mode: processing all entries with content")
                new_entries = collated_data[
                    (collated_data['Content'].notna()) & 
                    (collated_data['Content'].str.strip() != '') &
                    (collated_data['WordCount'] > 0)
                ].copy()
                # Reset processed IDs for force reprocessing
                self.save_processed_ids(set())
            else:
                new_entries = self.identify_new_entries(collated_data)
            
            if len(new_entries) == 0:
                logger.info("No new entries to process")
                results['success'] = True
                results['processed_count'] = 0
                return results
            
            # Step 4: Prepare LIWC input file
            temp_input_path = self.prepare_liwc_input_file(new_entries)
            temp_output_path = None
            
            try:
                # Step 5: Create temporary output file
                temp_fd, temp_output_path = tempfile.mkstemp(suffix='.csv', prefix='liwc_output_')
                os.close(temp_fd)  # Close file descriptor immediately
                
                # Step 6: Run LIWC analysis
                if not self.run_liwc_analysis(temp_input_path, temp_output_path):
                    results['error_message'] = "LIWC analysis failed"
                    return results
                
                # Step 7: Merge results
                merged_results = self.merge_liwc_results(new_entries, temp_output_path)
                
                # Step 8: Save results
                self.save_liwc_results(merged_results, is_new_file=force_reprocess)
                
                # Step 9: Update processed IDs
                processed_ids = merged_results['UniqueID'].tolist()
                self.update_processed_ids(processed_ids)
                
                results['success'] = True
                results['processed_count'] = len(processed_ids)
                
                logger.info(f"Successfully processed {len(processed_ids)} entries")
                
            finally:
                # Clean up temporary files
                if os.path.exists(temp_input_path):
                    os.unlink(temp_input_path)
                if temp_output_path and os.path.exists(temp_output_path):
                    os.unlink(temp_output_path)
            
        except Exception as e:
            logger.error(f"Error in processing pipeline: {e}")
            results['error_message'] = str(e)
        
        finally:
            results['processing_time'] = (datetime.now() - start_time).total_seconds()
            logger.info(f"Processing completed in {results['processing_time']:.2f} seconds")
        
        return results
    
    def get_processing_stats(self) -> Dict[str, any]:
        """
        Get statistics about the current processing state.
        
        Returns:
            Dict: Processing statistics
        """
        stats = {}
        
        try:
            # Count processed entries
            processed_ids = self.load_processed_ids()
            stats['total_processed'] = len(processed_ids)
            
            # Check if results file exists and get stats
            if self.liwc_results_path.exists():
                results_df = pd.read_csv(self.liwc_results_path)
                stats['results_file_entries'] = len(results_df)
                stats['results_file_size_mb'] = self.liwc_results_path.stat().st_size / (1024 * 1024)
            else:
                stats['results_file_entries'] = 0
                stats['results_file_size_mb'] = 0
            
            # Check collated data stats
            if self.collated_data_path.exists():
                collated_df = self.load_collated_data()
                if collated_df is not None:
                    total_with_content = len(collated_df[
                        (collated_df['Content'].notna()) & 
                        (collated_df['Content'].str.strip() != '') &
                        (collated_df['WordCount'] > 0)
                    ])
                    stats['total_entries_with_content'] = total_with_content
                    stats['unprocessed_entries'] = total_with_content - stats['total_processed']
            
        except Exception as e:
            logger.error(f"Error getting processing stats: {e}")
            stats['error'] = str(e)
        
        return stats


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Process Unity Scheduler entries through LIWC-22 CLI')
    parser.add_argument('--force-reprocess', action='store_true', 
                       help='Force reprocessing of all entries (creates new results file)')
    parser.add_argument('--stats-only', action='store_true', 
                       help='Only show processing statistics, do not process entries')
    parser.add_argument('--base-path', default='/data/jkuc/unity_scheduler',
                       help='Base path for Unity Scheduler data (default: /data/jkuc/unity_scheduler)')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = LIWCProcessor(base_path=args.base_path)
    
    if args.stats_only:
        # Show statistics only
        logger.info("=== LIWC Processing Statistics ===")
        stats = processor.get_processing_stats()
        
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        
        return
    
    # Process entries
    results = processor.process_new_entries(force_reprocess=args.force_reprocess)
    
    # Report results
    logger.info("=== Processing Results ===")
    logger.info(f"Success: {results['success']}")
    logger.info(f"Processed entries: {results['processed_count']}")
    
    if results['error_message']:
        logger.error(f"Error: {results['error_message']}")
        sys.exit(1)
    
    if results['processing_time']:
        logger.info(f"Processing time: {results['processing_time']:.2f} seconds")
    
    # Show updated statistics
    stats = processor.get_processing_stats()
    logger.info("=== Updated Statistics ===")
    for key, value in stats.items():
        if key != 'error':
            logger.info(f"{key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LIWC over project journals and write output/liwc/liwc.csv")
    parser.add_argument("--cli", default="LIWC-22-cli", help="Path or name of LIWC CLI executable")
    # Default: start the server unless explicitly disabled
    parser.add_argument("--no-start-server", action="store_false", dest="start_server", default=True, help="Do not start LIWC license server")
    parser.add_argument("--server-cli", default="LIWC-22-license-server", help="Path or name of LIWC license server CLI")
    parser.add_argument("--server-wait", type=float, default=2.0, help="Seconds to wait after starting server")
    args = parser.parse_args()
    # Prefer the shared starter script
    if args.start_server:
        try:
            start_liwc_server.start_liwc_server()
        except Exception:
            pass
    project_process_all_journals(
        cli_path=args.cli,
        start_server=False,
        server_cli=args.server_cli,
        server_wait=args.server_wait,
    )
