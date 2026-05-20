#!/usr/bin/env python3
"""
Interactive Manual Journal Review Tool
--------------------------------------
Displays journal entries one at a time for manual fraud detection.

Controls:
- ENTER: Accept entry (authentic)
- BACKSPACE/DELETE: Reject entry (fraudulent)
- SPACE: Mark as invalid
- LEFT ARROW: Go back and change previous rating
- q: Quit and save progress

Results are saved incrementally to avoid data loss.
"""

import pandas as pd
import os
import sys
import termios
import tty
from pathlib import Path
from datetime import datetime

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, 'input', 'journals.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'ratings')

# Terminal colors
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_single_key():
    """Get a single keypress from the user."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
        # Check for special keys (arrow keys, delete, etc.)
        if ch == '\x7f':  # Backspace/Delete
            return 'BACKSPACE'
        elif ch == '\r' or ch == '\n':  # Enter
            return 'ENTER'
        elif ch == ' ':  # Space
            return 'SPACE'
        elif ch == 'q' or ch == 'Q':
            return 'QUIT'
        elif ch == '\x1b':  # Escape sequence (arrow keys)
            next_chars = sys.stdin.read(2)
            if next_chars == '[D':  # Left arrow
                return 'LEFT'
            elif next_chars == '[C':  # Right arrow
                return 'RIGHT'
            elif next_chars == '[A':  # Up arrow
                return 'UP'
            elif next_chars == '[B':  # Down arrow
                return 'DOWN'
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def clear_screen():
    """Clear the terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')


def get_rater_id():
    """Prompt user for rater ID."""
    clear_screen()
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}MANUAL FRAUD REVIEW - RATER SETUP{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.ENDC}\n")
    
    while True:
        try:
            rater_id = input(f"{Colors.OKCYAN}Enter your rater ID (e.g., 0, 1, 2): {Colors.ENDC}").strip()
            if rater_id:
                return rater_id
            print(f"{Colors.FAIL}Please enter a valid rater ID.{Colors.ENDC}\n")
        except KeyboardInterrupt:
            print(f"\n{Colors.WARNING}Setup cancelled.{Colors.ENDC}")
            sys.exit(0)


def display_entry(row, idx, total, progress_stats):
    """Display a journal entry with formatting."""
    clear_screen()
    
    # Header
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}MANUAL FRAUD REVIEW{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.ENDC}\n")
    
    # Progress
    progress_pct = (idx / total) * 100
    print(f"{Colors.OKCYAN}Progress: {idx}/{total} ({progress_pct:.1f}%){Colors.ENDC}")
    print(f"{Colors.OKGREEN}Accepted: {progress_stats['accepted']}{Colors.ENDC} | "
          f"{Colors.FAIL}Rejected: {progress_stats['rejected']}{Colors.ENDC} | "
          f"{Colors.WARNING}Invalid: {progress_stats['invalid']}{Colors.ENDC}\n")
    
    # Entry metadata
    print(f"{Colors.BOLD}Entry #{idx}{Colors.ENDC}")
    print(f"Participant ID: {Colors.WARNING}{row['ParticipantID']}{Colors.ENDC}")
    print(f"Journal ID: {Colors.WARNING}{row['JournalUniqueID']}{Colors.ENDC}")
    print(f"Type: {Colors.OKBLUE}{row['Type']}{Colors.ENDC}")
    
    # Display submission date if available
    if 'Timestamp' in row and pd.notna(row['Timestamp']):
        # convert from unix ms to datetime
        timestamp = datetime.fromtimestamp(row['Timestamp'] / 1000)
        timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        print(f"Submitted: {Colors.OKCYAN}{timestamp}{Colors.ENDC}")
    
    # Determine text column
    text_col = 'JournalAnonymised' if 'JournalAnonymised' in row else 'JournalContent'
    content = row[text_col] if isinstance(row[text_col], str) else "[No content]"
    
    print(f"\n{Colors.BOLD}{Colors.UNDERLINE}Content:{Colors.ENDC}")
    print(f"\n{content}\n")
    
    # Instructions
    print(f"{Colors.BOLD}{'─'*80}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}[ENTER]{Colors.ENDC} Accept (authentic)  |  "
          f"{Colors.FAIL}[BACKSPACE]{Colors.ENDC} Reject (fraudulent)  |  "
          f"{Colors.WARNING}[SPACE]{Colors.ENDC} Invalid")
    print(f"{Colors.OKBLUE}[← LEFT]{Colors.ENDC} Go back  |  "
          f"{Colors.OKCYAN}[Q]{Colors.ENDC} Quit & Save")
    print(f"{Colors.BOLD}{'─'*80}{Colors.ENDC}\n")


def save_result(result, output_file):
    """Save a single result to CSV (append mode)."""
    df = pd.DataFrame([result])
    
    # Check if file exists
    if os.path.exists(output_file):
        # Append without header
        df.to_csv(output_file, mode='a', header=False, index=False, quoting=3, escapechar='\\')
    else:
        # Create new file with header
        df.to_csv(output_file, mode='w', header=True, index=False, quoting=3, escapechar='\\')


def load_progress(output_file):
    """Load previously reviewed entries to resume from checkpoint."""
    if not os.path.exists(output_file):
        return set(), {'accepted': 0, 'rejected': 0, 'invalid': 0}
    
    df = pd.read_csv(output_file, escapechar='\\')
    reviewed_ids = set(df['JournalUniqueID'].values)
    stats = {
        'accepted': (df['ManualReview'] == 'Authentic').sum(),
        'rejected': (df['ManualReview'] == 'Fraudulent').sum(),
        'invalid': (df['ManualReview'] == 'Invalid').sum()
    }
    return reviewed_ids, stats


def undo_last_entry(output_file):
    """Remove the last entry from the CSV file and return it."""
    if not os.path.exists(output_file):
        return None
    
    try:
        df = pd.read_csv(output_file, escapechar='\\')
    except Exception as e:
        print(f"{Colors.FAIL}Error reading CSV: {e}{Colors.ENDC}")
        return None
    
    if len(df) == 0:
        return None
    
    # Get the last entry
    last_entry = df.iloc[-1].to_dict()
    
    # Remove the last row and save
    df = df.iloc[:-1]
    if len(df) > 0:
        df.to_csv(output_file, index=False, quoting=3, escapechar='\\')
    else:
        # If empty, delete the file
        os.remove(output_file)
    
    return last_entry


def main():
    """Main interactive review loop."""
    # Get rater ID
    rater_id = get_rater_id()
    output_file = os.path.join(OUTPUT_DIR, f'rater_{rater_id}.csv')
    
    # Load data
    print(f"{Colors.OKCYAN}Loading journal entries...{Colors.ENDC}")
    journals = pd.read_csv(INPUT_FILE)
    
    # Filter out voice notes and non-text entries
    text_col = 'JournalAnonymised' if 'JournalAnonymised' in journals.columns else 'JournalContent'
    
    # Filter conditions
    mask = (
        (journals['Type'] != 'Voice') &  # Exclude voice notes
        (journals['Type'] != 'Summary') &  # Exclude summaries
        (journals[text_col].notna()) &  # Must have content
        (journals[text_col].apply(lambda x: isinstance(x, str)))  # Must be string
    )
    
    journals_to_review = journals[mask].copy()
    total_entries = len(journals_to_review)  # Total entries to review across all sessions
    
    print(f"{Colors.OKGREEN}✓ Loaded {total_entries} written entries for review{Colors.ENDC}")
    print(f"{Colors.WARNING}(Excluded Voice notes and Summaries){Colors.ENDC}\n")
    
    # Load progress (resume capability)
    reviewed_ids, progress_stats = load_progress(output_file)
    
    if reviewed_ids:
        print(f"{Colors.OKBLUE}✓ Resuming from previous session{Colors.ENDC}")
        print(f"  Previously reviewed: {len(reviewed_ids)} entries")
        print(f"  Accepted: {progress_stats['accepted']}")
        print(f"  Rejected: {progress_stats['rejected']}")
        print(f"  Invalid: {progress_stats['invalid']}\n")
        
        # Filter out already reviewed entries
        journals_to_review = journals_to_review[
            ~journals_to_review['JournalUniqueID'].isin(reviewed_ids)
        ]
    
    remaining = len(journals_to_review)
    already_reviewed = len(reviewed_ids)
    print(f"{Colors.BOLD}Remaining entries to review: {remaining}{Colors.ENDC}")
    print(f"\nPress any key to start...\n")
    get_single_key()
    
    # Review loop with history
    journals_list = list(journals_to_review.iterrows())
    idx = 0
    
    while idx < len(journals_list):
        _, row = journals_list[idx]
        cumulative_idx = already_reviewed + idx + 1  # Total progress across all sessions
        display_entry(row, cumulative_idx, total_entries, progress_stats)
        
        # Wait for user input
        while True:
            key = get_single_key()
            
            if key == 'ENTER':
                # Accept entry
                result = {
                    'ParticipantID': row['ParticipantID'],
                    'JournalUniqueID': row['JournalUniqueID'],
                    'ManualReview': 'Authentic',
                    'ReviewedAt': datetime.now().isoformat()
                }
                save_result(result, output_file)
                progress_stats['accepted'] += 1
                idx += 1
                break
            
            elif key == 'BACKSPACE':
                # Reject entry
                result = {
                    'ParticipantID': row['ParticipantID'],
                    'JournalUniqueID': row['JournalUniqueID'],
                    'ManualReview': 'Fraudulent',
                    'ReviewedAt': datetime.now().isoformat()
                }
                save_result(result, output_file)
                progress_stats['rejected'] += 1
                idx += 1
                break
            
            elif key == 'SPACE':
                # Mark as invalid
                result = {
                    'ParticipantID': row['ParticipantID'],
                    'JournalUniqueID': row['JournalUniqueID'],
                    'ManualReview': 'Invalid',
                    'ReviewedAt': datetime.now().isoformat()
                }
                save_result(result, output_file)
                progress_stats['invalid'] += 1
                idx += 1
                break
            
            elif key == 'LEFT':
                # Go back to previous entry
                if idx > 0:
                    undone = undo_last_entry(output_file)
                    if undone:
                        # Update stats
                        if undone['ManualReview'] == 'Authentic':
                            progress_stats['accepted'] -= 1
                        elif undone['ManualReview'] == 'Fraudulent':
                            progress_stats['rejected'] -= 1
                        elif undone['ManualReview'] == 'Invalid':
                            progress_stats['invalid'] -= 1
                        idx -= 1
                        break
            
            elif key == 'QUIT':
                # Save and exit
                clear_screen()
                print(f"\n{Colors.OKGREEN}✓ Progress saved!{Colors.ENDC}")
                print(f"\nReviewed so far:")
                print(f"  Total: {progress_stats['accepted'] + progress_stats['rejected'] + progress_stats['invalid']}")
                print(f"  Accepted: {progress_stats['accepted']}")
                print(f"  Rejected: {progress_stats['rejected']}")
                print(f"  Invalid: {progress_stats['invalid']}")
                print(f"\nResults saved to:")
                print(f"  {Colors.OKCYAN}{output_file}{Colors.ENDC}")
                print(f"\nRun this script again to resume from where you left off.\n")
                return
    
    # Completed all entries
    clear_screen()
    print(f"\n{Colors.BOLD}{Colors.OKGREEN}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKGREEN}✓ REVIEW COMPLETE!{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKGREEN}{'='*80}{Colors.ENDC}\n")
    
    total_reviewed = progress_stats['accepted'] + progress_stats['rejected'] + progress_stats['invalid']
    print(f"Total entries reviewed: {total_reviewed}")
    print(f"{Colors.OKGREEN}Accepted: {progress_stats['accepted']}{Colors.ENDC}")
    print(f"{Colors.FAIL}Rejected: {progress_stats['rejected']}{Colors.ENDC}")
    print(f"{Colors.WARNING}Invalid: {progress_stats['invalid']}{Colors.ENDC}")
    
    fraud_rate = (progress_stats['rejected'] / (progress_stats['accepted'] + progress_stats['rejected'])) * 100
    print(f"\nFraud rate: {Colors.WARNING}{fraud_rate:.1f}%{Colors.ENDC}")
    
    print(f"\nResults saved to:")
    print(f"  {Colors.OKCYAN}{output_file}{Colors.ENDC}\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        clear_screen()
        print(f"\n{Colors.WARNING}⚠ Interrupted by user{Colors.ENDC}")
        print(f"Progress has been saved. Run again to resume.\n")
        sys.exit(0)

