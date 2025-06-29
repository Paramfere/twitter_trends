#!/usr/bin/env python3

"""
Browse and manage organized fetch session folders.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

def list_sessions():
    """List all available fetch sessions."""
    data_dir = Path("data")
    if not data_dir.exists():
        print("No data directory found.")
        return
    
    session_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("fetch_session_")]
    
    if not session_dirs:
        print("No fetch sessions found.")
        return
    
    # Sort by creation time (newest first)
    session_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    print("📁 AVAILABLE FETCH SESSIONS:")
    print("=" * 60)
    
    for i, session_dir in enumerate(session_dirs):
        # Extract timestamp from folder name
        timestamp_str = session_dir.name.replace("fetch_session_", "")
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
            formatted_time = timestamp.strftime("%Y-%m-%d at %H:%M:%S")
        except ValueError:
            formatted_time = timestamp_str
        
        # Count files in each subdirectory
        raw_count = len(list((session_dir / "raw_data").glob("*"))) if (session_dir / "raw_data").exists() else 0
        analysis_count = len(list((session_dir / "analysis").glob("*"))) if (session_dir / "analysis").exists() else 0
        reports_count = len(list((session_dir / "reports").glob("*"))) if (session_dir / "reports").exists() else 0
        
        print(f"{i+1:2d}. {formatted_time}")
        print(f"    📂 {session_dir.name}")
        print(f"    📄 Files: {raw_count} raw, {analysis_count} analysis, {reports_count} reports")
        print()

def show_session_details(session_name: str):
    """Show detailed information about a specific session."""
    data_dir = Path("data")
    session_dir = data_dir / session_name
    
    if not session_dir.exists():
        print(f"Session '{session_name}' not found.")
        return
    
    print(f"📁 SESSION DETAILS: {session_name}")
    print("=" * 60)
    
    # Show folder structure with file details
    subdirs = ["raw_data", "analysis", "reports"]
    
    for subdir_name in subdirs:
        subdir = session_dir / subdir_name
        if subdir.exists():
            files = list(subdir.glob("*"))
            print(f"📂 {subdir_name}/ ({len(files)} files)")
            
            for file in sorted(files):
                size_kb = file.stat().st_size / 1024
                modified = datetime.fromtimestamp(file.stat().st_mtime)
                print(f"   📄 {file.name}")
                print(f"      Size: {size_kb:.1f} KB")
                print(f"      Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}")
                print()
        else:
            print(f"📂 {subdir_name}/ (empty)")
    
    print("🎯 QUICK ACCESS COMMANDS:")
    print(f"   View raw data:    cat '{session_dir}/raw_data/'*.csv")
    print(f"   View analysis:    cat '{session_dir}/analysis/'*.csv")
    print(f"   View reports:     cat '{session_dir}/reports/'*.txt")
    print(f"   View marketing:   cat '{session_dir}/reports/'*.md")

def clean_old_sessions(days: int = 7):
    """Clean up sessions older than specified days."""
    data_dir = Path("data")
    if not data_dir.exists():
        print("No data directory found.")
        return
    
    session_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("fetch_session_")]
    
    if not session_dirs:
        print("No fetch sessions found.")
        return
    
    cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
    old_sessions = [d for d in session_dirs if d.stat().st_mtime < cutoff_time]
    
    if not old_sessions:
        print(f"No sessions older than {days} days found.")
        return
    
    print(f"🗑️  Found {len(old_sessions)} sessions older than {days} days:")
    for session in old_sessions:
        timestamp_str = session.name.replace("fetch_session_", "")
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
            formatted_time = timestamp.strftime("%Y-%m-%d at %H:%M:%S")
        except ValueError:
            formatted_time = timestamp_str
        print(f"   📂 {formatted_time}")
    
    confirm = input(f"\nDelete these {len(old_sessions)} sessions? (y/N): ")
    if confirm.lower() == 'y':
        import shutil
        for session in old_sessions:
            shutil.rmtree(session)
            print(f"   ✅ Deleted {session.name}")
        print(f"🗑️  Cleaned up {len(old_sessions)} old sessions.")
    else:
        print("❌ Cleanup cancelled.")

def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description="Browse and manage fetch session folders")
    parser.add_argument("--list", "-l", action="store_true", help="List all sessions")
    parser.add_argument("--show", "-s", type=str, help="Show details for a specific session")
    parser.add_argument("--clean", "-c", type=int, nargs="?", const=7, help="Clean sessions older than N days (default: 7)")
    
    args = parser.parse_args()
    
    if args.list:
        list_sessions()
    elif args.show:
        show_session_details(args.show)
    elif args.clean is not None:
        clean_old_sessions(args.clean)
    else:
        # Default: list sessions
        list_sessions()

if __name__ == "__main__":
    main() 