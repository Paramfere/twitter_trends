#!/usr/bin/env python3

"""
Session Manager - Handle clean session numbering and folder organization.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SessionManager:
    """Manages session numbering and folder organization."""
    
    def __init__(self):
        """Initialize the session manager."""
        self.data_dir = Path("data")
        self.state_file = self.data_dir / "session_state.json"
        self.data_dir.mkdir(exist_ok=True)
    
    def load_session_state(self) -> Dict:
        """Load session state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load session state: {e}")
        
        return {
            'next_session_number': 1,
            'sessions': {},
            'created_at': datetime.now().isoformat()
        }
    
    def save_session_state(self, state: Dict):
        """Save session state to file."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save session state: {e}")
    
    def create_new_session(self) -> Tuple[str, Path]:
        """Create a new session with clean numbering."""
        state = self.load_session_state()
        
        session_number = state['next_session_number']
        timestamp = datetime.now()
        
        # Create clean session folder name
        session_name = f"session_{session_number:03d}"
        session_dir = self.data_dir / session_name
        
        # Create session directory structure
        session_dir.mkdir(exist_ok=True)
        (session_dir / "raw_data").mkdir(exist_ok=True)
        (session_dir / "analysis").mkdir(exist_ok=True)
        (session_dir / "reports").mkdir(exist_ok=True)
        
        # Update state
        state['sessions'][session_name] = {
            'session_number': session_number,
            'timestamp': timestamp.isoformat(),
            'created_at': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'session_dir': str(session_dir)
        }
        state['next_session_number'] = session_number + 1
        
        self.save_session_state(state)
        
        logger.info(f"Created new session: {session_name}")
        return session_name, session_dir
    
    def get_latest_session(self) -> Optional[Tuple[str, Path]]:
        """Get the most recent session."""
        state = self.load_session_state()
        
        if not state['sessions']:
            return None
        
        # Get the session with highest number
        latest_session_name = max(state['sessions'].keys(), 
                                key=lambda x: state['sessions'][x]['session_number'])
        latest_session_dir = Path(state['sessions'][latest_session_name]['session_dir'])
        
        return latest_session_name, latest_session_dir
    
    def list_sessions(self) -> Dict:
        """List all sessions with metadata."""
        state = self.load_session_state()
        return state['sessions']
    
    def cleanup_old_sessions(self, keep_count: int = 20) -> int:
        """Clean up old sessions, keeping only the most recent ones."""
        state = self.load_session_state()
        sessions = state['sessions']
        
        if len(sessions) <= keep_count:
            return 0
        
        # Sort sessions by number and keep only the most recent
        sorted_sessions = sorted(sessions.items(), 
                               key=lambda x: x[1]['session_number'], 
                               reverse=True)
        
        sessions_to_keep = dict(sorted_sessions[:keep_count])
        sessions_to_remove = dict(sorted_sessions[keep_count:])
        
        cleaned_count = 0
        for session_name, session_info in sessions_to_remove.items():
            session_dir = Path(session_info['session_dir'])
            if session_dir.exists():
                import shutil
                shutil.rmtree(session_dir)
                cleaned_count += 1
                logger.info(f"🗑️ Cleaned up old session: {session_name}")
        
        # Update state
        state['sessions'] = sessions_to_keep
        self.save_session_state(state)
        
        if cleaned_count > 0:
            logger.info(f"🧹 Cleaned up {cleaned_count} old sessions")
        
        return cleaned_count
    
    def migrate_old_sessions(self) -> int:
        """Migrate old timestamp-based sessions to new numbering system."""
        old_sessions = [d for d in self.data_dir.iterdir() 
                       if d.is_dir() and d.name.startswith("fetch_session_")]
        
        if not old_sessions:
            return 0
        
        logger.info(f"Found {len(old_sessions)} old sessions to migrate")
        
        # Sort by creation time
        old_sessions.sort(key=lambda x: x.stat().st_mtime)
        
        state = self.load_session_state()
        migrated_count = 0
        
        for old_session_dir in old_sessions:
            try:
                # Create new session
                session_number = state['next_session_number']
                new_session_name = f"session_{session_number:03d}"
                new_session_dir = self.data_dir / new_session_name
                
                # Move old session to new location
                old_session_dir.rename(new_session_dir)
                
                # Extract timestamp from old name
                timestamp_str = old_session_dir.name.replace("fetch_session_", "")
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
                except ValueError:
                    timestamp = datetime.fromtimestamp(old_session_dir.stat().st_mtime)
                
                # Update state
                state['sessions'][new_session_name] = {
                    'session_number': session_number,
                    'timestamp': timestamp.isoformat(),
                    'created_at': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'session_dir': str(new_session_dir),
                    'migrated_from': str(old_session_dir)
                }
                state['next_session_number'] = session_number + 1
                
                migrated_count += 1
                logger.info(f"📦 Migrated {old_session_dir.name} → {new_session_name}")
                
            except Exception as e:
                logger.error(f"Failed to migrate {old_session_dir}: {e}")
        
        self.save_session_state(state)
        
        if migrated_count > 0:
            logger.info(f"✅ Successfully migrated {migrated_count} sessions")
        
        return migrated_count

def main():
    """CLI interface for session management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Session management utilities")
    parser.add_argument("--migrate", action="store_true", help="Migrate old timestamp-based sessions")
    parser.add_argument("--list", action="store_true", help="List all sessions")
    parser.add_argument("--cleanup", type=int, nargs="?", const=20, help="Clean up old sessions (keep N most recent)")
    parser.add_argument("--create", action="store_true", help="Create a new session")
    
    args = parser.parse_args()
    
    manager = SessionManager()
    
    if args.migrate:
        count = manager.migrate_old_sessions()
        print(f"✅ Migrated {count} old sessions")
    
    elif args.list:
        sessions = manager.list_sessions()
        print("📁 SESSIONS:")
        print("=" * 50)
        for session_name, info in sorted(sessions.items(), 
                                       key=lambda x: x[1]['session_number'], 
                                       reverse=True):
            print(f"{info['session_number']:3d}. {session_name}")
            print(f"     📅 {info['created_at']}")
            print(f"     📂 {info['session_dir']}")
            print()
    
    elif args.cleanup is not None:
        count = manager.cleanup_old_sessions(args.cleanup)
        print(f"🧹 Cleaned up {count} old sessions")
    
    elif args.create:
        session_name, session_dir = manager.create_new_session()
        print(f"✅ Created new session: {session_name}")
        print(f"📂 Location: {session_dir}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 