import streamlit as st
import json
import os
import datetime
import hashlib
from typing import Dict, List, Any, Optional
import sqlite3
from pathlib import Path

class AuthSystem:
    """
    Authentication system with multiple sign-in options and chat management
    """
    
    def __init__(self):
        self.db_path = "user_data/chatbot_users.db"
        self.chats_path = "user_data/saved_chats/"
        self.ensure_directories()
        self.init_database()
    
    def ensure_directories(self):
        """Ensure necessary directories exist"""
        Path("user_data").mkdir(exist_ok=True)
        Path(self.chats_path).mkdir(exist_ok=True)
    
    def init_database(self):
        """Initialize SQLite database for users and chats"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE,
                auth_provider TEXT NOT NULL,
                auth_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create chats table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                chat_name TEXT NOT NULL,
                messages TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password: str) -> str:
        """Hash password for security"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_guest_user(self, username: str) -> Dict[str, Any]:
        """Create a guest user account"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO users (username, auth_provider, auth_id)
                VALUES (?, 'guest', ?)
            ''', (username, f"guest_{datetime.datetime.now().timestamp()}"))
            
            user_id = cursor.lastrowid
            conn.commit()
            
            return {
                'id': user_id,
                'username': username,
                'auth_provider': 'guest',
                'created_at': datetime.datetime.now().isoformat()
            }
        except sqlite3.IntegrityError:
            # Username already exists, try with timestamp
            new_username = f"{username}_{int(datetime.datetime.now().timestamp())}"
            cursor.execute('''
                INSERT INTO users (username, auth_provider, auth_id)
                VALUES (?, 'guest', ?)
            ''', (new_username, f"guest_{datetime.datetime.now().timestamp()}"))
            
            user_id = cursor.lastrowid
            conn.commit()
            
            return {
                'id': user_id,
                'username': new_username,
                'auth_provider': 'guest',
                'created_at': datetime.datetime.now().isoformat()
            }
        finally:
            conn.close()
    
    def save_chat_session(self, user_id: int, chat_name: str, messages: List[Dict]) -> bool:
        """Save chat session to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            messages_json = json.dumps(messages)
            
            cursor.execute('''
                INSERT INTO chats (user_id, chat_name, messages, updated_at)
                VALUES (?, ?, ?, ?)
            ''', (user_id, chat_name, messages_json, datetime.datetime.now()))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            st.error(f"Error saving chat: {e}")
            return False
    
    def load_user_chats(self, user_id: int) -> List[Dict[str, Any]]:
        """Load all chats for a user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, chat_name, messages, created_at, updated_at
                FROM chats
                WHERE user_id = ?
                ORDER BY updated_at DESC
            ''', (user_id,))
            
            chats = []
            for row in cursor.fetchall():
                chats.append({
                    'id': row[0],
                    'chat_name': row[1],
                    'messages': json.loads(row[2]) if row[2] else [],
                    'created_at': row[3],
                    'updated_at': row[4]
                })
            
            conn.close()
            return chats
        except Exception as e:
            st.error(f"Error loading chats: {e}")
            return []
    
    def delete_chat(self, chat_id: int) -> bool:
        """Delete a chat session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM chats WHERE id = ?', (chat_id,))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            st.error(f"Error deleting chat: {e}")
            return False

def render_auth_interface():
    """Render the authentication interface"""
    
    # Initialize auth system
    if 'auth_system' not in st.session_state:
        st.session_state.auth_system = AuthSystem()
    
    auth_system = st.session_state.auth_system
    
    # Check if user is already logged in
    if 'user' in st.session_state and st.session_state.user:
        return st.session_state.user
    
    st.markdown("## 🔐 Authentication")
    st.markdown("Sign in to save and manage your chat sessions.")
    
    # Authentication options
    auth_option = st.radio(
        "Choose your sign-in method:",
        ["👤 Guest User", "🔗 Google", "🔗 LinkedIn", "🔗 GitHub", "📧 Email/Password"],
        horizontal=True
    )
    
    if auth_option == "👤 Guest User":
        return render_guest_auth(auth_system)
    elif auth_option == "🔗 Google":
        return render_google_auth(auth_system)
    elif auth_option == "🔗 LinkedIn":
        return render_linkedin_auth(auth_system)
    elif auth_option == "🔗 GitHub":
        return render_github_auth(auth_system)
    elif auth_option == "📧 Email/Password":
        return render_email_auth(auth_system)
    
    return None

def render_guest_auth(auth_system: AuthSystem):
    """Render guest authentication"""
    st.markdown("### 👤 Guest Sign In")
    st.info("Create a guest account to save your chats. No email required!")
    
    username = st.text_input("Choose a username:", placeholder="Enter your username")
    
    if st.button("🚀 Continue as Guest", type="primary"):
        if username and len(username) >= 3:
            user = auth_system.create_guest_user(username)
            st.session_state.user = user
            st.success(f"Welcome, {user['username']}! 🎉")
            st.rerun()
        else:
            st.error("Please enter a username (at least 3 characters)")
    
    return None

def render_google_auth(auth_system: AuthSystem):
    """Render Google authentication"""
    st.markdown("### 🔗 Google Sign In")
    st.info("Sign in with your Google account for seamless access.")
    
    # Placeholder for Google OAuth
    st.markdown("""
    **Google OAuth Integration** (Coming Soon!)
    
    This would integrate with Google's OAuth 2.0 API to provide secure authentication.
    
    Features:
    - Secure OAuth 2.0 authentication
    - Automatic profile information
    - Single sign-on experience
    """)
    
    if st.button("🔗 Sign in with Google", disabled=True):
        st.info("Google OAuth integration will be available in the next update!")
    
    return None

def render_linkedin_auth(auth_system: AuthSystem):
    """Render LinkedIn authentication"""
    st.markdown("### 🔗 LinkedIn Sign In")
    st.info("Connect with your LinkedIn profile for professional networking.")
    
    # Placeholder for LinkedIn OAuth
    st.markdown("""
    **LinkedIn OAuth Integration** (Coming Soon!)
    
    This would integrate with LinkedIn's OAuth 2.0 API for professional authentication.
    
    Features:
    - Professional profile integration
    - Network-based features
    - Career-focused analytics
    """)
    
    if st.button("🔗 Sign in with LinkedIn", disabled=True):
        st.info("LinkedIn OAuth integration will be available in the next update!")
    
    return None

def render_github_auth(auth_system: AuthSystem):
    """Render GitHub authentication"""
    st.markdown("### 🔗 GitHub Sign In")
    st.info("Sign in with GitHub for developer-focused features.")
    
    # Placeholder for GitHub OAuth
    st.markdown("""
    **GitHub OAuth Integration** (Coming Soon!)
    
    This would integrate with GitHub's OAuth 2.0 API for developer authentication.
    
    Features:
    - Developer profile integration
    - Code repository linking
    - Technical collaboration tools
    """)
    
    if st.button("🔗 Sign in with GitHub", disabled=True):
        st.info("GitHub OAuth integration will be available in the next update!")
    
    return None

def render_email_auth(auth_system: AuthSystem):
    """Render email/password authentication"""
    st.markdown("### 📧 Email/Password Sign In")
    st.info("Create an account with email and password for full features.")
    
    # Placeholder for email authentication
    st.markdown("""
    **Email Authentication** (Coming Soon!)
    
    This would provide traditional email/password authentication with:
    
    Features:
    - Email verification
    - Password recovery
    - Account management
    - Enhanced security
    """)
    
    if st.button("📧 Create Account", disabled=True):
        st.info("Email authentication will be available in the next update!")
    
    return None

def render_chat_management():
    """Render chat management interface"""
    if 'user' not in st.session_state or not st.session_state.user:
        st.warning("Please sign in to manage your chats.")
        return
    
    user = st.session_state.user
    auth_system = st.session_state.auth_system
    
    st.markdown(f"## 💾 Chat Management - {user['username']}")
    
    # Load user's chats
    chats = auth_system.load_user_chats(user['id'])
    
    if not chats:
        st.info("No saved chats yet. Start chatting to save your conversations!")
        return
    
    # Display saved chats
    st.markdown("### 📚 Your Saved Chats")
    
    for chat in chats:
        with st.expander(f"💬 {chat['chat_name']} - {chat['updated_at'][:10]}"):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**Messages:** {len(chat['messages'])}")
                st.write(f"**Created:** {chat['created_at'][:10]}")
                st.write(f"**Updated:** {chat['updated_at'][:10]}")
            
            with col2:
                if st.button("📥 Load", key=f"load_{chat['id']}"):
                    st.session_state.messages = chat['messages']
                    st.success("Chat loaded successfully!")
                    st.rerun()
            
            with col3:
                if st.button("🗑️ Delete", key=f"delete_{chat['id']}"):
                    if auth_system.delete_chat(chat['id']):
                        st.success("Chat deleted successfully!")
                        st.rerun()

def save_chat_session(messages: List[Dict]):
    """Save current chat session"""
    if 'user' not in st.session_state or not st.session_state.user:
        st.warning("Please sign in to save your chat.")
        return
    
    user = st.session_state.user
    auth_system = st.session_state.auth_system
    
    # Get chat name from user
    chat_name = st.text_input("Enter a name for this chat session:", 
                             value=f"Chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}")
    
    if st.button("💾 Save Chat"):
        if chat_name:
            if auth_system.save_chat_session(user['id'], chat_name, messages):
                st.success(f"Chat '{chat_name}' saved successfully!")
            else:
                st.error("Failed to save chat.")
        else:
            st.error("Please enter a chat name.")

def load_chat_session():
    """Load a saved chat session"""
    if 'user' not in st.session_state or not st.session_state.user:
        st.warning("Please sign in to load your chats.")
        return
    
    user = st.session_state.user
    auth_system = st.session_state.auth_system
    
    # Load user's chats
    chats = auth_system.load_user_chats(user['id'])
    
    if not chats:
        st.info("No saved chats found.")
        return
    
    # Create selection interface
    chat_options = [f"{chat['chat_name']} ({chat['updated_at'][:10]})" for chat in chats]
    selected_chat = st.selectbox("Select a chat to load:", chat_options)
    
    if st.button("📥 Load Selected Chat"):
        selected_index = chat_options.index(selected_chat)
        selected_messages = chats[selected_index]['messages']
        st.session_state.messages = selected_messages
        st.success("Chat loaded successfully!")
        st.rerun()

def logout():
    """Logout current user"""
    if 'user' in st.session_state:
        del st.session_state.user
    st.success("Logged out successfully!")
    st.rerun()
