import os
import json
import hashlib
import secrets
import datetime

class UserAuth:
    def __init__(self, users_file="data/users.json"):
        self.users_file = users_file
        self.sessions = {}
        
        # Create users file if it doesn't exist
        os.makedirs(os.path.dirname(users_file), exist_ok=True)
        if not os.path.exists(users_file):
            with open(users_file, 'w') as f:
                json.dump({"users": []}, f)
    
    def _load_users(self):
        """Load users from file"""
        with open(self.users_file, 'r') as f:
            return json.load(f)
    
    def _save_users(self, data):
        """Save users to file"""
        with open(self.users_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _hash_password(self, password, salt=None):
        """Hash a password with a salt"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Hash the password with the salt
        hash_obj = hashlib.sha256((password + salt).encode())
        password_hash = hash_obj.hexdigest()
        
        return password_hash, salt
    
    def register_user(self, username, password, role="user"):
        """Register a new user"""
        data = self._load_users()
        
        # Check if username already exists
        for user in data["users"]:
            if user["username"] == username:
                return False, "Username already exists"
        
        # Hash the password
        password_hash, salt = self._hash_password(password)
        
        # Add the user
        data["users"].append({
            "username": username,
            "password_hash": password_hash,
            "salt": salt,
            "role": role,
            "created_at": datetime.datetime.now().isoformat()
        })
        
        self._save_users(data)
        return True, "User registered successfully"
    
    def authenticate(self, username, password):
        """Authenticate a user"""
        data = self._load_users()
        
        # Find the user
        for user in data["users"]:
            if user["username"] == username:
                # Hash the password with the user's salt
                password_hash, _ = self._hash_password(password, user["salt"])
                
                # Check if the password is correct
                if password_hash == user["password_hash"]:
                    # Generate a session token
                    session_token = secrets.token_hex(32)
                    
                    # Store the session
                    self.sessions[session_token] = {
                        "username": username,
                        "role": user["role"],
                        "created_at": datetime.datetime.now().isoformat()
                    }
                    
                    return True, session_token
        
        return False, "Invalid username or password"
    
    def validate_session(self, session_token):
        """Validate a session token"""
        if session_token in self.sessions:
            return True, self.sessions[session_token]
        
        return False, "Invalid session token"
    
    def logout(self, session_token):
        """Logout a user by invalidating their session token"""
        if session_token in self.sessions:
            del self.sessions[session_token]
            return True
        
        return False
    
    def get_user_role(self, session_token):
        """Get the role of a user from their session token"""
        if session_token in self.sessions:
            return self.sessions[session_token]["role"]
        
        return None
