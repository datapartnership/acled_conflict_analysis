#!/usr/bin/env python3
"""
ACLED API Authentication Module
This module handles OAuth authentication with ACLED API and credential management.
"""

import requests
import getpass
import json
import urllib3
import os
from datetime import datetime, timedelta

def get_credentials_path():
    """Get the path to store ACLED credentials."""
    if os.name == 'nt':  # Windows
        config_dir = os.path.join(os.environ['USERPROFILE'], '.config', 'acled')
    else:  # macOS/Linux
        config_dir = os.path.expanduser('~/.config/acled')
    
    os.makedirs(config_dir, exist_ok=True)
    return config_dir

def save_acled_oauth_credentials(email=None, password=None):
    """
    Save ACLED OAuth credentials (email and password) to a JSON file.
    
    Args:
        email (str, optional): Email address. If None, will prompt for it.
        password (str, optional): Password. If None, will prompt for it.
    
    Returns:
        bool: True if credentials were saved successfully, False otherwise
    """
    config_dir = get_credentials_path()
    credentials_path = os.path.join(config_dir, 'oauth_credentials.json')
    
    # Prompt for credentials if not provided
    if email is None:
        email = input("Enter your ACLED email: ")
    
    if password is None:
        password = getpass.getpass("Enter your ACLED password: ")
    
    try:
        credentials = {
            'email': email,
            'password': password,
            'created_at': datetime.now().isoformat()
        }
        
        with open(credentials_path, 'w') as file:
            json.dump(credentials, file, indent=2)
        
        # Set file permissions to be readable only by owner
        os.chmod(credentials_path, 0o600)
        
        print(f"✅ ACLED OAuth credentials saved to {credentials_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving credentials: {e}")
        return False

def get_acled_oauth_credentials():
    """
    Get ACLED OAuth credentials from the stored file.
    If credentials don't exist, prompt user to enter them.
    
    Returns:
        tuple: (email, password) or (None, None) if failed
    """
    config_dir = get_credentials_path()
    credentials_path = os.path.join(config_dir, 'oauth_credentials.json')
    
    # Check if credentials file exists
    if os.path.exists(credentials_path):
        try:
            with open(credentials_path, 'r') as file:
                credentials = json.load(file)
            
            email = credentials.get('email')
            password = credentials.get('password')
            
            if email and password:
                return email, password
                
        except Exception as e:
            print(f"❌ Error reading credentials: {e}")
    
    # If we get here, we need to prompt for and save credentials
    print("🔐 ACLED OAuth credentials not found. Please enter your credentials:")
    email = input("Enter your ACLED email: ")
    password = getpass.getpass("Enter your ACLED password: ")
    
    if save_acled_oauth_credentials(email, password):
        return email, password
    else:
        return None, None

def set_credentials_from_notebook(email, password):
    """
    Set ACLED OAuth credentials directly from notebook input.
    This function allows users to input credentials in a notebook cell.
    
    Args:
        email (str): Email address
        password (str): Password
    
    Returns:
        bool: True if credentials were saved successfully, False otherwise
    """
    if not email or not password:
        print("❌ Both email and password are required")
        return False
        
    return save_acled_oauth_credentials(email, password)

def get_oauth_token_with_credentials(email=None, password=None):
    """
    Get OAuth token using provided credentials or stored credentials.
    
    Args:
        email (str, optional): Email address
        password (str, optional): Password
        
    Returns:
        str: OAuth token or None if failed
    """
    # If credentials are provided, use them directly
    if email and password:
        # Save credentials for future use
        save_acled_oauth_credentials(email, password)
        
        # Get token with these credentials
        return authenticate_with_credentials(email, password)
    else:
        # Use existing get_oauth_token function
        return get_oauth_token()

def authenticate_with_credentials(email, password):
    """
    Authenticate with ACLED API using provided credentials.
    
    Args:
        email (str): Email address
        password (str): Password
        
    Returns:
        str: OAuth token or None if failed
    """
    # Suppress SSL warnings for corporate networks
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # API endpoint
    url = "https://acleddata.com/oauth/token"
    
    # Request headers
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    # Request data
    data = {
        "username": email,
        "password": password,
        "grant_type": "password",
        "client_id": "acled"
    }
    
    print(f"🔐 Authenticating with ACLED API for: {email}")
    
    try:
        # Make the POST request
        response = requests.post(url, headers=headers, data=data, verify=False)
        
        # Check if request was successful
        if response.status_code == 200:
            token_data = response.json()
            print("✅ Authentication successful!")
            print(f"Token expires in: {token_data.get('expires_in', 'Unknown')} seconds")
            
            return token_data.get('access_token')
        else:
            print(f"❌ Authentication failed!")
            print(f"Status Code: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error: {error_data.get('error_description', response.text)}")
            except:
                print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse response: {e}")
        return None

def get_oauth_token():
    """
    Get a valid OAuth token. Check for cached token first, authenticate if needed.
    
    Returns:
        str: Valid access token or None if authentication failed
    """
    import time
    
    # Check for cached token first
    config_path = os.path.join(get_credentials_path(), 'oauth_token.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as file:
                token_info = json.load(file)
                
            # Check if token is still valid (with 5 minute buffer)
            expires_at = token_info.get('expires_at', 0)
            current_time = time.time()
            
            if current_time < (expires_at - 300):  # 5 minute buffer
                print("🔄 Using cached OAuth token")
                return token_info.get('access_token')
            else:
                print("⏰ Cached token expired, refreshing...")
                
        except (json.JSONDecodeError, KeyError, OSError):
            print("⚠️  Invalid cached token, fetching new one...")
    
    # Get credentials
    email, password = get_acled_oauth_credentials()
    if not email or not password:
        return None
    
    # Suppress SSL warnings for corporate networks
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # API endpoint
    url = "https://acleddata.com/oauth/token"
    
    # Request headers
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    # Request data
    data = {
        "username": email,
        "password": password,
        "grant_type": "password",
        "client_id": "acled"
    }
    
    print(f"🔐 Authenticating with ACLED API for : {email}")
    
    try:
        # Make the POST request (disable SSL verification for corporate networks)
        response = requests.post(url, headers=headers, data=data, verify=False)
        
        # Check if request was successful
        if response.status_code == 200:
            token_data = response.json()
            print("✅ Authentication successful!")
            print(f"Token expires in: {token_data.get('expires_in', 'Unknown')} seconds")
            
            # Cache the token
            expires_in = token_data.get('expires_in', 86400)  # Default to 24 hours
            expires_at = time.time() + expires_in
            
            token_info = {
                'access_token': token_data.get('access_token'),
                'expires_in': expires_in,
                'expires_at': expires_at,
                'token_type': token_data.get('token_type', 'Bearer')
            }
            
            try:
                with open(config_path, 'w') as file:
                    json.dump(token_info, file)
                # Set secure permissions
                os.chmod(config_path, 0o600)
                print("💾 Token cached for future use")
            except OSError as e:
                print(f"⚠️  Could not cache token: {e}")
            
            return token_data.get('access_token')
        else:
            print(f"❌ Authentication failed!")
            print(f"Status Code: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error: {error_data.get('error_description', response.text)}")
            except:
                print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse response: {e}")
        return None

def get_oauth_email():
    """
    Get the email address from the stored credentials.
    
    Returns:
        str: Email address or None if not available
    """
    email, _ = get_acled_oauth_credentials()
    return email

if __name__ == "__main__":
    token = get_oauth_token()
    if token:
        print(f"🎉 Success! OAuth token is ready.")
        print(f"Token: {token[:20]}...")
    else:
        print("❌ Failed to get OAuth token")
