import os
import requests
import pandas as pd
import pkgutil
import io
import json
import getpass
from .acled_auth import get_oauth_token, get_oauth_email, authenticate_with_credentials
#import pycountry

# Optional dependency for ISO code lookups
try:
    import pycountry  # type: ignore
    #print("✅ 'pycountry' is available for ISO code lookups.")
except Exception:  # optional
    pycountry = None

def save_acled_credentials(email, api_key=None):
    """
    Save ACLED credentials (email and API key) to a JSON file.
    
    Args:
        email (str): Your registered email address
        api_key (str, optional): Your API key. If None, will prompt for it.
    
    Returns:
        bool: True if credentials were saved successfully, False otherwise
    """
    # Define the location of the credentials file
    if os.name == 'nt':  # Windows
        config_dir = os.path.join(os.environ['USERPROFILE'], '.config', 'acled')
        config_path = os.path.join(config_dir, 'credentials.json')
    else:  # macOS/Linux
        config_dir = os.path.expanduser('~/.config/acled')
        config_path = os.path.expanduser('~/.config/acled/credentials.json')
    
    # Create the directory if it doesn't exist
    os.makedirs(config_dir, exist_ok=True)
    
    # If API key not provided, prompt for it
    if api_key is None:
        api_key = getpass.getpass("Enter your ACLED API key: ")
    
    # Store values
    try:
        credentials = {'email': email, 'api_key': api_key}
        with open(config_path, 'w') as file:
            json.dump(credentials, file)
        print(f"Credentials saved to {config_path}")
        return True
    except Exception as e:
        print(f"Error saving credentials: {e}")
        return False
    
def get_acled_credentials():
    """
    Get the ACLED email and API key from the configuration file.
    If not found, prompt the user and save the credentials.
    
    Returns:
        tuple: (email, api_key)
    """
    # Define the location of the credentials file
    if os.name == 'nt':  # Windows
        config_path = os.path.join(os.environ['USERPROFILE'], '.config', 'acled', 'credentials.json')
    else:  # macOS/Linux
        config_path = os.path.expanduser('~/.config/acled/credentials.json')
    
    # Check if the file exists
    if os.path.exists(config_path):
        # Read the credentials from the file
        try:
            with open(config_path, 'r') as file:
                credentials = json.load(file)
                email = credentials.get('email')
                api_key = credentials.get('api_key')
                
                # Verify we have both values
                if email and api_key:
                    return email, api_key
        except (json.JSONDecodeError, KeyError, IOError) as e:
            print(f"Error reading credentials: {e}")
    
    # If we get here, we need to prompt for credentials
    print("ACLED API requires both an email and API key.")
    email = input("Enter your registered email: ")
    api_key = getpass.getpass("Enter your ACLED API key: ")
    
    # Save the credentials
    save_acled_credentials(email, api_key)
    
    return email, api_key

def load_country_centroids():
    # Access the file from the package using pkgutil
    data = pkgutil.get_data('acled_conflict_analysis', 'data/Country_ISO_Code_Map.csv')
    iso_country_map = pd.read_csv(io.BytesIO(data))
    return iso_country_map

iso_country_map = load_country_centroids()



def get_iso_code(country_names):
    iso_country = []
    for x in country_names:
        iso_country.append(iso_country_map[iso_country_map['COUNTRY']==x]['ISO_CODE'].iloc[0])
    
    return iso_country

# --- ISO numeric normalization helpers ---
def _to_numeric_iso(code_or_name: str) -> str:
    """
    Convert a country identifier (alpha-2, alpha-3, numeric, or common name)
    into ISO-3166-1 numeric code (as a zero-padded 3-character string).

    Tries pycountry first (if available), then falls back to the local
    iso_country_map (expects columns COUNTRY and ISO_CODE).
    """
    if code_or_name is None:
        raise ValueError("Empty country identifier provided")

    s = str(code_or_name).strip()
    if not s:
        raise ValueError("Empty country identifier provided")

    # Already numeric
    if s.isdigit():
        return s.zfill(3)

    # Try pycountry (preferred for alpha codes and names)
    if pycountry:
        c = (pycountry.countries.get(alpha_3=s.upper())
             or pycountry.countries.get(alpha_2=s.upper()))
        if not c:
            try:
                c = pycountry.countries.lookup(s)
            except Exception:
                c = None
        if c and getattr(c, "numeric", None):
            # pycountry returns numeric as a zero-padded string already
            return str(c.numeric).zfill(3)

    # Fallback: local mapping by country name
    try:
        if isinstance(iso_country_map, pd.DataFrame) and {'COUNTRY','ISO_CODE'}.issubset(iso_country_map.columns):
            # case-insensitive match on name
            mask = iso_country_map['COUNTRY'].astype(str).str.casefold() == s.casefold()
            if mask.any():
                val = iso_country_map.loc[mask, 'ISO_CODE'].iloc[0]
                return str(val).zfill(3)
    except Exception:
        pass

    raise ValueError(f"Could not convert '{code_or_name}' to ISO numeric code")


def _normalize_iso_numeric_list(countries=None, country_codes=None):
    """
    Build a list of ISO numeric codes from either countries (names) or country_codes
    (alpha-2/alpha-3/numeric). Skips blanks/None and deduplicates while preserving order.
    """
    raw = []
    if country_codes:
        raw.extend([c for c in country_codes if c])
    elif countries:
        raw.extend([c for c in countries if c])

    numeric = []
    seen = set()
    provided_count = len(raw)
    for item in raw:
        try:
            num = _to_numeric_iso(item)
        except ValueError:
            # Skip invalid entries but collect nothing here
            continue
        if num not in seen:
            seen.add(num)
            numeric.append(num)
    # If user provided at least one identifier but nothing could be mapped,
    # fail fast to avoid unfiltered API queries returning all countries.
    if provided_count > 0 and len(numeric) == 0:
        raise ValueError(
            "Could not convert provided country identifiers to ISO numeric codes. "
            "Install 'pycountry' or pass country names via 'countries' or numeric ISO codes via 'country_codes'."
        )
    return numeric


def acled_api(
    countries=None,
    country_codes =None,
    region=None,
    start_date=None,
    end_date=None,
    add_variables=None,
    all_variables=False,
    dyadic=False,
    interaction=None,
    other_query=None,
):
    """
    Query ACLED API using OAuth token authentication with Bearer token header.
    
    Args:
        countries (list): List of country names
        region (list): List of region names/codes
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        add_variables (list): Additional variables to include
        all_variables (bool): Include all available variables
        dyadic (bool): Return dyadic data format
        interaction (list): Interaction codes to filter by
        other_query (list): Additional query parameters
    
    Returns:
        pandas.DataFrame: ACLED conflict data
    """
    # Get OAuth token
    access_token = get_oauth_token()
    if not access_token:
        raise ValueError(
            "Failed to obtain OAuth token. Please check your ACLED credentials."
        )
    
    # Normalize to ISO numeric codes from either names or codes
    country_iso = _normalize_iso_numeric_list(countries=countries, country_codes=country_codes)
    
    # Building the URL for the new ACLED API
    url = "https://acleddata.com/api/acled/read"
    
    # Set up parameters
    params = {
        "_format": "json",
        "limit": 4000000
    }
    
    if country_iso:
        params["iso"] = "|".join(str(iso) for iso in country_iso)
    
    if region:
        params["region"] = "|".join(str(region) for region in region)
    
    if start_date and end_date:
        params["event_date"] = f"{start_date}|{end_date}"
        params["event_date_where"] = "BETWEEN"

    params['population'] = 'full'

    fields = "region|country|year|event_date|source|admin1|admin2|admin3|location|event_type|sub_event_type|interaction|fatalities|timestamp|latitude|longitude|actor1|actor2|notes|population_1km|population_5km|population_2km|population_best"
    if all_variables:
        # Don't specify fields to get all variables
        pass
    elif add_variables:
        fields += "|" + "|".join(add_variables)
    
    if not all_variables:
        params["fields"] = fields

    if interaction:
        params["interaction"] = ":".join(str(i) for i in interaction)

    if other_query:
        for query in other_query:
            key, value = query.split('=', 1)
            params[key] = value

    # Set up headers with Bearer token
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }

    # print(f"🌐 Making request to ACLED API...")
    print(f"URL: {url}")
    print(f"Parameters: {params}")

    # Making the GET request with Bearer token authentication
    response = requests.get(url, params=params, headers=headers, verify=False)
    
    if not response.ok:
        print(f"❌ GET request was unsuccessful. Status code: {response.status_code}")
        print(f"Response: {response.text}")
        raise Exception(
            f"GET request was unsuccessful. Status code: {response.status_code}"
        )

    # Parsing JSON response
    try:
        data = response.json()
    except json.JSONDecodeError:
        print(f"❌ Failed to parse JSON response")
        print(f"Raw response: {response.text[:500]}...")
        raise Exception("Failed to parse JSON response from ACLED API")
    
    # Check if response contains error information
    if isinstance(data, dict) and 'error' in data:
        print(f"❌ API request returned an error")
        print(f"Error: {data.get('error', 'Unknown error')}")
        raise Exception(f"API request failed. Error: {data.get('error', '')}")
    
    # Handle different response formats
    if isinstance(data, list):
        # Direct list of records
        records = data
    elif isinstance(data, dict) and 'data' in data:
        # Response with data wrapper
        records = data['data']
    elif isinstance(data, dict) and not data.get('success', True):
        # Explicit success=False
        print(f"❌ API request wasn't successful")
        print(f"Error: {data.get('error', 'Unknown error')}")
        raise Exception(f"GET request wasn't successful. Error: {data.get('error', '')}")
    else:
        # Assume the whole response is the data
        records = data if isinstance(data, list) else [data]

    print(f"✅ Successfully retrieved {len(records)} records")
    return pd.DataFrame(records)

def acled_api_with_credentials(
    email,
    password,
    countries=None,
    country_codes =None,
    region=None,
    start_date=None,
    end_date=None,
    add_variables=None,
    all_variables=False,
    dyadic=False,
    interaction=None,
    other_query=None,
):
    """
    Query ACLED API using email and password for OAuth authentication.
    This function is designed for notebook use where credentials are provided directly.
    
    Args:
        email (str): ACLED email address
        password (str): ACLED password
        countries (list): List of country names
        region (list): List of region names/codes
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        add_variables (list): Additional variables to include
        all_variables (bool): Include all available variables
        dyadic (bool): Return dyadic data format
        interaction (list): Interaction codes to filter by
        other_query (list): Additional query parameters
    
    Returns:
        pandas.DataFrame: ACLED conflict data
    """
    # Get OAuth token using provided credentials
    access_token = authenticate_with_credentials(email, password)
    if not access_token:
        raise ValueError(
            "Failed to obtain OAuth token. Please check your ACLED credentials."
        )
    
    # Normalize to ISO numeric codes from either names or codes
    country_iso = _normalize_iso_numeric_list(countries=countries, country_codes=country_codes)
    
    # Building the URL for the new ACLED API
    url = "https://acleddata.com/api/acled/read"
    
    # Set up parameters
    params = {
        "_format": "json",
        "limit": 4000000
    }
    
    if country_iso:
        params["iso"] = "|".join(str(iso) for iso in country_iso)
    
    if region:
        params["region"] = "|".join(str(region) for region in region)
    
    if start_date and end_date:
        params["event_date"] = f"{start_date}|{end_date}"
        params["event_date_where"] = "BETWEEN"

    fields = "region|country|year|event_date|source|admin1|admin2|admin3|location|event_type|sub_event_type|interaction|fatalities|timestamp|latitude|longitude|actor1|actor2|notes"
    if all_variables:
        # Don't specify fields to get all variables
        pass
    elif add_variables:
        fields += "|" + "|".join(add_variables)
    
    if not all_variables:
        params["fields"] = fields

    if interaction:
        params["interaction"] = ":".join(str(i) for i in interaction)

    if other_query:
        for query in other_query:
            key, value = query.split('=', 1)
            params[key] = value

    # Set up headers with Bearer token
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }

    # print(f"🌐 Making request to ACLED API...")
    # print(f"URL: {url}")
    # print(f"Parameters: {params}")

    # Making the GET request with Bearer token authentication
    response = requests.get(url, params=params, headers=headers, verify=False)
    
    if not response.ok:
        print(f"❌ GET request was unsuccessful. Status code: {response.status_code}")
        print(f"Response: {response.text}")
        raise Exception(
            f"GET request was unsuccessful. Status code: {response.status_code}"
        )

    # Parsing JSON response
    try:
        data = response.json()
    except json.JSONDecodeError:
        print(f"❌ Failed to parse JSON response")
        print(f"Raw response: {response.text[:500]}...")
        raise Exception("Failed to parse JSON response from ACLED API")
    
    # Check if response contains error information
    if isinstance(data, dict) and 'error' in data:
        print(f"❌ API request returned an error")
        print(f"Error: {data.get('error', 'Unknown error')}")
        raise Exception(f"API request failed. Error: {data.get('error', '')}")
    
    # Handle different response formats
    if isinstance(data, list):
        # Direct list of records
        records = data
    elif isinstance(data, dict) and 'data' in data:
        # Response with data wrapper
        records = data['data']
    elif isinstance(data, dict) and not data.get('success', True):
        # Explicit success=False
        print(f"❌ API request wasn't successful")
        print(f"Error: {data.get('error', 'Unknown error')}")
        raise Exception(f"GET request wasn't successful. Error: {data.get('error', '')}")
    else:
        # Assume the whole response is the data
        records = data if isinstance(data, list) else [data]

    print(f"✅ Successfully retrieved {len(records)} records")
    return pd.DataFrame(records)

# def acled_api(
#     email_address=None,
#     access_key=None,
#     countries=None,
#     region=None,
#     start_date=None,
#     end_date=None,
#     add_variables=None,
#     all_variables=False,
#     dyadic=False,
#     interaction=None,
#     other_query=None,
# ):
#     country_iso = get_iso_code(countries)
#     # Access key
#     access_key = access_key or os.getenv("ACLED_ACCESS_KEY")
#     if not access_key:
#         raise ValueError(
#             "ACLED requires an access key, which needs to be supplied. You can request an access key by registering on https://developer.acleddata.com/."
#         )

#     # Email address
#     email_address = email_address or os.getenv("ACLED_EMAIL_ADDRESS")
#     if not email_address:
#         raise ValueError(
#             "ACLED requires an email address for access. Use the email address you provided when registering on https://developer.acleddata.com/."
#         )

#     # Building the URL
#     url = "https://api.acleddata.com/acled/read/?key={}&email={}&limit=200000".format(
#         access_key, email_address
#     )

#     if country_iso:
#         url += "&iso=" + "|".join(str(iso) for iso in country_iso)

#     if region:
#         url += "&region=" + "|".join(str(region) for region in region)
#     if start_date and end_date:
#         url += "&event_date={}|{}&event_date_where=BETWEEN".format(start_date, end_date)

#     fields = "region|country|year|event_date|source|admin1|admin2|admin3|location|event_type|sub_event_type|interaction|fatalities|timestamp|latitude|longitude|actor1|actor2|notes"
#     if all_variables:
#         fields = ""
#     elif add_variables:
#         fields += "|" + "|".join(add_variables)
#     url += "&fields=" + fields

#     # if dyadic:
#     #     url += "&export_type=dyadic"
#     # else:
#     #     url += "&export_type=monadic"

#     if interaction:
#         url += "&interaction=" + ":".join(str(i) for i in interaction)

#     if other_query:
#         url += "&" + "&".join(other_query)

#     # Making the GET request
#     response = requests.get(url, verify=False)
#     if not response.ok:
#         raise Exception(
#             "GET request was unsuccessful. Status code: {}".format(response.status_code)
#         )

#     # Parsing JSON response
#     data = response.json()
#     if not data.get("success", False):
#         raise Exception(
#             "GET request wasn't successful. Error: {}".format(data.get("error", ""))
#         )

#     return pd.DataFrame(data["data"])