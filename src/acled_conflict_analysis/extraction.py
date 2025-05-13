import os
import requests
import pandas as pd
import pkgutil
import io
import json
import getpass

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

#iso_country_map = pd.read_csv('./data/Country_ISO_Code_Map.csv')


def get_iso_code(country_names):
    iso_country = []
    for x in country_names:
        iso_country.append(iso_country_map[iso_country_map['COUNTRY']==x]['ISO_CODE'].iloc[0])
    
    return iso_country

def acled_api(
    email_address=None,
    access_key=None,
    countries=None,
    region=None,
    start_date=None,
    end_date=None,
    add_variables=None,
    all_variables=False,
    dyadic=False,
    interaction=None,
    other_query=None,
):
    country_iso = get_iso_code(countries)
    # Access key
    access_key = access_key or os.getenv("ACLED_ACCESS_KEY")
    if not access_key:
        raise ValueError(
            "ACLED requires an access key, which needs to be supplied. You can request an access key by registering on https://developer.acleddata.com/."
        )

    # Email address
    email_address = email_address or os.getenv("ACLED_EMAIL_ADDRESS")
    if not email_address:
        raise ValueError(
            "ACLED requires an email address for access. Use the email address you provided when registering on https://developer.acleddata.com/."
        )

    # Building the URL
    url = "https://api.acleddata.com/acled/read/?key={}&email={}&limit=200000".format(
        access_key, email_address
    )

    if country_iso:
        url += "&iso=" + "|".join(str(iso) for iso in country_iso)

    if region:
        url += "&region=" + "|".join(str(region) for region in region)
    if start_date and end_date:
        url += "&event_date={}|{}&event_date_where=BETWEEN".format(start_date, end_date)

    fields = "region|country|year|event_date|source|admin1|admin2|admin3|location|event_type|sub_event_type|interaction|fatalities|timestamp|latitude|longitude|actor1|actor2|notes"
    if all_variables:
        fields = ""
    elif add_variables:
        fields += "|" + "|".join(add_variables)
    url += "&fields=" + fields

    # if dyadic:
    #     url += "&export_type=dyadic"
    # else:
    #     url += "&export_type=monadic"

    if interaction:
        url += "&interaction=" + ":".join(str(i) for i in interaction)

    if other_query:
        url += "&" + "&".join(other_query)

    # Making the GET request
    response = requests.get(url, verify=False)
    if not response.ok:
        raise Exception(
            "GET request was unsuccessful. Status code: {}".format(response.status_code)
        )

    # Parsing JSON response
    data = response.json()
    if not data.get("success", False):
        raise Exception(
            "GET request wasn't successful. Error: {}".format(data.get("error", ""))
        )

    return pd.DataFrame(data["data"])