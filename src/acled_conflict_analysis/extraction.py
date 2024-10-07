import os
import requests
import pandas as pd

iso_country_map = pd.read_csv('./data/Country_ISO_Code_Map.csv')

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
    url = "https://api.acleddata.com/acled/read/?key={}&email={}&limit=50000".format(
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
    response = requests.get(url)
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