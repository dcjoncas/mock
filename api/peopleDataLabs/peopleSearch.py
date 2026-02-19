import os
from dotenv import load_dotenv
import requests
import json
#from peopledatalabs import PDLPY

# Load environment variables
load_dotenv()

PDL_API_KEY = os.getenv("PDL_API_KEY")

def searchSkills(skillList: list[str], size: int = 5):
    url = "https://api.peopledatalabs.com/v5/person/search"

    headers = {
        'Content-Type': 'application/json',
        'X-Api-Key': PDL_API_KEY,
    }

    payload = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"skills": skill}} for skill in skillList
                ]
            }
        },
        "size": size
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()
    else:
        return f"Failed to retrieve data, status code: {response.status_code}"
    
def searchSkillsAndLocation(skillList: list[str], locationCity: str = "", locationState: str = "", locationCountry: str = "", size: int = 5):
    url = "https://api.peopledatalabs.com/v5/person/search"

    headers = {
        'Content-Type': 'application/json',
        'X-Api-Key': PDL_API_KEY,
    }

    print("Skill Input: " + str(skillList))
    print("Location City Input: " + locationCity)

    mustArray = []

    for skill in skillList:
        mustArray.append({"match": {"skills": skill.lower()}})
    
    if len(locationCity) > 0:
        mustArray.append({"match": {"location_locality": locationCity.lower()}})
    if len(locationState) > 0:
        mustArray.append({"match": {"location_region": locationState.lower()}})
    if len(locationCountry) > 0:
        mustArray.append({"match": {"location_country": locationCountry.lower()}})

    payload = {
        "query": {
            "bool": {
                "must": mustArray
            }
        },
        "size": size
    }

    print("Payload for PeopleDataLabs Search: " + str(payload))

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"PeopleDataLabs API Error: {response.status_code}, Response: {response.text}")
        return f"Failed to retrieve data, status code: {response.status_code}"
    