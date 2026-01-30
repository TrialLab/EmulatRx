import time
import requests

def get_trial_info(trial_id, verbose = 0):
    api_url = "https://clinicaltrials.gov/api/v2/studies/{}".format(trial_id)
    # params = {"nctId": trial_id}  # Use the term passed to the function
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }  # Fake browser request header
    # Throttle requests to avoid overloading the server
    time.sleep(1)

    # Make the GET request to the Clinicaltrial.gov API
    response = requests.get(api_url, headers=headers)
    response.raise_for_status()  # Raise an error for bad status codes

    # Parse the JSON response
    data = response.json()

    return data

def get_concept_id(term, verbose = 0):
    """
    Query the concept ID using the OHDSI Athena API for a fixed query term ('sterioad').
    Parses the returned result to find matching concept information.
    """
    api_url = "https://athena.ohdsi.org/api/v1/concepts"
    params = {"query": term}  # Use the term passed to the function
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }  # Fake browser request header
    try:
        # Throttle requests to avoid overloading the server
        time.sleep(1)

        # Make the GET request to the Athena API
        response = requests.get(api_url, params=params, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes

        # Parse the JSON response
        data = response.json()
        if verbose:
            print(f"API Response: {data}")

        # Check if the response contains valid content
        if "content" in data and data["content"]:
            # Extract the first matching concept's details
            concept = data["content"][0]
            concept_id = concept.get("id", "N/A")
            concept_name = concept.get("name", "Unknown")
            domain_name = concept.get("domain", "Unknown")
            if verbose:
                print(f"Found concept: {concept_name} (ID: {concept_id}, Domain: {domain_name})")
            return {"concept_id": concept_id, "concept_name": concept_name, "concept_domain": domain_name}
        else:
            if verbose:
                print(f"No concept found for term: {term}")
            return {"concept_id": None, "concept_name": "Unknown", "concept_domain": "Unknown"}

    except requests.exceptions.RequestException as e:
        if verbose:
            print(f"Error retrieving concept_id for '{term}': {e}")
        return {"concept_id": None, "concept_name": "Error", "concept_domain": "Error"}

