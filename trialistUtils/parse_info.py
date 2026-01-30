from openai import OpenAI
import json
import os.path as osp
import os
from datetime import datetime
from urllib.parse import quote
import spacy
import requests
import re
import time
nlp = spacy.load("en_core_web_sm")

api_key = "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"
client = OpenAI(api_key='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',
                organization='org-p7vlqGOYmscIzHdhbT7MJtEP',
                project='proj_6EKrvoAglwQOkOMLXTRd0SgD')

concept_set = {}
# "1) Annotate concepts with the domains 'Demographic', 'Condition', 'Device', 'Cost', 'Procedure', 'Drug', 'Episode', 'Measurement', 'Observation', 'Provider', 'Specimen', 'Visit', 'Value', 'Negation_cue', 'Temporal', and 'Quantity'. If you cannot annotate with the given domains, you can name a new one (e.g., Drug_cycle). \n" + \

parse_prompt =  "Annotate clinical concepts from the given text using the following rules: \n" + \
                  "1) Annotate concepts with the domains 'Demographic', 'Condition', 'Device', 'Procedure', 'Drug', 'Measurement', 'Observation', 'Visit', 'Value', 'Negation_cue', 'Temporal', and 'Quantity'. If you cannot annotate with the given domains, you can name a new one (e.g., Drug_cycle, Visit, Provider, etc.). \n" + \
                  "2) Split the concepts as detail as possible. Each concept can be annotated only once with a single domain. \n" + \
                  "3) Normalize clinical abbreviation and acronyms and attached behind the original abbreviation with parenthesis. \n" + \
                  "4) Return your response under [Annotation] section. \n" + \
                  "Following is not allowed examples: \n" + \
                  "1) <Measurement>EGFR <Value>triple postive</Value></Measurement> \n" + \
                  "2) <Condition>Hypertension, diabetes, heart failure, and dementia</Condition> \n" + \
                  "Below is allowed examples: \n" + \
                  "1) <Measurement>EGFR</Measurement> <Value>triple positive</Value> \n" + \
                  "2) <Condition>hypertension</Condition>, <Condition>T2DM (Type 2 Diabetes Mellitus)</Condition>, <Condition>heart failure</Condition>, and <Condition>dementia</Condition> \n" + \
                  "3) Patient <Demographic>aged<Demographic> > <Value>65 years old</Value> \n" + \
                  "4) <Drug>Metformin</Drug> <Value>500 mg</Value> <Temporal>daily</Temporal> \n" + \
                  "Following is information for each domain: \n" + \
                  "1) Condition is events of a Person suggesting the presence of a disease or medical condition stated as a diagnosis, a sign, or a symptom, which is either observed by a Provider or reported by the patient. \n" +\
                  "2) Drugs include prescription and over-the-counter medicines, vaccines, and large-molecule biologic therapies. Radiological devices ingested or applied locally do not count as Drugs. \n" +\
                  "3) Procedure is records of activities or processes ordered by, or carried out by, a healthcare provider on the patient with a diagnostic or therapeutic purpose. Lab tests are not a procedure, if something is observed with an expected resulting amount and unit then it should be a measurement. \n" +\
                  "4) Devices include implantable objects (e.g. pacemakers, stents, artificial joints), medical equipment and supplies (e.g. bandages, crutches, syringes), other instruments used in medical procedures (e.g. sutures, defibrillators) and material used in clinical care (e.g. adhesives, body material, dental material, surgical material). \n" +\
                  "5) Measurement contains both orders and results of such Measurements as laboratory tests, vital signs, quantitative findings from pathology reports, etc. OBSERVATION captures clinical facts about a Person obtained in the context of examination, questioning or a procedure. Any data that cannot be represented by any other domains, such as social and lifestyle facts, medical history, family history, etc. are recorded here. \n" +\
                  "6) Observations differ from Measurements in that they do not require a standardized test or some other activity to generate clinical fact. Typical observations are medical history, family history, the stated need for certain treatment, social circumstances, lifestyle choices, healthcare utilization patterns, etc. \n" +\
                  "7) Demographic can include factors of patient such as age, gender, race, ethnicity, education level, income, occupation, geographic location, marital status, and family size. Age term can be demographic but the specific age criteria should be annotated as value. \n" +\
                  "8) Negation_cue includes all information that negates clinical concepts. \n" +\
                  "9) Value is the numeric value or string test result of clinical concepts. Typicall values can be the results of Measurements such as Lab test, vital signs, and quantitative findings from pathology reports. It can also be the dosage of drugs, the frequency of drugs, positive/negative of Gene test or lab test, the duration of drugs or numeric criteria of age, weight, height, etc."



def get_chatgpt_response(str):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": parse_prompt
            },
            {
                "role": "user",
                "content": str
            }
        ],
        temperature=0,
        max_tokens=9999,
        top_p=1
    )

    return response.choices[0].message.content
from urllib.parse import quote
import time


def get_concept_id(term="sterioad"):
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
        print(f"API Response: {data}")

        # Check if the response contains valid content
        if "content" in data and data["content"]:
            # Extract the first matching concept's details
            concept = data["content"][0]
            concept_id = concept.get("id", "N/A")
            concept_name = concept.get("name", "Unknown")
            domain_id = concept.get("domain", "Unknown")
            print(f"Found concept: {concept_name} (ID: {concept_id}, Domain: {domain_id})")
        else:
            concept_id = "N/A"
            print(f"No concept found for term: {term}")

        # Return the concept ID (or cache it if needed)
        return concept_id

    except requests.exceptions.RequestException as e:
        print(f"Error retrieving concept_id for '{term}': {e}")
        return "N/A"


def formulate_ner_result(text, gpt_results):
    terms = [{"text": term.strip()} for term in text.split() if term]  # Split text into terms
    negate_cues = []  # Placeholder for negation cues
    return terms, negate_cues


def trans4display(text, terms, concept_set):
    """
    Map terms to their respective concept IDs, assign concept sets if new, and generate display text.
    """
    display = text

    for term in terms:
        concept_id = ""
        name = ""
        term_text = term["text"]
        term_domain = term.get("domain", "Unknown")

        try:
            # Map the term to a concept ID and get the name using the external API
            res = get_concept_id(term_text)  # Assume this returns a tuple (concept_id, name)
            if isinstance(res, str):
                concept_id = res
            elif isinstance(res, tuple) and len(res) > 0:
                concept_id, name = res[0], res[1]

            # Assign the concept ID and name to the term
            term["concept_id"] = int(concept_id) if concept_id.isdigit() else concept_id
            term["name"] = name

            # Check if the term needs a new concept set
            if term_text + " " + concept_id not in concept_set:
                concept_set_id = len(concept_set) + 1
                concept_set[term_text + " " + concept_id] = concept_set_id
                term["vocabulary_id"] = concept_set_id
            else:
                term["vocabulary_id"] = concept_set[term_text + " " + concept_id]
        except Exception as e:
            print(f"Error processing term: {term_text}, Error: {e}")
            continue

        # Generate display logic
        if concept_id:  # If the term can be mapped to a concept
            display += (
                f"\n<mark data-entity=\"{term_domain.lower()}\" "
                f"concept-id=\"{concept_id}\" "
                f"vocabulary-id=\"{term['vocabulary_id']}\">"
                f"{term_text} <b><i>{name}</i></b>"
                f"</mark>"
            )
        else:  # If the term cannot be mapped
            display += f"\n<mark data-entity=\"{term_domain.lower()}\">{term_text}</mark>"

    return display


def translate_by_block_seg_ner_concept_mapping(text):
    if not text:
        return []

    spas = []
    paragraphs = text.split('\n')
    for p in paragraphs:
        if not p.strip():
            continue

        pa = {'sents': []}
        block_text = [sent.text for sent in nlp(p).sents]

        for s in block_text:
            gpt_results = get_chatgpt_response(s)

            # Remove tags
            s = re.sub(r"<([\s\S]*?)>|</([\s\S]*?)>", " ", s)
            s = re.sub(r" {2,}", " ", s)
            doc = nlp(" " + s + " ")

            # Extract terms and negation cues
            terms, negate_cues = formulate_ner_result(doc.text, gpt_results)

            # Display with dynamic concept mapping
            try:
                display = trans4display(doc.text, terms, concept_set)
            except Exception as ex:
                print(f"Error in trans4display: {ex}")
                display = ""

            sent = {'text': doc.text, 'terms': terms, 'display': display, 'negate_cues': negate_cues}
            pa['sents'].append(sent)

        spas.append(pa)
    return spas


def process_trial(trial_id, trial_path):
    time_str = datetime.now().strftime("%Y%m%d%H%M%S")

    # 读取 trial info 文件
    trial_info_path = osp.join(trial_path, "info")
    if not os.path.exists(trial_info_path):
        print(f"Info file not found for trial {trial_id}")
        return

    with open(trial_info_path, "r") as fin:
        trial_info = json.loads(fin.read())

    # 处理 InclusionCriteria
    parsed_in_criteria = []
    in_criteria = trial_info.get('InclusionCriteria', [])
    for in_criterion in in_criteria:
        response = get_chatgpt_response(in_criterion)
        parsed_in_criteria.append({'text': in_criterion, 'parsed_result': response})

    # 保存 InclusionCriteria 结果
    in_criteria_output_path = osp.join(trial_path, f"parsed_in_criteria_{time_str}.json")
    with open(in_criteria_output_path, "w") as fout:
        fout.write(json.dumps(parsed_in_criteria, indent=4))

    # 处理 ExclusionCriteria
    parsed_ex_criteria = []
    ex_criteria = trial_info.get('ExclusionCriteria', [])
    for ex_criterion in ex_criteria:
        response = get_chatgpt_response(ex_criterion)
        parsed_result = translate_by_block_seg_ner_concept_mapping(response)
        parsed_ex_criteria.append({'text': ex_criterion, 'parsed_result': parsed_result})

    # 保存 ExclusionCriteria 结果
    ex_criteria_output_path = osp.join(trial_path, f"parsed_ex_criteria_{time_str}.json")
    with open(ex_criteria_output_path, "w") as fout:
        fout.write(json.dumps(parsed_ex_criteria, indent=4))

    print(f"Processed trial {trial_id} successfully.")


# 遍历所有 trial 文件夹并处理
base_path = "data/trial_info"
trial_dirs = [d for d in os.listdir(base_path) if os.path.isdir(osp.join(base_path, d))]

# for trial_id in trial_dirs:
for trial_id in ['NCT03710187']:
    trial_path = osp.join(base_path, trial_id)
    print(f"Processing trial: {trial_id}")
    process_trial(trial_id, trial_path)

