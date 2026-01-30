import json
import os.path as osp
import os
from openai_client import OpenaiClient
from trialutils import extract_json_substring
from trial_classes import TrialEligibilityCriterion, TrialInfo, TrialOutcome, TrialTreatment
import pickle
import requests

from nlp_utils import extract_json_substring, extract_domain_text
from web_utils import get_trial_info, get_concept_id
from prompts import TR_SEL_PROMPT, EC_EXPORT_PROMPT, PARSE_PROMPT
from copy import deepcopy

def parse_clinical_trial(trial_id, trial_dir=None):
    client = OpenaiClient(api_key='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',
        organization='org-p7vlqGOYmscIzHdhbT7MJtEP',
        project='proj_6EKrvoAglwQOkOMLXTRd0SgD')

    if trial_dir is None:
        trial_dir = osp.join("results/trial_info", trial_id)
    os.makedirs(trial_dir, exist_ok=True)

    # pdf_file = osp.join(trial_dir, "webpage.pdf")
    # if not osp.exists(pdf_file):
    #     print(f"Export the trial information from the website and convert to PDF: {trial_id}")
    #     trial_url = "https://clinicaltrials.gov/study/{}".format(trial_id)
    #     save_webpage_as_pdf(trial_url, pdf_file)
    # else:
    #     print(f"PDF exists: {trial_id}")

    # info_file = osp.join(trial_dir, "info.json")
    # if not osp.exists(info_file):
    #     print(f"Upload the pdf and extract the components: {trial_id}")
    #     res = client.get_chatgpt_response_with_file(EXPORT_PROMPT, pdf_file)
    #     with open(info_file, "w") as fout:
    #         fout.write(extract_json_substring(res))
    # else:
    #     print(f"Info json exists: {trial_id}")

    # with open(info_file, "r") as fin:
        # trial_info = json.loads(fin.read())


    api_file = osp.join(trial_dir, "api_info.json")
    if not osp.exists(api_file):
        print(f"Export the trial information from the website and convert to PDF: {trial_id}")
        try:
            data = get_trial_info(trial_id)
            with open(api_file, "w") as fout:
                json.dump(data, fout, indent=4)
        except requests.exceptions.RequestException as e:
            print(f"Error retrieving concept_id for '{trial_id}': {e}")
            return
    else:
        print(f"API json exists: {trial_id}")


    info_file = osp.join(trial_dir, "info.json")
    if not osp.exists(info_file):
        with open(api_file, "r") as fin:
            api_trial_info = json.load(fin)
        trial_info = {}

        eligibility_criteria = api_trial_info.get('protocolSection', {}).get('eligibilityModule', {}).get(
            'eligibilityCriteria', "")
        response = client.get_chatgpt_response(EC_EXPORT_PROMPT, eligibility_criteria)
        ec_dict = json.loads(extract_json_substring(response))
        trial_info.update(ec_dict)

        tmp_treatments = []
        arms_interventions = api_trial_info.get('protocolSection', {}).get('armsInterventionsModule', {})

        if 'armGroups' in arms_interventions:
            for tmp_item in arms_interventions['armGroups']:
                tmp_item['Participant Group/Arm'] = "{}:{}\n{}".format(
                    tmp_item.get('type', ''), tmp_item.get('label', ''), tmp_item.get('description', '')
                )
                tmp_item['Intervention/Treatment'] = ""

                if 'interventions' in arms_interventions:
                    for tmp_inter in arms_interventions['interventions']:
                        if tmp_item['label'] in tmp_inter.get('armGroupLabels', []):
                            tmp_item['Intervention/Treatment'] = "{}:{}\n{}".format(
                                tmp_inter.get('type', ''), tmp_inter.get('name', ''), tmp_inter.get('description', '')
                            )
                            tmp_item['intervention_dict'] = tmp_inter

                tmp_treatments.append(tmp_item)

        trial_info['Treatments'] = tmp_treatments

        tmp_outcomes = []
        outcomes_module = api_trial_info.get('protocolSection', {}).get('outcomesModule', {})

        if 'primaryOutcomes' in outcomes_module:
            for outcome_item in outcomes_module['primaryOutcomes']:
                tmp_outcomes.append({
                    "Outcome Measure": outcome_item.get('measure', ''),
                    "Measure Description": outcome_item.get('description', ''),
                    "Time Frame": outcome_item.get('timeFrame', '')
                })
        trial_info['Primary Outcome Measures'] = tmp_outcomes

        tmp_outcomes = []
        if 'secondaryOutcomes' in outcomes_module:
            for outcome_item in outcomes_module['secondaryOutcomes']:
                tmp_outcomes.append({
                    "Outcome Measure": outcome_item.get('measure', ''),
                    "Measure Description": outcome_item.get('description', ''),
                    "Time Frame": outcome_item.get('timeFrame', '')
                })
        trial_info['Secondary Outcome Measures'] = tmp_outcomes

        with open(info_file, "w") as fout:
            json.dump(trial_info, fout, indent=4)
    else:
        print(f"Info json exists: {trial_id}")

    with open(info_file, "r") as fin:
        trial_info = json.load(fin)


    print(f"Preprocess the components: {trial_id}")
    key_name_dict = {"Inclusion Criteria": "in_criteria", "Exclusion Criteria": "ex_criteria",
                     "Primary Outcome Measures": "pr_outcomes", "Treatments": "treatments"}

    for trial_key in key_name_dict:
        parsed_file = osp.join(trial_dir, f"parsed_{key_name_dict[trial_key]}.json")

        if not osp.exists(parsed_file):
            parsed_list = []
            com_list = trial_info.get(trial_key, [])

            if trial_key in ["Inclusion Criteria", "Exclusion Criteria"]:
                for com_item in com_list:
                    response = client.get_chatgpt_response(PARSE_PROMPT, com_item)
                    parsed_list.append({'text': com_item, 'parsed_result': response})

            elif trial_key == "Treatments":
                for com_item in com_list:
                    com_text = com_item.get('description', '')

                    if 'intervention_dict' in com_item and 'description' in com_item['intervention_dict']:
                        additional_desc = com_item['intervention_dict']['description']
                        com_text = client.get_chatgpt_response(TR_SEL_PROMPT, json.dumps([com_text, additional_desc]))

                    response = client.get_chatgpt_response(PARSE_PROMPT, com_text)
                    parsed_list.append({'ori_dict': com_item, 'text': com_text, 'parsed_result': response})

            elif trial_key == "Primary Outcome Measures":
                for com_item in com_list:
                    com_text = com_item.get('Outcome Measure', '')
                    response = client.get_chatgpt_response(PARSE_PROMPT, com_text)
                    parsed_list.append({'ori_dict': com_item, 'text': com_text, 'parsed_result': response})

            with open(parsed_file, "w") as fout:
                json.dump(parsed_list, fout, indent=4)


        with open(parsed_file, "r") as fin:
            if trial_key == 'Inclusion Criteria':
                in_criteria = []
                with open(parsed_file, 'r') as fin:
                    in_list = json.loads(fin.read())
                for item in in_list:
                    in_criteria.append(TrialEligibilityCriterion(text=item['text']))
            elif trial_key == 'Exclusion Criteria':
                ex_criteria = []
                with open(parsed_file, 'r') as fin:
                    in_list = json.loads(fin.read())
                for item in in_list:
                    ex_criteria.append(TrialEligibilityCriterion(text=item['text']))
            elif trial_key == 'Primary Outcome Measures':
                with open(parsed_file, 'r') as fin:
                    in_list = json.loads(fin.read())
                pr_outcomes = [TrialOutcome(text=in_item['text']) for in_item in in_list]
            elif trial_key == 'Treatments':
                with open(parsed_file, 'r') as fin:
                    in_list = json.loads(fin.read())
                treatments = [TrialTreatment(text=in_item['text']) for in_item in in_list]
            else:
                raise NotImplementedError

        # query_file = osp.join(trial_dir, "query_{}.json".format(key_name_dict[trial_key]))
        #
        # if not osp.exists(query_file):
        #     query_list = []
        #     for parsed_item in parsed_list:
        #         query_item = deepcopy(parsed_item)
        #         parsed_tuples = extract_domain_text(query_item['parsed_result'])
        #         sel_tuples = []
        #         for domain, concept in parsed_tuples:
        #             if domain not in ['Value', 'Temporal']:
        #                 concept_data = get_concept_id(concept)
        #                 sel_tuples.append({"stand_name": concept_data['concept_name'],
        #                                    "ori_name": concept, "domain": domain, "concept_id":concept_data["concept_id"]})
        #         query_item['query_results'] = sel_tuples
        #         query_list.append(query_item)
        #         with open(query_file, "w") as fout:
        #             json.dump(query_list, fout, indent=4)

    trial_info = TrialInfo(inclusion_criteria=in_criteria,
                           exclusion_criteria=ex_criteria,
                           treatments=treatments,
                           outcomes=pr_outcomes)

    with open(osp.join(trial_dir, f"trial_info_{trial_id}.pkl"), "wb") as f:
        pickle.dump(trial_info, f)

if __name__ == "__main__":
    parse_clinical_trial('NCT03872011')









