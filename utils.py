from typing import Optional, List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
import numpy as np
import re
import copy
import pandas as pd
import requests

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

def split_documents(
    chunk_size: int,
    knowledge_base: List[Document],
    tokenizer_name: Optional[str] = 'sentence-transformers/all-MiniLM-L6-v2',
) -> List[Document]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique

def process_df(df):
    column_names = df.columns
    new_column_names = []
    for name in column_names:
        name = name.replace("\n", " ")
        new_column_names.append(name)
    df.columns = new_column_names
    return df

def add_space_around_period(s):
    # Add a space before the period if it is preceded by a letter and not already preceded by a space
    s = re.sub(r'(?<=[a-zA-Z])\.(?![ ])', r' .', s)
    # Add a space after the period if it is followed by a letter or space and not already followed by a space
    s = re.sub(r'\.(?=[a-zA-Z ])(?![ ])', r'. ', s)
    return s

def clean_text(text):
    text = text.replace("\n", " ")
    text = text.replace("_", " ")
    text = add_space_around_period(text)
    text = text.replace("  ", " ")
    text = text.replace(" . ", ". ")
    text = text.replace("e. g .", "e.g.")
    text = text.replace("i. e .", "i.e.")
    text = text.strip()
    return text

def clean_pandas_table(df):
    column_names = df.columns
    new_column_names = []
    for name in column_names:
        name = name.replace("\n", " ")
        new_column_names.append(name)
    df.columns = new_column_names

    for col in df:
        values_for_col = df[col].values 
        new_vals = []
        for val in values_for_col:
            if val == None:
                new_vals.append(val)
            else:
                if val == "":
                    new_vals.append(None)
                else:
                    val = val.replace("\n", " ")
                    val = val.replace("t=", "")
                    val = val.replace("0a", "0")
                    new_vals.append(val.replace("\n", " "))
        df[col] = new_vals
    df = df.dropna(axis=1, how='all')
    return df

def clean_table(markdown_df):
    elements = markdown_df.split('|')
    new_elements = []
    for element in elements:
        element = element.strip()
        new_elements.append(element)
    markdown_df = "|".join(new_elements)
    return markdown_df

def clean_text_table(text_table, table_header):
    table_prose = text_table[text_table.find("<Q1>")+4:]
    table_prose = table_prose[:table_prose.find("</Q1>")]
    table_summary = text_table[text_table.find("<Q2>")+4:]
    table_summary = table_summary[:table_summary.find("</Q2>")]
    table_prose = table_prose.replace("\n\n", " ")
    table_summary = table_summary.replace("\n\n", " ")
    table_prose = "<TABLE EXPLANATION> " + table_prose + " </TABLE EXPLANATION>"
    if "summary" in table_header.lower():
       table_summary = "<TABLE SUMMARY>" + "" + " </TABLE SUMMARY>"
    else:
        table_summary = "<TABLE SUMMARY>" + table_summary + " </TABLE SUMMARY>"
    table_text = table_summary + " " + table_prose
    return table_text

def clean_page_text(text):
    text = text.replace("\n\n", " \n\n ")
    pattern = re.compile('regeneron confidential', re.IGNORECASE)
    text = pattern.sub(' ', text)
    pattern = re.compile('regeneron - confidential', re.IGNORECASE)
    text = pattern.sub(' ', text)
    text = re.sub(' +', ' ', text)
    text = re.sub('samplesa', 'samplesb', text)
    text = text.strip()
    return text

def filter_chunks(chunks, minimum_sentence_length):
    output_chunks = []
    for chunk in chunks:
        flags = (chunk != "") and (len(chunk) > minimum_sentence_length)
        if flags:
            output_chunks.append(chunk)
    return output_chunks

def sort_by_y_then_x(points):
  return sorted(points, key=lambda p: (p[1], p[0]))

def distance(p1, p2):
    # This function calculates the Euclidean distance between two points
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def sort_by_distance(points, reference):
  # Create a list with tuples of (point, distance)
  distances = [(p, distance(p, reference)) for p in points]
  # Sort the list by distance (ascending order)
  distances.sort(key=lambda p: p[1])
  # Extract the sorted points from the list of tuples
  return [p[0] for p in distances]

def sort_blocks(final_blocks_for_page):
  sorted_final_blocks_for_page = sort_by_y_then_x(final_blocks_for_page.copy())
  sorted_points = sort_by_distance(sorted_final_blocks_for_page.copy(), sorted_final_blocks_for_page[0])
  return sorted_points

def check_if_table_is_good_pos(current_table_bbox, previous_bbox, current_bbox):
   sorted_positions = sort_by_distance([previous_bbox, current_bbox], current_table_bbox)
   if sorted_positions[0][1] < sorted_positions[1][1]:
      return True
   else:
      return False
   
def remove_spaces_from_patterns(text):
    # This pattern is designed to match:
    # 1. A single letter followed by an optional space, a period, and an optional space.
    # 2. A number followed by an optional space, a period, and an optional space.
    # 3. A sequence that can be either a number or multiple letters.
    # The pattern ensures that it starts with a single letter to avoid matching longer words.
    pattern = r'\b([a-zA-Z])\s*\.\s*(\d+)\.\s*([a-zA-Z]+|\d+)\b'
    # This function will be used to replace the matched object m with a string where the spaces are removed
    def replacer(m):
        return f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
    # Using re.sub() to replace the occurrences found in the text with the pattern
    result = re.sub(pattern, replacer, text)
    return result

def remove_linebreaks_and_spaces(text):
    # This pattern matches: a substring followed by a hyphen, a space, a linebreak, a space, and another substring
    pattern = r'(\S)- \n (\S)'
    # This function will be used to replace the matched object m with a string where the linebreak and spaces are removed
    def replacer(m):
        return f"{m.group(1)}-{m.group(2)}"
    # Using re.sub() to replace the occurrences found in the text with the pattern
    result = re.sub(pattern, replacer, text)
    return result

def get_table_ref_names(other_doc_name, sent):
    # Getting the table that is referenced in the sentence of the prose+table
    substring = other_doc_name
    text = sent
    pattern_segment = re.escape(substring) + "(.*)"
    segment_match = re.search(pattern_segment, text, re.DOTALL)
    if segment_match:
        segment = segment_match.group(1)
        # Now find all "Table X" mentions in the segment
        pattern_tables = r"Table \d+"
        tables = re.findall(pattern_tables, segment)
    else:
        tables = []
    return tables

def get_tokens_with_digits(text):
  """
  This function takes a string as input and returns a list of all tokens 
  in the string that contain at least one digit.

  Args:
      text: The input string.

  Returns:
      A list of tokens containing digits.
  """
  tokens_with_digits = []
  for token in text.split():
    if any(char.isdigit() for char in token):
      tokens_with_digits.append(token)
  return tokens_with_digits

def get_numeric_tokens(page_strings):
  numeric_tokens_per_page = []
  for page_string in page_strings:
    numeric_tokens = get_tokens_with_digits(page_string)
    final_numeric_tokens = []
    for token in numeric_tokens:
      if token.count(".") >= 2:
        continue
      possible_tokens = token.split("/")
      for t in possible_tokens:
        if any(char.isdigit() for char in t):
          token = t
          break
      if "REGN" in token:
        continue
      token = token.strip(",")
      token = token.strip(".")
      token = token.strip("|")
      token = token.strip(")")
      token = token.strip("(")
      token = token.strip(";")
      token = token.strip(":")
      token = token.strip("\"")
      token = token.strip("\'")
      token = token.strip()
      final_numeric_tokens.append(token)
    numeric_tokens_per_page.append(final_numeric_tokens)
  return numeric_tokens_per_page

def determine_if_find_contradiction(sent, numeric_words):
    should_find_contradiction_for_claim = False
    sent = sent.replace("-", "")
    sent = sent.replace("/", "")
    pattern = r"Table \d+"
    sent = re.sub(pattern, "", sent)
    pattern = r"Figure \d+"
    sent = re.sub(pattern, "", sent)
    sent = re.sub(r'\b8\w{6,}\b', '', sent)
    sent = re.sub(r'\bR\w*\b', "", sent)
    sent = re.sub(r"\bREGN\w+", "", sent)
    sent = re.sub(r"\bRS\w+", "", sent)
    sent = re.sub(r"\bL\d+\b", "", sent)
    sent = re.sub(r"\b[A-Z]\.\d+\.\d+\b", "", sent)

    if any(char.isdigit() for char in sent):
        should_find_contradiction_for_claim = True

    for numeric_word in numeric_words:
        if numeric_word in sent.lower().split():
            should_find_contradiction_for_claim = True
    return should_find_contradiction_for_claim

def create_extraction_components(doc, keep_pages=None, remove_pages=None, run_llms=0, remove_tables=True):
    # Getting all tables and blocks of text
    blocks_for_pages = []
    tables_for_pages = []
    table_bboxs_for_pages = []
    num_tables_per_page = []
    bool_table_on_page = []
    converted_markdown_tables_and_headers = []

    blocks_for_pages = []
    for page in doc:
        text = page.get_text("blocks")
        tabs = page.find_tables()
        page_pandas_tables = []
        page_table_bboxs = []
        if len(tabs.tables) > 0:
            bool_table_on_page.append(True)
            num_tables_per_page.append(len(tabs.tables))
            for table in tabs.tables:
                pandas_table = table.to_pandas()
                table_bbox = table.bbox
                page_pandas_tables.append(pandas_table)
                page_table_bboxs.append(table_bbox)
        else:
            bool_table_on_page.append(False)
            num_tables_per_page.append(0)
            page_pandas_tables.append([])
            page_table_bboxs.append([])
        blocks_for_pages.append(text)
        tables_for_pages.append(page_pandas_tables)
        table_bboxs_for_pages.append(page_table_bboxs)

    # Removing text if it overlaps with tables
    final_blocks_for_pages = []
    for page_num in range(0, len(blocks_for_pages)):
        if num_tables_per_page[page_num] == 0:
            final_blocks_for_pages.append(blocks_for_pages[page_num])
            continue
        table_bounding_boxes = []
        for table_bbox in table_bboxs_for_pages[page_num]:
            table_bounding_boxes.append(table_bbox)
        blocks_in_page = copy.deepcopy(blocks_for_pages[page_num])
        index_elements_to_delete = []
        for table_bounding_box in table_bounding_boxes:
            for i, blocks in enumerate(blocks_in_page):
                if ((blocks[0] >= table_bounding_box[0]) and (blocks[1] >= table_bounding_box[1])) and ((blocks[2] <= table_bounding_box[2]) and (blocks[3] <= table_bounding_box[3])):
                    index_elements_to_delete.append(i)
        final_blocks_in_page = []
        for j in range(0, len(blocks_in_page)):
            if j not in index_elements_to_delete:
                final_blocks_in_page.append(blocks_in_page[j])
        final_blocks_for_pages.append(final_blocks_in_page)

        
    # Putting all text in order
    page_texts = []
    only_prose_page_texts = []
    uncleaned_page_texts = []
    all_table_headers = []
    previous_page_texts = None
    for page_num in range(0, len(final_blocks_for_pages)):
        has_table = bool_table_on_page[page_num]
        if has_table:
            table_index = 0
        texts = []
        only_texts = []
        uncleaned_texts = []
        previous_text = None
        if len(final_blocks_for_pages[page_num]) == 0:
            page_texts.append([])
            uncleaned_page_texts.append([])
            only_prose_page_texts.append([])
            continue
        sorted_final_blocks_for_page = sort_blocks(final_blocks_for_pages[page_num].copy())
        for l, text_block in enumerate(sorted_final_blocks_for_page):
            if has_table:
                page_tables = tables_for_pages[page_num]
                page_table_bboxs = table_bboxs_for_pages[page_num]
                current_earliest_table = page_tables[table_index]
                current_table_bbox = page_table_bboxs[table_index]
                if (l!= 0) and check_if_table_is_good_pos(current_table_bbox, sorted_final_blocks_for_page[l-1], text_block):
                    # Getting previous text for table header
                    if (previous_text != None) and ("table" in previous_text.lower()):
                        table_header = previous_text
                    elif previous_page_texts != None:
                            for s in reversed(previous_page_texts):
                                if "table" in s.lower():
                                    table_header = s
                                    break
                    else:
                        table_header = None
                    all_table_headers.append(table_header)
                    table_df = clean_pandas_table(current_earliest_table)
                    markdown_table = table_df.to_markdown(index=False)
                    markdown_table = clean_text(markdown_table)
                    markdown_table = clean_table(markdown_table)
                    study_nos = []
                    df = table_df
                    if "Table" in df.columns[0]:
                        new_header = df.iloc[0] #grab the first row for the header
                        df = df[1:] #take the data less the header row
                        df.columns = new_header
                    # Check if there is a column for study numbers
                    if 'Study No.' in df.columns:
                        study_nos.extend(df['Study No.'].dropna().values)
                    if 'Study Number' in df.columns:
                        study_nos.extend(df['Study Number'].dropna().values)
                    # Also check if any of the columns themselves are study numbers
                    ## Two potential manners
                    cols = df.columns
                    for col in cols:
                        if col != None:
                            delimited_col_ents = col.split("-")
                            if len(delimited_col_ents)==3 and delimited_col_ents[0][0] == "R" and delimited_col_ents[1][0] == "R":
                                study_nos.append(col)
                            if len(delimited_col_ents)==2 and delimited_col_ents[0][0:4] == "REGN" and delimited_col_ents[1][0:2] == "SS":
                                study_nos.append(col)
                    converted_markdown_tables_and_headers.append([table_df, markdown_table, table_header, study_nos])
                    if run_llms:
                        print("Using LLM for table conversion")
                        input_data = markdown_table
                        table_conversion_prompt = f" \
                            # Instruction  \
                            You are an expert at converting tables to text without leaving any detail behind. All the information within each cell of the dataframe needs to be represented in the text conversion, with each cell being individually specified. You will also provide a summary the includes some statements concerning general analysis of the table. If there is temporal information in the table, describe what time points there is currently data available for. Also describe the TOTAL time data can be gathered for. The time data is currently available for may be different from the total time as there may be some missing data due to incompletion. \
                            - **Goal**: Convert the given markdown table to a text format, where the output is a paragraph. \
                            - **Data**: \
                            - table_name: Name and title of the table \
                            - table: A markdown table containing both textual and numerical cells. \
                            \
                            # Data \
                            table_name: {table_header} \
                            table: {input_data} \
                            \
                            # Questions \
                            ## Q1: Convert the input markdown table into a text of paragraph where each cell is explicitly represented. \
                            Tips \
                            - Be concise and clear. Do not add phrases like “This is..” or “Conclusion…”. \
                            - Your output will be used for downstream analysis and each cell needs to be represented accurately for the analysis to work. \
                            - Do not list multiple values within one sentence even if they all follow a similar pattern. For example, do not do something like “Other conditions include X1, X2, X3 for Y1, Y2, Y3 respectively…” \
                            - For sections of the table which present numerical values for an attribute across time, add statements on whether the values increase or decrease over time. \
                            - Include all information provided in the table in your output. Do not summarize a group of cells to make a shorter output, without including specific information from each cell.  \
                            - Provide your answer in **English** only. \
                            ## Q2: Provide a summary of what the table presents overall using a maximum of 50 words. \
                            Tips \
                            - Use the table_title as well as the answer to Q1 to provide context of what the table is showing \
                            - If there is temporal information in the table, add two statements to the summary: \
                            -- Describe what time points there is currently data available for. \
                            -- Describe the TOTAL time data can be gathered for. \
                            - The time data is currently available for may be different from the total time as there may be some missing data due to incompletion. \
                            - Be concise and clear. Do not add phrases like “This is..” or “Conclusion…”. \
                            - Provide your answer in **English** only. \
                            ## Provide your answers between the tags: <Q1>your answer to Q1</Q1><Q2>your answer to Q2</Q2> \
                            \
                            # Output \
                            "
                        # text_table = llm.predict(table_conversion_prompt)
                        # table_text = clean_text_table(text_table, table_header)
                        # texts.append(table_text)
                        # uncleaned_texts.append(table_text)
                    else:
                        if not remove_tables:
                            texts.append(markdown_table)
                            uncleaned_texts.append(markdown_table)
                    table_index += 1
                    if table_index >= num_tables_per_page[page_num]:
                        has_table = 0
            initial_text = text_block[4]
            cleaned_text = clean_text(initial_text)
            if len(cleaned_text) != 0:
                texts.append(cleaned_text)
                only_texts.append(cleaned_text)
                uncleaned_texts.append(initial_text)
            previous_text = cleaned_text
        page_texts.append(texts)
        uncleaned_page_texts.append(uncleaned_texts)
        only_prose_page_texts.append(only_texts)
        previous_page_texts = only_texts

    # Combining all pages
    page_strings = []
    uncleaned_page_strings = []
    for i, page_text in enumerate(page_texts):
        page_string = " \n ".join(page_text)
        page_string = clean_page_text(page_string)
        page_strings.append(page_string)
        uncleaned_page_strings.append(" \n ".join(uncleaned_page_texts[i]))

    if keep_pages != None:
        kept_page_strings = []
        uncleaned_kept_page_strings = []
        keep_pages_array = np.array(keep_pages) - 1
        for i, page_string in enumerate(page_strings):
            if i in keep_pages_array:
                kept_page_strings.append(page_string)
                uncleaned_kept_page_strings.append(uncleaned_page_strings[i])
        page_strings = kept_page_strings
        uncleaned_page_strings = uncleaned_kept_page_strings

    if remove_pages != None:
        kept_page_strings = []
        uncleaned_kept_page_strings = []
        remove_pages_array = np.array(remove_pages) - 1
        for i, page_string in enumerate(page_strings):
            if i not in remove_pages_array:
                kept_page_strings.append(page_string)
                uncleaned_kept_page_strings.append(uncleaned_page_strings[i])
        page_strings = kept_page_strings
        uncleaned_page_strings = uncleaned_kept_page_strings

    doc_string = " \n ".join(page_strings)
    return doc_string, page_strings, uncleaned_page_strings, all_table_headers, converted_markdown_tables_and_headers


def custom_python_executor(code: str, external_vars: dict = None, extract_vars: list = None):
    """
    Executes Python code with access to external variables and extracts specified variables.
    
    Parameters:
    - code (str): The Python code to execute.
    - external_vars (dict): A dictionary of external variables to include in the execution context.
    - extract_vars (list): A list of variable names to extract from the local scope.
    
    Returns:
    - A dictionary containing the requested variables, or an error message if execution fails.
    """
    global_variables = {}
    local_variables = external_vars if external_vars else {}

    try:
        # Execute the code
        exec(code, global_variables, local_variables)

        # Merge global and local variables
        all_variables = {**global_variables, **local_variables}

        # Extract the requested variables
        if extract_vars:
            return {var: all_variables.get(var, None) for var in extract_vars}
        else:
            return all_variables  # Return all variables if none are specified
    except Exception as e:
        return f"Error during execution: {str(e)}"


# Function to flatten nested dictionaries
def flatten_dict(d, parent_key='', sep='_'):
    """
    Recursively flattens a nested dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Handle lists (convert to a string or process further)
            items.append((new_key, ', '.join(map(str, v)) if all(isinstance(i, str) for i in v) else str(v)))
        else:
            items.append((new_key, v))
    return dict(items)

def download(disease, treatment, other_keywords=None, outcome=None):
    # combine the keywords with AND
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    params = {"query.cond":disease,  "query.intr": treatment, "format": "json", "pageSize": 1000}
    if other_keywords:
        params["query.term"] = other_keywords
    if outcome:
        params["query.outc"] = outcome
    params['sort'] = '@relevance'
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def get_clinical_trials_data(essex_disease_str, essex_trt_str, other_keywords_str=None, outcome=None):
    # Download the clinical trials
    result = download(essex_disease_str, essex_trt_str, other_keywords=other_keywords_str, outcome=outcome)
    clinical_data = result['studies']

    # Flatten each study and collect the results
    flattened_data = [flatten_dict(trial) for trial in clinical_data]
    # Convert to a DataFrame
    df_clinical = pd.DataFrame(flattened_data)

    column_rename_map = {
        'protocolSection_identificationModule_nctId': 'NCT ID',
        'protocolSection_identificationModule_orgStudyIdInfo_id': 'Organization Study ID',
        'protocolSection_identificationModule_organization_fullName': 'Organization Full Name',
        'protocolSection_identificationModule_organization_class': 'Organization Class',
        'protocolSection_identificationModule_briefTitle': 'Brief Title',
        'protocolSection_identificationModule_officialTitle': 'Official Title',
        'protocolSection_identificationModule_acronym': 'Study Acronym',
        'protocolSection_statusModule_statusVerifiedDate': 'Status Verified Date',
        'protocolSection_statusModule_overallStatus': 'Overall Status',
        'protocolSection_statusModule_expandedAccessInfo_hasExpandedAccess': 'Expanded Access Available',
        'protocolSection_statusModule_startDateStruct_date': 'Start Date',
        'protocolSection_statusModule_startDateStruct_type': 'Start Date Type',
        'protocolSection_statusModule_primaryCompletionDateStruct_date': 'Primary Completion Date',
        'protocolSection_statusModule_primaryCompletionDateStruct_type': 'Primary Completion Date Type',
        'protocolSection_statusModule_studyFirstSubmitDate': 'Study First Submit Date',
        'protocolSection_statusModule_studyFirstSubmitQcDate': 'Study First Submit QC Date',
        'protocolSection_statusModule_studyFirstPostDateStruct_date': 'Study First Post Date',
        'protocolSection_statusModule_studyFirstPostDateStruct_type': 'Study First Post Date Type',
        'protocolSection_statusModule_lastUpdateSubmitDate': 'Last Update Submit Date',
        'protocolSection_statusModule_lastUpdatePostDateStruct_date': 'Last Update Post Date',
        'protocolSection_statusModule_lastUpdatePostDateStruct_type': 'Last Update Post Date Type',
        'protocolSection_sponsorCollaboratorsModule_responsibleParty_type': 'Responsible Party Type',
        'protocolSection_sponsorCollaboratorsModule_leadSponsor_name': 'Lead Sponsor Name',
        'protocolSection_sponsorCollaboratorsModule_leadSponsor_class': 'Lead Sponsor Class',
        'protocolSection_oversightModule_oversightHasDmc': 'Has Data Monitoring Committee',
        'protocolSection_descriptionModule_briefSummary': 'Brief Summary',
        'protocolSection_conditionsModule_conditions': 'Conditions',
        'protocolSection_designModule_studyType': 'Study Type',
        'protocolSection_designModule_phases': 'Study Phases',
        'protocolSection_designModule_designInfo_allocation': 'Allocation Type',
        'protocolSection_designModule_designInfo_interventionModel': 'Intervention Model',
        'protocolSection_designModule_designInfo_maskingInfo_masking': 'Masking Type',
        'protocolSection_designModule_enrollmentInfo_count': 'Enrollment Count',
        'protocolSection_designModule_enrollmentInfo_type': 'Enrollment Type',
        'protocolSection_armsInterventionsModule_armGroups': 'Arm Groups',
        'protocolSection_armsInterventionsModule_interventions': 'Interventions',
        'protocolSection_outcomesModule_primaryOutcomes': 'Primary Outcomes',
        'protocolSection_eligibilityModule_eligibilityCriteria': 'Eligibility Criteria',
        'protocolSection_eligibilityModule_healthyVolunteers': 'Healthy Volunteers Allowed',
        'protocolSection_eligibilityModule_sex': 'Eligible Sex',
        'protocolSection_eligibilityModule_minimumAge': 'Minimum Age',
        'protocolSection_eligibilityModule_maximumAge': 'Maximum Age',
        'protocolSection_eligibilityModule_stdAges': 'Standard Ages',
        'protocolSection_contactsLocationsModule_locations': 'Locations',
        'derivedSection_miscInfoModule_versionHolder': 'Version Holder',
        'derivedSection_conditionBrowseModule_meshes': 'Condition Mesh Terms',
        'derivedSection_conditionBrowseModule_ancestors': 'Condition Ancestors',
        'derivedSection_conditionBrowseModule_browseLeaves': 'Condition Browse Leaves',
        'derivedSection_conditionBrowseModule_browseBranches': 'Condition Browse Branches',
        'hasResults': 'Has Results',
        'protocolSection_statusModule_completionDateStruct_date': 'Completion Date',
        'protocolSection_statusModule_completionDateStruct_type': 'Completion Date Type',
        'protocolSection_sponsorCollaboratorsModule_responsibleParty_investigatorFullName': 'Responsible Investigator Full Name',
        'protocolSection_sponsorCollaboratorsModule_responsibleParty_investigatorTitle': 'Responsible Investigator Title',
        'protocolSection_sponsorCollaboratorsModule_responsibleParty_investigatorAffiliation': 'Responsible Investigator Affiliation',
        'protocolSection_oversightModule_isFdaRegulatedDrug': 'FDA Regulated Drug',
        'protocolSection_oversightModule_isFdaRegulatedDevice': 'FDA Regulated Device',
        'protocolSection_descriptionModule_detailedDescription': 'Detailed Description',
        'protocolSection_designModule_patientRegistry': 'Patient Registry',
        'protocolSection_designModule_designInfo_observationalModel': 'Observational Model',
        'protocolSection_designModule_designInfo_timePerspective': 'Time Perspective',
        'protocolSection_designModule_bioSpec_retention': 'Biospecimen Retention',
        'protocolSection_designModule_bioSpec_description': 'Biospecimen Description',
        'protocolSection_outcomesModule_secondaryOutcomes': 'Secondary Outcomes',
        'protocolSection_eligibilityModule_studyPopulation': 'Study Population',
        'protocolSection_eligibilityModule_samplingMethod': 'Sampling Method',
        'protocolSection_contactsLocationsModule_overallOfficials': 'Overall Officials',
        'protocolSection_ipdSharingStatementModule_ipdSharing': 'IPD Sharing Statement',
        'derivedSection_interventionBrowseModule_browseLeaves': 'Intervention Browse Leaves',
        'derivedSection_interventionBrowseModule_browseBranches': 'Intervention Browse Branches',
        'protocolSection_identificationModule_secondaryIdInfos': 'Secondary ID Infos',
        'protocolSection_sponsorCollaboratorsModule_responsibleParty_oldNameTitle': 'Responsible Party Name Title',
        'protocolSection_sponsorCollaboratorsModule_collaborators': 'Collaborators',
        'protocolSection_conditionsModule_keywords': 'Condition Keywords',
        'protocolSection_designModule_designInfo_primaryPurpose': 'Primary Purpose',
        'protocolSection_referencesModule_references': 'References'
    }
    additional_column_rename_map = {
        'protocolSection_designModule_designInfo_interventionModelDescription': 'Intervention Model Description',
        'protocolSection_contactsLocationsModule_centralContacts': 'Central Contacts',
        'protocolSection_designModule_designInfo_maskingInfo_maskingDescription': 'Masking Description',
        'protocolSection_designModule_designInfo_maskingInfo_whoMasked': 'Who Masked',
        'derivedSection_interventionBrowseModule_meshes': 'Intervention Mesh Terms',
        'derivedSection_interventionBrowseModule_ancestors': 'Intervention Ancestors',
        'protocolSection_ipdSharingStatementModule_description': 'IPD Sharing Description',
        'protocolSection_statusModule_resultsFirstSubmitDate': 'Results First Submit Date',
        'protocolSection_statusModule_resultsFirstSubmitQcDate': 'Results First Submit QC Date',
        'protocolSection_statusModule_resultsFirstPostDateStruct_date': 'Results First Post Date',
        'protocolSection_statusModule_resultsFirstPostDateStruct_type': 'Results First Post Date Type',
        'resultsSection_participantFlowModule_groups': 'Participant Flow Groups',
        'resultsSection_participantFlowModule_periods': 'Participant Flow Periods',
        'resultsSection_baselineCharacteristicsModule_groups': 'Baseline Groups',
        'resultsSection_baselineCharacteristicsModule_denoms': 'Baseline Denominators',
        'resultsSection_baselineCharacteristicsModule_measures': 'Baseline Measures',
        'resultsSection_outcomeMeasuresModule_outcomeMeasures': 'Outcome Measures',
        'resultsSection_adverseEventsModule_frequencyThreshold': 'Adverse Events Frequency Threshold',
        'resultsSection_adverseEventsModule_timeFrame': 'Adverse Events Time Frame',
        'resultsSection_adverseEventsModule_eventGroups': 'Adverse Event Groups',
        'resultsSection_adverseEventsModule_seriousEvents': 'Serious Adverse Events',
        'resultsSection_adverseEventsModule_otherEvents': 'Other Adverse Events',
        'resultsSection_moreInfoModule_certainAgreement_piSponsorEmployee': 'PI Sponsor Employee Agreement',
        'resultsSection_moreInfoModule_certainAgreement_restrictiveAgreement': 'Restrictive Agreement',
        'resultsSection_moreInfoModule_pointOfContact_title': 'Point of Contact Title',
        'resultsSection_moreInfoModule_pointOfContact_organization': 'Point of Contact Organization',
        'resultsSection_moreInfoModule_pointOfContact_email': 'Point of Contact Email',
        'resultsSection_moreInfoModule_pointOfContact_phone': 'Point of Contact Phone',
        'protocolSection_statusModule_lastKnownStatus': 'Last Known Status',
        'protocolSection_oversightModule_isUsExport': 'Is US Export',
        'protocolSection_ipdSharingStatementModule_infoTypes': 'IPD Info Types',
        'protocolSection_ipdSharingStatementModule_timeFrame': 'IPD Time Frame',
        'protocolSection_ipdSharingStatementModule_accessCriteria': 'IPD Access Criteria',
        'protocolSection_statusModule_whyStopped': 'Reason Stopped',
        'resultsSection_baselineCharacteristicsModule_populationDescription': 'Population Description',
        'resultsSection_adverseEventsModule_description': 'Adverse Events Description',
        'resultsSection_moreInfoModule_limitationsAndCaveats_description': 'Limitations and Caveats',
        'protocolSection_eligibilityModule_genderBased': 'Gender Based',
        'protocolSection_eligibilityModule_genderDescription': 'Gender Description',
        'protocolSection_outcomesModule_otherOutcomes': 'Other Outcomes',
        'documentSection_largeDocumentModule_largeDocs': 'Large Documents',
        'derivedSection_miscInfoModule_submissionTracking_firstMcpInfo_postDateStruct_date': 'First MCP Post Date',
        'derivedSection_miscInfoModule_submissionTracking_firstMcpInfo_postDateStruct_type': 'First MCP Post Date Type',
        'protocolSection_designModule_targetDuration': 'Target Duration',
        'protocolSection_referencesModule_seeAlsoLinks': 'See Also Links',
        'protocolSection_sponsorCollaboratorsModule_responsibleParty_oldOrganization': 'Responsible Party Old Organization',
        'derivedSection_miscInfoModule_removedCountries': 'Removed Countries',
        'resultsSection_participantFlowModule_recruitmentDetails': 'Recruitment Details',
        'annotationSection_annotationModule_unpostedAnnotation_unpostedResponsibleParty': 'Unposted Responsible Party',
        'annotationSection_annotationModule_unpostedAnnotation_unpostedEvents': 'Unposted Events',
        'derivedSection_miscInfoModule_submissionTracking_estimatedResultsFirstSubmitDate': 'Estimated Results First Submit Date',
        'derivedSection_miscInfoModule_submissionTracking_submissionInfos': 'Submission Infos',
        'resultsSection_participantFlowModule_preAssignmentDetails': 'Pre-Assignment Details',
        'protocolSection_ipdSharingStatementModule_url': 'IPD Sharing URL'
    }
    column_rename_map.update(additional_column_rename_map)

    # Rename columns in the DataFrame
    df_clinical.rename(columns=column_rename_map, inplace=True)

    relevant_columns = ['NCT ID', 'Brief Title', 'Official Title', 'FDA Regulated Drug', 'FDA Regulated Device',
                            'Detailed Description', 'Conditions', 'Condition Keywords', 'Study Type',
                            'Observational Model', 'Time Perspective',
                            'Biospecimen Description', 'Enrollment Count', 'Enrollment Type', 'Arm Groups',
                            'Primary Outcomes', 'Secondary Outcomes', 'Eligibility Criteria',
                            'Eligible Sex', 'Minimum Age', 'Maximum Age',
                            'Standard Ages', 'Study Population', 'Sampling Method',
                            'Interventions', 'Study Acronym', 'Study Phases', 'Allocation Type', 'Intervention Model',
                            'Intervention Model Description', 'Primary Purpose', 'Baseline Groups', 'Baseline Denominators', 'Baseline Measures',
                            'Outcome Measures', 'Adverse Events Frequency Threshold', 'Adverse Events Time Frame',
                            'Adverse Event Groups', 'Serious Adverse Events', 'Other Adverse Events', 'Reason Stopped',
                            'Population Description', 'Adverse Events Description',
                            'Limitations and Caveats', 'Gender Based', 'Gender Description', 'Other Outcomes', 'Target Duration']
    relevant_columns = list(set(relevant_columns) & set(df_clinical.columns))
    df_clinical = df_clinical[relevant_columns]
    # create string for each row, with <Column Name>: <Value> for each column, if nan, skip
    def create_string(row):
        string = ''
        for column in df_clinical.columns:
            if pd.notna(row[column]):
                string += f'{column}: {row[column]}\n'
        return string

    df_clinical['string'] = df_clinical.apply(create_string, axis=1)
    return df_clinical