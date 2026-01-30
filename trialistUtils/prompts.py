TR_SEL_PROMPT = "Select an item from the given list which include the most detailed information of the treatment which contain dosage, frequency information. Only reply the selected item."

EXPORT_PROMPT = "Extract the components of a clinical trial from the uploaded files using the following rules:\n" + \
                "1) Return a dictionary formatted as a json string, the keys in the dictionary of the are the names of the components.\n" + \
                "2) If a component contains multiple items, organize them as a list.\n" + \
                "3) Ignore the components which can not be found in a file.\n" + \
                "Following re the components to be extracted and specific format requirements:\n" + \
                "1) Inclusion Criteria.\n" + \
                "2) Exclusion Criteria.\n" + \
                "3) Primary Outcome Measures: Organize each measure with a dictionary with the keys as the columns names of the corresponding table.\n" + \
                "4) Secondary Outcome Measures: Organize each measure with a dictionary with the keys as the columns names of the corresponding table.\n" + \
                "5) Treatment: Organize each treatment with a dictionary with the keys as the columns names of the corresponding table."

EC_EXPORT_PROMPT = "Extract the components of a clinical trial from the given text using the following rules:\n" + \
                    "1) Return a dictionary formatted as a json string, the keys in the dictionary of the are the names of the components.\n" + \
                    "2) If a component contains multiple items, organize them as a list.\n" + \
                    "3) Ignore the components which can not be found in a file.\n" + \
                    "4) A line started by a single space “ ” is not a new item but part of the item in the previous line.\n" + \
                    "Following is the components to be extracted and specific format requirements:\n" + \
                    "1) Inclusion Criteria.\n" + \
                    "2) Exclusion Criteria."

PARSE_PROMPT =  "Annotate clinical concepts from the given text using the following rules: \n" + \
                  "1) Annotate concepts with the domains 'Demographic', 'Condition', 'Device', 'Procedure', 'Drug', 'Measurement', 'Observation', 'Visit', 'Value', 'Negation_cue', 'Temporal', and 'Quantity'. If you cannot annotate with the given domains, you can name a new one (e.g., Drug_cycle, Visit, Provider, etc.). \n" + \
                  "2) Split the concepts as detail as possible. Each concept can be annotated only once with a single domain. \n" + \
                  "3) Normalize clinical abbreviation and acronyms and attached behind the original abbreviation with parenthesis. \n" + \
                  "4) Return your response under [Annotation] section. \n" + \
                  "Following is not allowed examples: \n" + \
                  "1) <Measurement>EGFR <Value>triple postive</Value></Measurement> \n" + \
                  "2) <Condition>Hypertension, diabetes, heart failure, and dementia</Condition> \n" + \
                  "3) <Observation>allergy</Observation> to <Drug>X</Drug>, <Drug>Y</Drug>, or <Drug>Z</Drug> \n" + \
                  "Below is allowed examples: \n" + \
                  "1) <Measurement>EGFR</Measurement> <Value>triple positive</Value> \n" + \
                  "2) <Condition>hypertension</Condition>, <Condition>T2DM (Type 2 Diabetes Mellitus)</Condition>, <Condition>heart failure</Condition>, and <Condition>dementia</Condition> \n" + \
                  "3) Patient <Demographic>aged<Demographic> > <Value>65 years old</Value> \n" + \
                  "4) <Drug>Metformin</Drug> <Value>500 mg</Value> <Temporal>daily</Temporal> \n" + \
                  "5) <Observation>allergy to X</Observation>, <Observation>allergy to Y</Observation>, or <Observation>allergy to Z</Observation>  \n" + \
                  "Following is information for each domain: \n" + \
                  "1) Condition is events of a Person suggesting the presence of a disease or medical condition stated as a diagnosis, a sign, or a symptom, which is either observed by a Provider or reported by the patient. \n" +\
                  "2) Drugs include prescription and over-the-counter medicines, vaccines, and large-molecule biologic therapies. Radiological devices ingested or applied locally do not count as Drugs. \n" +\
                  "3) Procedure is records of activities or processes ordered by, or carried out by, a healthcare provider on the patient with a diagnostic or therapeutic purpose. Lab tests are not a procedure, if something is observed with an expected resulting amount and unit then it should be a measurement. \n" +\
                  "4) Devices include implantable objects (e.g. pacemakers, stents, artificial joints), medical equipment and supplies (e.g. bandages, crutches, syringes), other instruments used in medical procedures (e.g. sutures, defibrillators) and material used in clinical care (e.g. adhesives, body material, dental material, surgical material). \n" +\
                  "5) Measurement contains both orders and results of such Measurements as laboratory tests, vital signs, quantitative findings from pathology reports, etc. OBSERVATION captures clinical facts about a Person obtained in the context of examination, questioning or a procedure. Any data that cannot be represented by any other domains, such as social and lifestyle facts, medical history, family history, etc. are recorded here. \n" +\
                  "6) Observations differ from Measurements in that they do not require a standardized test or some other activity to generate clinical fact. Typical observations are medical history, family history, the stated need for certain treatment, social circumstances, lifestyle choices, healthcare utilization patterns, etc. \n" +\
                  "7) Demographic can include factors of patient such as age, gender, race, ethnicity, education level, income, occupation, geographic location, marital status, and family size. Age term can be demographic but the specific age criteria should be annotated as value. Demographic term only includes the above factors, the word as 'patients' or 'patient' should not be annotated. \n" +\
                  "8) Some demographic factors are not explicitly included in the text, such as 'patients who are at least 18 years old' or 'less than 18 years old'. In such cases, the factor 'age' should be additional annotated. \n" +\
                  "9) Negation_cue includes all information that negates clinical concepts. \n" +\
                  "10) Value is the numeric value or string test result of clinical concepts. Typicall values can be the results of Measurements such as Lab test, vital signs, and quantitative findings from pathology reports. It can also be the dosage of drugs, the frequency of drugs, positive/negative of Gene test or lab test, the duration of drugs or numeric criteria of age, weight, height, etc. 9) Value is the numeric value or string test result of clinical concepts. Typicall values can be the results of Measurements such as Lab test, vital signs, and quantitative findings from pathology reports. It can also be the dosage of drugs, the frequency of drugs, positive/negative of Gene test or lab test, the duration of drugs or numeric criteria of age, weight, height, etc. The relational operator such as \"<\", \">\", \">=\", \"<=\" should be recognized as part of the VALUE information."


TEMPORAL_PROMPT = "Annotate the temporal text with the given phrase representing the temporal relation and potential related events, as well as the origin text\n" + \
                    "1) Categorize the temporal relation into one of the following relations: \n" + \
                    "Event X within t time unitsa before/after event Y; Cumulative duration t time units of an event X;  \n" \
                    "Event X before/after event Y, Event X at the time of event Y.\n" + \
                    "2) Specify the corresponding Event X and/or Y for each category.\n" + \
                    "3) Output the results in a JSON format."


OLD_PARSE_PROMPT = "Annotate clinical concepts from the given text using the following rules: \n" + \
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
                   "1) Condition is events of a Person suggesting the presence of a disease or medical condition stated as a diagnosis, a sign, or a symptom, which is either observed by a Provider or reported by the patient. \n" + \
                    "2) Drugs include prescription and over-the-counter medicines, vaccines, and large-molecule biologic therapies. Radiological devices ingested or applied locally do not count as Drugs. \n" + \
                    "3) Procedure is records of activities or processes ordered by, or carried out by, a healthcare provider on the patient with a diagnostic or therapeutic purpose. Lab tests are not a procedure, if something is observed with an expected resulting amount and unit then it should be a measurement. \n" + \
                    "4) Devices include implantable objects (e.g. pacemakers, stents, artificial joints), medical equipment and supplies (e.g. bandages, crutches, syringes), other instruments used in medical procedures (e.g. sutures, defibrillators) and material used in clinical care (e.g. adhesives, body material, dental material, surgical material). \n" + \
                    "5) Measurement contains both orders and results of such Measurements as laboratory tests, vital signs, quantitative findings from pathology reports, etc. OBSERVATION captures clinical facts about a Person obtained in the context of examination, questioning or a procedure. Any data that cannot be represented by any other domains, such as social and lifestyle facts, medical history, family history, etc. are recorded here. \n" + \
                    "6) Observations differ from Measurements in that they do not require a standardized test or some other activity to generate clinical fact. Typical observations are medical history, family history, the stated need for certain treatment, social circumstances, lifestyle choices, healthcare utilization patterns, etc. \n" + \
                    "7) Demographic can include factors of patient such as age, gender, race, ethnicity, education level, income, occupation, geographic location, marital status, and family size. Age term can be demographic but the specific age criteria should be annotated as value. \n" + \
                    "8) Negation_cue includes all information that negates clinical concepts. \n" + \
                    "9) Value is the numeric value or string test result of clinical concepts. Typicall values can be the results of Measurements such as Lab test, vital signs, and quantitative findings from pathology reports. It can also be the dosage of drugs, the frequency of drugs, positive/negative of Gene test or lab test, the duration of drugs or numeric criteria of age, weight, height, etc."
