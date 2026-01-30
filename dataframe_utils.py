# %%
import os
import os.path as osp
import json
import pandas as pd
import numpy as np
import psycopg2
import pickle
from itertools import combinations, chain
from sqlalchemy import create_engine
from llm_zoo_rlhf import LLMZoo, GPT4AzureModel, OllamaModel, OpenaiClient, LLMwithReward, LLMwithInteractive

import sys
import os
sys.path.insert(0, os.path.abspath("./trialistUtils"))
from trialist_utils import *

# Read variables from .env file into the environment
env_file = '.env'
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            parts = line.strip().split('=')
            if len(parts) == 2:
                key, value = parts
                print(f"Setting {key} to {value}")
                os.environ[key] = value

# %%
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')
host = os.getenv('DB_HOST')
port = os.getenv('DB_PORT')
database = os.getenv('DB_NAME')

assert user and password and host and port and database

# %%


import xml.etree.ElementTree as ET
from xml.dom import minidom

def to_xml(obj, root_tag):
    root = ET.Element(root_tag)
    for attr, value in vars(obj).items():
        if attr != "text" and value is not None:  # exclude text
        # if value is not None:
            child = ET.SubElement(root, attr)
            child.text = str(value)
    return root

def convert_to_xml(trial_criteria, name='criteria'):
    if type(trial_criteria) is not list:
        trial_criteria = [trial_criteria]
        
    root = ET.Element(name)
    # root 
    
    criteria_element = ET.SubElement(root, None)
    for criterion in trial_criteria:
        criteria_element.append(to_xml(criterion, None))
    
    xml_string = ET.tostring(root, encoding='utf-8', method='xml').decode('utf-8')
    return minidom.parseString(xml_string).toprettyxml(indent="  ")


def load_sql_files_from_folder(folder_path):
    sql_dict = {}
    
    # Check if the folder exists
    if not os.path.exists(folder_path):
        return None
        # raise FileNotFoundError(f"The folder {folder_path} does not exist.")
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"{folder_path} is not a directory.")
    
    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        # Construct the full file path
        file_path = os.path.join(folder_path, file_name)
        
        # Check if the file is a .sql file
        if os.path.isfile(file_path) and file_name.endswith('.sql'):
            # Read the content of the .sql file
            with open(file_path, 'r', encoding='utf-8') as f:
                sql_content = f.read()
                # Use the file name as the key and file content as the value
                sql_dict[file_name.replace('.sql', '')] = sql_content

    return sql_dict


def extract_valid_adverse_events(df: pd.DataFrame) -> list:
    """
    Extract adverse event names that have both 'adverseevent_XXX' and 
    'adverseevent_XXX_durationtime' columns in the DataFrame.

    Args:
        df (pd.DataFrame): A DataFrame containing adverse event columns.

    Returns:
        List[str]: A sorted list of valid adverse event names.
    """
    # Get all column names that start with 'adverseevent_'
    ae_cols = [col for col in df.columns if col.startswith('adverseevent_')]

    # Sets to store base event names and those with durationtime
    events = set()
    duration_events = set()

    for col in ae_cols:
        if col.endswith('_durationtime'):
            # Strip prefix and suffix to get the base event name
            event_name = col[len('adverseevent_'):-len('_durationtime')]
            duration_events.add(event_name)
        else:
            # Strip only the prefix to get the base event name
            event_name = col[len('adverseevent_'):]
            events.add(event_name)

    # Return events that have both base and duration columns
    valid_events = sorted(events & duration_events)
    return valid_events

def load_or_run_and_save(path, func, *args, **kwargs):
    if os.path.exists(path):
        print(f"Loading cached result from {path}")
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"File not found. Running function and saving result to {path}")
        result = func(*args, **kwargs)
        with open(path, 'wb') as f:
            pickle.dump(result, f)
        return result
    

# %%
class DataframeBuilder:
    def __init__(self, trialID, trial_info, dataset, user, password, host, port, database, root_dir, **kwargs):
        
        self.trialID = trialID
        self.trial_info = trial_info
        self.dataset = dataset

        if self.dataset == 'insight':
            database = 'insight'

        print('trial_info...')
        print(self.trial_info)

        # load sql queries
        self.treatment_sql_dict = load_sql_files_from_folder(osp.join(root_dir, f'{self.trialID}', 'treatment'))
        self.outduration_sql_dict = load_sql_files_from_folder(osp.join(root_dir, f'{self.trialID}', 'outduration'))
        self.include_sql_dict = load_sql_files_from_folder(osp.join(root_dir, f'{self.trialID}', 'include'))
        self.exclude_sql_dict = load_sql_files_from_folder(osp.join(root_dir, f'{self.trialID}', 'exclude'))
        self.adverseevents_sql_dict = load_sql_files_from_folder(osp.join(root_dir, f'{self.trialID}', 'adverseevents'))

        
        # print(self.treatment_sql_dict.keys(), self.outduration_sql_dict.keys(), self.include_sql_dict.keys(), self.exclude_sql_dict.keys())

        # self.treatment_sql_dict = {ec: self.treatment_sql_dict[ec] for ec in trial_info['treatment_definition'] if ec in self.treatment_sql_dict} 
        # self.outduration_sql_dict = {ec: self.outduration_sql_dict[ec] for ec in trial_info['outcome_definition'] if ec in self.outduration_sql_dict} 
        # self.include_sql_dict = {ec: self.include_sql_dict[ec] for ec in trial_info['inclusion_criteria'] if ec in self.include_sql_dict} 
        # self.exclude_sql_dict = {ec: self.exclude_sql_dict[ec] for ec in trial_info['exclusion_criteria'] if ec in self.exclude_sql_dict} 

        # print(self.treatment_sql_dict.keys(), self.outduration_sql_dict.keys(), self.include_sql_dict.keys(), self.exclude_sql_dict.keys())

        try:
            # connect to database
            connect_string = f'postgresql://{user}:{password}@{host}:{port}/{database}'
            self.engine_sqlalchemy = create_engine(connect_string)
            self.engine_psycopg2 = psycopg2.connect(
                host=host,
                database=database,
                user=user,
                password=password,
                port=port,
                connect_timeout=10  # Wait 10s to connect database
            )
        except:
            print('Not Connected to Database!!')

        if self.dataset == 'insight':
            self.scheme_name = 'merge' # mimiciv_derived scheme
            self.X_table_names = ['covariates'] # list of X table names
        elif self.dataset == 'mimic':
            self.scheme_name = 'mimiciv_derived' # mimiciv_derived scheme
            self.X_table_names = ['first_day_demographics', 'first_day_bg', 'first_day_gcs', 'first_day_lab', 'first_day_vitalsign', 'first_day_urine_output', 'first_day_sofa'] # list of X table names
        else:
            assert False, f"Unknown dataset: {self.dataset}"

        self.dataframe_folder = osp.join(root_dir, f'{self.trialID}', 'dataframes')

        if not osp.exists(self.dataframe_folder):
            os.makedirs(self.dataframe_folder)

        self.df = None
        self.covariate_cols = None
        self.treatment_cols = None
        self.outcome_cols = None
        self.duration_cols = None


    def _fetch_table(self, table_name, chunksize=-1):
        df_path = osp.join(self.dataframe_folder, f'{table_name}.csv')
        if osp.exists(df_path):
            df = pd.read_csv(df_path)
        else:
            sql_query = f"SELECT * FROM {table_name}"
            df = self._sql2df(sql_query)
            df.to_csv(df_path, index=False)
        return df
    

    def _remove_sql_markers(self, input_str):
        """
        Removes the starting ```sql and ending ``` from the input string if they exist.

        Args:
            input_str (str): The input string to process.

        Returns:
            str: The processed string without ```sql at the start and ``` at the end.
        """
        start_marker = "```sql"
        end_marker = "```"

        if input_str.startswith(start_marker) and input_str.endswith(end_marker):
            # Remove the markers by slicing
            return input_str[len(start_marker):-len(end_marker)].strip()
        
        return input_str

    def _sql2df(self, sql_query, engine='psycopg2'):
        """
        engine: ['psycopg2', 'sqlalchemy']
        """
        sql_query = self._remove_sql_markers(sql_query)
        print('sql_query:', sql_query)

        if engine == 'psycopg2':
            # try:
            with self.engine_psycopg2.cursor() as cursor:
                cursor.execute(sql_query)
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
            df = pd.DataFrame(data, columns=columns)
            # except:
            #     print("You have to connect the database")
            #     assert False
        else:
            df = pd.read_sql_query(sql_query, con=self.engine)
        return df
    

    def _load_df_from_sql(self, sql_definition, sql_query):
        df_path = osp.join(self.dataframe_folder, f'{sql_definition}.csv')
        if osp.exists(df_path):
            df = pd.read_csv(df_path)
        else:
            df = self._sql2df(sql_query)
            df.to_csv(df_path, index=False)
        return df

    def _build_dataframe_X(self):
        merge_df_path = osp.join(self.dataframe_folder, 'merged_X.csv')
        if osp.exists(merge_df_path):
            merged_df = pd.read_csv(merge_df_path)
            return merged_df
        else:
            df_X_list = []
            for table_name in self.X_table_names:
                df_i = self._fetch_table(f'{self.scheme_name}.{table_name}')
                df_X_list.append(df_i)
            
            merged_df = df_X_list[0]
            for _df in df_X_list[1:]:
                merged_df = pd.merge(merged_df, _df.drop(columns=['subject_id']), on='stay_id', how='outer', suffixes=("", "_dup"))
                merged_df = merged_df.drop(columns=[col for col in merged_df.columns if col.endswith("_dup")])

            variables_keep_min = ['gcs', 'hemoglobin', 'pao2fio2ratio', 'platelets', 'so2']
            cols_kept = []
            for col in merged_df.columns:
                if not (col.endswith('_max') or col.endswith('_min') or col.endswith('_mean') or col.endswith('_sum')):
                    cols_kept.append(col)
                elif col.endswith('_max') and col.replace('_max', '') not in variables_keep_min:
                    cols_kept.append(col)
                elif col.endswith('_min') and col.replace('_min', '') in variables_keep_min:
                    cols_kept.append(col)
            merged_df = merged_df[cols_kept]
            merged_df = merged_df.rename(columns=lambda col: col.removesuffix("_max").removesuffix("_min"))

            merged_df["abs_basophils"] = merged_df["abs_basophils"].apply(lambda x: float(x) if x is not None else np.nan)
            merged_df["abs_monocytes"] = merged_df["abs_monocytes"].apply(lambda x: float(x) if x is not None else np.nan)
            merged_df["abs_eosinophils"] = merged_df["abs_eosinophils"].apply(lambda x: float(x) if x is not None else np.nan)
            merged_df["abs_lymphocytes"] = merged_df["abs_lymphocytes"].apply(lambda x: float(x) if x is not None else np.nan)
            merged_df["abs_neutrophils"] = merged_df["abs_neutrophils"].apply(lambda x: float(x) if x is not None else np.nan)
            merged_df["sofa"] = merged_df["sofa"].apply(lambda x: float(x) if x is not None else np.nan)

            merged_df = merged_df.drop_duplicates(subset="stay_id", keep="last")
            merged_df.to_csv(merge_df_path, index=False)
            return merged_df
        

    def _build_dataframe_treatment(self):

        def clean_treatment_time(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
            """
            Clean the 'treatment_time' column in the given DataFrame:
            
            1. For rows where treatment == 1:
                - Ensure 'treatment_time' is not null and within the range [intime, outtime].
                - If not, set 'treatment_time' to 'intime' and count these corrections.
            
            2. For rows where treatment == 0:
                - Ensure 'treatment_time' is null. If not, set it to NaT.

            Args:
                df: A pandas DataFrame with the columns 'intime', 'outtime', 
                    'treatment_time', and 'treatment'.

            Returns:
                A tuple of:
                - The cleaned DataFrame.
                - The number of rows where 'treatment_time' was replaced with 'intime'.
            """
            df = df.copy()

            # Ensure datetime format for time-related columns
            df['intime'] = pd.to_datetime(df['intime'])
            df['outtime'] = pd.to_datetime(df['outtime'])
            df['treatment_time'] = pd.to_datetime(df['treatment_time'], errors='coerce')

            # Case 1: treatment == 1 but 'treatment_time' is missing or outside [intime, outtime]
            condition_treat1_invalid = (
                (df['treatment'] == 1) & (
                    df['treatment_time'].isna() |
                    (df['treatment_time'] < df['intime']) |
                    (df['treatment_time'] > df['outtime'])
                )
            )
            count_replaced = condition_treat1_invalid.sum()
            df.loc[condition_treat1_invalid, 'treatment_time'] = df.loc[condition_treat1_invalid, 'intime']

            # Case 2: treatment == 0 but 'treatment_time' is not null
            condition_treat0_has_time = (df['treatment'] == 0) & df['treatment_time'].notna()
            df.loc[condition_treat0_has_time, 'treatment_time'] = pd.NaT

            df['treatment_time_delta'] = (
                (df['treatment_time'] - df['intime']).dt.total_seconds() / 86400
            )

            return df, count_replaced


        treament_df_path = osp.join(self.dataframe_folder, 'treatment_df.csv')
        if osp.exists(treament_df_path):
            df_T = pd.read_csv(treament_df_path)
            df_T, error_cnt = clean_treatment_time(df_T)
            print('Number of treatment_time replaced with intime:', error_cnt)
            return df_T
        else:
            df_T_list = []
            for sql_definition, sql_query in self.treatment_sql_dict.items():
                _df_T = self._load_df_from_sql(sql_definition, sql_query)
                df_T_list.append(_df_T)
            
            df_T = df_T_list[0]  # TODO: merge multiple treatment tables

            df_T, error_cnt = clean_treatment_time(df_T)
            print('Number of treatment_time replaced with intime:', error_cnt)

            df_T.to_csv(treament_df_path, index=False)
            return df_T
        

    def _build_dataframe_outduration(self):
        outduration_df_path = osp.join(self.dataframe_folder, 'outduration_df.csv')
        if osp.exists(outduration_df_path):
            df_Y = pd.read_csv(outduration_df_path)
            return df_Y
        else:
            df_Y_list = []
            for sql_definition, sql_query in self.outduration_sql_dict.items():
                _df_Y = self._load_df_from_sql(sql_definition, sql_query)
                df_Y_list.append(_df_Y)
            
            df_Y = df_Y_list[0]
            df_Y.to_csv(outduration_df_path, index=False)
            return df_Y

    def _build_dataframe_adverseevents(self):
        df_Y_list, adverseevent_definition_list = [], []
        if self.adverseevents_sql_dict is None:
            return [], []
        
        for sql_definition, sql_query in self.adverseevents_sql_dict.items():
            idx, adverseevent_defition = sql_definition.split('-', 1)
            adverseevents_df_path = osp.join(self.dataframe_folder, f'adverseevent{idx}.csv')
            adverseevent_definition_list.append(adverseevent_defition)
            if osp.exists(adverseevents_df_path):
                _df_Y = pd.read_csv(adverseevents_df_path)
            else:
                _df_Y = self._load_df_from_sql(sql_definition, sql_query)
                _df_Y.to_csv(adverseevents_df_path, index=False)
            df_Y_list.append(_df_Y)
        return df_Y_list, adverseevent_definition_list

    def _build_big_dataframe(self):
        big_df_path = osp.join(self.dataframe_folder, 'big_df.csv')
        if osp.exists(big_df_path):  # TODO: remove False
            merged_df = pd.read_csv(big_df_path)
            return merged_df
        else:
            df_X = self._build_dataframe_X()
            df_T = self._build_dataframe_treatment()
            df_Y = self._build_dataframe_outduration()

            merged_df = pd.merge(df_Y, df_T, on='stay_id', how='inner', suffixes=("", "_dup"))
            merged_df = merged_df.drop(columns=[col for col in merged_df.columns if col.endswith("_dup")])
        
            merged_df = pd.merge(merged_df, df_X, on='stay_id', how='inner', suffixes=("", "_dup"))
            merged_df = merged_df.drop(columns=[col for col in merged_df.columns if col.endswith("_dup")])

            # clincal notes
            if osp.exists(osp.join(self.dataframe_folder, 'all_notes.csv')):
                df_notes = pd.read_csv(osp.join(self.dataframe_folder, 'all_notes.csv'))
                merged_df = pd.merge(merged_df, df_notes, on='stay_id', how='left', suffixes=("", "_dup"))
                merged_df = merged_df.drop(columns=[col for col in merged_df.columns if col.endswith("_dup")])

            # adverse events
            df_Y_list, adverseevent_definition_list = self._build_dataframe_adverseevents()
            for df_Y_adverse, adverseevent_definition in zip(df_Y_list, adverseevent_definition_list):
                df_Y_adverse_new = df_Y_adverse.iloc[:, [0, 2, 3]].copy()
                df_Y_adverse_new.columns = ['stay_id', f'adverseevent_{adverseevent_definition}', f'adverseevent_{adverseevent_definition}_durationtime']  # 新的列名
                merged_df = pd.merge(merged_df, df_Y_adverse_new, on='stay_id', how='left', suffixes=("", "_dup"))
                merged_df = merged_df.drop(columns=[col for col in merged_df.columns if col.endswith("_dup")])            

            merged_df = merged_df.drop_duplicates(subset="stay_id", keep="last")
            merged_df.to_csv(big_df_path, index=False)
            return merged_df


    def _get_matched_patients_by_ec(self, ec, sql=None):
        matched_patients_df_path = osp.join(self.dataframe_folder, f'matched_patients_ec_{ec}.csv')
        if osp.exists(matched_patients_df_path):
            matched_patients = pd.read_csv(matched_patients_df_path)
            return matched_patients['stay_id'].to_list()
        else:
            assert sql is not None
            patients = self._sql2df(sql)
            matched_patients = patients['stay_id'].to_list()
            matched_patients_df = pd.DataFrame({'stay_id': matched_patients})
            matched_patients_df.to_csv(matched_patients_df_path, index=False)
            return matched_patients


    def _get_matched_patients(self):
        matched_patient_df_path = osp.join(self.dataframe_folder, 'matched_patients.csv')
        if osp.exists(matched_patient_df_path):
            matched_patients = pd.read_csv(matched_patient_df_path)
            return matched_patients['stay_id'].to_list()
        else:
            include_patients_list = []
            for ec, sql in self.include_sql_dict.items():
                include_patients_list.append(self._get_matched_patients_by_ec(ec, sql))
            include_patients_set = set.intersection(*map(set, include_patients_list)) if len(include_patients_list) > 0 else set()

            exclude_patients_list = []
            for ec, sql in self.exclude_sql_dict.items():
                exclude_patients_list.append(self._get_matched_patients_by_ec(ec, sql))
            exclude_patients_set = set.union(*map(set, exclude_patients_list)) if len(exclude_patients_list) > 0 else set()

            matched_patients = list(include_patients_set -  exclude_patients_set)
            matched_patients_df = pd.DataFrame({'stay_id': matched_patients})
            matched_patients_df.to_csv(matched_patient_df_path, index=False)
            return matched_patients

    def get_dataframe(self, matched_patients=None):
        df_path = osp.join(self.dataframe_folder, 'dataframe.csv')
        col_dict_path = osp.join(self.dataframe_folder, 'col_dict.json')
        if osp.exists(df_path) and osp.exists(col_dict_path):  # TODO: remove False
            df = pd.read_csv(df_path)
            with open(col_dict_path, 'r') as json_file:
                col_dict = json.load(json_file)
        else:
            df_all = self._build_big_dataframe()
            if matched_patients is None:
                matched_patients = self._get_matched_patients()
            else:
                assert isinstance(matched_patients, list), "matched_patients should be a list of stay_ids"

            # get the matched patients
            df = df_all[df_all['stay_id'].isin(matched_patients)]
            df.to_csv(df_path.replace('dataframe.csv', 'dataframe_all_cols.csv'), index=False)

            # 
            adverse_events = extract_valid_adverse_events(df)

            col_dict = {
                'covariate_cols': [col for col in df.columns if 'adverseevent' not in col and col not in ['stay_id', 'subject_id', 'treatment', 'treatment_time', "total_administered_dose", "treatment_time_delta",'intime', 'outtime', 'total_adjusted_dose', 'hospitaldischargestatus', 'duration_day', 'censoring_day', 'hadm_id', 'admittime', 'anchor_age', 'anchor_year', 'hospitaldischargeoffset', 'hospitaldischargelocation', 'charttime_discharge', 'text_discharge', 'charttime_radiology', 'text_radiology']],
                'treatment_cols': 'treatment',
                'outcome_cols': ['hospitaldischargestatus'] + [f'adverseevent_{ae}' for ae in adverse_events],
                'duration_cols': ['duration_day'] + [f'adverseevent_{ae}_durationtime' for ae in adverse_events],
                'treatment_time_cols': 'treatment_time_delta',
                'patient_id': 'stay_id',
            }
            df = df[['stay_id', 'subject_id'] + [col_dict['treatment_cols']] + [col_dict['treatment_time_cols']] + col_dict['outcome_cols'] + col_dict['duration_cols'] + col_dict['covariate_cols'] ]
            
            # save the dataframe and col_dict
            df.to_csv(df_path, index=False)
            with open(col_dict_path, 'w') as json_file:
                json.dump(col_dict, json_file, indent=4)

        return df, col_dict

    def get_treatment_sql_dict(self):
        return self.treatment_sql_dict
    def get_outduration_sql_dict(self):
        return self.outduration_sql_dict
    def get_include_sql_dict(self):
        return self.include_sql_dict
    def get_exclude_sql_dict(self):
        return self.exclude_sql_dict
    
    def get_big_dataframe(self):
        return self._build_big_dataframe()
    
    def get_big_notes_dataframe(self, notes_folder='notes', notes_filename='all_notes.csv'):
        notes_df_path = osp.join(notes_folder, notes_filename)
        if osp.exists(notes_df_path):
            try:
                df_notes = pd.read_csv(notes_df_path)
                return df_notes
            except Exception as e:
                print(f"Error reading {notes_df_path}: {e}")
                return None
        else:
            return None
    
    def load_matched_patients_by_ec(self, ec):
        return self._get_matched_patients_by_ec(ec=ec, sql=None)
    


# %%
# df_builder = DataframeBuilder(trialID='NCT00000000', user=user, password=password, host=host, port=port, database=database, root_dir='./trials')
# df_NCT00000000, cols_dict_NCT00000000 = df_builder.get_dataframe()
# df_NCT00000000

# %%
# cols_dict_NCT00000000

# %%
class Informatician:
    # def __init__(self, trialID, trial_info, llm, root_dir='./trials', trial_templates=['NCT00000000']):
    def __init__(self, trialID, trial_info, llm, dataset='mimic', root_dir='./trials', trial_templates=['NCT00000000', 'NCT03872011']):

        self.root_dir = root_dir
        self.trialID = trialID
        self.trial_info = trial_info
        self.llm = llm
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.database = database
        self.root_dir = root_dir
        self.dataset = dataset # 'mimic' or 'insight'
        if self.dataset is None:
            self.dataset = 'mimic'  # default to mimic if not specified
        self.trial_templates = trial_templates

        self.all_trials_info = self.get_all_trial_info()
        self.df_builder = None

    def _replace_spaces_in_columns_and_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace spaces in all column names of a DataFrame with underscores.
        Additionally, replace spaces with underscores in all string values within the DataFrame.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame.
        
        Returns:
        pd.DataFrame: New DataFrame with updated column names and string values.
        """
        df_renamed = df.rename(columns=lambda x: x.replace(" ", "_"))
        df_replaced = df_renamed.applymap(lambda x: x.replace(" ", "_") if isinstance(x, str) else x)
        return df_replaced

    def _replace_spaces_in_dict_values(self, data: dict) -> dict:
        """
        Replace spaces with underscores in all string elements within list values of a dictionary.
        Also attempts to replace spaces in single string values.
        
        Parameters:
        data (dict): Input dictionary where values can be lists of strings or single strings.
        
        Returns:
        dict: New dictionary with modified values.
        """
        new_data = {}
        for key, value in data.items():
            if isinstance(value, list):
                new_data[key] = [item.replace(" ", "_") for item in value]
            elif isinstance(value, str):
                new_data[key] = value.replace(" ", "_")
            else:
                new_data[key] = value
        return new_data

    def get_df_dict(self, patient_id_list=None):
        try:
            # directly get the dataframe and col_dict
            df_builder = DataframeBuilder(trialID=self.trialID, trial_info=self.trial_info, dataset=self.dataset, user=self.user, password=self.password, host=self.host, port=self.port, database=self.database, root_dir=self.root_dir)
            self.df_builder = df_builder
            df, cols_dict = df_builder.get_dataframe(matched_patients=patient_id_list)
            df = self._replace_spaces_in_columns_and_values(df)
            cols_dict = self._replace_spaces_in_dict_values(cols_dict)
            return df, cols_dict
        except:
            print('except')
            # build sqls before get the dataframe
            self.build_sqls_for_trial()
            df_builder = DataframeBuilder(trialID=self.trialID, trial_info=self.trial_info, dataset=self.dataset, user=self.user, password=self.password, host=self.host, port=self.port, database=self.database, root_dir=self.root_dir)
            self.df_builder = df_builder
            df, cols_dict = df_builder.get_dataframe(matched_patients=patient_id_list)
            df = self._replace_spaces_in_columns_and_values(df)
            cols_dict = self._replace_spaces_in_dict_values(cols_dict)
            return df, cols_dict

    def get_bigdf_ec_opti(self):
        assert self.df_builder is not None
        df_big = self.df_builder.get_big_dataframe()
        return df_big
    
    def get_notes(self):
        assert self.df_builder is not None
        df_notes = self.df_builder.get_big_notes_dataframe()
        return df_notes
    
    def get_permuted_ec_for_optimize(self):
        assert self.df_builder is not None
        
        df_big = self.get_bigdf_ec_opti()
        include_sql_dict = self.df_builder.include_sql_dict
        exclude_sql_dict = self.df_builder.exclude_sql_dict

        ec_patients_list = []
        for ec in include_sql_dict.keys():
            patient_list = self.df_builder.load_matched_patients_by_ec(ec=ec)
            ec_patients_list.append((ec, set(patient_list)))

        for ec in exclude_sql_dict.keys():
            patient_list_excluded = self.df_builder.load_matched_patients_by_ec(ec=ec)
            patient_list = list(set(df_big['stay_id'].to_list())-set(patient_list_excluded))
            ec_patients_list.append((ec, set(patient_list)))

        # Function to generate the power set of indices [0, ..., n-1]
        def powerset(iterable):
            s = list(iterable)
            return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))

        # Example input list: each tuple contains (eligibility criteria, matched patients set)
        # eligibility_criteria_list = [
        #     ("ec_0", {1, 2, 3, 4}),
        #     ("ec_1", {3, 4, 5, 6}),
        #     ("ec_2", {1, 3, 5, 7})
        # ]
        eligibility_criteria_list = ec_patients_list

        n = len(eligibility_criteria_list)

        # Generate the power set of indices
        index_powerset = powerset(range(n))

        # Define the universal set as the union of all matched patients
        universal_set = set.union(*(ec[1] for ec in eligibility_criteria_list))

        # Generate corresponding EC and matched patients sets
        ec_powerset = [[eligibility_criteria_list[i][0] for i in subset] for subset in index_powerset]
        matched_patients_powerset = [
            set.intersection(*(eligibility_criteria_list[i][1] for i in subset)) if subset else universal_set
            for subset in index_powerset
        ]
        index_powerset = [list(_) for _ in index_powerset]
        matched_patients_powerset = [list(_) for _ in matched_patients_powerset]
        matched_patients_number = [len(_) for _ in matched_patients_powerset]

        # Create and display the DataFrame
        df_powerset = pd.DataFrame({
            "IndexSet": index_powerset,
            "EligibilityCriteria": ec_powerset,
            "MatchedPatients": matched_patients_powerset,
            "MatchedPatientsNumber": matched_patients_number
        })

        state = {'variables': {}}
        
        # state['variables']['cleaned_dataframe'] = df
        # state['variables']['cols_dict'] = cols_dict
        # state['variables']['Covariates'] = cols_dict['covariate_cols']
        # state['variables']['Treatment'] = cols_dict['treatment_cols']
        # state['variables']['Outcome'] = cols_dict['outcome_cols']
        # state['variables']['Duration']  = cols_dict['duration_cols']

        state['variables']['big_dataframe'] = self._replace_spaces_in_columns_and_values(df_big)
        state['variables']['ec_patients_powerset_dataframe'] = df_powerset

        return state

        # # Function to generate the power set of a given list
        # def generate_power_set(lst):
        #     power_set = []
        #     n = len(lst)
        #     for i in range(n + 1):
        #         for combo in combinations(range(n), i):
        #             power_set.append(list(combo))
        #     return power_set

        # example_list = ec_patients_list
        # # example_list = [
        # #     ("ec_1", [1,2]),
        # #     ("ec_2", [11,22]),
        # #     ("ec_3", [111,222]),
        # #     ("ec_4", [1111,2222])
        # # ]

        # # Generate the power set of indices
        # power_set_indices = generate_power_set(example_list)

        # # Generate the corresponding ECs and matched patients lists
        # power_set_ecs = [[example_list[i][0] for i in subset] for subset in power_set_indices]
        # power_set_patients = [[example_list[i][1] for i in subset] for subset in power_set_indices]

        # # Flatten matched patient lists
        # power_set_patients = [[patient for sublist in group for patient in sublist] for group in power_set_patients]

        # # Display results
        # assert len(power_set_indices) == len(power_set_ecs) == len(power_set_patients)
        # for ind, ec, patients in zip(power_set_indices, power_set_ecs, power_set_patients):
        #     print(ind, ec, len(patients))

    def get_all_trial_info(self):
        all_trials = [_ for _ in os.listdir(self.root_dir) if _.startswith('NCT')]
        all_trials = [_ for _ in all_trials if _ in self.trial_templates]
        all_trials_info = {}
        for trialID in all_trials:
            try:
                treatment_sql_dict = load_sql_files_from_folder(osp.join(self.root_dir, f'{trialID}', 'treatment'))
                outduration_sql_dict = load_sql_files_from_folder(osp.join(self.root_dir, f'{trialID}', 'outduration'))
                include_sql_dict = load_sql_files_from_folder(osp.join(self.root_dir, f'{trialID}', 'include'))
                exclude_sql_dict = load_sql_files_from_folder(osp.join(self.root_dir, f'{trialID}', 'exclude'))
                all_trials_info[trialID] = {'treatment': treatment_sql_dict, 'outduration': outduration_sql_dict, 'include': include_sql_dict, 'exclude': exclude_sql_dict}
            except:
                continue
        return all_trials_info

    def build_sqls_for_trial(self):
        # build sqls for the trial
        for key, ec in self.trial_info.items():
            print('key ec', key, ec)
            if key == 'inclusion_criteria':
                for index, ec in enumerate(self.trial_info[key]):
                    try:
                        formated_ec = self.trial_info[f'{key}_formated'][index]
                    except:
                        formated_ec = None
                    sql_query, prompt = self.ask_ec_sql(ec, formated_ec=formated_ec, include_or_exclude='include')
                    # print('>>>>>>>>')
                    # print('prompt:', prompt)
                    # print('---------------------------')
            elif key == 'exclusion_criteria':
                for index, ec in enumerate(self.trial_info[key]):
                    try:
                        formated_ec = self.trial_info[f'{key}_formated'][index]
                    except:
                        formated_ec = None
                    sql_query, prompt = self.ask_ec_sql(ec, formated_ec=formated_ec, include_or_exclude='exclude')
                    # print('>>>>>>>>')
                    # print('prompt:', prompt)
                    # print('---------------------------')
            elif key == 'treatment_definition':
                for index, treatment_definition in enumerate(self.trial_info[key]):
                    try:
                        formated_treatment = self.trial_info[f'{key}_formated'][index]
                    except:
                        formated_treatment = None
                    sql_query, prompt = self.ask_treatment_sql(treatment_definition, formated_treatment=formated_treatment)
                    # print('>>>>>>>>')
                    # print('prompt:', prompt)
                    # print('---------------------------')
            elif key == 'outcome_definition':
                for index, outcome_definition in enumerate(self.trial_info[key]):
                    try:
                        formated_outcome = self.trial_info[f'{key}_formated'][index]
                    except:
                        formated_outcome = None
                    sql_query, prompt = self.ask_outcome_sql(outcome_definition, formated_outcome=formated_outcome)
                    # print('>>>>>>>>')
                    # print('prompt:', prompt)
                    # print('---------------------------')
            # else:
            #     assert False, f"key {key} is not recognized."



    def ask_ec_sql(self, ec, formated_ec=None, include_or_exclude='include'):
        if include_or_exclude == 'include':
            ec_path = osp.join(self.root_dir, self.trialID, 'include', f'{ec}.sql')
        else:
            ec_path = osp.join(self.root_dir, self.trialID, 'exclude', f'{ec}.sql')

        if osp.exists(ec_path):
            with open(ec_path, 'r') as f:
                sql_query = f.read()
                return sql_query, f"load sql from {ec_path}"
        else:       
            prompt = f"""Convert the following eligibility criteria (EC) into a sql that can be queried on a postgresSQL database in MIMIC-2.2 format. You need to match patients by stay_id instead of subject_id. The database has two schemes: mimic_hosp and mimic_icu. I will give you some examples (EC and its SQL)

            EC: {ec}
            Formated EC: {formated_ec}
            SQL:

            Provide ONLY your generated runnable SQL and NOTHING else. You also do not include sql```SQL``` in your response. Directly output runnable SQL.
            
            Examples: 

            """
            for trialID, trial_info in self.all_trials_info.items():
                for include_ec, sql_query in trial_info['include'].items():
                    prompt += f"EC:{include_ec}\nSQL:{sql_query}\n\n"
                for exclude_ec, sql_query in trial_info['exclude'].items():
                    prompt += f"EC:{exclude_ec}\nSQL:{sql_query}\n\n" 
            sql_query = self.llm.invoke(prompt, use_cache=False)  
            
            if not os.path.exists(os.path.dirname(ec_path)):
                os.makedirs(os.path.dirname(ec_path))
            with open(ec_path, 'w') as f:
                f.write(sql_query)

            with open('tmpsql.txt', 'a') as f:
                f.write(prompt)
                f.write('\n\n')

            return sql_query, prompt      

    def ask_treatment_sql(self, treatment_definition, formated_treatment=None):
        treatment_path = osp.join(self.root_dir, self.trialID, 'treatment', f'{treatment_definition}.sql')
        if osp.exists(treatment_path):
            with open(treatment_path, 'r') as f:
                sql_query = f.read()
                return sql_query, f"load sql from {treatment_path}"
        else:
            prompt = f"""Convert the following treatment definition into a sql that can be queried on a postgresSQL database in MIMIC-2.2 format. I will give you some examples (treatment and its SQL).

            Treatment: {treatment_definition}
            Formated Treatment: {formated_treatment}
            SQL:

            Provide ONLY your generated runnable SQL and NOTHING else. You also do not include sql```SQL``` in your response. Directly output runnable SQL.
            
            Examples: 

            """
            for trialID, trial_info in self.all_trials_info.items():
                for treatment_definition, sql_query in trial_info['treatment'].items():
                    prompt += f"Treatment:{treatment_definition}\nSQL:{sql_query}\n\n"
            
            sql_query = self.llm.invoke(prompt)

            if not os.path.exists(os.path.dirname(treatment_path)):
                os.makedirs(os.path.dirname(treatment_path))
            with open(treatment_path, 'w') as f:
                f.write(sql_query)
            return sql_query, prompt
        
    def ask_outcome_sql(self, outcome_definition, formated_outcome=None):
        outcome_path = osp.join(self.root_dir, self.trialID, 'outduration', f'{outcome_definition}.sql')
        if osp.exists(outcome_path):
            with open(outcome_path, 'r') as f:
                sql_query = f.read()
                return sql_query, f"load sql from {outcome_path}"
        else:
            prompt = f"""Convert the following outcome definition into a sql that can be queried on a postgresSQL database in MIMIC-2.2 format. I will give you some examples (outcome and its SQL)

            Outcome: {outcome_definition}
            Formated Outcome: {formated_outcome}
            SQL:

            Provide ONLY your generated runnable SQL and NOTHING else. You also do not include sql```SQL``` in your response. Directly output runnable SQL.
            
            Examples:

            """
            for trialID, trial_info in self.all_trials_info.items():
                for outcome_definition, sql_query in trial_info['outduration'].items():
                    prompt += f"Outcome:{outcome_definition}\nSQL:{sql_query}\n\n"
            
            sql_query = self.llm.invoke(prompt)

            if not os.path.exists(os.path.dirname(outcome_path)):
                os.makedirs(os.path.dirname(outcome_path))
            with open(outcome_path, 'w') as f:
                f.write(sql_query)
            return sql_query, prompt      

def clean_ec(ec):
    # if 'Septic shock admitted to the ICU within 24 hours as defined by SEPSIS-3' in ec:
    #     ec = 'Septic shock admitted to the ICU within 24 hours as defined by SEPSIS-3'
    if len(ec) > 140:
        ec = ec[:140]
    replace_dict = {'mg/d': 'mg per day', 'mmol/L': 'mmol per liter', '\n': ' '}
    for key, value in replace_dict.items():
        ec = ec.strip().replace('\n', ' ')
        ec = ec.replace(key, value)
    return ec

if __name__ == "__main__":
    # %%
    # trialID = 'NCT00000000'
    # trialID = 'NCT00475852'
    # trialID = 'NCT02856698'
    # trialID = 'NCT03872011'
    trialID = 'NCT00000011'
    trialID = 'NCT00000012'
    # trialID = 'NCT04134403'
    # trialID = 'NCT06091982'

    # trialID = 'NCT00934011'

    # trialID = 'NCT00000001'
    trialID = 'NCT04691505'
    # %%

    
    with open(osp.join('trials', f"{trialID}", f"trial_info_{trialID}.pkl"), "rb") as fin:
        trial = pickle.load(fin)
        if len(trial.treatments) == 1:
            treatment_definition = [f"{trial.treatments[0].text}"]
        elif len(trial.treatments) == 2:
            treatment_definition = [f"Treated: {trial.treatments[0].text}; Control: {trial.treatments[1].text}"]
        else:
            assert False, "Trialist should only have two treatments, one for treated and one for control."
        include_ec = [clean_ec(item.text) for item in trial.inclusion_criteria]
        exclude_ec = [clean_ec(item.text) for item in trial.exclusion_criteria]
        outcome_definition = [clean_ec(item.text) for item in trial.outcomes]
        include_ec_formated = [convert_to_xml(item, name='criteria') for item in trial.inclusion_criteria]
        exclude_ec_formated = [convert_to_xml(item, name='criteria') for item in trial.exclusion_criteria]
        treatment_definition_formated = [convert_to_xml(item, name='treatment') for item in trial.treatments]
        outcome_definition_formated = [convert_to_xml(item, name='outcome') for item in trial.outcomes]

    print(treatment_definition)
    print(include_ec)
    print(exclude_ec)
    print(outcome_definition)


    # %%
    # Defining large language model
    # Example usage
    from llm_zoo import LLMZoo, GPT4AzureModel, OllamaModel

    # Defining large language model
    # Example usage
    llm = LLMZoo()

    api_key = os.getenv('AZURE_OPENAI_API_KEY') + '1'
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    api_version = "2023-05-15"
    assert api_key and endpoint and api_key

    # openai gpt4 api
    openai_api_key = os.getenv('OPENAI_API_KEY')
    openai_organization = os.getenv('OPENAI_ORGANIZATION')
    openai_project = os.getenv('OPENAI_PROJECT')

    # Register models with initialization parameters
    # llm.register_model("GPT-4o", GPT4AzureModel(api_key=api_key, endpoint=endpoint, api_version=api_version))
    # llm.select_model("GPT-4o")

    # llm.register_model("OllamaModel", OllamaModel(model='phi4', temperature=0.5))
    # llm.select_model("OllamaModel")

    if openai_api_key and openai_organization and openai_project:
        llm.register_model("gpt4openai", OpenaiClient(api_key=openai_api_key, organization=openai_organization, project=openai_project))
    llm.select_model("gpt4openai")

    trial_info = {
        'treatment_definition': treatment_definition,
        'outcome_definition': outcome_definition,
        'inclusion_criteria': include_ec,
        'exclusion_criteria': exclude_ec,
        'treatment_definition_formated': treatment_definition_formated,
        'outcome_definition_formated': outcome_definition_formated,
        'inclusion_criteria_formated': include_ec_formated,
        'exclusion_criteria_formated': exclude_ec_formated

    }
    # informatician = Informatician(trialID='NCT00000000', llm=llm, trial_info=trial_info, user=user, password=password, host=host, port=port, database=database, root_dir='./trials')
    dataset = 'insight' if trialID in ['NCT04691505'] else 'mimic'
    informatician = Informatician(trialID=trialID, trial_info=trial_info, llm=llm, dataset=dataset, root_dir='./trials')

    # %%
    # informatician.build_sqls_for_trial()

    # %%
    df, cols_dict = informatician.get_df_dict()
    print(df, cols_dict)
    print('-----------------------')
    # df_big = informatician.get_bigdf_ec_opti()
    # print(df_big)
    # state = informatician.get_permuted_ec_for_optimize()
    # with open(f"trials/{trialID}/state_ec_optim.pkl", "wb") as f:
    #     pickle.dump(state, f)

    # df_notes = informatician.get_notes()
    # print(df_notes)

    # %%
    # response, prompt = informatician.ask_treatment_sql(treatment_definition[0])
    # print(response)
    # print(prompt)

    # %%
    # response, prompt = informatician.ask_outcome_sql(outcome_definition[0])
    # print(response)
    # print(prompt)

    # %%
    # response, prompt = informatician.ask_ec_sql(exclude_ec[1])
    # # print(response)
    # print(prompt)

    # %%
    # informatician.all_trials_info

    # %%
    # df, cols_dict = informatician.get_df_dict()
    # df


