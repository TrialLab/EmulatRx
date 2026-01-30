# %%
import json
import os
import numpy as np
import logging
from ollama import chat
from ollama import ChatResponse
from ollama._types import Options
from langchain_community.llms.llamafile import Llamafile
from openai import AzureOpenAI
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# %%
api_key = os.getenv('AZURE_OPENAI_API_KEY')
endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
api_version = "2023-05-15"

# %%
# Configure the global logger
logger = logging.getLogger("LLMZoo")
# logger.setLevel(logging.DEBUG)  # Set the log level
# logger.setLevel(logging.INFO)  # Set the log level
# logger.setLevel(logging.ERROR)  # Set the log level
logger.setLevel(logging.CRITICAL)  # Set the log level
handler = logging.StreamHandler()  # Log to console
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# %%
class LLMZoo:
    def __init__(self, cache_file="./llm_cache.json"):
        """
        Initialize the LLMZoo with an empty model registry, no selected model, and cache support.
        :param cache_file: Path to the JSON file for storing cached responses
        """
        self.models = {}  # Dictionary to store registered LLM models
        self.current_model = None  # Currently selected model
        self.cache_file = cache_file  # Path to the cache file
        self.cache = self._load_cache()  # Load cache from the file
        assert self.cache is not None

    def _load_cache(self):
        """
        Load cached responses from the JSON file.
        :return: A dictionary with cached responses
        """
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r") as f:
                    logger.info(f"Loading cache from {self.cache_file}")
                    return json.load(f)
            else:
                logger.info("Cache file not found. Starting with an empty cache.")
                return {}
        except:
            logger.info("Cache file loaded error. Starting with an empty cache.")
            return {}

    def _save_cache(self):
        """
        Save cached responses to the JSON file.
        """
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, indent=4)
        logger.info(f"Cache saved to {self.cache_file}")

    def register_model(self, name, model_instance):
        """
        Register a new LLM model.
        :param name: Name of the model (string)
        :param model_instance: The class instance that represents the model
        """
        self.models[name] = model_instance
        logger.debug(f"Model '{name}' registered successfully.")

    def select_model(self, name):
        """
        Select an LLM model to use.
        :param name: Name of the model to select
        :raises ValueError: If the specified model is not registered
        """
        if name not in self.models:
            logger.error(f"Model '{name}' not found. Please register it first.")
            raise ValueError(f"Model '{name}' not found. Please register it first.")
        self.current_model = self.models[name]
        self.current_model_name = name
        logger.info(f"Model '{name}' selected.")

    def invoke(self, prompt, *args, **kwargs):
        """
        Execute the selected LLM model with optional caching.
        :param prompt: The input prompt for the model
        :param args: Additional positional arguments for the model
        :param kwargs: Additional keyword arguments for the model
        :return: The result of the model execution
        :raises RuntimeError: If no model is selected
        """
        if 'use_cache' in kwargs and kwargs['use_cache'] is True:
            use_cache = True
        else:
            use_cache = False

        if not self.current_model:
            logger.error("No model selected. Please select a model first.")
            raise RuntimeError("No model selected. Please select a model first.")

        # Create cache key
        cache_key = (self.current_model_name, prompt, args, frozenset(kwargs.items()))
        cache_key_str = json.dumps([str(k) for k in cache_key])  # Serialize for caching

        # Check if the result is in the cache
        if use_cache and cache_key_str in self.cache:
            logger.debug(f"Cache hit for model '{self.current_model_name}' and prompt: {prompt}")
            return self.cache[cache_key_str]

        # Execute the model and cache the result
        response = self.current_model(prompt, *args, **kwargs)
        logger.debug(f"Response: {response}")


        # # Ask for human feedback (simple rating)
        # try:
        #     feedback = input("Please rate this response (1-5): ")
        #     feedback = int(feedback)
        #     assert 1 <= feedback <= 5
        # except:
        #     feedback = None  # Ignore invalid feedback

        # # Store feedback
        # if feedback:
        #     with open("rlhf_feedback.json", "a") as f:
        #         json.dump({"prompt": prompt, "response": response, "rating": feedback}, f)
        #         f.write("\n")

        if use_cache:
            self.cache[cache_key_str] = response
            self._save_cache()  # Save updated cache to file
            logger.debug(f"Cache miss. Response saved for model '{self.current_model_name}' and prompt: {prompt}")
        else:
            logger.debug(f"Cache bypassed for model '{self.current_model_name}' and prompt: {prompt}")
        return response


# define your own llm here
class GPT4AzureModel:
    def __init__(self, api_key, endpoint, api_version):        
        self.model_name = 'gpt-4o'
        self.llm = AzureOpenAI(api_key = api_key, api_version = api_version, azure_endpoint = endpoint)
        
    def __call__(self, prompt, *args, **kwargs):
        system_contet = kwargs.get("system_contet", None)
        if system_contet is None:
            messages=[
                {"role": "user", "content": prompt}
            ]
        else:
            messages=[
                {"role": "system", "content": system_contet},
                {"role": "user", "content": prompt}
            ]            

        results = self.llm.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
        response = results.choices[0].message.content
        return response


class OllamaModel:
    def __init__(self, model="llama3.2", temperature=0.7):
        self.model = model
        self.temperature = temperature

    def __call__(self, prompt, *args, **kwargs):

        response: ChatResponse = chat(
            model=self.model,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ],
            options=Options(temperature=self.temperature)
        )

        final_response = response.message.content

        if "deepseek" in self.model:
            final_response = final_response[final_response.find("</think>")+len("</think>"):].strip()
        
        return final_response 


class LlamafileModel:
    def __init__(self):
        self.llm = Llamafile()

    def __call__(self, prompt, *args, **kwargs):
        response =  self.llm.invoke(prompt)
        return response


class OpenaiClient:
    def __init__(self, api_key, organization=None, project=None):
        self.client = OpenAI(api_key=api_key, organization=organization, project=project)
        self.model_name = "gpt-4o"
    
    def __call__(self, prompt, *args, **kwargs):
        system_content = kwargs.get("system_content", None)
        if system_content is None:
            messages = [
                {"role": "user", "content": prompt}
            ]
        else:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=kwargs.get("temperature", 0),
            max_tokens=kwargs.get("max_tokens", 9999),
            top_p=kwargs.get("top_p", 1)
        )
        
        return response.choices[0].message.content

# %% [markdown]
# ### User Interface to collect ranking and rating data

# %%
import json
import ipywidgets as widgets
from IPython.display import display, clear_output
import os

class LLMwithInteractive:
    def __init__(self, llm, feedback_file="rlhf_feedback.json"):
        self.llm = llm
        self.feedback_file = feedback_file
        self.feedback_data = self._load_feedback()  # Ensure existing feedback data is loaded

    def _load_feedback(self):
        """Load existing feedback data"""
        if os.path.exists(self.feedback_file):
            with open(self.feedback_file, "r") as f:
                try:
                    return json.load(f)  # Read JSON and convert it into a list of dictionaries
                except json.JSONDecodeError:
                    return []  # Return an empty list if JSON parsing fails
        return []

    def _save_feedback(self):
        """Save feedback data to a file, ensuring existing feedback is not overwritten"""
        existing_data = self._load_feedback()  # Load existing data first
        all_data = {f"{entry['prompt']}|{entry['response']}": entry for entry in existing_data}  # Create a dictionary to remove duplicates
        
        for entry in self.feedback_data:
            all_data[f"{entry['prompt']}|{entry['response']}"] = entry  # Update or add new data

        # Store JSON as a list to ensure historical data is preserved
        with open(self.feedback_file, "w") as f:
            json.dump(list(all_data.values()), f, indent=4)

    def _update_feedback(self, prompt, response, feedback):
        """
        Update or add user feedback:
        - If `prompt` and `response` exist, update `feedback`
        - If `prompt` and `response` do not exist, append a new entry
        """
        for entry in self.feedback_data:
            if entry["prompt"] == prompt and entry["response"] == response:
                entry["feedback"] = feedback  # Update rating or ranking
                self._save_feedback()
                return

        # Append new feedback if no matching entry is found
        self.feedback_data.append({"prompt": prompt, "response": response, "feedback": feedback})
        self._save_feedback()

    def invoke(self, prompt, mode="rating", response_b=None, callback=lambda x: print(f"✅ Ranking received: {x}"), *args, **kwargs):
        """
        Run LLM to generate a response and provide rating or ranking feedback.
        - mode="rating"  ->  Allow users to rate from 1 to 5
        - mode="ranking" ->  Allow users to choose the better response between Response A and B
        """
        response_a = self.llm.invoke(prompt, *args, **kwargs)  # Generate Response A

        # Detect whether running in Jupyter Notebook
        try:
            get_ipython  # Check if running in Jupyter Notebook
            is_jupyter = True
        except NameError:
            is_jupyter = False

        if mode == "rating":
            if is_jupyter:
                self._rating_mode_jupyter(prompt, response_a, callback)
            else:
                self._rating_mode_terminal(prompt, response_a, callback)

        elif mode == "ranking":
            if response_b is None:
                raise ValueError("mode='ranking' requires response_b")
            if is_jupyter:
                self._ranking_mode_jupyter(prompt, response_a, response_b, callback)
            else:
                self._ranking_mode_terminal(prompt, response_a, response_b, callback)
        
        return response_a  # Return the LLM's response

    def _rating_mode_jupyter(self, prompt, response, callback):
        """Rating mode for Jupyter Notebook"""
        output_box = widgets.Output()
        
        with output_box:
            print("\n❓ **Prompt (User Question):**")
            print(prompt)
            print("\n📢 **Model Response:**")
            print(response)

        # Rating slider
        rating_slider = widgets.IntSlider(min=1, max=5, step=1, value=3, description="Rating:")
        submit_button = widgets.Button(description="Submit Feedback", button_style='success')

        def on_submit(b):
            """Handle user rating submission, clear the UI, and continue execution"""
            rating = rating_slider.value
            self._update_feedback(prompt, response, rating)

            with output_box:
                clear_output(wait=True)  # Only clear `output_box`, does not affect other interactive UI elements
                print("\n❓ Prompt (User Question):")
                print(prompt)
                print("\n📢 Model Response:")
                print(response)
                print(f"\n✅ Feedback saved! You rated: {rating}/5")

            if callback:
                callback(rating)

        submit_button.on_click(on_submit)

        ui_container = widgets.VBox([output_box, rating_slider, submit_button])  # Keep UI components independent
        display(ui_container)  # Create a new UI container each time to avoid overwriting previous interactions

    def _ranking_mode_jupyter(self, prompt, response_a, response_b, callback):
        """Ranking mode for Jupyter Notebook"""
        output_box = widgets.Output()
        
        with output_box:
            print("\n❓ **Prompt (User Question):**")
            print(prompt)
            print("\n📢 **Response A:**")
            print(response_a)
            print("\n📢 **Response B:**")
            print(response_b)
        
        # Ranking buttons
        ranking_buttons = widgets.ToggleButtons(
            options=["A is better", "B is better"],
            description="Select:"
        )
        submit_button = widgets.Button(description="Submit Ranking", button_style='success')

        def on_submit(b):
            """Handle user ranking submission, clear the UI, and continue execution"""
            ranking = ranking_buttons.value
            self._update_feedback(prompt, f"A: {response_a} | B: {response_b}", ranking)

            with output_box:
                clear_output(wait=True)
                print("\n❓ Prompt (User Question):")
                print(prompt)
                print("\n📢 Response A:")
                print(response_a)
                print("\n📢 Response B:")
                print(response_b)
                print(f"\n✅ Feedback saved! You ranked: {ranking}")

            if callback:
                callback(ranking)

        submit_button.on_click(on_submit)

        ui_container = widgets.VBox([output_box, ranking_buttons, submit_button])
        display(ui_container)


# %% [markdown]
# ### Train with the user data

# %%
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import json

class FeedbackDataset(Dataset):
    def __init__(self, file_path):
        self.data = []

        assert os.path.exists(file_path), f"Error: {file_path} does not exist. Please provide a valid file path, otherwise click the score system above."
        with open(file_path, "r") as f:
            feedback_data = json.load(f)

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        for entry in feedback_data:
            if isinstance(entry["feedback"], int):  # Rating data
                self.data.append({
                    "text": entry["response"],
                    "label": entry["feedback"] - 1  # Convert to range 0-4
                })
            elif isinstance(entry["feedback"], str) and " is better" in entry["feedback"]:  # Ranking data
                ranking = entry["feedback"]
                label = 0 if ranking == "A is better" else 1  # A vs B ranking
                text = f"A: {entry['response'].split('| B: ')[0]} B: {entry['response'].split('| B: ')[1]}"
                self.data.append({
                    "text": text,
                    "label": label
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoded = self.tokenizer(
            item["text"], truncation=True, padding="max_length", max_length=256, return_tensors="pt"
        )
        label = torch.tensor(item["label"])
        return encoded["input_ids"].squeeze(0), encoded["attention_mask"].squeeze(0), label

def train_reward_model():
    dataset = FeedbackDataset("rlhf_feedback.json")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Choose 5 classes (ratings 1-5) or 2 classes (A vs B ranking)
    labels = [dataset[i][2].item() for i in range(len(dataset))]  # Extract all labels
    num_labels = 5 if max(labels) > 1 else 2  # Determine whether it's rating data (5 classes) or ranking data (2 classes)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(5):  # Train for 5 epochs
        for input_ids, attention_mask, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask).logits
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1} Loss: {loss.item()}")

    model.save_pretrained("reward_model")
    print("✅ Reward model saved.")

# %% [markdown]
# ### Score the response

# %%
from transformers import BertForSequenceClassification, BertTokenizer
import torch

class LLMwithReward:
    def __init__(self, llm, llm_name_list, reward_model_path="reward_model"):
        """
        :param llm: Instance of LLMZoo
        :param llm_name_list: List of LLM names to iterate over
        :param reward_model_path: Path to the trained reward model
        """
        self.llm = llm  # Pass in the LLM instance
        self.llm_name_list = llm_name_list  # Store multiple LLM names
        self.reward_model = BertForSequenceClassification.from_pretrained(reward_model_path)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.reward_model.eval()

        # Get num_labels to determine whether it is a rating task (5 classes) or a ranking task (2 classes)
        self.num_labels = self.reward_model.config.num_labels

    def invoke(self, prompt, *args, **kwargs):
        """
        1. Iterate over `llm_name_list`, selecting a different LLM each time to generate responses.
        2. Use the reward model to compute the score for each response.
        3. Return the response with the highest score.
        """
        candidate_responses = []
        llm_names = []

        # **Iterate through LLM names and generate responses with each model**
        for llm_name in self.llm_name_list:
            self.llm.select_model(llm_name)  # Select LLM
            response = self.llm.invoke(prompt)  # Generate response

            candidate_responses.append(response)
            llm_names.append(llm_name)  # Record LLM name

        # **Compute reward scores**
        scores = []
        for response in candidate_responses:
            encoded = self.tokenizer(
                response, return_tensors="pt", truncation=True, padding="max_length", max_length=256
            )
            with torch.no_grad():
                logits = self.reward_model(**encoded).logits[0]  # Get logits vector
                
                if self.num_labels == 5:
                    score = torch.argmax(logits).item() + 1  # Rating task, take the highest index + 1 (restore 1-5)
                else:
                    score = torch.softmax(logits, dim=0)[1].item()  # Ranking task, take the probability of B being selected

            scores.append(score)

        # **Select the LLM with the highest score**
        best_index = scores.index(max(scores))
        best_response = candidate_responses[best_index]
        best_llm = llm_names[best_index]

        # print(f"✅ Best Response selected from {best_llm} with score {scores[best_index]:.2f}")
        return best_response


class MoEModel:
    def __init__(
        self,
        zoo: LLMZoo,
        expert_names: list[str],
        expert_descriptions: dict[str, str],
        top_k=1,
        aggregation_method="first",  # "first", "concat", "vote", "average"
    ):
        self.zoo = zoo
        self.expert_names = expert_names
        self.expert_descriptions = expert_descriptions
        self.top_k = top_k
        self.aggregation_method = aggregation_method
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        self.expert_vectors = {
            name: self.embedder.encode(desc, normalize_embeddings=True)
            for name, desc in expert_descriptions.items()
        }

    def __call__(self, prompt, *args, **kwargs):
        prompt_vec = self.embedder.encode(prompt, normalize_embeddings=True)
        scores = {
            name: np.dot(prompt_vec, self.expert_vectors[name])
            for name in self.expert_names
        }

        selected = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:self.top_k]

        results = {}
        for name, _ in selected:
            model = self.zoo.models[name]
            result = model(prompt, *args, **kwargs)
            results[name] = result

        return self.aggregate_results(results)

    def aggregate_results(self, results: dict[str, str]):
        if self.aggregation_method == "first":
            return next(iter(results.values()))

        elif self.aggregation_method == "concat":
            return "\n\n".join(
                [f"[{name}]: {res}" for name, res in results.items()]
            )

        elif self.aggregation_method == "vote":
            from collections import Counter
            votes = list(results.values())
            winner = Counter(votes).most_common(1)[0][0]
            return winner

        elif self.aggregation_method == "average":
            # This assumes results are numbers or strings convertible to float
            try:
                numbers = [float(v) for v in results.values()]
                return sum(numbers) / len(numbers)
            except:
                return "[Aggregation Error: average method failed]"

        else:
            return "[Aggregation Error: unknown aggregation method]"


# from sentence_transformers import SentenceTransformer
# import numpy as np
# from collections import Counter

# class MoEModel:
#     def __init__(
#         self,
#         zoo,
#         expert_names: list[str],
#         expert_descriptions: dict[str, str],
#         top_k=1,
#         aggregation_method="first",  # "first", "concat", "vote", "average"
#         use_learned_router=False,
#         learned_router=None,
#     ):
#         self.zoo = zoo
#         self.expert_names = expert_names
#         self.expert_descriptions = expert_descriptions
#         self.top_k = top_k
#         self.aggregation_method = aggregation_method
#         self.use_learned_router = use_learned_router
#         self.learned_router = learned_router
#         self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

#         self.expert_vectors = {
#             name: self.embedder.encode(desc, normalize_embeddings=True)
#             for name, desc in expert_descriptions.items()
#         }

#     def set_learned_router(self, router):
#         self.learned_router = router
#         self.use_learned_router = True

#     def __call__(self, prompt, *args, **kwargs):
#         if self.use_learned_router and self.learned_router:
#             selected = [self.learned_router.predict(prompt)]
#         else:
#             prompt_vec = self.embedder.encode(prompt, normalize_embeddings=True)
#             scores = {
#                 name: np.dot(prompt_vec, self.expert_vectors[name])
#                 for name in self.expert_names
#             }
#             selected = [name for name, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:self.top_k]]

#         results = {}
#         for name in selected:
#             model = self.zoo.models[name]
#             result = model(prompt, *args, **kwargs)
#             results[name] = result

#         return self.aggregate_results(results)

#     def aggregate_results(self, results: dict[str, str]):
#         if self.aggregation_method == "first":
#             return next(iter(results.values()))
#         elif self.aggregation_method == "concat":
#             return "\n\n".join([f"[{name}]: {res}" for name, res in results.items()])
#         elif self.aggregation_method == "vote":
#             votes = list(results.values())
#             winner = Counter(votes).most_common(1)[0][0]
#             return winner
#         elif self.aggregation_method == "average":
#             try:
#                 numbers = [float(v) for v in results.values()]
#                 return sum(numbers) / len(numbers)
#             except:
#                 return "[Aggregation Error: average method failed]"
#         else:
#             return "[Aggregation Error: unknown aggregation method]"

#     def gpt4o_score(self, prompt, results: dict[str, str], gpt4_model):
#         """
#         Use GPT-4o to pick the best result.
#         :param gpt4_model: a model instance with __call__(prompt) interface
#         :return: best model name
#         """
#         eval_prompt = f"""You are a helpful evaluator. Below is a user prompt and several AI-generated answers.

#         ## Prompt:
#         {prompt}

#         ## Answers:
#         """
#         for name, response in results.items():
#             eval_prompt += f"\n### Answer from [{name}]:\n{response.strip()}\n"

#         eval_prompt += "\nPlease select the best answer based on factuality, clarity, and completeness. Only reply with the model name like [phi4], [gemma3], etc."

#         judgment = gpt4_model(eval_prompt)

#         for name in results:
#             if f"[{name}]" in judgment:
#                 return name
#         return list(results.keys())[0]  # fallback

# %%
if __name__ == "__main__":
    llm = LLMZoo()

    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    api_version = "2023-05-15"
    assert api_key and endpoint and api_key

    # Register models with initialization parameters
    # llm.register_model("GPT-4o", GPT4AzureModel(api_key=api_key, endpoint=endpoint, api_version=api_version))
    # llm.select_model("GPT-4o")

    llm.register_model("OllamaModelt00", OllamaModel(model='phi4', temperature=0.0))
    llm.register_model("OllamaModelt02", OllamaModel(model='phi4', temperature=0.2))
    llm.register_model("OllamaModelt04", OllamaModel(model='phi4', temperature=0.4))
    llm_name_list = ['OllamaModelt00', 'OllamaModelt02', 'OllamaModelt04']

    llm.select_model("OllamaModelt00")
    print(llm.invoke("Who is the founder of Microsoft?"))

    llm.select_model("OllamaModelt02")
    print(llm.invoke("Who is the founder of Microsoft?"))

    llm.select_model("OllamaModelt04")
    print(llm.invoke("Who is the founder of Microsoft?"))

    # %%
    rlhf_interactive = LLMwithInteractive(llm)
    rlhf_interactive.invoke("What is Federated Learning?", mode="rating", callback=lambda x: print(f"✅ Rating received: {x}"))
    rlhf_interactive.invoke("What is Continue Learning?", mode="rating", callback=lambda x: print(f"✅ Rating received: {x}"))

    # %%
    rlhf_interactive = LLMwithInteractive(llm)
    response_b = "Federated Learning enables multiple devices to collaboratively train a model."
    rlhf_interactive.invoke("What is Federated Learning?", mode="ranking", response_b=response_b, callback=lambda x: print(f"✅ Ranking received: {x}"))

    # %%
    rlhf_interactive.invoke("What is Apple")

    # %%
    # Train the reward model
    train_reward_model()

    # %%
    # select llm with highest score
    rlhf_optimized_llm = LLMwithReward(llm=llm, llm_name_list=llm_name_list)

    # invoke the llm with highest score
    response = rlhf_optimized_llm.invoke("What is Federated Learning?")
    print("Final Optimized Response:", response)

# %%


# %%


