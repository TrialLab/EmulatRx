# EmulatRx Framework

**EmulatRx** (EmulatRx) is an agentic framework designed to automate **Clinical Trial Design (CTD)** and **Target Trial Emulation (TTE)**. By leveraging a multi-agent architecture (Supervisor, Trialist, Informatician, Statistician, Clinician), it autonomously maps natural language trial protocols to structured Real-World Data (RWD) and performs causal inference analysis.

This repository contains the implementation of the EmulatRx framework as described in the manuscript *"EmulatRx: Empowering Clinical Trial Design with Agentic Intelligence and Real World Data"*.

## 🚀 Features

* **Multi-Agent Workflow:** Orchestrates specialized agents via [LangGraph](https://langchain-ai.github.io/langgraph/) to handle trial parsing, cohort construction, and statistical analysis.
* **Hybrid LLM Support:** Seamlessly switch between cloud-based models (Azure OpenAI, GPT-4o) and local models (Phi-4, DeepSeek-R1, Llama 3) via **Ollama**.
* **RLHF & Optimization:** Includes a native Reinforcement Learning from Human Feedback (RLHF) loop (`llm_zoo_rlhf.py`) to train reward models and optimize agent responses based on user ratings.
* **Automated Trial Parsing:** Extracts and standardizes eligibility criteria from ClinicalTrials.gov using LLMs.
* **Real-World Evidence Generation:** Maps criteria to OMOP/MIMIC-IV schemas and performs covariate balancing (PSM/IPTW) and survival analysis.

## 🛠️ Prerequisites & Installation

### 1. System Requirements
* **Python 3.10+**
* **Ollama** (Required for local model execution)
* Access to MIMIC-IV or a compatible OMOP-CDM database.

### 2. Install Dependencies
Clone the repository and install the required Python packages:

```bash
git clone [https://github.com/your-username/triallab.git](https://github.com/your-username/triallab.git)
cd triallab
```

### 3. Setting up Local Models (Ollama)
To use local models like Phi-4 or DeepSeek-R1 as described in the paper, you must install and run Ollama.

1.  **Download Ollama:** Visit ollama.com and install the version for your OS.
2.  **Pull Required Models:** Run the following commands in your terminal to download the weights:

```bash
ollama pull phi4
```

**Verify Server:** Ensure the Ollama server is running (default: localhost:11434). The `llm_zoo_rlhf.py` script connects to this local server automatically.

### 4. API Configuration
Create a `.env` file in the root directory to configure your API keys for cloud models (if used):

```ini
# Standard OpenAI Configuration
OPENAI_API_KEY=your_key_here

# Project Settings
LOG_LEVEL=INFO
```

## 🏃 Usage
The core workflow is driven by the Jupyter Notebook `v1-codebase.ipynb`.

### Running an Emulation
Launch Jupyter:
```bash
jupyter notebook v1-codebase.ipynb
```

**Select LLM Backend:** In the configuration cell, you can choose between OpenAI or local Ollama models using the LLMZoo registry:

Run the cells sequentially.

## 📂 Repository Structure

* `v1-codebase.ipynb`: Main entry point controlling the LangGraph state machine and agent coordination.
* `llm_zoo_rlhf.py`: The core Model Registry.
    * Handles connections to **Ollama** (Local) and **Azure OpenAI** (Cloud).
    * Implements **RLHF** data collection and reward model training (FeedbackDataset, LLMwithReward).
    * Implements **Mixture of Experts (MoE)** routing logic.
* `trialistUtils/`:
    * `main.py`: Logic for fetching and parsing ClinicalTrials.gov data.
    * `prompts.py`: System prompts for extracting and standardizing eligibility criteria.
    * `nlp_utils.py`: NLP helper functions for text processing.
* `dataframe_utils.py`: Helper functions for the **Informatician Agent** (SQL generation, data cleaning, cohort building).
* `emulation_utils.py`: Statistical tools for the **Statistician Agent** (Propensity Score Matching, SMD calculation, Survival Analysis).
* `trials/`: Output directory where generated trial data (CSVs, JSONs) and analysis reports are cached.