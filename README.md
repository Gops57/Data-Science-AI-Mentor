# Data Science & AI Mentor

An AI-powered learning assistant designed to help students understand
core Data Science topics through conversational questions and
student-friendly explanations.

## Project overview

The application uses a fine-tuned **Microsoft Phi-3 Mini** causal
language model and a **Streamlit** interface. The model was fine-tuned
using **QLoRA** with 4-bit quantization as part of the project workflow.

The training dataset contains **3,800 instruction--response pairs**
across seven subject areas:

  Subject                Number of pairs
  -------------------- -----------------
  Python                             800
  SQL                                800
  Statistics                         500
  Machine Learning                   500
  Deep Learning                      500
  Data Visualization                 400
  Data Wrangling                     300
  **Total**                    **3,800**

The dataset was split into training (80%), validation (10%), and test
(10%) sets.

## Features

-   Chat-based question and answer interface.
-   Seven selectable Data Science subjects.
-   Quick sample question for each subject.
-   Dark and light themes.
-   Adjustable text size, response style, and response length.
-   Optional timestamps and subject labels.
-   Conversation reset and print options.
-   Welcome animation using a Lottie JSON asset.

## Technology stack

-   Python
-   Streamlit
-   Microsoft Phi-3 Mini
-   PyTorch
-   Accelerate
-   Safetensors
-   QLoRA and 4-bit quantization (model fine-tuning workflow)

## Repository structure

The application code expects the animation file at
`assets/welcome_animation.json`.

``` text
Data-Science-AI-Mentor/
├── app_final.py
├── requirements.txt
├── README.md
├── assets/
│   └── welcome_animation.json
├── screenshots/
│   ├── screenshot_dark.png
│   ├── screenshot_light.png
│   └── screenshot_print.png
└── datasets/
    ├── testing_sample.json
    ├── training_sample.json
    └── validation_sample.json
```

Use your actual screenshot and dataset filenames. The dataset is not
loaded by `app_final.py` during normal chatbot inference; it is included
only if you choose to share the project data.

## Model files

**The trained model is not included in this GitHub repository.** The
model files are large and must be obtained separately.

The current `app_final.py` uses this model directory:

``` python
MODEL_PATH = "/content/merged_model"
```

Before running the app, place the extracted merged model in the
directory expected by your environment, or edit `MODEL_PATH` in
`app_final.py` to point to your model folder.

The model directory should contain the files required by Hugging Face
Transformers, including the model configuration, tokenizer files, and
model weight shards.

## Installation and setup

### 1. Clone or download this repository

``` bash
git clone <https://github.com/Gops57/Data-Science-AI-Mentor.git>
cd Data-Science-AI-Mentor
```

Alternatively, download the repository as a ZIP file and extract it.

### 2. Create and activate a virtual environment (recommended)

**Windows:**

``` bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

``` bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install the PyTorch build appropriate for your hardware from the
official PyTorch installation selector if the default installation does
not match your CPU/GPU setup.

### 4. Configure the model path

Open `app_final.py` and update `MODEL_PATH` to the location of your
extracted merged model.

For example, on Windows:

``` python
MODEL_PATH = r"C:\path\to\merged_model"
```

Ensure that the `assets` folder contains `welcome_animation.json`.

### 5. Run the Streamlit application

``` bash
streamlit run app_final.py
```

Streamlit will display a local URL in the terminal. Open that URL in
your browser.

## Important deployment note

The application loads the model when it starts and uses PyTorch and
Transformers for inference. Running it requires access to the model
files and sufficient system memory / GPU memory. The original project
was run in a Google Colab environment with GPU access; a temporary Colab
deployment link is not a permanent hosting solution.

## Training and evaluation

The model fine-tuning workflow used QLoRA and 4-bit quantization. The
dataset was divided into training, validation, and test sets (80:10:10).

Reported project evaluation results:

  Metric                         Result
  -------------------------- ----------
  Final training loss          0.898241
  Validation loss              0.994369
  Test loss                    1.003348
  Test mean token accuracy     0.736212

These are the recorded results from the project's training and
evaluation run; they are not a guarantee of performance on every prompt.

## Notes

-   Do not commit the large trained model files to this repository.
-   Do not upload API keys, passwords, access tokens, private files, or
    personal data.
-   Check dataset licensing and remove any material you do not have
    permission to redistribute before publishing dataset files.
-   The GitHub repository contains the application source and project
    documentation; it does not itself provide a hosted chatbot unless
    the model is separately configured and the app is deployed.

## Acknowledgements

-   Microsoft Phi-3 Mini
-   Hugging Face Transformers
-   Streamlit
-   PyTorch
