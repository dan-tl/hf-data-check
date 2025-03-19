# HF Dataset Sample Extractor

A tool to randomly extract and save samples from various image-text datasets on Hugging Face.

## Supported Datasets

- dandelin/redcaps
- dandelin/cc12m
- dandelin/wit
- dandelin/vg
- dandelin/sbu
- dandelin/cocokt
- dandelin/cc3m

## Installation

### Standard Installation (using pip)

```bash
# Clone the repository
git clone https://github.com/dan-tl/hf-data-check.git
cd hf-data-check

# Install dependencies
pip install -r requirements.txt

# Prepare configuration file
cp config.py.example config.py
# Edit config.py file to input your Hugging Face API key
```

### Using uv Package Manager (recommended)

[uv](https://github.com/astral-sh/uv) is a fast and reliable Python package manager.

```bash
# Install uv (if not already installed)
curl -sSf https://astral.sh/uv/install.sh | bash

# Clone the repository
git clone https://github.com/dan-tl/hf-data-check.git
cd hf-data-check

# Install dependencies
uv pip install -r requirements.txt

# Prepare configuration file
cp config.py.example config.py
# Edit config.py file to input your Hugging Face API key
```

## Usage

1. Set your Hugging Face API key in the `config.py` file.
2. Run the program with the following command:

```bash
python dataset_check.py
```

## Output

The following outputs are generated for each dataset:

- `samples_dataset_name` folder: Location where each dataset's samples are stored
- Each sample is saved as an image (`image_number.jpg`) and text (`caption_number.txt`) pair
- If the text has multiple items, each item is displayed on a separate line with an index

## Notes

- Some datasets are private and require login to Hugging Face.
- Downloading large datasets may take time.
