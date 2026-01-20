# NSFC-Writer: Intelligent Writing Assistant for NSFC Grant Proposals

An AI-powered writing assistant system for National Natural Science Foundation of China (NSFC) grant proposals, built on Large Language Model technology to enhance research proposal writing efficiency and quality.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Web Interface](#web-interface)
- [Advanced: Model Fine-tuning](#advanced-model-fine-tuning)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Best Practices](#best-practices)

---

## Overview

This system provides an intelligent writing support framework that enables:
- **Automated Generation** of grant proposal content
- **Scientific Review** with actionable feedback
- **Smart Literature Management** for reference integration

> Core entry point implementation: `main.py:17-19`

---

## Features

### 🖊️ Intelligent Content Generation

Automated generation of six core proposal modules covering the entire grant application workflow:

- Research Background & Significance
- Research Content
- Research Methodology
- Innovation Points
- Expected Outcomes
- Research Foundation

> Implementation: `generator.py:19-80`

### 📝 Smart Proposal Review

NLP-powered deep analysis of proposal drafts with comprehensive review support:

- Quantitative scoring for each module
- Logic and compliance diagnostics
- Targeted improvement suggestions
- Auto-optimized version output

> Implementation: `app.py:593-651`

### 📚 Intelligent Literature Management

Full lifecycle management of multi-format literature resources with vector database technology:

- Support for PDF, Word, Markdown, and other formats
- Automatic content parsing and vector database construction
- Topic-relevant literature retrieval
- Automatic citation recommendations during generation

> Configuration: `config.yaml:70-75`

### 🤖 Dual Model Architecture

Flexible and extensible model framework supporting two deployment modes:

| Mode | Description |
|------|-------------|
| **Ollama Mode** | Compatible with various open-source LLMs, enabling rapid deployment and switching |
| **Local Fine-tuned Mode** | Train custom models with domain-specific data for improved field adaptation |

> Implementation: `main.py:75-88`

---

## Quick Start

### Prerequisites

#### 1. Conda Environment Setup

```bash
conda create -n nsfc python=3.10
conda activate nsfc

# 进入 Pytorch 官网
# 根据 CUDA 版本选择对应的 Pytorch 版本
# 建议使用 pip 命令安装

pip install -r requirements.txt
```


#### 2. Ollama Installation (Recommended)

1. Download and install the Ollama client from [ollama.ai](https://ollama.ai)
2. Start the service and download the recommended model:

```bash
ollama serve                  # Start Ollama service
ollama pull qwen2.5:14b       # Download Qwen2.5-14B model
```

---

## Usage

### System Status Check

```bash
python main.py info
```

Displays Ollama service status, available models, and literature index status.

> Implementation: `main.py:160-223`

### Literature Import

**Single file:**
```bash
python main.py add_literature paper.pdf
```

**Batch import (directory):**
```bash
python main.py add_literature --directory ./papers/
```

> Implementation: `main.py:136-158`

### Web Interface (Recommended)

```bash
python main.py run
```

Access the visual interface at: `http://127.0.0.1:7860`

> Configuration: `config.yaml:86-88`

### Command Line Generation

**Single module:**
```bash
python main.py generate -t "Deep Learning-based Medical Image Analysis" -s research_background
```

**Complete proposal:**
```bash
python main.py generate -t "Your Research Topic" -s all -o output.md
```

> Implementation: `main.py:101-134`

---

## Web Interface

The web interface features a modular design with four core functional tabs:

| Tab | Description | Reference |
|-----|-------------|-----------|
| **Model Settings** | Switch between Ollama and local fine-tuned models, monitor model status | `app.py:799-844` |
| **Proposal Review** | Upload Word/PDF drafts, generate scoring reports, diagnostics, and export optimized versions | `app.py:846-920` |
| **Literature Management** | Drag-and-drop upload, automatic parsing, indexing, and smart citation matching | `app.py:922-952` |
| **Module Generation** | Input research topics, select modules, add domain-specific information, online editing and multi-format export | - |

---

## Advanced: Model Fine-tuning

For domain-specific customization, the system supports local model fine-tuning:

### Step 1: Prepare Training Data

Collect domain-relevant proposal samples in Markdown format and place them in the `data/raw/` directory.

### Step 2: Data Preprocessing

```bash
python main.py process_data --input_dir ./data/raw
```

Performs data cleaning, format standardization, and feature extraction.

> Implementation: `main.py:49-62`

### Step 3: Model Training

```bash
python main.py train --data ./data/processed/training_data.json
```

Fine-tunes the model with preprocessed data. The web interface will launch automatically after training completes.

> Implementation: `main.py:64-73`

---

## Project Structure

```
nsfc_writer/
├── configs/                    # Configuration files
│   └── config.yaml             # Main config (model, service, literature settings)
├── src/                        # Source code
│   ├── app.py                  # Web application core
│   ├── generator.py            # Proposal content generation
│   ├── literature_manager.py   # Literature management & retrieval
│   ├── trainer.py              # Model training & fine-tuning
│   └── ...                     # Other utility modules
├── data/                       # Data storage
│   ├── raw/                    # Raw training samples
│   └── literature_db/          # Literature vector database
├── models/                     # Model files (pretrained & fine-tuned)
├── main.py                     # Main entry point (CLI interface)
└── environment.yml             # Environment dependencies
```

---

## Configuration

All core configurations are managed in `configs/config.yaml`:

| Category | Parameters |
|----------|------------|
| **Model** | Base model selection, fine-tuning parameters, generation strategy |
| **Ollama** | Service address, port, model name |
| **Literature** | Vector model, chunk size, indexing parameters |
| **Generation** | Temperature, max tokens, output format |

> Full configuration: `config.yaml:1-88`

---

## Best Practices

| Recommendation | Details |
|----------------|---------|
| **First-time Users** | Start with Ollama models for quick setup without local training |
| **Literature Database** | Import high-quality journal papers, reviews, and successful proposals from your research field |
| **Writing Strategy** | Use module-by-module generation → individual optimization → final integration for better quality |
| **Review Process** | Utilize the review feature for multiple rounds of refinement, focusing on research background logic and innovation significance |
| **Parameter Tuning** | Higher temperature = more creative output; Lower temperature = more rigorous and standardized content |

---

