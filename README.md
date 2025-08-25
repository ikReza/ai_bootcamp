# AI Bootcamp Journey

Welcome to my AI/ML bootcamp learning repository! This project tracks my progress through various machine learning modules, from regression to deep learning.

## 🗂️ Project Structure

```
ai-bootcamp/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Project configuration
└── modules/                 # Learning modules
    ├── 01-regression/
    ├── 02-classification/
    ├── 03-unsupervised-learning/
    └── 04-deep-learning/
```

Each module contains:

- **assignments/**: Weekly assignments and solutions
- **notes/**: Personal learning notes and summaries
- **experiments/**: Code experiments and practice work
- **README.md**: Module-specific documentation

## 🚀 Setup Instructions

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. **Clone the repository:**

```bash
git clone <your-repo-url>
cd ai-bootcamp
```

2. **Set up the environment:**

```bash
# Install dependencies using uv
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install Jupyter kernel (for notebooks):**

```bash
uv run python -m ipykernel install --user --name ai-bootcamp
```

4. **Launch Jupyter:**

```bash
uv run jupyter lab
# or
uv run jupyter notebook
```

## 📚 Modules Overview

### Module 1: Regression Models and Evaluation

- Linear Regression
- Polynomial Regression
- Regularization (Ridge, Lasso)
- Model Evaluation Metrics

### Module 2: Classification Models and Evaluation

- Logistic Regression
- Decision Trees
- Random Forest
- Classification Metrics

### Module 3: Unsupervised Learning & Dimensionality Reduction

- K-Means Clustering
- Hierarchical Clustering
- PCA, t-SNE
- DBSCAN

### Module 4: Introduction to Deep Learning

- Neural Network Fundamentals
- TensorFlow/PyTorch Basics
- CNN, RNN Introduction

## 🛠️ Development Workflow

### Adding New Dependencies

````bash
# Add a new package
uv add package-name

# or if you have requirementx.txt file
uv add -r requirements.txt

### Running Code
```bash
# Run Python scripts
uv run python script.py

# Run Jupyter notebooks
uv run jupyter lab
````

## 🤝 Contributing / Forking

If you want to fork this repository:

1. **Fork the repo** on GitHub
2. **Clone your fork:**

```bash
git clone https://github.com/ikReza/ai-bootcamp.git
cd ai-bootcamp
```

3. **Set up the environment:**

```bash
uv sync
source .venv/bin/activate
```

4. **Create your own branch:**

```bash
git checkout -b your-learning-branch
```

5. **Start learning and coding!**

Feel free to adapt the structure to your learning style and add your own modules or experiments.

## 📊 Progress Tracking

- [x] Module 1: Regression Models
- [ ] Module 2: Classification Models
- [ ] Module 3: Unsupervised Learning
- [ ] Module 4: Deep Learning

## 🔗 Resources

- [Course Materials](#)
- [Additional Reading](#)
- [Useful Tools](#)

## 📄 License

This project is for educational purposes. Feel free to use and modify as needed for your learning journey.

---

**Happy Learning!** 🚀
