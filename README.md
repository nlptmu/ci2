# Complementary Intelligence Theory: When Does Human Oversight Add Value to AI?

## Overview

This project implements the theoretical framework and experimental methodology from the paper *"When Does Human Oversight Add Value to AI? A Theory of Complementary Intelligence in Content Generation"*. By analyzing large-scale TripAdvisor hotel review data and conducting user studies, we validate the **Complementary Intelligence Theory (CIT)** to understand when human-AI collaboration effectively improves content quality.

## Key Findings

- **Lexical Distinctiveness Matters**: The uniqueness of target audience vocabulary determines AI classification accuracy
  - Business/Solo travelers: High distinctiveness (77.5% F1-score)
  - Family travelers: Moderate distinctiveness (67.6%)
  - Friend/Couple travelers: Low distinctiveness (61.4%)

- **Non-Monotonic Human-in-the-Loop (HITL) Effectiveness**: Human oversight benefits follow an inverted-U pattern
  - Moderate distinctiveness (Family): Maximum improvement (+15.4 pp)
  - High distinctiveness (Business/Solo): Moderate improvement (+11.9 pp)
  - Low distinctiveness (Friend/Couple): No significant benefit (-2.5 pp)

- **Boundary Conditions**: When vocabulary overlap is too high, human experts cannot extract value from non-existent signals

## Project Structure

```
.
├── data_collection/          # Data collection module
│   └── TripAdvisorRequest.py  # Scrape hotel reviews from TripAdvisor
├── preprocessing/             # Data preprocessing module
│   └── DataPreprocessing.py   # Data cleaning, splitting, feature extraction
├── topic_modeling/            # Topic modeling and keyword extraction
│   ├── vis_business_pos.ipynb # Business/Solo positive reviews topic analysis
│   ├── vis_business_neg.ipynb # Business/Solo negative reviews topic analysis
│   ├── vis_family_pos.ipynb   # Family positive reviews topic analysis
│   ├── vis_family_neg.ipynb   # Family negative reviews topic analysis
│   ├── vis_friend_pos.ipynb   # Friend/Couple positive reviews topic analysis
│   └── vis_friend_neg.ipynb   # Friend/Couple negative reviews topic analysis
├── multitask_learning/        # Multi-task learning module
│   ├── DatasetModule.py       # Data loading and preparation
│   ├── ModelModule.py         # Deep learning model architecture
│   ├── Trainer.py             # Model training and validation
│   └── Predictor.py           # Model prediction
├── explainability/            # Model interpretability module
│   └── ModelExplainer.py      # LIME and SHAP explanations
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Core Modules

### 1. Data Collection (data_collection)

**TripAdvisorRequest.py**
- Scrape hotel reviews from TripAdvisor via API
- Support pagination and data validation
- Handle traveler type labels and multi-dimensional ratings (Value, Location, Rooms, Service, Sleep Quality, Cleanliness)

### 2. Data Preprocessing (preprocessing)

**DataPreprocessing.py**
- Load raw review data
- Clean missing values and outliers
- Create train/validation/test splits (60%/20%/20%)
- Extract relevant features and construct labels

```python
python preprocessing/DataPreprocessing.py \
    --input_file data/raw/reviews.txt \
    --output_dir data/processed/ \
    --test_size 0.2 \
    --eval_size 0.1 \
    --seed 1207
```

### 3. Topic Modeling (topic_modeling)

This module implements **BERTopic** for unsupervised topic modeling and keyword extraction from hotel reviews. The analysis is conducted separately for each traveler segment (Business/Solo, Family, Friend/Couple) and sentiment polarity (positive/negative), resulting in six Jupyter notebooks.

**Key Techniques:**
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2) for semantic text representation
- **Dimensionality Reduction**: UMAP (Uniform Manifold Approximation and Projection)
- **Clustering**: HDBSCAN (Hierarchical Density-Based Spatial Clustering)
- **Topic Representation**: Class-based TF-IDF with KeyBERT-inspired fine-tuning

**Workflow:**
1. **Load and Filter Data**: Select reviews by traveler type and rating thresholds
   - Positive: 4-5 stars on key aspects (Value, Location, Rooms)
   - Negative: 1-2 stars on key aspects
2. **Text Preprocessing**: Tokenization, lemmatization, stopword removal
3. **Topic Modeling**: BERTopic pipeline extracts semantic topics
4. **Topic Reduction**: Consolidate similar topics (default: 8 topics per segment)
5. **Keyword Extraction**: Extract top 30-100 keywords per topic
6. **Visualization**: Generate interactive topic visualizations
   - Topic distance maps (intertopic distance)
   - Document-topic mappings
   - Topic bar charts
   - Topic similarity heatmaps

**Notebooks:**
- `vis_business_pos.ipynb` / `vis_business_neg.ipynb`: Business/Solo traveler topics
- `vis_family_pos.ipynb` / `vis_family_neg.ipynb`: Family traveler topics
- `vis_friend_pos.ipynb` / `vis_friend_neg.ipynb`: Friend/Couple traveler topics

### 4. Multi-Task Learning (multitask_learning)

This module uses the PyTorch Lightning framework to implement joint traveler type classification and aspect-based sentiment analysis.

**Architecture Features:**
- **Encoder**: RoBERTa-base transforms text into contextualized embeddings
- **Feature Extraction**: TextCNN layers (filter sizes 3, 4, 5) extract n-gram features
- **Multi-Task Learning**: Simultaneously predict traveler type and 6-dimensional aspect ratings, learning segment-specific relationships through shared representations

**Model Architecture:**
```
RoBERTa Encoder
        ↓
TextCNN Layers (3, 4, 5-grams)
        ↓
    ├─→ Traveler Type Classifier (3 classes)
    └─→ Aspect Sentiment Predictors (6 tasks)
```

**Training Example:**
```python
python multitask_learning/Trainer.py \
    --num_labels 3 \
    --train_file data/processed/train.json \
    --eval_file data/processed/val.json \
    --test_file data/processed/test.json \
    --output_dir outputs/ \
    --model bert-cnn \
    --max_epochs 10 \
    --train_batch_size 16 \
    --learning_rate 2e-5 \
    --multi_task
```

**Supported Models:**
- `bert-cnn`: BERT + TextCNN (recommended)
- `bert`: Standard BERT

### 5. Model Explainability (explainability)

**ModelExplainer.py**

Uses LIME and SHAP to provide local and global explanations for model predictions:

- **LIME** (Local Interpretable Model-agnostic Explanations)
  - Generate feature importance for each prediction
  - Output interactive HTML visualizations
  
- **SHAP** (SHapley Additive exPlanations)
  - Game-theoretic approach to feature attribution
  - Support detailed explanations for multi-class tasks

```python
python explainability/ModelExplainer.py \
    --input_file data/processed/test.json \
    --output_dir outputs/explanations/ \
    --model_path outputs/checkpoint/ \
    --num_instances 100
```

## Dependencies

```
pandas                  # Data manipulation
torch                   # Deep learning framework
pytorch-lightning       # Training management
transformers           # Pre-trained language models
datasets               # Dataset handling
lime                   # Model interpretability
shap                   # Shapley explanations
numpy                  # Numerical computing
scikit-learn           # Machine learning tools
matplotlib             # Plotting
seaborn                # Statistical visualization
tqdm                   # Progress bars
bertopic               # Topic modeling
sentence-transformers  # Semantic embeddings
umap-learn             # Dimensionality reduction
hdbscan                # Clustering algorithm
```

## Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## License

See [LICENSE](LICENSE) file