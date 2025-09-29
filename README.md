# GCP MLOps Pipeline

A Kubeflow-based machine learning pipeline for Iris classification using Google Cloud Platform.

## Setup

1. Copy the environment variables template:
```bash
cp .env.example .env
```

2. Update the `.env` file with your GCP project details:
   - Set your GCP project ID
   - Configure your GCS bucket name
   - Adjust other parameters as needed

3. Install dependencies:
```bash
pip install -r pipelines/requirements.txt
```

4. Set up GCP resources by running the commands in [create_gcp_components.md](pipelines/create_gcp_components.md)

5. Run the pipeline:
```bash
cd pipelines
python iris_pipeline.py
```

## Pipeline Components

- **Data Loading**: Load Iris dataset from BigQuery
- **Model Training**: Train Decision Tree and Random Forest models
- **Model Evaluation**: Compare models and select the best performing one
- **Model Registration**: Upload the best model to Vertex AI Model Registry
