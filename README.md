# GCP MLOps Pipeline

A production-ready Kubeflow pipeline for Iris classification on Google Cloud Platform, featuring automated model training, evaluation, registration, and deployment to Vertex AI.

## Features

- **Automated ML Pipeline**: End-to-end workflow from data loading to model deployment
- **Model Comparison**: Trains and compares Decision Tree and Random Forest models
- **Vertex AI Integration**: Automatic model registration and deployment
- **Flexible Configuration**: Easy configuration via environment variables
- **Conditional Deployment**: Optional model deployment based on configuration

## Prerequisites

- Google Cloud Platform account with billing enabled
- Python 3.10 or higher
- GCP CLI (`gcloud`) installed and authenticated
- Required GCP APIs enabled:
  - Vertex AI API
  - BigQuery API
  - Cloud Storage API

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example environment file and update with your GCP project details:

```bash
cp .env.example .env
```

Edit `.env` and set:
- `GCP_PROJECT_ID`: Your GCP project ID
- `GCP_LOCATION`: Your preferred region (e.g., `us-central1`)
- `GCP_BUCKET_NAME`: Unique name for your GCS bucket
- `PIPELINE_ROOT`: GCS path (e.g., `gs://your-bucket-name`)

### 3. Set Up GCP Resources

Run the setup commands to create required GCP resources:

```bash
# Follow the instructions in SETUP.md
bash -c "$(cat SETUP.md | grep -A 100 '```bash' | tail -n +2 | head -n -1)"
```

Or manually run the commands in [SETUP.md](SETUP.md).

### 4. Run the Pipeline

```bash
cd pipelines
python pipeline.py
```

The pipeline will:
1. Load data from BigQuery
2. Train Decision Tree and Random Forest models
3. Evaluate and select the best model
4. Register the model to Vertex AI Model Registry
5. Deploy the model to a Vertex AI endpoint (if `DEPLOY_MODEL=true`)

## Project Structure

```
gcp_mlops/
├── pipelines/
│   ├── components/         # Pipeline component definitions
│   │   ├── data.py        # Data loading from BigQuery
│   │   ├── models.py      # Model training (Decision Tree & Random Forest)
│   │   ├── evaluation.py  # Model evaluation and selection
│   │   └── deploy.py      # Model registration and deployment
│   └── pipeline.py        # Main pipeline definition
├── requirements.txt       # Python dependencies
├── SETUP.md              # GCP resource setup instructions
├── .env.example          # Environment variables template
└── README.md            # This file
```

## Pipeline Components

### 1. Data Loading
Loads the Iris dataset from BigQuery and splits it into training and test sets.

### 2. Model Training
- **Decision Tree**: Trains a decision tree classifier
- **Random Forest**: Trains a random forest classifier

### 3. Model Evaluation
Compares both models on the test set and selects the best performer based on accuracy.

### 4. Model Registration
Uploads the best model to Vertex AI Model Registry with proper metadata.

### 5. Model Deployment (Optional)
Deploys the registered model to a Vertex AI endpoint for online predictions.

## Configuration

Key environment variables in `.env`:

| Variable | Description | Example |
|----------|-------------|---------|
| `GCP_PROJECT_ID` | Your GCP project ID | `my-project-123` |
| `GCP_LOCATION` | GCP region | `us-central1` |
| `GCP_BUCKET_NAME` | GCS bucket name | `my-mlops-bucket` |
| `PIPELINE_ROOT` | Pipeline artifacts path | `gs://my-mlops-bucket` |
| `MODEL_DISPLAY_NAME` | Model name in registry | `iris-classifier` |
| `ENDPOINT_DISPLAY_NAME` | Endpoint name | `iris-endpoint` |
| `DEPLOY_MODEL` | Enable deployment | `true` or `false` |
| `BASE_IMAGE` | Container base image | `python:3.10-slim` |

## Usage

### Run Pipeline Without Deployment

Set `DEPLOY_MODEL=false` in `.env` to skip the deployment step:

```bash
DEPLOY_MODEL=false python pipelines/pipeline.py
```

### Monitor Pipeline Execution

View pipeline runs in the [Vertex AI Pipelines Console](https://console.cloud.google.com/vertex-ai/pipelines).

### Make Predictions

After deployment, use the Vertex AI endpoint to make predictions:

```python
from google.cloud import aiplatform

aiplatform.init(project="your-project-id", location="us-central1")
endpoint = aiplatform.Endpoint.list(
    filter='display_name="iris-endpoint"'
)[0]

predictions = endpoint.predict([[5.1, 3.5, 1.4, 0.2]])
print(predictions)
```

## Troubleshooting

### Pipeline Fails to Compile
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that environment variables are properly set in `.env`

### BigQuery Table Not Found
- Run the setup commands in [SETUP.md](SETUP.md) to create the BigQuery table

### Model Deployment Fails
- Verify Vertex AI API is enabled
- Check that your service account has necessary permissions
- Ensure the model was successfully registered

## Contributing

Contributions are welcome! Please ensure all changes:
- Follow PEP 8 style guidelines
- Include appropriate comments and documentation
- Are tested in a GCP environment

## License

This project is provided as-is for educational and demonstration purposes.
