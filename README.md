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
- Python 3.11 or higher
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

First, get your GCP project ID from your current gcloud configuration:

```bash
gcloud config get-value project
```

Copy the example environment file and update with your GCP project details:

```bash
cp .env.example .env
```

Edit `.env` and set:
- `GCP_PROJECT_ID`: Your GCP project ID (use the output from the command above)
- `GCP_LOCATION`: Your preferred region (e.g., `us-central1`)
- `GCP_BUCKET_NAME`: Unique name for your GCS bucket (e.g., `mlops-bucket-<your-project-id>`)
- `PIPELINE_ROOT`: GCS path (e.g., `gs://your-bucket-name`)
- `BQ_DATASET`: BigQuery dataset name (e.g., `iris_dataset`)
- `BQ_TABLE`: BigQuery table name (e.g., `iris_data`)

### 3. Set Up GCP Resources

After configuring your `.env` file, run the following commands to create required GCP resources:

```bash
# Load environment variables first
source .env

# Create GCS bucket (update .env with the bucket name before running)
gsutil mb -p ${GCP_PROJECT_ID} -l ${GCP_LOCATION} gs://${GCP_BUCKET_NAME}

# Grant Cloud Storage Admin role to Compute Engine default service account
# This is required for the pipeline to read/write artifacts to GCS
PROJECT_NUMBER=$(gcloud projects describe ${GCP_PROJECT_ID} --format="value(projectNumber)")
gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/storage.admin"

# Download sample data
curl https://huggingface.co/datasets/scikit-learn/iris/raw/main/Iris.csv -o iris-setosa.csv

# Create BigQuery dataset and table
bq mk --dataset --location=${GCP_LOCATION} ${GCP_PROJECT_ID}:${BQ_DATASET}
bq load --source_format=CSV --autodetect ${GCP_PROJECT_ID}:${BQ_DATASET}.${BQ_TABLE} ./iris-setosa.csv
```

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
| `BASE_IMAGE` | Container base image | `python:3.11-slim` |

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
- Run the setup commands in step 3 to create the BigQuery table

### Model Deployment Fails

**"Model server never became ready" Error:**
- This occurs when there's a version mismatch between training and serving containers
- **Critical:** All components must use the same scikit-learn version (1.3.2) to match the serving container
- The fix includes:
  1. Pinning `scikit-learn==1.3.2` in all training/evaluation components
  2. Using the matching Vertex AI pre-built container `sklearn-cpu.1-3:latest`
  3. Properly structuring model artifacts with `model.joblib` in GCS
  4. Using `Model.upload()` instead of `upload_scikit_learn_model_file()`

**Other deployment issues:**
- Verify Vertex AI API is enabled
- Check that your service account has necessary permissions
- Ensure the model was successfully registered

## Clean Up

To avoid incurring charges, delete the GCP resources when you're done:

```bash
# Load environment variables
source .env

# Delete the Vertex AI endpoint (if deployed)
ENDPOINT_ID=$(gcloud ai endpoints list --region=${GCP_LOCATION} \
  --filter="displayName:${ENDPOINT_DISPLAY_NAME}" \
  --format="value(name)")
if [ ! -z "$ENDPOINT_ID" ]; then
  gcloud ai endpoints delete $ENDPOINT_ID --region=${GCP_LOCATION} --quiet
fi

# Undeploy and delete models from Vertex AI
MODEL_ID=$(gcloud ai models list --region=${GCP_LOCATION} \
  --filter="displayName:${MODEL_DISPLAY_NAME}" \
  --format="value(name)")
if [ ! -z "$MODEL_ID" ]; then
  gcloud ai models delete $MODEL_ID --region=${GCP_LOCATION} --quiet
fi

# Delete BigQuery dataset and all tables
bq rm -r -f -d ${GCP_PROJECT_ID}:${BQ_DATASET}

# Delete GCS bucket and all contents
gsutil rm -r gs://${GCP_BUCKET_NAME}

# Delete downloaded sample data
rm -f iris-setosa.csv

echo "Clean up completed!"
```

**Note**: These commands will permanently delete all resources and data. Make sure you have backed up anything you need before running them.

## Contributing

Contributions are welcome! Please ensure all changes:
- Follow PEP 8 style guidelines
- Include appropriate comments and documentation
- Are tested in a GCP environment

## License

This project is provided as-is for educational and demonstration purposes.
