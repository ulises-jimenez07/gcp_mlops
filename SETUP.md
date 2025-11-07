```bash
# Load environment variables first
source .env

# Create GCS bucket
gsutil mb gs://${GCP_BUCKET_NAME}

# Download sample data
curl https://huggingface.co/datasets/scikit-learn/iris/raw/main/Iris.csv -o iris-setosa.csv

# Create BigQuery dataset and table
bq mk --dataset ${BQ_DATASET}
bq load --source_format=CSV --autodetect ${BQ_DATASET}.${BQ_TABLE} ./iris-setosa.csv
```