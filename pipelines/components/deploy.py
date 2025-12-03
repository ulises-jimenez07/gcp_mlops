import os
from kfp.dsl import Input, Model, Output, Artifact, component
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@component(
    base_image=os.getenv("BASE_IMAGE", "python:3.11-slim"),
    packages_to_install=["google-cloud-aiplatform", "google-cloud-storage", "scikit-learn==1.3.2", "joblib"],
)
def upload_model(
    project_id: str,
    location: str,
    model_display_name: str,
    model: Input[Model],
    vertex_model: Output[Artifact],
):
    """Upload trained model to Vertex AI Model Registry with custom serving container."""
    from google.cloud import aiplatform, storage
    import os
    import shutil
    import time
    import re

    aiplatform.init(project=project_id, location=location)

    # Create proper model artifact structure for sklearn serving
    model_dir = "/tmp/model_artifacts"
    os.makedirs(model_dir, exist_ok=True)
    model_artifact_path = os.path.join(model_dir, "model.joblib")

    # Copy the model to the expected location
    shutil.copy(model.path, model_artifact_path)

    # Upload model artifacts to GCS
    # Get bucket from the model's existing URI (which is already in GCS from KFP)
    storage_client = storage.Client(project=project_id)

    # Extract bucket name from model.uri
    # Model.uri looks like: gs://bucket-name/path/to/model
    model_uri = model.uri if hasattr(model, 'uri') and model.uri else model.path
    if model_uri.startswith('gs://'):
        match = re.match(r'gs://([^/]+)', model_uri)
        bucket_name = match.group(1) if match else f"{project_id}-mlops"
    else:
        bucket_name = f"{project_id}-mlops"

    # Create a unique GCS path for the model artifacts
    timestamp = int(time.time())
    gcs_blob_path = f"models/{model_display_name}/{timestamp}"
    gcs_model_path = f"gs://{bucket_name}/{gcs_blob_path}"

    # Get bucket (should already exist from pipeline setup)
    bucket = storage_client.bucket(bucket_name)

    # Upload model.joblib to GCS
    blob = bucket.blob(f"{gcs_blob_path}/model.joblib")
    blob.upload_from_filename(model_artifact_path)

    print(f"Model uploaded to: {gcs_model_path}")

    # Upload using Vertex AI with custom serving container
    # Using sklearn 1.3 pre-built container for better compatibility
    uploaded_model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=gcs_model_path,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        project=project_id,
        location=location,
    )

    # Store the model resource name for deployment
    vertex_model.uri = uploaded_model.resource_name
    vertex_model.metadata["model_id"] = uploaded_model.name
    vertex_model.metadata["display_name"] = model_display_name


@component(
    base_image=os.getenv("BASE_IMAGE", "python:3.11-slim"),
    packages_to_install=["google-cloud-aiplatform"],
)
def deploy_model(
    project_id: str,
    location: str,
    endpoint_display_name: str,
    model: Input[Artifact],
    machine_type: str = "n1-standard-2",
    min_replica_count: int = 1,
    max_replica_count: int = 1,
):
    """Deploy model to Vertex AI endpoint."""
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=location)

    # Get the model from the registry
    vertex_model = aiplatform.Model(model.uri)

    # Create or get endpoint
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_display_name}"',
        order_by="create_time desc",
        project=project_id,
        location=location,
    )

    if len(endpoints) > 0:
        endpoint = endpoints[0]
        print(f"Using existing endpoint: {endpoint.display_name}")
    else:
        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_display_name,
            project=project_id,
            location=location,
        )
        print(f"Created new endpoint: {endpoint.display_name}")

    # Deploy model to endpoint
    vertex_model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=model.metadata.get("display_name", "iris-model"),
        machine_type=machine_type,
        min_replica_count=min_replica_count,
        max_replica_count=max_replica_count,
        traffic_percentage=100,
    )

    print(f"Model deployed to endpoint: {endpoint.resource_name}")
