import os
from kfp.dsl import Input, Model, Output, Artifact, component
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@component(
    base_image=os.getenv("BASE_IMAGE", "python:3.10-slim"),
    packages_to_install=["google-cloud-aiplatform", "scikit-learn", "joblib"],
)
def upload_model(
    project_id: str,
    location: str,
    model_display_name: str,
    model: Input[Model],
    vertex_model: Output[Artifact],
):
    """Upload trained model to Vertex AI Model Registry."""
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=location)

    # Upload the scikit-learn model to Vertex AI
    uploaded_model = aiplatform.Model.upload_scikit_learn_model_file(
        model_file_path=model.path,
        display_name=model_display_name,
        project=project_id,
        location=location,
    )

    # Store the model resource name for deployment
    vertex_model.uri = uploaded_model.resource_name
    vertex_model.metadata["model_id"] = uploaded_model.name
    vertex_model.metadata["display_name"] = model_display_name


@component(
    base_image=os.getenv("BASE_IMAGE", "python:3.10-slim"),
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
