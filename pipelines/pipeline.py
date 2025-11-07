import os
import sys
import google.cloud.aiplatform as aip
import kfp
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

sys.path.append("src")

PIPELIE_NAME = os.getenv("PIPELINE_NAME", "The-Iris-Pipeline-v1")
PIPELINE_ROOT = os.getenv("PIPELINE_ROOT", "gs://your-bucket-name")


@kfp.dsl.pipeline(name=PIPELIE_NAME, pipeline_root=PIPELINE_ROOT)
def pipeline(
    project_id: str,
    location: str,
    bq_dataset: str,
    bq_table: str,
    model_display_name: str,
    endpoint_display_name: str,
    deploy_model_flag: bool = True,
):
    from components.data import load_data
    from components.evaluation import choose_best_model
    from components.models import decision_tree, random_forest
    from components.deploy import upload_model, deploy_model

    # Load data from BigQuery
    data_op = load_data(
        project_id=project_id, bq_dataset=bq_dataset, bq_table=bq_table
    ).set_display_name("Load Data from BigQuery")

    # Train Decision Tree model
    dt_op = decision_tree(
        train_dataset=data_op.outputs["train_dataset"]
    ).set_display_name("Train Decision Tree")

    # Train Random Forest model
    rf_op = random_forest(
        train_dataset=data_op.outputs["train_dataset"]
    ).set_display_name("Train Random Forest")

    # Choose best model based on evaluation
    choose_model_op = choose_best_model(
        test_dataset=data_op.outputs["test_dataset"],
        decision_tree_model=dt_op.outputs["output_model"],
        random_forest_model=rf_op.outputs["output_model"],
    ).set_display_name("Select Best Model")

    # Upload model to Vertex AI Model Registry
    upload_op = upload_model(
        project_id=project_id,
        location=location,
        model_display_name=model_display_name,
        model=choose_model_op.outputs["best_model"],
    ).set_display_name("Register Model to Vertex AI")

    # Conditionally deploy model to endpoint
    with kfp.dsl.Condition(deploy_model_flag == True, name="Deploy Model"):
        deploy_model(
            project_id=project_id,
            location=location,
            endpoint_display_name=endpoint_display_name,
            model=upload_op.outputs["vertex_model"],
        ).set_display_name("Deploy Model to Endpoint")


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline, package_path=f"pipeline.yaml"
    )

    # Initialize the AI Platform SDK
    aip.init(
        project=os.getenv("GCP_PROJECT_ID"),
        location=os.getenv("GCP_LOCATION"),
        staging_bucket=os.getenv("PIPELINE_ROOT"),
    )

    # Create an AI Platform PipelineJob
    job = aip.PipelineJob(
        display_name=os.getenv("PIPELINE_DISPLAY_NAME", "iris pipeline"),
        template_path=f"pipeline.yaml",
        pipeline_root=os.getenv("PIPELINE_ROOT"),
        parameter_values={
            "project_id": os.getenv("GCP_PROJECT_ID"),
            "location": os.getenv("GCP_LOCATION"),
            "bq_dataset": os.getenv("BQ_DATASET"),
            "bq_table": os.getenv("BQ_TABLE"),
            "model_display_name": os.getenv("MODEL_DISPLAY_NAME", "iris-classifier"),
            "endpoint_display_name": os.getenv("ENDPOINT_DISPLAY_NAME", "iris-endpoint"),
            "deploy_model_flag": os.getenv("DEPLOY_MODEL", "true").lower() == "true",
        },
        enable_caching=False,
    )

    # Run the pipeline job
    job.run(
    )
