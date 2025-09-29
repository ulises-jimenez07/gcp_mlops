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
def pipeline(project_id: str, location: str, bq_dataset: str, bq_table: str):
    from components.data import load_data
    from components.evaluation import choose_best_model
    from components.models import decision_tree, random_forest
    from components.register import upload_model

    data_op = load_data(
        project_id=project_id, bq_dataset=bq_dataset, bq_table=bq_table
    ).set_display_name("Load data from BigQuery")

    dt_op = decision_tree(
        train_dataset=data_op.outputs["train_dataset"]
    ).set_display_name("Decision Tree")

    rf_op = random_forest(
        train_dataset=data_op.outputs["train_dataset"]
    ).set_display_name("Random Forest")

    choose_model_op = choose_best_model(
        test_dataset=data_op.outputs["test_dataset"],
        decision_tree_model=dt_op.outputs["output_model"],
        random_forest_model=rf_op.outputs["output_model"],
    ).set_display_name("Select best Model")

    upload_model(
        project_id=project_id,
        location=location,
        model=choose_model_op.outputs["best_model"],
    ).set_display_name("Register Model")


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
                "bq_table": os.getenv("BQ_TABLE")
            },
        enable_caching=False,
    )

    # Run the pipeline job
    job.run(
    )
