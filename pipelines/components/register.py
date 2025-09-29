import os
from kfp.dsl import Input, Model, component
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@component(
    base_image=os.getenv("BASE_IMAGE", "gcr.io/deeplearning-platform-release/tf2-cpu.2-6:latest"),
    packages_to_install=["google-cloud-aiplatform", "python-dotenv"],
)
def upload_model(
    project_id: str,
    location: str,
    model: Input[Model],
):
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=location)

    # Load environment variables inside the component
    import os
    from dotenv import load_dotenv
    load_dotenv()

    aiplatform.Model.upload_scikit_learn_model_file(
        model_file_path=model.path,
        display_name=os.getenv("MODEL_DISPLAY_NAME", "IrisModelv3"),
        project=project_id,
    )
