from kfp import dsl

# Define a component to add two numbers.
@dsl.component
def add(a: int, b: int) -> int:
    return a + b

# Define a simple pipeline using the component.
@dsl.pipeline
def addition_pipeline(x: int, y: int, z: int) -> int:
    task1 = add(a=x, b=y)
    task2 = add(a=task1.output, b=z)
    return task2.output

from kfp import local

local.init(runner=local.DockerRunner())

pipeline_task = addition_pipeline(x=1, y=2, z=3)
print(f'Result: {pipeline_task.output}')