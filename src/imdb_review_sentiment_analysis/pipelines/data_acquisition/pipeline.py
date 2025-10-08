from kedro.pipeline import Node, Pipeline
from .nodes import extract_data_from_files

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(func=extract_data_from_files, inputs=None, outputs='raw_data', name='extract_data_node')
    ])