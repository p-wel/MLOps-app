"""
This is a boilerplate pipeline
generated using Kedro 0.18.10
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import make_predictions, report_accuracy, load_and_split_data, prepare_raw_data, train_model, \
    send_data_to_wandb, create_model, train_model_automl


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=prepare_raw_data,
                inputs=[],
                outputs='source_file',
                name="data_preparation"
            ),
            node(
                func=load_and_split_data,
                inputs=["source_file", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="load_and_split",
            ),

            node(
                func=create_model,
                inputs=None,
                outputs="MODEL",
                name="model_creation"
            ),
            node(
                func=train_model,
                inputs=["MODEL", 'X_train', 'y_train'],
                outputs=["TRAINED_MODEL", "traininglog_file"],
                name="model_training"
            ),
            node(
                func=train_model_automl,
                inputs=["X_train", "y_train"],
                outputs="AutomlModel",
                name="train_autogluon_model",
            ),
            node(
                func=send_data_to_wandb,
                inputs='traininglog_file',
                outputs=None,
                name="sending_data_to_wandb"
            ),
            node(
                func=make_predictions,
                inputs=["TRAINED_MODEL", "X_test", "y_test"],
                outputs="y_pred_normal",
                name="make_predictions_normal_model",
            ),
            node(
                func=report_accuracy,
                inputs="y_pred_normal",
                outputs=None,
                name="report_accuracy",
            )
        ]
    )
