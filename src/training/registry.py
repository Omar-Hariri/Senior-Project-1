from src.pipelines.sensor.rf_pipeline import run as run_rf_pipeline
from src.pipelines.sensor.svm_pipeline import run as run_svm_pipeline
from src.pipelines.sensor.xgb_pipeline import run as run_xgb_pipeline
from src.pipelines.sensor.lr_pipeline import run as run_lr_pipeline
from src.pipelines.sensor.lstm_pipeline import run as run_lstm_pipeline
from src.pipelines.vision.yolo_pipeline import run as run_yolo_pipeline

PIPELINES = {
    "rf": run_rf_pipeline,
    "svm": run_svm_pipeline,
    "xgb": run_xgb_pipeline,
    "lr": run_lr_pipeline,
    "lstm": run_lstm_pipeline,
    "yolo": run_yolo_pipeline,
}
