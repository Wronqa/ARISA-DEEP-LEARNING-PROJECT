import argparse
import os
import json
import time
from datetime import datetime
import neptune
import matplotlib.pyplot as plt





from src.MedMnist.entity.dataset import load_data
from src.MedMnist.pipeline.train import optimize_hyperparameters, optimize_one
from src.MedMnist.conponents.custom_cnn import experiment_models
from src.MedMnist.logging.logger import logger
from src.MedMnist.conponents.image_net import use_image_net
from src.MedMnist.pipeline.train_choosen import train_choosen_model
from src.MedMnist.pipeline.train_all import train_all_models    
from src.MedMnist.pipeline.fine_tune import fine_tune_model
from dotenv import load_dotenv







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate models.")
    parser.add_argument("--experiment-name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--mode", type=str, default='train_all', help="Mode: train_all, train_single, fine_tune...")
    parser.add_argument("--choosen-model", type=str, help="Specify a model version to train/fine_tune")
    parser.add_argument("--fine-tune-params-path", type=str, help="Path to the fine-tune parameters JSON file.")
    args = parser.parse_args()

    experiment_name = args.experiment_name
    choosen_model = args.choosen_model
    mode = args.mode
    fine_tune_params_path = args.fine_tune_params_path

    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = f"experiments/{experiment_name}"
    os.makedirs(experiment_dir, exist_ok=True)

    
    logger.info("Starting pipeline for experiment: %s at %s", experiment_name, current_date)

    load_dotenv()
    api_token = os.getenv("NEPTUNE_API_TOKEN")
    if not api_token:
        raise ValueError("NEPTUNE_API_TOKEN not found in environment variables.")

    run = neptune.init_run(
    project="pgawzynski.backup/ArisaDeepLearning",
    api_token=api_token,
    )
    
    run["sys/name"] = experiment_name  # nazwa runa

    try:
        if mode == "train_single":
            logger.info("Training single model: %s", choosen_model)
            train_choosen_model(experiment_name, experiment_dir, choosen_model, run)

        elif mode == "train_all":
            logger.info("Training all models.")
            train_all_models(experiment_name, experiment_dir, run)

        elif mode == "fine_tune":
            logger.info("Fine-tuning model: %s", choosen_model)
            fine_tune_model(experiment_name, experiment_dir, choosen_model, fine_tune_params_path, run)
        elif mode == 'use_image_net':
            logger.info("Using ImageNet model")
            x_train, y_train, x_test, y_test, num_classes = load_data()
            x_train, y_train, x_test, y_test, num_classes = load_data()
            model, history, test_loss, test_acc = use_image_net(x_train, y_train, x_test, y_test, num_classes, run)

        else:
            logger.error(f"Unknown mode: {mode}. Available modes: train_all, train_single, fine_tune.")

    finally:

        run.stop()

    logger.info("Pipeline completed.")