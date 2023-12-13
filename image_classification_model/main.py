import logging

from train import TrainTask


logging.basicConfig(handlers=[logging.StreamHandler()], level=logging.INFO)
logger = logging.getLogger()


if __name__ == '__main__':
    logger.info("Training model")
    train_task = TrainTask()
    train_task.train()
    logger.info("Training complete")
    
    # Load the saved model
    # train_task.load_saved_model()

    # Save the model checkpoint
    logger.info("Saving model checkpoint")
    train_task.save()
    logger.info("Model checkpoint saved")

    # Test Prediction on a sample batch
    logger.info("Making predictions on sample batch")
    train_task.predict_on_one_random_batch()
    logger.info("Predictions complete")

    # Get the model accuracy
    logger.info("Getting model accuracy")
    train_task.get_model_accuracy()
    logger.info("Accuracy calculated")
