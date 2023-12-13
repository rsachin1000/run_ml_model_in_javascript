import torch
from train import TrainTask


def convert_model_to_onnx(model):
    ''' converts the model to ONNX format '''
    # set the model to inference mode
    model.eval()

    # create a dummy input
    dummy_input = torch.randn(1, 3, 32, 32)
    # export the model
    torch.onnx.export(model, dummy_input, "../model.onnx")


if __name__ == "__main__":
    # load the model
    train_task = TrainTask()
    train_task.load_saved_model()

    # # convert the model to ONNX
    convert_model_to_onnx(train_task.model)