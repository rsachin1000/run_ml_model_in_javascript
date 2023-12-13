import logging
import torch
import torch.nn as nn
import torch.optim as optim

from model import Net
from data import load_data, imshow, CLASSES


logger = logging.getLogger(__name__)

MODEL_PATH = "./CIFAR_net.pth"


class TrainTask:
    def __init__(
        self, 
        learning_rate: float = 0.001, 
        momentum: float = 0.9,
    ) -> None:
        self.model = Net()

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=learning_rate, 
            momentum=momentum
        )
        self.trainloader, self.testloader = load_data()

    def load_saved_model(self) -> None:
        self.model.load_state_dict(torch.load(MODEL_PATH))
        logger.info(f"Loaded model from checkpoint: {MODEL_PATH}")

    def train(self, num_epochs: int = 2):
        ''' Trains the model 
        :param num_epochs: number of epochs to train the model
        '''
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:
                    # print every 2000 mini-batches
                    logger.info(f"[{epoch + 1}, {i + 1}], Loss: {running_loss / 2000}")
                    running_loss = 0.0
    
    def save(self, path: str = MODEL_PATH) -> None:
        ''' saves the model '''
        torch.save(self.model.state_dict(), path)

    def predict_on_one_random_batch(self):
        """ Predicts on one random batch """
        dataiter = iter(self.testloader)
        images, labels = next(dataiter)
        # imshow(images=images)

        logger.info(
            'GroundTruth: ' + ' '.join(CLASSES[labels[j]] for j in range(4))
        )
        outputs = self.model(images)
        print(outputs)
        _, predicted = torch.max(outputs, 1)
        print(predicted)
        logger.info(
            'Predicted: ' + ' '.join(CLASSES[predicted[j]] for j in range(4))
        )

    def get_model_accuracy(self):
        ''' Gets the model accuracy '''
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        logger.info(
            f"Accuracy of the network on the 10000 test images: {100 * correct / total} %"
        )


if __name__ == "__main__":
    train_task = TrainTask()
    train_task.load_saved_model()
    train_task.predict_on_one_random_batch()