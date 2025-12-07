import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# Hyperparameters
NUM_CLIENTS = 10
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# Split dataset among clients
client_datasets = random_split(dataset, [len(dataset) // NUM_CLIENTS] * NUM_CLIENTS)

# Define SARAHClient
class SARAHClient:
    def __init__(self, client_id, dataset):
        self.client_id = client_id
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.current_weights = None
        self.previous_weights = None
        self.v_prev = None

    def receive_weights(self, current_weights, previous_weights, v_prev, need_full_grad):
        self.current_weights = current_weights
        self.previous_weights = previous_weights
        self.v_prev = v_prev
        self.need_full_grad = need_full_grad

    def compute_gradient(self):
        model = models.resnet18()
        model.load_state_dict(self.current_weights)
        model.to(DEVICE)
        model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

        if self.need_full_grad:
            # Compute full gradient
            full_gradient = None
            for inputs, targets in self.dataloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                if full_gradient is None:
                    full_gradient = [param.grad.clone() for param in model.parameters()]
                else:
                    for fg, param in zip(full_gradient, model.parameters()):
                        fg += param.grad
            full_gradient = [fg / len(self.dataloader) for fg in full_gradient]
            return full_gradient, True
        else:
            # Compute recursive gradient
            batch = next(iter(self.dataloader))
            inputs, targets = batch
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # Grad on current weights
            model.load_state_dict(self.current_weights)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            grad_current = [param.grad.clone() for param in model.parameters()]

            # Grad on previous weights
            model.load_state_dict(self.previous_weights)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            grad_previous = [param.grad.clone() for param in model.parameters()]

            # Compute v_new
            v_new = [gc - gp + vp for gc, gp, vp in zip(grad_current, grad_previous, self.v_prev)]
            return v_new, False

# Define SARAHServer
class SARAHServer:
    def __init__(self, model, clients):
        self.model = model
        self.clients = clients
        self.current_weights = model.state_dict()
        self.previous_weights = None
        self.v_prev = None

    def send_weights_to_client(self, client_id, need_full_grad):
        client = self.clients[client_id]
        client.receive_weights(self.current_weights, self.previous_weights, self.v_prev, need_full_grad)

    def receive_gradient_from_client(self, client_id, gradient, is_full_grad):
        if is_full_grad:
            self.v_prev = gradient
        else:
            self.v_prev = gradient

    def update_model(self, gradient):
        with torch.no_grad():
            for param, grad in zip(self.model.parameters(), gradient):
                param -= LEARNING_RATE * grad
        self.previous_weights = self.current_weights
        self.current_weights = self.model.state_dict()

# Initialize server and clients
global_model = models.resnet18(num_classes=10).to(DEVICE)
clients = [SARAHClient(client_id, client_datasets[client_id]) for client_id in range(NUM_CLIENTS)]
server = SARAHServer(global_model, clients)

# Training loop
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    permutation = torch.randperm(NUM_CLIENTS)
    server.previous_weights = server.current_weights
    server.v_prev = None

    for i, client_id in enumerate(permutation):
        need_full_grad = (i == 0)
        server.send_weights_to_client(client_id, need_full_grad)
        gradient, is_full_grad = clients[client_id].compute_gradient()
        server.receive_gradient_from_client(client_id, gradient, is_full_grad)
        server.update_model(gradient)

# Evaluate the model
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
global_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = global_model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

print(f"Test Accuracy: {100. * correct / total:.2f}%")