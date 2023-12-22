import torch
import torch.nn as nn
import torch.utils.data as data

class RandomSequenceDataset(data.Dataset):
    def __init__(self, num_samples, sequence_length, input_size):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.input_size = input_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # Generate a random input sequence
        input_sequence = torch.rand(self.sequence_length, self.input_size)
        return input_sequence



class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        # If no hidden state is provided, initialize it
        if hidden is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            hidden = (h0, c0)
        else:
            # Adjust the dimensions of the hidden states
            h, c = hidden
            hidden = (h[:, :x.size(0), :], c[:, :x.size(0), :])
        
        # LSTM forward pass
        out, hidden = self.lstm(x, hidden)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out, hidden


input_size = 8  # Size of your input (e.g., one-hot encoded vector size)
hidden_size = 128  # Number of LSTM hidden units
output_size = 16  # Size of your vocabulary
num_layers = 1  # Number of LSTM layers

# Parameters
num_samples = 1000  # Number of random sequences
sequence_length = 10  # Length of each sequence

# Create the dataset and dataloader
dataset = RandomSequenceDataset(num_samples, sequence_length, input_size)
dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)


model = LSTMModel(input_size, hidden_size, output_size, num_layers)



# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# # Assuming each sequence has a fixed length of 10 for simplicity
# input_sequences = torch.rand((100, 10, input_size))  # 100 sequences, each of length 10
# target_sequences = torch.randint(0, output_size, (100, 10))  # Random target sequences

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for input_sequences in dataloader:
        # Generate random target sequences for each batch
        target_sequences = torch.randint(0, output_size, (input_sequences.size(0), sequence_length))
        
        outputs, _ = model(input_sequences)
        loss = criterion(outputs, target_sequences[:, -1])  # Using the last token as target for simplicity
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
