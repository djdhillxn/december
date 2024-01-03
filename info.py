from model import HangmanGRUNet
model = HangmanGRUNet(hidden_dim=768, gru_layers=3) # replace with your actual model and parameters
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_parameters}")

