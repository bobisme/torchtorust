import torch
import torch.nn as nn
import torch.optim as optim

from model import XORNet

MAX_ITERS = 10


def train(model, criterion, optimizer, num_epochs=1000, batch_size=64):
    for epoch in range(num_epochs):
        model.train()

        # Generating random data for training
        seq1 = torch.randint(0, 2, (batch_size, 8)).float().cuda()  # Batch size of 64
        seq2 = torch.randint(0, 2, (batch_size, 8)).float().cuda()
        inputs = torch.cat((seq1, seq2), dim=1)
        targets = (seq1 != seq2).float()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")


def save_model(model, file_name="xor_model.pth"):
    torch.save(model.state_dict(), file_name)
    print(f"Model saved to {file_name}")


if __name__ == "__main__":
    model = XORNet().cuda()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    train(model, criterion, optimizer, num_epochs=50_000, batch_size=1024)

    print("sampling model")
    for _ in range(4):
        seq1 = torch.randint(0, 2, (1, 8)).float().cuda()  # Batch size of 64
        seq2 = torch.randint(0, 2, (1, 8)).float().cuda()
        inputs = torch.cat((seq1, seq2), dim=1)
        targets = (seq1 != seq2).float()
        sample = model(inputs)
        print(f"{seq1.round().int().tolist()}")
        print(f"{seq2.round().int().tolist()}")
        print(f"expected: {targets.round().int().tolist()}")
        print(f"actual:   {sample.round().int().tolist()}")

    save_model(model, "xor_model.pth")
    print("saved to xor_model.pth")
    seq1 = torch.randint(0, 2, (1, 8)).float().cuda()  # Batch size of 64
    seq2 = torch.randint(0, 2, (1, 8)).float().cuda()
    inputs = torch.cat((seq1, seq2), dim=1)
    torch.onnx.export(
        model,
        inputs,
        "xor_model.onnx",
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print("saved to xor_model.onnx")
