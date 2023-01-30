import torch
from torch import nn, optim

import matplotlib.pyplot as plt

from neuralode.neuralode import NeuralODE


DYNAMICS_A = torch.tensor([[-0.1, -0.5], [0.5, -0.1]])


def gen_data(n_samples, n_trajectories, batch_size, t_final):
    t = torch.linspace(0, 10, n_samples)

    ys = torch.zeros((batch_size, 2, n_samples, n_trajectories))
    ys[:, :, 0, :] = torch.rand(
        (batch_size, 2, n_trajectories)
    )  # random starting position

    for i in range(1, n_samples):
        ys[:, :, i, :] = (
            ys[:, :, i - 1, :] + DYNAMICS_A @ ys[:, :, i - 1, :] * t_final / n_samples
        )

    return t, ys


def save_ckp(path, model, optimizer, epoch):
    ckp_data = {
        "epoch": epoch + 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(ckp_data, path)


def load_ckp(path, model, optimizer):
    ckp = torch.load(path)
    model.load_state_dict(ckp["state_dict"])
    optimizer.load_state_dict(ckp["optimizer"])
    return model, optimizer, ckp["epoch"]


def train_spiral(model, path, n_samp, n_traj, batch_size, epochs):
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    try:
        print("Trying to load model weights ...")
        model, optimizer, start_epoch = load_ckp(path, model, optimizer)
        print("Model loaded from weights file!")
    except:
        print("No model weights found. Creating a new model.")
        start_epoch = 0

    train_t, train_data = gen_data(n_samp, n_traj, batch_size, 10)
    val_t, val_data = gen_data(n_samp, n_traj // 5, 1, 10)

    for epoch in range(start_epoch, start_epoch + epochs):

        # Train
        total_loss = 0.0
        for traj in range(n_traj):
            x = train_data[:, :, 0, traj].detach().requires_grad_(True)
            y = train_data[:, :, :, traj]

            model.zero_grad()
            outputs = model(x, train_t.detach().requires_grad_(True))
            loss = loss_function(outputs, y)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            total_loss += current_loss

            print(f"Loss: {total_loss / (traj + 1)}")

        # Validate
        val_loss = 0.0
        with torch.no_grad():
            for traj in range(n_traj // 5):
                x = val_data[:, :, 0, traj]
                y = val_data[:, :, :, traj]

                outputs = model(x, val_t)
                val_loss += loss_function(outputs, y)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Training loss: {total_loss/n_traj}")
        print(f"Validation loss: {val_loss/n_traj}")

    save_ckp(path, model, optimizer, start_epoch + epochs)


def eval_spiral(model):
    t, traj = gen_data(100, 1, 1, 10)

    x = traj[:, :, 0, 0]
    y = traj[0, :, :, 0]

    pred_traj = model(x, t).detach().numpy()[0]

    plt.subplot(2, 1, 1)
    plt.scatter(y[0, :], y[1, :])
    plt.title("True Trajectory")
    plt.subplot(2, 1, 2)
    plt.scatter(pred_traj[0, :], pred_traj[1, :])
    plt.title("Learned Trajectory")
    plt.show()
