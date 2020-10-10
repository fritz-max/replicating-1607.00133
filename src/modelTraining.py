from torch.utils.data import TensorDataset
def train(model, minibatch_loader, micro_loader_func, optimizer, criterion, iterations, device='cpu'):
    """TODO::
    """
    # sample a Lot
    for X_minibatch, y_minibatch in minibatch_loader:
        running_loss = 0 
        # the DPSGD optimizer uses two gradients
        # setting the first to zero
        optimizer.zero_grad()

        for X_microbatch, y_microbatch in micro_loader_func(TensorDataset(X_minibatch, y_minibatch)):

            # moving data onto device
            X_microbatch = X_microbatch.to(device)
            y_microbatch = y_microbatch.to(device)

            # zeroing second gradient
            optimizer.zero_microbatch_grad()

            prediction = model.forward(X_microbatch.float())
            loss = criterion(prediction, y_microbatch.long())

            loss.backward()
            optimizer.microbatch_step()
            running_loss += loss.item()

        optimizer.step()
        print(f"Loss: {running_loss/len(minibatch_loader)}")