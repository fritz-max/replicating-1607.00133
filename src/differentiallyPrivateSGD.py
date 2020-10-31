from torch.utils.data import TensorDataset

def train(model, minibatch_loader, micro_loader_func, optimizer, criterion, callback, callback_per_iteration, device='cpu'):
    """Implements "Algorithm 1 Differentially private SGD (Outline)" as described in
    the paper, using the pyvacy library.
    """
    iteration = 0
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

            prediction = model.forward(X_microbatch)
            loss = criterion(prediction, y_microbatch)

            loss.backward()
            optimizer.microbatch_step()
            running_loss += loss.item()

        optimizer.step()
        # print(f"Loss: {running_loss/len(minibatch_loader)}")

        if iteration % callback_per_iteration == 0:
            print("Iterations: ", iteration)
            callback()     
        
        iteration += 1


def get_accuracy(predictions, targets):
    return round((sum((predictions == targets)).float()/len(predictions)*100).item(), 2)

def create_callback(model, train_dataset, test_dataset, train_acc_list, test_acc_list):
    
    def eval_callback():
        train_predictions = model.predict_class(train_dataset[:][0])
        train_accuracy = get_accuracy(train_predictions, train_dataset[:][1])
        train_acc_list.append(train_accuracy)

        test_predictions = model.predict_class(test_dataset[:][0])
        test_accuracy = get_accuracy(test_predictions, test_dataset[:][1])
        test_acc_list.append(test_accuracy)

        print(f"Training Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")

    return eval_callback