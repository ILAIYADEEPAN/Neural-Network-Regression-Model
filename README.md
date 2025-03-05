# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

![image](https://github.com/user-attachments/assets/5e98f7b5-8198-4af8-a199-3c58746e3669)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: ILAIYADEEPAN K
### Register Number:212223230080
```python
class NeuralNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(1, 8)
    self.fc2 = nn.Linear(8, 10)
    self.fc3 = nn.Linear(10, 1)
    self.relu = nn.ReLU()
    self.history = {'loss': []} 

  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.fc3(x)  
    return x

ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(), lr=0.001)

def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
  for epoch in range(epochs):
    optimizer.zero_grad()
  
    output = ai_brain(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    ai_brain.history['loss'].append(loss.item())
    if epoch % 200 == 0:
      print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')



```
## Dataset Information

![image](https://github.com/user-attachments/assets/fd411bd3-6bc0-4523-9ce8-8de44eca31e1)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/7351e5df-496c-4cb4-9936-ad19d62328e8)


### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/24c2b24a-6369-468d-b7b3-f35130b1e8b0)
![image](https://github.com/user-attachments/assets/b60bbb62-bf95-40c1-965b-3a5bd1739bf8)



## RESULT

The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
