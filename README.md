### Deep Learning with Python
Steps:

1. create virtual environment
2. ...

Conceptual notes:

#### Tensor Basics:
- create empty tensor of size 'size' with torch.empty(...size)
- can do multiple dimensions like numpy array: torch.empty(dim1, dim2, dim3)
- torch.rand(...size)
    - randomly generated values based on the uniform distrubution
- torch.zeros(...size)
- torch.ones(...size)
- Optional parameter during initiation: dtype=<type>
- tensor_name.dtype returns datatype stored in tensor
- tensor_name.size()
- make tensor from list: torch.tensor([...])
- +, -, *, /, torch.add(), torch.sub(), torch.mul(), torch.div(), ... between tensors will be element wise operation
- tensor_name.add_(tensor_name2) is inplace version of add (same exist for sub, mul, div)
- torch.add(), torch.sub(), torch.mul(), torch.div() return new Tensors
- Tensor index/slicing, similar to numpy: tensor_name[row, col, ...]
    - Indexing/slicing returns a tensor as well
- tensor_name[index].item(): gets the actual element of the indexed element (only use if tensor has one element)
- tensor_name.view(...size): Formats and returns Tensor to specified ...size dimensions 
    - ex. tensor_name.view(1, 2): return a 2D tensor with 1 row and 2 cols of the elements in Tensor
        - num elements must match original tensor
    - tensor_name.view(-1, size) will automatically shape dim1 so dim2 is length size
    - tensor_name.view(...size) is a good way to resize tensors

#### Tensors on CPU vs GPU
- tensor to numpyarray: tensor_name.numpy()
    - if tensor on CPU, both would share same memory location, changing one changes the other!!!
    - ex. a = torch.ones(5); b = a.numpy(): a and b point to same memory location
    - does not work if tensor is on GPU: numpy only handle CPU
        - move to CPU first, then to numpy
- numpyarray to tensor: torch.from_numpy(numpy_name)
    - datatype float64 by default, can specify datatype in initialization
    - if stored on CPU, both red on the CPUshare same memory too!
- By default, Tensors are sto
- Check if PyTorch is using CUDA
    - torch.cuda.is_available()
- create tensor on GPU:
    - if torch.cuda.is_available():
        device_name = torch.device("cuda")
        x = torch.ones(5,device=device_name) # puts x on GPU
- move tensor to GPU:
    - tensor_name.to(device_name) # device_name is cuda or smtg
- requires_grad=True in tensor initialization
    - gradient calculation on tensor will be required later
        - pytorch will create backpropagation computaitonal graph
    - false by default

#### Autograd
- torch.randn(...size): random tensor with values based on the normal distribution
- backpropagation:
    - forward pass: applies operation, stores output
        - PyTorch will create gradient function grad_fn as attribute when requires_grad=True
    - backward pass: calculates gradient using grad_fn
        - tensor_name.backward() will store gradient as attribute tensor_name.grad
        - Under the hood, gradients are computed with Jacobian vector matrix
        - if tensor_name not a scalar, need to define vector and pass into .backward(vector)
    - ex. y = x + 2
        - y will have attribute grad_fn=<AddBackward0>
    - .backward() requires requires_grad=True during initialization
        - pass vector if tensor_name is not scalar
        - accumulates gradient onto existing .grad, so zero out gradients if used in a loop
            - zero out gradients with tensor_name.grad.zero_()
    - prevent gradient tracking:
        - 1. tensor_name.requires_grad_(False) modifies in-place requires_grad to False
        - 2. tensor_name.detach() makes new tensor with requires_grad False
        - 3. with torch.no_grad():
                ... do operation without grad ...

#### Backpropagation (Conceptual)
- dericative chain rule: for z(y(x)), dz/dx = dz/dy * dy/dx
- Computation Graph
    - Every operation done on a Tensor, PyTorch does on computation graph
    - for each operation, define local gradient
    - for overall gradient, use local gradients and chain rule
- 1. Forward pass: compute loss
- 2. Compute local gradients
- 3. Backward pass: compute dLoss / dWeights using the Chain rule

#### Training Loop
- training pipeline:
    - 1. design model (inputs, output size, forward pass)
    - 2. construct loss, optimizer
    - 3. training loop
- in loop:
- 1. prediction: forward pass
- 2. calculate loss
- 3. gradient: backward pass
- 4. update weights
    - use torch.no_grad() to not affect computational graph
    - zero out graident before next iteration

#### pytorch loss computation
- import torch.nn
- nn has predefined loss functions
    - loss = nn.MSELoss()
- define optimizer
    - stochastic gradient descent: optimizer = torch.optim.SGD([list_of_variables_to_optimize], lr=learning_rate)
        - get variables with model.prameters()
- update weights in training loop: optimizer.step()
- IMPORTANT: optimizer.zero_grad() to zero out gradient in training loop
- can use pytorch models: model = nn.Linear(input_size, output_size)
- or make custom model:
    - class Model_name(nn.Module):
        - def __init__(self, input_dim, output_dim):
        - def forward(self, x):
    - model = Model_name(input_dim, output_dim)
- model(x_i) to make prediction

#### Linear regression
- from sklearn import datasets to generate regression dataset
    - x_numpy, y_numpy = datasets.make_regression(n_samples, n_features, noise, random_state)
- make model with: model = nn.Linear(input_size, output_size)

#### Logistic regression
- from sklearn import datasets to generate binary classification dataset
    - bc = datasets.load_breast_cancer()
    - X, y = bc.data, bc.target
- from sklearn.preprocessing import StandardScaler
    - scale features
- from sklearn.model_selection import train_test_split
    - split dataset
    - X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, random_State)
- for logistic regression, should make features 0 mean, 1 variance
    - sc = StandardScalar()
    - different functions for training & testing data:
    - X_train = sc.fit_transform(X_train)
    - X_test = sc.transform(X_test)
    - convert X_train, X_test, y_train, y_test to tensors
    - reshape y tensors with .view()
- make custom model class:
    - class LogisticRegression(nn.Module):
        - def __init__(self, n_input_features):
            super(LogisticRegression, self).__init__()
            self.linear = nn.Linear(n_input_features, 1) # layer corresponding to linear regression
        
        - def forward(self, x):
            y_predicted = torch.sigmoid(self.linear(x)) # apply sigmoid activation
            return y_predicted
- use custom model: model = LogisticRegression(n_features)
- use binary cross-entropy loss:
    - criterion = nn.BCELoss()
- prediction:
    - with torch.no_grad():
        y_predicted = model(X_test)
        y_predicted_cls = y_predicted.round() # convert predicted probability to class

#### Dataset, dataloader
- can use DataSet & DataLoader classes to do batch training
    - help calculate batches
- terminology:
    - epoch = 1 forward and backward pass of all training samples
    - batch_size = number of training samples in one forward and backword pass
    - number of iterations = number of passes, each pass using batch_size number of samples
    - e.g. 100 samples, batch_size = 20 -> 100/20 = 5 iterations per epoch
- make dataset class:
    import torch
    import torchvision
    from torch.utils.data import Dataset, Dataloader
    import numpy as np
    import math

    class WineDataset(Dataset): # custom dataset class

        def __init__(self): # data loading
            xy = np.loadtxt(location, delimiter=",", dtype=np.float32, skiprows=1)
            self.x = torch.from_numpy(xy[:, 1:])
            self.y = torch.from_numpy(xy[:, [0]]) # n_samples, 1
            self.n_samples = xy.shape[0]

        def __getitem__(self, index): # get observation
            return self.x[index], self.y[index]

        def __len__(self): # dataset length
            return self.n_samples
- use dataloader:
    dataset = WineDataset()
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2) # num_workers makes it multi-threaded
- use in training loop:
    num_epochs = 2
    total_samples = len(dataset)
    n_iterations = math.ceil(total_samples/batch_size)

    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(dataloader):
            # forward pass -> loss calculation -> backward pass -> update weight
