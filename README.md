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