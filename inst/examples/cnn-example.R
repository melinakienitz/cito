\donttest{
if(torch::torch_is_installed()){
library(cito)

# Example workflow in cito

## Data
### We generate our own data:
### 320 images (size 50x50) of either a rectangle or ellipsoid
shapes <- simulate_shapes(320, 50)
X <- shapes$data
Y <- shapes$labels

## Architecture
### Declare the architecture of the CNN
### Note that the output layer is added automatically by cnn()
architecture <- create_architecture(conv(5), maxPool(), conv(5), maxPool(), linear(10))

## Build and train network
### softmax is used for classification
cnn.fit <- cnn(X, Y, architecture, loss = "softmax", epochs = 50, validation = 0.1, lr = 0.05)

## The training loss is below the baseline loss but at the end of the
## training the loss was still decreasing, so continue training for another 50
## epochs
cnn.fit <- continue_training(cnn.fit, epochs = 50)

# Structure of Neural Network
print(cnn.fit)

# Plot Neural Network
plot(cnn.fit)

## Convergence can be tested via the analyze_training function
analyze_training(cnn.fit)
}
}
