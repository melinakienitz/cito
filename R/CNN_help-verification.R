library(cito)

# data

mnist <- torchvision::mnist_dataset("home/data/mnist", download=T)

X <- array(mnist$data, dim=c(60000,1,28,28))
Y <- factor(mnist$targets)

# Achitecture of the CNN
# outputlayer is automatically added

architecture <- create_architecture(conv(5), maxPool(), conv(5), maxPool(), linear (10))

# Build & train the network

cnn.fit <- cnn(X, Y, architecture, loss = "softmax", epochs = 5, validation = 0.1, lr = 0.05, device = "cuda")

print(cnn.fit)
plot(cnn.fit)

analyze_training(cnn.fit)




