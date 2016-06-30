import mxnet as mx


train = mx.io.MNISTIter(
    image = 'mnist/train-images-idx3-ubyte',
    label = 'mnist/train-labels-idx1-ubyte',
    batch_size = 128,
    data_shape = (784,))

val = mx.io.MNISTIter(...)

data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullConnected(data = data, num_hidden = 128)
act1 = mx.symbol.Activation(data = fc1, act_type = 'relu')
fc2 = mx.symbol.FullConnected(data = act1, num_hidden = 64)
act2 = mx.symbol.Activation(data = fc2, act_type = 'relu')
fc3 = mx.symbol.FullConnected(data = act2, num_hidden = 10)
mlp = mx.symbol.Activation(data =fc3, act_type = 'softmax')


# train a model
model = mx.model.FeedForward(
    symbol = mlp,
    num_epoch = 20,
    learning_rate = 0.1)

model.fit(X = train, eval_data = val)


# predict
test = mx.io.MNISTIter(...)
model.predict(X = test)
