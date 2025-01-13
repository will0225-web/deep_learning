def get_input_shape(model):
    layer1 = model.get_layer(index=0)
    input_shape = layer1.get_config()['batch_input_shape']
    return input_shape[1]