"""Get NN-PES tf model info"""
# %%
if __name__ == "__main__":
    # With tensorflow>2.16
    # pip install tf-keras~=2.16
    import os

    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    import tf_keras as keras
    import numpy as np

    model = keras.models.load_model("./ch5_model_ready", compile=False)
    sequential_layer = model.get_layer("sequential_1")

    print("*" * 15, "Model info", "*" * 15)
    print("\t--------------------")
    print("\t||model.summary()||")
    print("\t--------------------")
    model.summary()
    print("\t-----------------------------------")
    print("\t||The sequential_1 in model summary:||")
    print("\t----------------------------------")
    sequential_layer.summary()
    print("\t-----------------------------------")
    print("\t||Dense in sequential_layer:||")
    print("\t----------------------------------")
    for layer in sequential_layer.layers:
        print("Layer name:", layer.name)
        print("  Type:", layer.__class__.__name__)
        print("  Units:", layer.units)
        print(
            "  Activation:",
            layer.activation.__name__ if layer.activation is not None else "None",
        )
        print("  Use bias:", layer.use_bias)

    print("*" * 15, "Saving model to graphs", "*" * 15)
    keras.utils.plot_model(model, "NN_PES_model_with_shape_info.png", show_shapes=True)
    keras.utils.plot_model(
        sequential_layer,
        "NN_PES_sequential_1_model_with_shape_info.png",
        show_shapes=True,
    )
    keras.utils.plot_model(
        model,
        "nested_info.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=600,
        show_layer_activations=True,
    )
    print("*" * 15, "Done", "*" * 15)

# %%
