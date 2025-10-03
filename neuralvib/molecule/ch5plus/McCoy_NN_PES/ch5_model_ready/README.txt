NOTE: Please use tensorflow <= 2.15 for compatibility!

================================================================
Original README
================================================================
These TensorFlow Keras models are loaded in using:

	import tensorflow as tf
	model = tf.keras.models.load_model('./ch5_model_ready',compile=False)

They can then be called using model.predict(descriptor).

The atom-atom distances for the CM calculation are in Bohr.
Since the models are trained on the shifted, log energies, one has to
take the exponential value of the predicted value and shift it by 100:

	import numpy as np
    v = model.predict(coulomb,batch_size=batch_size)
    v = v.flatten()		  # Flatten to one dimension
    v = np.exp(v)-100     # The resultant energy value is in cm-1
    v = v*wvn_to_hartree  # Convert to Eh
	
One can use these models for DMC simulations by using the NN_Potential 
simulation utility in PyVibDMC. For more information, see:
https://pyvibdmc.readthedocs.io/en/latest/potentials.html#tensorflow-keras-neural-network-potentials
================================================================
Original README
================================================================

if tensorflow >= 2.16 then
pip install tf-keras~=2.16

```python
    import os;os.environ["TF_USE_LEGACY_KERAS"]="1"
    import tf_keras as keras
    import tensorflow as tf

    model = keras.models.load_model('./ch5_model_ready',compile=False)
```