# Input metadata for SqueezeNet 1.1


# This follows the style of existing samples in the repo
class Program_input_tensor_image:
    name = "image"
    shape = [1, 3, 224, 224]
    dtype = "float32"
    data = None


# The validation script needs this dict to match with model.forward(**state_dict)
input_meta = {"image": {"shape": [1, 3, 224, 224], "dtype": "float32"}}
