class ImageRelay:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "relay"

    CATEGORY = "utils"

    def relay(self, image):
        return (image,)

class MaskRelay:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASK",)
    FUNCTION = "relay"

    CATEGORY = "utils"

    def relay(self, mask):
        return (mask,)

class IntRelay:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "int_": ("INT",{ "default": 0, "step": 1 }),
            }
        }
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("INT",)
    FUNCTION = "relay"

    CATEGORY = "utils"

    def relay(self, int_):
        return (int_,)
    
class StringRelay:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string_": ("STRING",{ "default": "" }),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "relay"

    CATEGORY = "utils"

    def relay(self, string_):
        return (string_,)
    
class FloatRelay:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "float_": ("FLOAT",{ "default": 1.0, "step": 0.01 }),
            }
        }
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("FLOAT",)
    FUNCTION = "relay"

    CATEGORY = "utils"

    def relay(self, float_):
        return (float_,)


NODE_CLASS_MAPPINGS = {
    "Image Relay":ImageRelay,
    "Mask Relay": MaskRelay,
    "Int Relay": IntRelay,
    "String Relay": StringRelay,
    "Float Relay": FloatRelay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Image Relay":"Image Relay",
    "Mask Relay": "Mask Relay",
    "Int Relay":  "Int Relay",
    "String Relay": "String Relay",
    "Float Relay": "Float Relay",

}