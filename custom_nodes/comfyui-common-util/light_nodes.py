from .light_utils import EnhancedRandomLightSourceGenerator
from .layer_utils import pilimage_to_tensor, tensor_to_pilimage


class EnhancedRandomLightSource:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "generate_fc"

    CATEGORY = "utils"

    def generate_fc(self, image, seed):
        image = tensor_to_pilimage(image)
        generator = EnhancedRandomLightSourceGenerator(
            image.size[0], image.size[1])
        light_source = generator.generate_light_source(image, seed)
        light_source, _ = pilimage_to_tensor(light_source)
        return (light_source,)


NODE_CLASS_MAPPINGS = {
    "Enhanced Random Light Source": EnhancedRandomLightSource,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Enhanced Random Light Source": "Enhanced Random Light Source",
}
