import os
import json
import torch
import folder_paths
from .layer_utils import fuse_layer,pilimage_to_tensor
from PIL import Image

class InitLayerInfoArray:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "width": ("INT", {"default": 0}),
                "height": ("INT", {"default": 0}),
                "prompt": ("STRING", {
                    "dynamicPrompts": False,
                    "multiline": True,
                    "default": ""
                }),
                "reference_bg": (sorted(files), {"image_upload": False}),
                "number_of_results": ("INT", {"default": 0}),
                "correct_color":("BOOLEAN", {"default": True,"lable_on":"active","lable_off":"inactive"}),
                "render_strength": (["Default", "Extra Weak", "Weak", "Strong", "Extra Stong"],),
                "color_strength": (["Default", "None", "Weak", "Strong", "Extra Stong"],),
                "outline_strength": (["Default", "None","Extra Weak", "Weak", "Strong", "Extra Stong"],),
            }
        }
    RETURN_TYPES = ("LAYER_INFO_ARRAY","STRING",)
    RETURN_NAMES = ("layerInfoArray","layerInfoArrayJson",)
    FUNCTION = "init_fc"

    CATEGORY = "utils"

    def init_fc(self,width,height,prompt,reference_bg,number_of_results,correct_color,render_strength,color_strength,outline_strength,):
        #将上面的内容转换成json对象
        layerInfoArray = {
            "width": width,
            "height": height,
            "prompt": prompt,
            "reference_bg": reference_bg,
            "number_of_results": number_of_results,
            "correct_color": correct_color,
            "render_strength": render_strength,
            "color_strength": color_strength,
            "outline_strength": outline_strength,
            "layers": []
        }
        return (layerInfoArray, json.dumps(layerInfoArray),)

class AddedLayerInfoToArray:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "layerInfoArray":('LAYER_INFO_ARRAY', {}),
                "type":  (["Product", "Prop", "Human-Hand", "Human-Male", "Human-Female"],),
                "image": (sorted(files), {"image_upload": False}),
                "width": ("INT", {"default": 0}),
                "height": ("INT", {"default": 0}),
                "rotation": ("INT", {"default": 0,"min":0,"max":360,"step":1}),
                "position_x": ("INT", {"default": 0}),
                "position_y": ("INT", {"default": 0}),
            }
        }
    RETURN_TYPES = ("LAYER_INFO_ARRAY","STRING",)
    RETURN_NAMES = ("layerInfoArray","layerInfoArrayJson",)
    FUNCTION = "add_fc"

    CATEGORY = "utils"

    def add_fc(self,layerInfoArray,type,image,width,height,rotation,position_x,position_y):
        layer = {
            "type": type,
            "image":image,
            "width": width,
            "height": height,
            "rotation": rotation,
            "position_x": position_x,
            "position_y": position_y,
        }
        layerInfoArray = json.loads(json.dumps(layerInfoArray))
        
        layerInfoArray["layers"].append(layer)
        return (layerInfoArray, json.dumps(layerInfoArray),)    
    
class LayerInfoArrayFuse:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "layerInfoArrayJson":("STRING",{"default": ""}),
            }
        }
    RETURN_TYPES = ("IMAGE","MASK","IMAGE","MASK","IMAGE","LAYER_IMAGES","STRING","BOOLEAN","INT","INT","INT")
    RETURN_NAMES = ("fuseImage","fuseMask","fuseProductImage","fuseProductMask","referenceBGImage","layerImages","prompt","correctColor","renderStrength","colorStrength","outlineStrength")
    FUNCTION = "fuse_fc"

    CATEGORY = "utils"

    def fuse_fc(self, layerInfoArrayJson,):
        layerInfoArray = json.loads(layerInfoArrayJson)

        fuse_image, fuse_product_image,reference_bg_image,layer_index_images = fuse_layer(layerInfoArray)
        fuse_image, fuse_mask = pilimage_to_tensor(fuse_image, needMask=True)
        
        fuse_product_image, fuse_product_mask = pilimage_to_tensor(fuse_product_image, needMask=True)
        reference_bg_image, _ = pilimage_to_tensor(reference_bg_image)
        return (fuse_image, fuse_mask, fuse_product_image, fuse_product_mask,reference_bg_image, layer_index_images,  layerInfoArray["prompt"], layerInfoArray["correct_color"], 0, 0, 0,)

class LayerImageSeleted:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "layer_images": ("LAYER_IMAGES",),
                "layer_filter_type":  (["all", "Product", "Prop", "Human-Hand", "Human-Male", "Human-Female"],),
                "layer_need_fuse": ("BOOLEAN", {"default": True, "lable_on": "yes", "lable_off": "no"}),
                "layer_canvas": ("IMAGE",),
            },
        }
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE",)
    RETURN_NAMES = ("processedImage", "processedMask","originalImage",)
    FUNCTION = "select_fc"

    CATEGORY = "utils"

    def select_fc(self, layer_images, layer_filter_type, layer_need_fuse, layer_canvas):
        layer_filter_data = []
        for layer in layer_images:
            if layer_filter_type == "all" or layer["type"] == layer_filter_type:
                layer_filter_data.append(layer)
        if len(layer_filter_data) == 0:
            image, mask = pilimage_to_tensor(
                layer_canvas, needMask=True, justMask=False, empty=True)
            image = image.to(layer_canvas.device, dtype=layer_canvas.dtype)
            mask = mask.to(layer_canvas.device, dtype=layer_canvas.dtype)
            return (image, mask, image,)

        fuse_img = None
        org_img = None
        for layer in layer_filter_data:
            if fuse_img == None:
                fuse_img = layer["deformationImage"]
                org_img = layer["originalImage"]
            else:
                fuse_img = Image.alpha_composite(fuse_img, layer['deformationImage'])
                org_img = Image.alpha_composite(org_img, layer['originalImage'])
        if fuse_img is not None:
            layer_img, layer_mask = pilimage_to_tensor(fuse_img, needMask=True)
            # 取反 0 1 互换
            layer_mask = 1. - layer_mask
            layer_org_img,_ = pilimage_to_tensor(org_img)
            return (layer_img, layer_mask,layer_org_img,)
        else:
            image, mask = pilimage_to_tensor(
                layer_canvas, needMask=True, justMask=False, empty=True)
            image = image.to(layer_canvas.device, dtype=layer_canvas.dtype)
            mask = mask.to(layer_canvas.device, dtype=layer_canvas.dtype)
            return (image, mask, image,)

#this class code from ComfyUI_IPAdapter_plus.IPAdapterPlus.IPAdapterAdvanced
class LayerImagesIPAdapterAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        _WEIGHT_TYPES = []
        try:
            from custom_nodes.ComfyUI_IPAdapter_plus.IPAdapterPlus import WEIGHT_TYPES
            _WEIGHT_TYPES.extend(WEIGHT_TYPES)
        except:
            pass
        return {
            "required": {
                "layer_images": ("LAYER_IMAGES",),
                "layer_filter_type":  (["all","Product", "Prop", "Human-Hand", "Human-Male", "Human-Female"],),
                "layer_need_fuse":("BOOLEAN", {"default": True,"lable_on":"yes","lable_off":"no"}),
                "layer_canvas": ("IMAGE",),
                "model": ("MODEL", ),
                "ipadapter": ("IPADAPTER", ),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 5, "step": 0.05 }),
                "weight_type": (_WEIGHT_TYPES, ),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "image_negative": ("IMAGE",),
                "clip_vision": ("CLIP_VISION",),
            }
        }
    RETURN_TYPES = ("MODEL","LAYER_IMAGES","MASK")
    RETURN_NAMES = ("MODEL","layerImages","extendedMASK")
    FUNCTION = "apply_ipadapter"

    CATEGORY = "ipadapter"

    def apply_ipadapter(self,layer_images,layer_filter_type,layer_need_fuse,layer_canvas, model, ipadapter, start_at=0.0, end_at=1.0, weight=1.0, weight_style=1.0, weight_composition=1.0, expand_style=False, weight_type="linear", combine_embeds="concat", weight_faceidv2=None, image_style=None, image_composition=None, image_negative=None, clip_vision=None, insightface=None, embeds_scaling='V only', layer_weights=None, ipadapter_params=None, encode_batch_size=0, style_boost=None):
        layer_filter_data = []
        remain = []
        for layer in layer_images:
            if layer_filter_type=="all" or layer["type"] == layer_filter_type:
                layer_filter_data.append(layer)
            else:
                remain.append(layer)
        if len(layer_filter_data) == 0:
            _,extended_mask = pilimage_to_tensor(layer_canvas, needMask=True,justMask=True,empty=True)
            extended_mask = extended_mask.to(layer_canvas.device,dtype=layer_canvas.dtype)
            return (model,remain,extended_mask,)
        
        need_process_images = []
        if layer_need_fuse:
            extended_mask = None
            fuse_img = None
            for layer in layer_filter_data:
                if fuse_img == None:
                    fuse_img = layer["deformationImage"]
                else:
                    fuse_img = Image.alpha_composite(fuse_img, layer['deformationImage'])
            if fuse_img is not None:
                layer_img,layer_mask = pilimage_to_tensor(fuse_img, needMask=True)
                #取反 0 1 互换
                layer_mask = 1. - layer_mask
                need_process_images.append((layer_img,layer_mask))
                extended_mask = layer_mask
        else:
            fuse_img = None
            for layer in layer_filter_data:
                img_pil = layer["deformationImage"]
                if fuse_img == None:
                    fuse_img = img_pil
                else:
                    fuse_img = Image.alpha_composite(fuse_img, img_pil)
                layer_img,_ = pilimage_to_tensor(layer["originalImage"])
                _,layer_mask = pilimage_to_tensor(img_pil, needMask=True,justMask=True)
                layer_mask = 1. - layer_mask
                need_process_images.append((layer_img,layer_mask))
            if fuse_img is not None:
                _,extended_mask = pilimage_to_tensor(fuse_img, needMask=True,justMask=True)
                extended_mask = 1. - extended_mask
        if extended_mask is None:
            _,extended_mask = pilimage_to_tensor(layer_canvas, needMask=True,justMask=True,empty=True)
            extended_mask = extended_mask.to(layer_canvas.device,dtype=layer_canvas.dtype)
        if len(need_process_images) == 0:
            return (model,remain,extended_mask,)        
        try:
            from custom_nodes.ComfyUI_IPAdapter_plus.IPAdapterPlus import IPAdapterAdvanced
        except:
            raise Exception("ComfyUI_IPAdapter_plus.IPAdapterPlus not found,you should install it first.")
        ipadapterObj = IPAdapterAdvanced()
        
        for layer in need_process_images:
            model ,_ = ipadapterObj.apply_ipadapter(model, ipadapter, start_at, end_at, weight, weight_style, weight_composition, expand_style, weight_type, combine_embeds, weight_faceidv2, layer_img, image_style, image_composition, image_negative, clip_vision, layer_mask, insightface, embeds_scaling, layer_weights, ipadapter_params, encode_batch_size, style_boost)
        return (model,remain,extended_mask,)


NODE_CLASS_MAPPINGS = {
    "Init Layer Info Array":InitLayerInfoArray,
    "Added Layer Info To Array":AddedLayerInfoToArray,
    "Layer Info Array Fuse":LayerInfoArrayFuse,
    "Layer Image Seleted":LayerImageSeleted,
    "Layer Images IPAdapter Advanced":LayerImagesIPAdapterAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Init Layer Info Array":"Init Layer Info Array",
    "Added Layer Info To Array": "Added Layer Info To Array",
    "Layer Info Array Fuse": "Layer Info Array Fuse",
    "Layer Image Seleted":"Layer Image Seleted",
    "Layer Images IPAdapter Advanced": "Layer Images IPAdapter Advanced",

}