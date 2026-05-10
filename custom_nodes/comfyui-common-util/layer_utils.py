from PIL import Image, ImageOps
import os
import numpy as np
import torch
import folder_paths
import node_helpers


def open_image_from_inputdir(filename):
    input_dir = folder_paths.get_input_directory()
    img_path = os.path.join(input_dir, filename)
    img = node_helpers.pillow(Image.open, img_path)
    img = node_helpers.pillow(ImageOps.exif_transpose, img)
    return img
    
def _process_layer(layer):
    img_org = open_image_from_inputdir(layer['image'])
    img = img_org.resize((layer['width'], layer['height']),Image.LANCZOS)
    img = img.rotate(layer['rotation'], expand=True)
    return img,img_org

def _paste_image(img, position,size):
    full_size_img = Image.new('RGBA',size, (0, 0, 0, 0))
    full_size_img.paste(img, position, img)
    return full_size_img
    
def fuse_layer(layerInfoArray):
    canvas = Image.new('RGBA', (layerInfoArray['width'], layerInfoArray['height']), (0, 0, 0, 0))
    layer_index_images = []
    for index,layer in enumerate(layerInfoArray['layers']):
        layer_img,img_org = _process_layer(layer)
        position = (layer['position_x'], layer['position_y'])
        
        layer_fft_img = _paste_image(layer_img, position,canvas.size)
        layer_index_images.append({
            'originalImage':img_org,
            'deformationImage':layer_fft_img,
            'type':layer['type']
        })
        
        canvas = Image.alpha_composite(canvas, layer_fft_img)
        
        # 使用当前图层的 alpha 通道作为蒙版
        layer_fft_mask = layer_fft_img.split()[3] 
        layer_fft_mask = Image.eval(layer_fft_mask, lambda x: 255 - x)
        
        #遍历layer_index_image 除了当前
        for i in range(len(layer_index_images)):
            if i==index:
                continue
            #使用当前非透明内容来删除之前图层中被遮挡部分
            index_img = layer_index_images[i]
            temp_img = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
            temp_img.paste(index_img['deformationImage'], (0, 0), layer_fft_mask)
            index_img['deformationImage'] = temp_img
    product_image = Image.new('RGBA', (layerInfoArray['width'], layerInfoArray['height']), (0, 0, 0, 0))
    for i in range(len(layer_index_images)):
        #使用当前非透明内容来删除之前图层中被遮挡部分
        index_img = layer_index_images[i]
        if index_img['type'] == 'Product':
            product_image = Image.alpha_composite(product_image, index_img['deformationImage'])
    if layerInfoArray["reference_bg"] is not None and layerInfoArray["reference_bg"] != "":
        reference_bg_image = open_image_from_inputdir(layerInfoArray["reference_bg"])
    else:
        reference_bg_image = Image.new(
            'RGBA', (100, 100), (255, 255, 255, 255))
    return (canvas, product_image,reference_bg_image,layer_index_images)
    

def pilimage_to_tensor(image, needMask=False, justMask=False, empty=False):
    if not isinstance(image, Image.Image):
        raise ValueError("`image` must be a PIL Image object.")
    if needMask:
        mask = np.array(image.getchannel('A')).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask)
        mask = 1. - mask
        mask = mask.unsqueeze(0)
    else:
        mask = None
    if empty and mask is not None:
        mask = torch.zeros_like(mask)
    if justMask:
        return None, mask
    # 转换为RGB并转为numpy数组
    image_array = np.array(image.convert("RGB")).astype(np.float32) / 255.0

    # 转换为torch tensor
    image_tensor = torch.from_numpy(image_array).unsqueeze(0)
    if empty:
        image_tensor = torch.zeros_like(image_tensor)
    return image_tensor, mask
    
def tensor_to_pilimage(tensor):
    if tensor.ndim == 4 and tensor.shape[0] == 1:  # Check for batch dimension
        tensor = tensor.squeeze(0)  # Remove batch dimension
    if tensor.dtype == torch.float32:  # Check for float tensors
        tensor = tensor.mul(255).byte()  # Convert to range [0, 255] and change to byte type
    elif tensor.dtype != torch.uint8:  # If not float and not uint8, conversion is needed
        tensor = tensor.byte()  # Convert to byte type

    numpy_image = tensor.cpu().numpy()

    # Determine the correct mode based on the number of channels
    if tensor.ndim == 3:
        if tensor.shape[2] == 1:
            mode = 'L'  # Grayscale
        elif tensor.shape[2] == 3:
            mode = 'RGB'  # RGB
        elif tensor.shape[2] == 4:
            mode = 'RGBA'  # RGBA
        else:
            raise ValueError(f"Unsupported channel number: {tensor.shape[2]}")
    else:
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

    pil_image = Image.fromarray(numpy_image, mode)
    return pil_image