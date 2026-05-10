import random
import math
import numpy as np
from PIL import Image, ImageDraw,ImageStat
from scipy.ndimage import gaussian_filter


class EnhancedRandomLightSourceGenerator:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.diagonal = np.sqrt(width**2 + height**2)

    def generate_light_source(self, base_image, seed=None):
        if seed is not None:
            valid_seed = seed % (2**32)
            random.seed(seed)
            np.random.seed(valid_seed)

        base_brightness = self._get_base_brightness(base_image)
        
        num_sources = random.randint(3, 7)  # Multiple light sources
        light_array = np.zeros((self.height, self.width))

        for _ in range(num_sources):
            if random.random() < 0.7:  # 70% chance for edge source
                source, direction = self._setup_edge_light()
            else:
                source, direction = self._setup_center_light()
            
            light_array += self._create_single_light(source, direction)

        light_array = self._normalize_and_smooth(light_array)
        return self._finalize_light_array(light_array, base_brightness)

    def _setup_edge_light(self):
        angle = random.uniform(0, 2*np.pi)
        distance = random.uniform(1.5, 4) * self.diagonal
        source = (
            self.width / 2 + np.cos(angle) * distance,
            self.height / 2 + np.sin(angle) * distance
        )
        direction = (-np.cos(angle), -np.sin(angle))
        return source, direction

    def _setup_center_light(self):
        source = (
            random.uniform(-0.5*self.width, 1.5*self.width),
            random.uniform(-0.5*self.height, 1.5*self.height)
        )
        direction = None
        return source, direction

    def _create_single_light(self, light_source, direction):
        y, x = np.ogrid[:self.height, :self.width]
        
        dx, dy = x - light_source[0], y - light_source[1]
        distances = np.sqrt(dx**2 + dy**2)
        
        if direction is not None:
            dot_product = dx*direction[0] + dy*direction[1]
            angles = np.arccos(np.clip(dot_product / (distances * np.linalg.norm(direction)), -1, 1))
            intensity = np.maximum(0, np.cos(angles)) * self._distance_falloff(distances)
        else:
            intensity = self._distance_falloff(distances)
        
        return intensity

    def _distance_falloff(self, distances):
        return 1 / (1 + (distances / (self.diagonal / 4))**2)

    def _normalize_and_smooth(self, light_array):
        light_array = light_array / np.max(light_array)
        light_array = gaussian_filter(light_array, sigma=min(self.width, self.height) / 20)
        return light_array

    def _finalize_light_array(self, light_array, base_brightness):
        ambient = 0.2
        light_array = ambient + (1 - ambient) * light_array
        
        max_intensity = min(255, max(128, base_brightness * 2))
        light_array = light_array * (max_intensity / 255)
        
        rgba_array = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        rgba_array[:,:,:3] = (255 * light_array[:,:,np.newaxis]).astype(np.uint8)
        rgba_array[:,:,3] = (255 * light_array).astype(np.uint8)
        
        return Image.fromarray(rgba_array)

    def _get_base_brightness(self, image):
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        img_array = np.array(image)
        rgb = img_array[:, :, :3]
        alpha = img_array[:, :, 3]
        
        # Calculate brightness for non-transparent pixels
        brightness = np.mean(rgb[alpha > 0], axis=1)
        
        # Remove pure white pixels (255, 255, 255)
        non_white = brightness < 255
        brightness = brightness[non_white]
        
        if len(brightness) == 0:
            return 128  # Default to mid-gray if all pixels are white or transparent
        
        # Get the top 20% brightest pixels
        brightness_threshold = np.percentile(brightness, 80)
        top_20_percent = brightness[brightness >= brightness_threshold]
        
        # Calculate the average brightness of the top 20%
        return np.mean(top_20_percent)