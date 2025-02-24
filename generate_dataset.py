import numpy as np
from pathlib import Path
import json
import cv2
from typing import List, Dict, Tuple
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import random
import textwrap
import albumentations as A
import os
import csv

class KazakhTextGenerator:
    """Generate Kazakh language sentences (no context meaning reserved)"""
    
    def __init__(self):
        self.common_words = [
            'және', 'бірақ', 'немесе', 'сондықтан', 'себебі',
            'мен', 'сен', 'ол', 'біз', 'сіз', 'олар',
            'бар', 'жоқ', 'керек', 'қажет', 'болады',
            'үлкен', 'кіші', 'жақсы', 'жаман', 'әдемі'
        ]
        
        self.sentence_templates = [
            "Бүгін {time} {location} {action}.",
            "{person} {location} {action} {object}.",
            "{time} {person} {action}.",
            "{location} {object} {action}.",
        ]
        
        self.locations = ['үйде', 'мектепте', 'дүкенде', 'паркте', 'қалада']
        self.actions = ['жұмыс істеді', 'оқыды', 'жазды', 'сөйледі', 'ойнады']
        self.times = ['таңертең', 'түсте', 'кешке', 'түнде']
        self.objects = ['кітапты', 'қаламды', 'телефонды', 'компьютерді']
        self.persons = ['студент', 'оқушы', 'мұғалім', 'дәрігер', 'инженер']

    def generate_sentence(self) -> str:
        template = random.choice(self.sentence_templates)
        return template.format(
            time=random.choice(self.times),
            location=random.choice(self.locations),
            action=random.choice(self.actions),
            object=random.choice(self.objects),
            person=random.choice(self.persons)
        )

    def generate_text_sample(self, min_words: int = 3, max_words: int = 10) -> str:
        if random.random() < 0.7:
            return self.generate_sentence()
        else:
            num_words = random.randint(min_words, max_words)
            return ' '.join(random.sample(self.common_words, num_words))

class ImageGenerator:
    """Generate images"""
    
    def __init__(self):
        #These are my chosen fonts to be applied to a text
        self.font_paths = ['/System/Library/Fonts/Supplemental/Arial.ttf',
        "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
         "/System/Library/Fonts/Supplemental/SnellRoundhand.ttc",
        "/System/Library/Fonts/Supplemental/Noteworthy.ttc",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
         "/System/Library/Fonts/Supplemental/SnellRoundhand.ttc",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
        "/System/Library/Fonts/Supplemental/Verdana.ttf",
        "/System/Library/Fonts/Supplemental/Georgia.ttf",
        "/System/Library/Fonts/Supplemental/Tahoma.ttf",
        "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
         "/System/Library/Fonts/Supplemental/SnellRoundhand.ttc",]
        # self.font_paths = os.listdir('/System/Library/Fonts/Supplemental/')
        self.font_sizes = range(32, 48, 4)
        self.background_patterns = ['noise', 'grid', 'dots', 'lines']
        self.shapes = ['circle', 'rectangle', 'triangle', 'line']
        
        # Initialize augmentations
        self.transform = A.Compose([
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
            ], p=0.8),
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.5),
            ], p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
            ], p=0.5),
        ])

    def _add_random_shapes(self, draw, width, height, num_shapes=5):
        """Add random geometric shapes"""
        for _ in range(num_shapes):
            shape = random.choice(self.shapes)
            color = (
                random.randint(100, 200),
                random.randint(100, 200),
                random.randint(100, 200)
            )
            
            if shape == 'circle':
                radius = random.randint(10, 30)
                x = random.randint(0, width)
                y = random.randint(0, height)
                draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
            elif shape == 'rectangle':
                w = random.randint(20, 60)
                h = random.randint(20, 60)
                x = random.randint(0, width-w)
                y = random.randint(0, height-h)
                draw.rectangle([x, y, x+w, y+h], fill=color)
            elif shape == 'triangle':
                points = [
                    (random.randint(0, width), random.randint(0, height)),
                    (random.randint(0, width), random.randint(0, height)),
                    (random.randint(0, width), random.randint(0, height))
                ]
                draw.polygon(points, fill=color)

    def _add_background_pattern(self, img, pattern_type):
        """Add background pattern"""
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        if pattern_type == 'grid':
            for x in range(0, width, 30):
                draw.line([(x, 0), (x, height)], fill='lightgray', width=1)
            for y in range(0, height, 30):
                draw.line([(0, y), (width, y)], fill='lightgray', width=1)
        elif pattern_type == 'dots':
            for x in range(0, width, 20):
                for y in range(0, height, 20):
                    draw.ellipse([x-1, y-1, x+1, y+1], fill='lightgray')
        elif pattern_type == 'lines':
            for _ in range(5):
                start = (random.randint(0, width), random.randint(0, height))
                end = (random.randint(0, width), random.randint(0, height))
                draw.line([start, end], fill='lightgray', width=1)
        
        return img

    def create_text_image(self, text: str) -> Tuple[Image.Image, List[int]]:
        """Create image with text and background elements"""
        font_size = random.choice(self.font_sizes)
        path = random.choice(self.font_paths)
        print(path)
        font = ImageFont.truetype(path, font_size)
        # font = ImageFont.truetype('DejaVuSans.ttf', font_size)
        
        text_lines = textwrap.wrap(text, width=30)
        
        # Calculate image size
        text_width = max(font.getbbox(line)[2] for line in text_lines)
        text_height = sum(font.getbbox(line)[3] - font.getbbox(line)[1] for line in text_lines)

        img_width = 600
        img_height = 200
        
        # Create image
        bg_color = (random.randint(240, 255), random.randint(240, 255), random.randint(240, 255))
        img = Image.new('RGB', (img_width, img_height), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Add background elements
        pattern = random.choice(self.background_patterns)
        if pattern != 'noise':
            img = self._add_background_pattern(img, pattern)
        self._add_random_shapes(draw, img_width, img_height)
        
        # Draw text
        x_offset = 30
        y_offset = 30
        bboxes = []
        
        for line in text_lines:
            bbox = [
                x_offset,
                y_offset,
                x_offset + font.getbbox(line)[2],
                y_offset + font.getbbox(line)[3] - font.getbbox(line)[1]
            ]
            draw.text((x_offset, y_offset), line, fill='black', font=font)
            bboxes.append(bbox)
            y_offset += font.getbbox(line)[3] - font.getbbox(line)[1]
        
        # Apply augmentations
        img_array = np.array(img)
        img_array = self.transform(image=img_array)['image']
        
        return Image.fromarray(img_array), bboxes

def generate_dataset(num_samples: int = 200, output_dir: str = 'kazakh_dataset'):
    """Generate dataset"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'images').mkdir(exist_ok=True)
    
    text_gen = KazakhTextGenerator()
    img_gen = ImageGenerator()
    labels = []
    train_samples = int(num_samples*0.8)
    val_samples = num_samples - train_samples
    train_data = []
    val_data = []

    for i in tqdm(range(num_samples), desc="Generating dataset"):
        # Generate text and image
        text = text_gen.generate_text_sample()
        img, bboxes = img_gen.create_text_image(text)
        
        if i < train_samples:
            folder = 'kk_train'
            train_data.append((f'sample_{i:05d}.jpg', text))
        else:
            folder = 'kk_val'
            val_data.append((f'sample_{i:05d}.jpg', text))
        
        image_path = str(output_dir / 'images' / folder / f'sample_{i:05d}.jpg')
        img.save(image_path)
        
        labels.append({
            'image_path': image_path,
            'text': text,
            'bboxes': bboxes
        })

    with open(output_dir / 'labels.json', 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    # Save train CSV
    train_csv_path = output_dir / 'images/kk_train/kk_train.csv'
    with open(train_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'words'])
        writer.writerows(train_data)

    # Save val CSV
    val_csv_path = output_dir / 'images/kk_val/kk_val.csv'
    with open(val_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'words'])
        writer.writerows(val_data)
    return labels

if __name__ == "__main__":
    # Generate dataset
    labels = generate_dataset(num_samples=1000)
    print(f"Generated {len(labels)} samples")