import logging
from PIL import Image
import os
import torch
from transformers import AutoProcessor, CLIPModel

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def evaluate_clip(original_frame_dir: str, output_frame_dir: str, text_prompt: str, batch_size: int = 5):
    """
    Evaluates temporal consistency and frame accuracy using CLIP.
    :param original_frame_dir: Directory for original frames
    :param output_frame_dir:  Directory for output frames
    :param text_prompt: Text prompt used for video generation
    :param batch_size: Number of images to process in each batch (more frames require more memory)
    :return: (Temporal consistency, Frame accuracy)
    """

    logger.debug(f"Original Frame Directory: {original_frame_dir}")
    logger.debug(f"Output Frame Directory: {output_frame_dir}")

    logger.info("Loading CLIP model and processor...")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    logger.info("Fetching image URLs...")

    relative_image_urls = [filename for filename in os.listdir(output_frame_dir) if filename.endswith('.jpg')]

    relative_image_urls = sorted(relative_image_urls)
    image_urls = [os.path.join(output_frame_dir, filename) for filename in relative_image_urls]
    total_images = len(image_urls)
    logger.info(f"Fetched {total_images} image URLs.")
    logger.debug(f"Sample Image URLs: {image_urls[:5]}")

    print("Processing output images...")
    image_features_tensor = torch.empty(total_images, 1, 768)

    def process_image(image_url):
        image = Image.open(image_url)
        inputs = processor(images=image, return_tensors="pt")
        image_features = model.get_image_features(**inputs)
        return image_features

    for batch_start in range(0, total_images, batch_size):
        batch_end = min(batch_start + batch_size, total_images)
        batch_image_urls = image_urls[batch_start:batch_end]
        batch_image_features = torch.stack([process_image(image_url) for image_url in batch_image_urls])
        image_features_tensor[batch_start:batch_end] = batch_image_features
        logger.debug(f"Processed images [{batch_start}, {batch_end}).")

    cosine_similarities = torch.cosine_similarity(image_features_tensor[:-1], image_features_tensor[1:], dim=-1)
    average_similarity = torch.sum(cosine_similarities) / (total_images - 1)
    print(f"Temporal consistency (Tem-Con): {average_similarity.item(): .4f}")

    print("Processing original images...")
    original_image_urls = [os.path.join(original_frame_dir, filename[:-4] + '.png') for filename in relative_image_urls]
    logger.debug(f"Sample Image URLs: {original_image_urls[:5]}")
    original_features_tensor = torch.empty(total_images, 1, 768)

    for batch_start in range(0, total_images, batch_size):
        batch_end = min(batch_start + batch_size, total_images)
        batch_image_urls = original_image_urls[batch_start:batch_end]
        batch_image_features = torch.stack([process_image(image_url) for image_url in batch_image_urls])
        original_features_tensor[batch_start:batch_end] = batch_image_features
        logger.debug(f"Processed images [{batch_start}, {batch_end}).")

    text = processor(text=text_prompt, return_tensors="pt", padding=True)
    text_features = model.get_text_features(**text)

    prompt_new_cosine_similarities = torch.cosine_similarity(image_features_tensor, text_features, dim=2).squeeze()
    prompt_old_cosine_similarities = torch.cosine_similarity(original_features_tensor, text_features, dim=2).squeeze()

    frame_acc = torch.mean((prompt_new_cosine_similarities > prompt_old_cosine_similarities).float())
    print(f"Frame accuracy (Frame-Acc): {frame_acc.item(): .4f}")

    return average_similarity.item(), frame_acc.item()
