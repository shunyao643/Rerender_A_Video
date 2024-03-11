from PIL import Image
import os
import torch
from transformers import AutoProcessor, CLIPModel


def evaluate_clip(original_frame_dir: str, output_frame_dir: str, text_prompt: str, batch_size: int = 5):
    """
    Evaluates temporal consistency and frame accuracy using CLIP.
    :param original_frame_dir: Directory for original frames
    :param output_frame_dir:  Directory for output frames
    :param text_prompt: Text prompt used for video generation
    :param batch_size: Number of images to process in each batch (more frames require more memory)
    :return: (Temporal consistency, Frame accuracy)
    """

    # Load the CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # Fetching image URLs from the folder (assuming all images are in jpg format)
    relative_image_urls = [filename for filename in os.listdir(output_frame_dir) if filename.endswith('.png')]
    relative_image_urls = sorted(relative_image_urls)
    # First process the images in the output_frame_dir
    image_urls = [os.path.join(output_frame_dir, filename) for filename in relative_image_urls]
    total_images = len(image_urls)

    # Initialize a tensor to store all image features
    image_features_tensor = torch.empty(total_images, 1, 768)  # 768 is the CLIP feature size

    def process_image(image_url):
        image = Image.open(image_url)
        inputs = processor(images=image, return_tensors="pt")
        image_features = model.get_image_features(**inputs)
        return image_features

    # Batch process output images to get their features
    for batch_start in range(0, total_images, batch_size):
        batch_end = min(batch_start + batch_size, total_images)
        batch_image_urls = image_urls[batch_start:batch_end]
        batch_image_features = torch.stack([process_image(image_url) for image_url in batch_image_urls])
        image_features_tensor[batch_start:batch_end] = batch_image_features

    # Frame Consistency
    # _Tem - Con_
    cosine_similarities = torch.cosine_similarity(image_features_tensor[:-1], image_features_tensor[1:], dim=-1)
    average_similarity = torch.sum(cosine_similarities) / (total_images - 1)
    print(f"Tem-Con: {average_similarity.item(): .4f}")  # Output average cosine similarity

    # Prompt Consistency
    # _Frame - Acc_
    original_image_urls = [os.path.join(original_frame_dir, filename) for filename in relative_image_urls]
    # Initialize a tensor to store all image features
    original_features_tensor = torch.empty(total_images, 1, 768)  # 768 is the CLIP feature size

    # Batch process original (input) images to get their features
    for batch_start in range(0, total_images, batch_size):
        batch_end = min(batch_start + batch_size, total_images)
        batch_image_urls = original_image_urls[batch_start:batch_end]
        batch_image_features = torch.stack([process_image(image_url) for image_url in batch_image_urls])
        original_features_tensor[batch_start:batch_end] = batch_image_features

    # Process the text prompt
    text = processor(text_prompt, return_tensors="pt", padding=True)
    text_features = model.get_text_features(**text)

    # Calculate pairwise cosine similarity between (prompt and each NEW image) AND (prompt and each OLD image)
    prompt_new_cosine_similarities = torch.cosine_similarity(image_features_tensor, text_features, dim=2).squeeze()
    prompt_old_cosine_similarities = torch.cosine_similarity(original_features_tensor, text_features, dim=2).squeeze()

    frame_acc = torch.mean((prompt_new_cosine_similarities > prompt_old_cosine_similarities).float())
    print(f"Frame-Acc: {frame_acc.item(): .4f}")

    return average_similarity.item(), frame_acc.item()
