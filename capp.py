import os
import urllib
from functools import lru_cache
from random import randint
from typing import Any, Callable, Dict, List, Tuple

import clip
import cv2
import gradio as gr
import numpy as np
import PIL
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

#CHECKPOINT_PATH = os.path.join(os.path.expanduser("~"), ".cache", "SAM")
CHECKPOINT_PATH = "models"
CHECKPOINT_NAME = "sam_vit_h_4b8939.pth"
#CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
MODEL_TYPE = "default"
MAX_WIDTH = MAX_HEIGHT = 1024
TOP_K_OBJ = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache
def load_mask_generator() -> SamAutomaticMaskGenerator:
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
    checkpoint = os.path.join(CHECKPOINT_PATH, CHECKPOINT_NAME)
    if not os.path.exists(checkpoint):
        raise ValueError("Please download the checkpoint from the link below and put it in ~/.cache/SAM")
        urllib.request.urlretrieve(CHECKPOINT_URL, checkpoint)
    sam = sam_model_registry[MODEL_TYPE](checkpoint=checkpoint).to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator


@lru_cache
def load_clip(
    name: str = "ViT-B/32",
) -> Tuple[torch.nn.Module, Callable[[PIL.Image.Image], torch.Tensor]]:
    model, preprocess = clip.load(name, device=device, download_root=CHECKPOINT_PATH)
    return model.to(device), preprocess


def adjust_image_size(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    if height > width:
        if height > MAX_HEIGHT:
            height, width = MAX_HEIGHT, int(MAX_HEIGHT / height * width)
    else:
        if width > MAX_WIDTH:
            height, width = int(MAX_WIDTH / width * height), MAX_WIDTH
    image = cv2.resize(image, (width, height))
    return image


@torch.no_grad()
def get_score(crop: PIL.Image.Image, texts: List[str]) -> torch.Tensor:
    model, preprocess = load_clip()
    preprocessed = preprocess(crop).unsqueeze(0).to(device)
    tokens = clip.tokenize(texts).to(device)
    logits_per_image, _ = model(preprocessed, tokens)
    similarity = logits_per_image.softmax(-1).cpu()
    return similarity[0, 0]


def crop_image(image: np.ndarray, mask: Dict[str, Any]) -> PIL.Image.Image:
    x, y, w, h = mask["bbox"]
    masked = image * np.expand_dims(mask["segmentation"], -1)
    crop = masked[y : y + h, x : x + w]
    if h > w:
        top, bottom, left, right = 0, 0, (h - w) // 2, (h - w) // 2
    else:
        top, bottom, left, right = (w - h) // 2, (w - h) // 2, 0, 0
    # padding
    crop = cv2.copyMakeBorder(
        crop,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    crop = PIL.Image.fromarray(crop)
    return crop


def get_texts(query: str) -> List[str]:
    return [f"a picture of {query}", "a picture of background"]

@torch.no_grad()
def embed_text_only(queries: List[str]) -> List[Dict[str, Any]]:
    model, preprocess = load_clip()
    tokens = clip.tokenize(queries).to(device)
    #clip.tokenize(get_texts(queries)).to(device)
    embs =  model.encode_text(tokens).to(device).cpu()
    return embs / embs.norm(dim=1, keepdim=True)
    #  image_features = image_features / image_features.norm(dim=1, keepdim=True)
    #     text_features = text_features / text_features.norm(dim=1, keepdim=True)

@torch.no_grad()
def embed_img_only(
    image: np.ndarray,
    masks: List[Dict[str, Any]],
    predicted_iou_threshold: float,
    stability_score_threshold: float,
    queries: List[str],
    clip_threshold: float, 
) -> List[Dict[str, Any]]:
    embedded_pictures, crops = [], []
    model, preprocess = load_clip()
    for mask in sorted(masks, key=lambda mask: mask["area"])[-TOP_K_OBJ:]:
        if (
            mask["predicted_iou"] < predicted_iou_threshold
            or mask["stability_score"] < stability_score_threshold
            or image.shape[:2] != mask["segmentation"].shape[:2]
        ):
            continue
        crop = crop_image(image, mask)
        img_processed = preprocess(crop).unsqueeze(0).to(device)
        emds = model.encode_image(img_processed)
        emds = emds / emds.norm(dim=1, keepdim=True)
        embedded_pictures.append(emds)
        crops.append(crop)

    return torch.cat(embedded_pictures, dim=0).cpu(), crops

def filter_masks(
    image: np.ndarray,
    masks: List[Dict[str, Any]],
    predicted_iou_threshold: float,
    stability_score_threshold: float,
    query: str,
    clip_threshold: float,
) -> List[Dict[str, Any]]:
    filtered_masks: List[Dict[str, Any]] = []

    for mask in sorted(masks, key=lambda mask: mask["area"])[-TOP_K_OBJ:]:
        if (
            mask["predicted_iou"] < predicted_iou_threshold
            or mask["stability_score"] < stability_score_threshold
            or image.shape[:2] != mask["segmentation"].shape[:2]
            or query
            and get_score(crop_image(image, mask), get_texts(query)) < clip_threshold
        ):
            continue

        filtered_masks.append(mask)

    return filtered_masks

def filter_masks_multi(
    image: np.ndarray,
    masks: List[Dict[str, Any]],
    predicted_iou_threshold: float,
    stability_score_threshold: float,
    query: str,
    clip_threshold: float,
) -> List[Dict[str, Any]]:
    filtered_masks: List[Dict[str, Any]] = []

    for mask in sorted(masks, key=lambda mask: mask["area"])[-TOP_K_OBJ:]:
        if (
            mask["predicted_iou"] < predicted_iou_threshold
            or mask["stability_score"] < stability_score_threshold
            or image.shape[:2] != mask["segmentation"].shape[:2]
            or query
            and get_score(crop_image(image, mask), get_texts(query)) < clip_threshold
        ):
            continue

        filtered_masks.append(mask)

    return filtered_masks


def draw_masks(
    image: np.ndarray, masks: List[np.ndarray], alpha: float = 0.7
) -> np.ndarray:
    for mask in masks:
        color = [randint(127, 255) for _ in range(3)]

        # draw mask overlay
        colored_mask = np.expand_dims(mask["segmentation"], 0).repeat(3, axis=0)
        colored_mask = np.moveaxis(colored_mask, 0, -1)
        masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
        image_overlay = masked.filled()
        image = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

        # draw contour
        contours, _ = cv2.findContours(
            np.uint8(mask["segmentation"]), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
    return image

def segment_cluster(
    predicted_iou_threshold: float,
    stability_score_threshold: float,
    clip_threshold: float,
    image_path: str,
    query_list: List[str],
    mask_generator: SamAutomaticMaskGenerator = None,
    return_masks: bool = False,
) -> PIL.ImageFile.ImageFile:    
    if not mask_generator:
        mask_generator = load_mask_generator()
    assert isinstance(mask_generator, SamAutomaticMaskGenerator), "mask_generator must be a SamAutomaticMaskGenerator"
    
    if isinstance(image_path, str):    
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(image_path, np.ndarray):
        image = image_path
    elif isinstance(image_path, torch.Tensor):
        image = image_path.numpy()
    masks = mask_generator.generate(image)
    


def segment_multi(
    predicted_iou_threshold: float,
    stability_score_threshold: float,
    clip_threshold: float,
    image_path: str,
    query_list: List[str],
    mask_generator: SamAutomaticMaskGenerator = None,
    return_masks: bool = False,
) -> PIL.ImageFile.ImageFile:
    
    if not mask_generator:
        mask_generator = load_mask_generator()
    assert isinstance(mask_generator, SamAutomaticMaskGenerator), "mask_generator must be a SamAutomaticMaskGenerator"
    
    if isinstance(image_path, str):    
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(image_path, np.ndarray):
        image = image_path
    else:
        raise ValueError("image_path must be a str or np.ndarray")
    masks = mask_generator.generate(image)
    masks = filter_masks_multi(
        image,
        masks,
        predicted_iou_threshold,
        stability_score_threshold,
        query_list,
        clip_threshold,
    )
    image = draw_masks(image, masks)
    image = PIL.Image.fromarray(image)
    if return_masks:
        return image, masks
    return image

def segment(
    predicted_iou_threshold: float,
    stability_score_threshold: float,
    clip_threshold: float,
    image_path: str,
    query: str,
    mask_generator: SamAutomaticMaskGenerator = None,
    return_masks: bool = False,
) -> PIL.ImageFile.ImageFile:
    
    if not mask_generator:
        mask_generator = load_mask_generator()
    assert isinstance(mask_generator, SamAutomaticMaskGenerator), "mask_generator must be a SamAutomaticMaskGenerator"
    
    if isinstance(image_path, str):    
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(image_path, np.ndarray):
        image = image_path
    else:
        raise ValueError("image_path must be a str or np.ndarray")
    # reduce the size to save gpu memory
    image = adjust_image_size(image)
    masks = mask_generator.generate(image)
    masks = filter_masks(
        image,
        masks,
        predicted_iou_threshold,
        stability_score_threshold,
        query,
        clip_threshold,
    )
    image = draw_masks(image, masks)
    image = PIL.Image.fromarray(image)
    if return_masks:
        return image, masks
    return image




if __name__ == "__main__":
    demo = gr.Interface(
    fn=segment,
    inputs=[
        gr.Slider(0, 1, value=0.9, label="predicted_iou_threshold"),
        gr.Slider(0, 1, value=0.8, label="stability_score_threshold"),
        gr.Slider(0, 1, value=0.85, label="clip_threshold"),
        gr.Image(type="filepath"),
        "text",
    ],
    outputs="image",
    allow_flagging="never",
    title="Segment Anything with CLIP",
    examples=[
        [
            0.9,
            0.8,
            0.99,
            os.path.join(os.path.dirname(__file__), "examples/dog.jpg"),
            "dog",
        ],
        [
            0.9,
            0.8,
            0.75,
            os.path.join(os.path.dirname(__file__), "examples/city.jpg"),
            "building",
        ],
        [
            0.9,
            0.8,
            0.998,
            os.path.join(os.path.dirname(__file__), "examples/food.jpg"),
            "strawberry",
        ],
        [
            0.9,
            0.8,
            0.75,
            os.path.join(os.path.dirname(__file__), "examples/horse.jpg"),
            "horse",
        ],
        [
            0.9,
            0.8,
            0.99,
            os.path.join(os.path.dirname(__file__), "examples/bears.jpg"),
            "bear",
        ],
        [
            0.9,
            0.8,
            0.99,
            os.path.join(os.path.dirname(__file__), "examples/cats.jpg"),
            "cat",
        ],
        [
            0.9,
            0.8,
            0.99,
            os.path.join(os.path.dirname(__file__), "examples/fish.jpg"),
            "fish",
        ],
    ],
    )   
    demo.launch()
