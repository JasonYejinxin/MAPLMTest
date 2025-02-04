import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model and processor
model_path = "./blip2_flan_t5_epoch_5_concatenate"  # replace with your trained model path
processor = BlipProcessor.from_pretrained(model_path)
model = BlipForConditionalGeneration.from_pretrained(model_path)
model = model.to(device)
model.eval()

# Load images (during inference, load multiple frames)
def load_images_for_inference(test_dir):
    images = []
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory {test_dir} not found.")
    
    # Traverse all subfolders (fr1, fr2, etc.)
    subfolders = sorted(os.listdir(test_dir))
    frame_names = []
    for folder in subfolders:
        folder_path = os.path.join(test_dir, folder)
        if os.path.isdir(folder_path):  # Ensure it's a subfolder
            img_names = sorted(os.listdir(folder_path))  # Sort files by name
            images_in_frame = []
            for img_name in img_names[:4]:  # Load at most 4 images
                img_path = os.path.join(folder_path, img_name)
                if img_path.endswith(".jpg") or img_path.endswith(".png"):  # Only load jpg/png images
                    try:
                        img = Image.open(img_path).convert("RGB")
                        images_in_frame.append(img)
                    except Exception as e:
                        print(f"Failed to open image {img_path}: {e}")
            if images_in_frame:
                images.append(images_in_frame)
                frame_names.append(folder)  # Save the frame name
    if not images:
        raise ValueError(f"No valid images found in {test_dir}.")
    return images, frame_names

# Feature concatenation (consistent with training)
def concatenate_image_features(images):
    pixel_values = []
    for img in images:
        inputs = processor(images=img, return_tensors="pt").pixel_values.to(device)
        pixel_values.append(inputs)
    pixel_values = torch.cat(pixel_values, dim=1)  # [1, 12, H, W]
    # Use a convolution layer to reduce channels to 3 (ensure consistency with model input)
    channel_reduction = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=1).to(device)
    reduced_pixel_values = channel_reduction(pixel_values)
    return reduced_pixel_values

# Inference function
def generate_answer(test_dir, question, feature_method="concatenate"):
    # Load images
    images, frame_names = load_images_for_inference(test_dir)

    answers = {}
    # Perform inference for each group of images
    for i, frame_images in enumerate(images):
        # Feature processing
        if feature_method == "concatenate":
            pixel_values = concatenate_image_features(frame_images)
        else:
            raise ValueError("Only 'concatenate' feature method is supported in this inference code.")

        # Process question
        text_inputs = processor(text=question, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

        # Model inference
        with torch.no_grad():
            outputs = model.generate(input_ids=text_inputs, pixel_values=pixel_values, max_length=50)

            print(f"Raw model outputs for frame {frame_names[i]}: {outputs}")  # Add logging to inspect outputs
            if outputs is not None:
                answer = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                print(f"No output generated for frame {frame_names[i]}")
                continue

        # Store the answer with the corresponding frame name
        answers[frame_names[i]] = answer

    return answers

# Example inference
if __name__ == "__main__":
    test_dir = "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/data/maplm_v0.1/test"
    question = "How many lanes in current road?"  # Replace with your question

    try:
        answers = generate_answer(test_dir, question)
        for frame_name, answer in answers.items():
            print(f"Frame: {frame_name}")
            print(f"Answer: {answer}")
    except Exception as e:
        print(f"Error occurred during inference: {e}")
