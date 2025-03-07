from transformers import ViTForImageClassification

def build_vit(num_classes):
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=num_classes)
    return model
