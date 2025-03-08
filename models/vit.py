from transformers import ViTModel
from torch import nn

class ViTForRegression(nn.Module):
    """
    Custom Vision Transformer (ViT) model for regression.
    """
    def __init__(self):
        super(ViTForRegression, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.regression_head = nn.Linear(self.vit.config.hidden_size, 1)  # ✅ Single output for regression

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values)
        return self.regression_head(outputs.last_hidden_state[:, 0, :])  # ✅ Extract regression output

def build_vit():
    """
    Build and return the ViT model for regression.

    Returns:
        model: A ViT model adapted for root volume regression.
    """
    return ViTForRegression()
