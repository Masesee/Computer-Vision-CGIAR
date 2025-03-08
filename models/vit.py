from transformers import ViTModel
from torch import nn

class ViTForRegression(nn.Module):
    def __init__(self, input_shape=None, num_classes=1, regression=True):
        super(ViTForRegression, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.regression_head = nn.Linear(self.vit.config.hidden_size, num_classes)
        self.regression = regression
        
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values)
        outputs = self.regression_head(outputs.last_hidden_state[:, 0, :])
        return outputs
        
def build_vit(input_shape, num_classes=1, regression=True):
    return ViTForRegression()
