import torch

class ClassificationHead(torch.nn.Module):
    """Classification Head for  transformer encoders"""

    def __init__(self, class_size, embed_size):
        super(ClassificationHead, self).__init__()
        self.class_size = class_size
        self.embed_size = embed_size
        # self.mlp1 = torch.nn.Linear(embed_size, embed_size)
        # self.mlp2 = (torch.nn.Linear(embed_size, class_size))
        self.mlp = torch.nn.Linear(embed_size, class_size)

    def forward(self, hidden_state):
        # hidden_state = F.relu(self.mlp1(hidden_state))
        # hidden_state = self.mlp2(hidden_state)
        logits = self.mlp(hidden_state)
        return logits


class NoLinClassificationHead(torch.nn.Module):
    """Classification Head for  transformer encoders With relu sub layer"""

    def __init__(self, class_size, embed_size):
        super(NoLinClassificationHead, self).__init__()
        self.class_size = class_size
        self.embed_size = embed_size

        self.sub_layer = torch.nn.Linear(embed_size, embed_size, bias=False)

        # self.mlp1 = torch.nn.Linear(embed_size, embed_size)
        # self.mlp2 = (torch.nn.Linear(embed_size, class_size))
        self.mlp = torch.nn.Linear(embed_size, class_size)

    def forward(self, hidden_state):
        # hidden_state = F.relu(self.mlp1(hidden_state))
        # hidden_state = self.mlp2(hidden_state)
        sub_repr = torch.relu(self.sub_layer(hidden_state))

        logits = self.mlp(sub_repr)
        return logits
