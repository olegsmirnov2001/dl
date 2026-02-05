import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from util import MLPRegressor


class SentimentClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        hidden_size: int,
        device: torch.device | str,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.model_name = model_name

        self.encoder = AutoModel.from_pretrained(
            model_name,
            device_map=self.device,
            trust_remote_code=True,
        ).requires_grad_(False)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        self.head = MLPRegressor(
            input_size=self.encoder.config.hidden_size,
            hidden_size=hidden_size,
        ).to(self.device)

    def mean_pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
        sum_embeddings = (hidden_states * mask_expanded).sum(dim=1)
        token_counts = attention_mask.sum(dim=1, keepdim=True)
        return sum_embeddings / token_counts

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs).last_hidden_state
        embeddings = self.mean_pool(hidden_states, attention_mask)
        return self.head(embeddings)

    def predict(self, texts: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt',
        ).to(self.device)

        with torch.no_grad():
            logits = self.forward(**inputs)
        return torch.sigmoid(logits)

    def save_head(self, path: str) -> None:
        torch.save(self.head.state_dict(), path)

    def load_head(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.head.load_state_dict(state)
