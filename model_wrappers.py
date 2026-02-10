from tqdm import tqdm
import open_clip
import sentence_transformers
import torch
import torch.nn as nn
import constants


class CLIPTextEncoderWrapper(nn.Module):

    def __init__(self, clip_architecture="ViT-L-14", pretrained="openai"):
        super().__init__()
        self.architecture = clip_architecture
        self.pretrained = pretrained
        self.clip, _, _ = open_clip.create_model_and_transforms(
            clip_architecture, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(clip_architecture)

        del self.clip.visual  # remove the vision encoder

    def encode_texts_batched(self, texts: list[str], device, bs=128):
        self.to(device)
        self.eval()
        text_embeddings = []

        with torch.no_grad():
            for idx in tqdm(range(0, len(texts), bs)):
                input = self.tokenizer(texts[idx : min(idx + bs, len(texts))]).to(
                    device
                )
                batch_embeddings = self.clip.encode_text(input).cpu()
                text_embeddings.append(batch_embeddings)

        text_embeddings = torch.cat(text_embeddings, dim=0)
        return text_embeddings

    def forward(self, x):
        device = list(self.clip.transformer.parameters())[0].device
        return self.encode_texts_batched(x, device=device)


class SentenceEncoderWrapper(nn.Module):

    def __init__(self, model_name="Alibaba-NLP/gte-large-en-v1.5"):
        super().__init__()
        self.model_name = model_name
        self.encoder = sentence_transformers.SentenceTransformer(
            model_name, trust_remote_code=True
        )

    def encode_texts_batched(self, texts: list[str], device, bs=128):
        self.to(device)
        self.eval()
        text_embeddings = []
        with torch.no_grad():
            for idx in tqdm(range(0, len(texts), bs)):
                batch_embeddings = self.encoder.encode(
                    texts[idx : min(idx + bs, len(texts))],
                    convert_to_tensor=True,
                    device=device,
                ).cpu()
                text_embeddings.append(batch_embeddings)

        text_embeddings = torch.cat(text_embeddings, dim=0)
        return text_embeddings

    def forward(self, x):
        device = list(self.encoder.parameters())[0].device
        return self.encode_texts_batched(x, device=device)


class TemperatureScaledLinearLayer(nn.Linear):

    def __init__(self, input_dim, output_dim, init_temp=None):
        super().__init__(input_dim, output_dim, bias=False)
        if init_temp is None:
            init_temp = constants.TEMP_INIT
        self.temperature = nn.Parameter(torch.ones(1, dtype=torch.float32) * init_temp)

    def forward(self, x):
        out = super().forward(x)
        return out * self.temperature.exp()
