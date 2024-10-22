class TextEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", pretrained=True, trainable=False):
        super().__init__()
        """
        Create the model and set its weights frozen. 
        Use Transformers library docs to find out how to do this.
        """
        # use the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

        if pretrained:
            self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        else:
            configuration = DistilBertConfig()
            self.model = DistilBertModel(configuration)

        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Pass the arguments through the model and make sure to return CLS token embedding
        """
        result = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return result.last_hidden_state[:, self.target_token_idx]
