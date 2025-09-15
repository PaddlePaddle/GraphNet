def get_auto_model_and_inputs(model_name, text, dtype):
    from paddlenlp.transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    model = model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(
        text, return_tensors="pd", padding=True, truncation=True, max_length=2048
    )
    return model, inputs


def get_bert_model_and_inputs(model_name, text, dtype):
    from paddlenlp.transformers import BertModel, BertTokenizer

    model = BertModel.from_pretrained(model_name)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pd")
    return model, inputs


def get_convbert_model_and_inputs(model_name, text, dtype):
    from paddlenlp.transformers import ConvBertModel as ModelClass
    from paddlenlp.transformers import ConvBertTokenizer as TokenizerClass

    model = ModelClass.from_pretrained(model_name)
    model.eval()

    tokenizer = TokenizerClass.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pd")
    return model, inputs


def get_ernie_model_and_inputs(model_name, text, dtype):
    from paddlenlp.transformers import ErnieModel, ErnieTokenizer

    model = ErnieModel.from_pretrained(model_name)
    tokenizer = ErnieTokenizer.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pd")
    return model, inputs


def get_ernie_m_model_and_inputs(model_name, text, dtype):
    from paddlenlp.transformers import ErnieMModel as ModelClass
    from paddlenlp.transformers import ErnieMTokenizer as TokenizerClass

    model = ModelClass.from_pretrained(model_name)
    model.eval()

    tokenizer = TokenizerClass.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pd")
    return model, inputs


def get_gpt_model_and_inputs(model_name, text, dtype):
    from paddlenlp.transformers import GPTModel, GPTTokenizer

    model = GPTModel.from_pretrained(model_name)
    model.eval()

    tokenizer = GPTTokenizer.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pd")
    inputs.pop("token_type_ids")
    return model, inputs


def get_nezha_model_and_inputs(model_name, text, dtype):
    from paddlenlp.transformers import NeZhaModel as ModelClass
    from paddlenlp.transformers import NeZhaTokenizer as TokenizerClass

    model = ModelClass.from_pretrained(model_name)
    tokenizer = TokenizerClass.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pd")
    return model, inputs


def get_ppminilm_model_and_inputs(model_name, text, dtype):
    from paddlenlp.transformers import PPMiniLMModel as ModelClass
    from paddlenlp.transformers import PPMiniLMTokenizer as TokenizerClass

    model = ModelClass.from_pretrained(model_name)
    tokenizer = TokenizerClass.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pd")
    return model, inputs


def get_reformer_model_and_inputs(model_name, text, dtype):
    from paddlenlp.transformers import RoFormerModel as ModelClass
    from paddlenlp.transformers import RoFormerTokenizer as TokenizerClass

    model = ModelClass.from_pretrained(model_name)
    tokenizer = TokenizerClass.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pd")
    return model, inputs


def get_skep_model_and_inputs(model_name, text, dtype):
    from paddlenlp.transformers import SkepModel as ModelClass
    from paddlenlp.transformers import SkepTokenizer as TokenizerClass

    model = ModelClass.from_pretrained(model_name)
    tokenizer = TokenizerClass.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pd")
    return model, inputs
