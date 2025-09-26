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


def get_bart_model_and_inputs(model_name, text, dtype):
    from paddlenlp.transformers import BartModel, BartTokenizer

    model = BartModel.from_pretrained(model_name)
    model.eval()

    tokenizer = BartTokenizer.from_pretrained(model_name)

    inputs = tokenizer(
        text,
        return_tensors="pd",
        padding=True,
        truncation=True,
        max_length=512,
    )
    inputs.pop("token_type_ids", None)

    return model, inputs


def get_xlnet_model_and_inputs(model_name, text, dtype):
    import paddle
    from paddlenlp.transformers import XLNetModel, XLNetTokenizer, XLNetConfig

    config = XLNetConfig.from_pretrained(model_name)
    model = XLNetModel(config)
    if dtype == "float16":
        model = model.astype(paddle.float16)
    model.eval()

    tokenizer = XLNetTokenizer.from_pretrained(model_name)

    enc = tokenizer(
        text,
        return_tensors="pd",
        padding=True,
        truncation=True,
        # max_length=512,
    )
    if "attention_mask" not in enc:
        input_ids = enc["input_ids"]
        pad_id = tokenizer.pad_token_id
        enc["attention_mask"] = (input_ids != pad_id).astype("int64")

    return model, enc


def get_t5_model_and_inputs(model_name, text, dtype):
    import paddle
    from paddlenlp.transformers import T5ForConditionalGeneration, T5Tokenizer

    # 1) 分词器（先建 tokenizer 方便取 pad/eos id）
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # 2) 编码输入（支持单条或批量 text）
    enc = tokenizer(
        text,
        return_tensors="pd",
        padding=True,
        truncation=True,
        max_length=512,
    )

    # 补 attention_mask（pad 处为 0，其他为 1）
    if "attention_mask" not in enc:
        input_ids = enc["input_ids"]
        attn_mask = (input_ids != tokenizer.pad_token_id).astype("int64")
        enc["attention_mask"] = attn_mask

    # 构造 decoder_input_ids：
    # T5 以 pad_token_id 作为 decoder_start_token_id
    batch_size = enc["input_ids"].shape[0]
    decoder_input_ids = paddle.full(
        shape=[batch_size, 1],
        fill_value=tokenizer.pad_token_id,
        dtype="int64",
    )

    # 3) 加载模型
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    if dtype == "float16":
        model = model.astype(paddle.float16)
    model.eval()

    # 4) 组装喂给模型的输入
    inputs = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "decoder_input_ids": decoder_input_ids,
    }
    return model, inputs


def get_albert_model_and_inputs(model_name, text, dtype):
    """
    加载 ALBERT backbone（AlbertModel）并构造输入。
    - model_name 例如: "albert-base-v2", "albert-xxlarge-v1"（PaddleNLP 内置名称）
    - dtype: "float32" 或 "float16"
    返回: (model, inputs_dict)
    """
    import paddle
    from paddlenlp.transformers import AlbertConfig, AlbertModel, AlbertTokenizer

    # 1) 读取配置（不触发权重下载）
    config = AlbertConfig.from_pretrained(model_name)

    # 2) 模型
    #    若你只需要网络结构，可改成: model = AlbertModel(config)
    model = AlbertModel(config)
    if dtype == "float16":
        model = model.astype(paddle.float16)
    model.eval()

    # 3) 分词器
    tokenizer = AlbertTokenizer.from_pretrained(model_name)

    # 若无 pad_token，则回退到 unk_token（ALBERT 没有 eos_token，别设 pad=eos）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    enc = tokenizer(
        text,
        return_tensors="pd",
        padding=True,
        truncation=True,
        max_length=512,
    )

    if "attention_mask" not in enc:
        input_ids = enc["input_ids"]
        enc["attention_mask"] = (input_ids != tokenizer.pad_token_id).astype("int64")

    return model, enc
