import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

# 0 Attentions

class BahdanauAttention(nn.Module):
    def __init__(self, item_dim, query_dim, attention_dim):
        super(BahdanauAttention, self).__init__()
        self.item_dim = item_dim            
        self.query_dim = query_dim          
        self.attention_dim = attention_dim  

        self.W = nn.Linear(self.query_dim,  self.attention_dim, bias=False)
        self.U = nn.Linear(self.item_dim,   self.attention_dim, bias=False)
        
        self.v = nn.Parameter(torch.randn(1, attention_dim, dtype=torch.float))

    def _calculate_reactivity(self, query_vector, multiple_items):
        B, N, H = multiple_items.shape  

        query_vector    = query_vector.unsqueeze(1)
        projected_q     = self.W(query_vector)   
        projected_item  = self.U(multiple_items)  


        added_itmes = projected_q + projected_item  
        tanh_items  = torch.tanh(added_itmes)     
       
        v_t = self.v.transpose(1,0)
        batch_v = v_t.expand(B,self.attention_dim,1)      
        reactivity_scores = torch.bmm(tanh_items, batch_v)  
        reactivity_scores = reactivity_scores.squeeze(-1)  
        return reactivity_scores 

    def forward(self, query_vector, multiple_items, mask):

        assert mask is not None

        B, N, H = multiple_items.size() 
        
        reactivity_scores = self._calculate_reactivity(query_vector, multiple_items)
        reactivity_scores.data.masked_fill_(mask == 0, -float('inf'))

        attention_scores = F.softmax(reactivity_scores, dim=1) 
        attention_scores = attention_scores.unsqueeze(1)

        blendded_vector = torch.matmul(attention_scores, multiple_items) 
        blendded_vector = blendded_vector.squeeze(1) 

        return blendded_vector, attention_scores

class DotAttention(nn.Module):
    def __init__(self, item_dim, query_dim, attention_dim):
        super(DotAttention, self).__init__()
        self.item_dim = item_dim        
        self.query_dim = query_dim         
        self.attention_dim = attention_dim  

        assert query_dim == item_dim

    def _calculate_reactivity(self, query_vector, multiple_items):
        query_vector = query_vector.unsqueeze(-1) 

        reactivity_scores = torch.bmm(multiple_items, query_vector)
        reactivity_scores = reactivity_scores.squeeze(-1) 
        return reactivity_scores

    def forward(self, query_vector, multiple_items, mask):
        assert mask is not None

        B, N, H = multiple_items.size() 

        reactivity_scores = self._calculate_reactivity(query_vector, multiple_items)
        reactivity_scores.data.masked_fill_(mask == 0, -float('inf'))

        attention_scores = F.softmax(reactivity_scores, dim=1)
        attention_scores = attention_scores.unsqueeze(1) 

        blendded_vector = torch.matmul(attention_scores, multiple_items) 
        blendded_vector = blendded_vector.squeeze(1)

        return blendded_vector, attention_scores

def scaled_dot_product_attention(   q: torch.Tensor, 
                                    k: torch.Tensor, 
                                    v: torch.Tensor,                                  
                                    mask: torch.Tensor = None,
                                    dropout: float = 0.1,
                                 ) -> torch.Tensor:

    d_k = k.size()[-1]
    attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask != None:
        inverted_mask = 1.0 - mask
        inverted_mask = inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(attn.dtype).min)
        attn = attn + inverted_mask  

    attention_weights = F.softmax(attn, dim=-1) 

    if type(dropout) == float : 
        attention_weights = F.dropout(attention_weights, dropout)
    else: 
        attention_weights = dropout(attention_weights)

    output = torch.matmul(attention_weights, v)
    return output, attention_weights

class Attention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, use_bias=True):
        super(Attention, self).__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads

        self.d_k = d_model // num_heads 
        self.d_v = d_model // num_heads 

        self.wq = nn.Linear(d_model, d_model, bias=use_bias) 
        self.wk = nn.Linear(d_model, d_model, bias=use_bias) 
        self.wv = nn.Linear(d_model, d_model, bias=use_bias) 

        self.dropout = nn.Dropout(dropout)

        self.wo = nn.Linear(d_model, d_model, bias=use_bias)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.d_k) 
        x = x.transpose(1,2).contiguous()  
        return x

    def forward(self, query, key, value, mask=None):
        q = self.wq(query)     
        k = self.wk(key)      
        v = self.wv(value)   

        qS, B= q.size()[1], k.size()[0]
        
        q = self.split_heads(q, B)
        k = self.split_heads(k, B) 
        v = self.split_heads(v, B) 

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask, self.dropout)
        scaled_attention = scaled_attention.transpose(1,2)   
        merge = scaled_attention.reshape(B, qS, -1) 

        output = self.wo(merge) 

        return output, attention_weights 

# 1 Transformer

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_head, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.dropout = dropout

        self.self_attn = Attention(d_model, num_head, dropout)

        self.act_fc = nn.GELU()
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)

        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.final_layer_norm     = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        residual = x
        x, attn_scores = self.self_attn(query=x, key=x, value=x, mask=mask)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x 
        x = self.self_attn_layer_norm(x) 

        residual = x
        x = self.act_fc(self.fc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x 
        x = self.final_layer_norm(x)   

        return x, attn_scores

import copy 
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dropout, dim_feedforward=None):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        if dim_feedforward == None: dim_feedforward = 4*d_model 
        
        a_layer = TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout)

        self.layers = clones(a_layer, self.num_layers)
        
    def forward(self, x, mask=None):
        layers_attn_scores = []
        for layer in self.layers:
            x, attn_scores = layer(x, mask)
            layers_attn_scores.append(attn_scores)
        return x, layers_attn_scores


def cp_weight(src, tar, copy_bias=True, include_eps=False):
    assert tar.weight.size() == src.weight.size()
    tar.load_state_dict( src.state_dict() )
    
    if include_eps:
        with torch.no_grad():
            tar.eps = src.eps  

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_head, droput, dim_feedforward, eps=1e-12):
        super(TransformerDecoderLayer, self).__init__()
        self.embed_dim = d_model 
        self.dropout = droput

        self.self_attn = Attention(self.embed_dim, num_head, droput)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.encoder_attn = Attention(self.embed_dim, num_head, droput)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim, eps=eps)

        self.act_fc = nn.GELU()
        self.activation_dropout = droput 

        self.fc1 = nn.Linear(self.embed_dim, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=eps)
      
    def forward(self, x, enc_output, look_ahead_mask, enc_pad_mask):
        residual = x
        x, dec_attn_scores = self.self_attn(query=x, key=x, value=x, mask=look_ahead_mask)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x 
        x = self.self_attn_layer_norm(x)

        residual = x
        x, cross_attn_scores = self.encoder_attn(query=x, key=enc_output, value=enc_output, mask=enc_pad_mask)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x 
        x = self.encoder_attn_layer_norm(x)

        residual = x
        x = self.act_fc(self.fc1(x))
        x = F.dropout(x, self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x 
        x = self.final_layer_norm(x)
        
        return x, dec_attn_scores, cross_attn_scores

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dropout, dim_feedforward=None):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers

        if dim_feedforward == None: dim_feedforward = 4*d_model
        a_layer = TransformerDecoderLayer(d_model, num_heads, dropout, dim_feedforward)

        self.layers = clones(a_layer, self.num_layers)
        
    def forward(self, x, enc_output, look_ahead_mask=None, enc_pad_mask=None):
        layers_attn_scores_1 = []
        layers_attn_scores_2 = []

        for layer in self.layers:
            x, attn_scores_1, attn_scores_2 = layer(x, enc_output, look_ahead_mask, enc_pad_mask)
            layers_attn_scores_1.append(attn_scores_1)
            layers_attn_scores_2.append(attn_scores_2)

        return x, layers_attn_scores_1, layers_attn_scores_2
    
class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, pad_token_id, max_length_size, layer_norm_eps, hidden_dropout_prob):
        super().__init__()
        self.word_embeddings        = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings    = nn.Embedding(max_length_size, hidden_size)

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout   = nn.Dropout(hidden_dropout_prob)

        self.register_buffer("position_ids", torch.arange(max_length_size).expand((1, -1)))
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
            persistent=False,
        )

        self.position_embedding_type = "absolute"

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds 
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dropout, dim_feedforward=None):
        super().__init__()

        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, dropout, dim_feedforward)
        self.decoder = TransformerDecoder(num_layers, d_model, num_heads, dropout, dim_feedforward)

    def create_padding_mask(self, mask):
        return mask[:, None, None, :] 

    def create_look_ahead_mask(self, seq_len):  
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.int), diagonal=1)
        mask = 1 - mask 
        return mask

    def forward(self, enc_input, dec_input, enc_pad_mask):
        enc_pad_mask = self.create_padding_mask(enc_pad_mask)
        enc_output, enc_layer_att_scores = self.encoder(enc_input, enc_pad_mask)

        dec_length = dec_input.size()[1] 
        look_ahead_mask = self.create_look_ahead_mask(dec_length).to(dec_input.device)

        dec_output, dec_layer_att_scores, dec_layer_cross_att_scores = self.decoder(
                                                                                    dec_input, 
                                                                                    enc_output, 
                                                                                    look_ahead_mask=look_ahead_mask, 
                                                                                    enc_pad_mask=enc_pad_mask
                                                                                )
        return enc_output, dec_output, enc_layer_att_scores, dec_layer_att_scores, dec_layer_cross_att_scores


# 2 BERT

class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, pad_token_id, max_bert_length_size, layer_norm_eps, hidden_dropout_prob):
        super().__init__()
        self.word_embeddings        = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings    = nn.Embedding(max_bert_length_size, hidden_size)
        self.token_type_embeddings  = nn.Embedding(2, hidden_size) # why 2 ? 0 and 1 

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout   = nn.Dropout(hidden_dropout_prob)

        self.register_buffer("position_ids", torch.arange(max_bert_length_size).expand((1, -1)))
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
            persistent=False,
        )

        self.position_embedding_type = "absolute"

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BERT_CONFIG():
    def __init__(self, vocab_size, padding_idx, max_seq_length, 
                       d_model, layer_norm_eps, emb_hidden_dropout,
                       num_layers, num_heads, att_prob_dropout, dim_feedforward
                 ):
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        self.layer_norm_eps = layer_norm_eps
        self.emb_hidden_dropout = emb_hidden_dropout
        self.num_layers=num_layers
        self.num_heads = num_heads
        self.att_prob_dropout = att_prob_dropout
        self.dim_feedforward = dim_feedforward

class BERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = BertEmbeddings(
                                            config.vocab_size ,
                                            config.d_model,
                                            config.padding_idx,
                                            config.max_seq_length,
                                            config.layer_norm_eps,    
                                            config.emb_hidden_dropout 
                                    )
        
        self.encoder = TransformerEncoder(
                                        num_layers=config.num_layers, 
                                        d_model=config.d_model, 
                                        num_heads=config.num_heads,
                                        dropout=config.att_prob_dropout,
                                        dim_feedforward=config.dim_feedforward
                                )

        self.pooler = BertPooler(config.d_model)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        attention_mask = attention_mask[:, None, None, :] 
        seq_embs   = self.embeddings(input_ids, token_type_ids)
        output     = self.encoder(seq_embs, attention_mask)
        pooled_out = self.pooler(output[0]) 

        layer_attention_scores = output[1]
        return pooled_out, layer_attention_scores

    def cp_encoder_block_weights_from_huggingface(self, src_encoder, tar_encoder):
        for layer_num, src_layer in enumerate(src_encoder.layer):
            cp_weight(src_layer.attention.self.query,   tar_encoder.layers[layer_num].self_attn.wq) 
            cp_weight(src_layer.attention.self.key,     tar_encoder.layers[layer_num].self_attn.wk) 
            cp_weight(src_layer.attention.self.value,   tar_encoder.layers[layer_num].self_attn.wv) 
            cp_weight(src_layer.attention.output.dense, tar_encoder.layers[layer_num].self_attn.wo) 

            cp_weight(src_layer.intermediate.dense, tar_encoder.layers[layer_num].fc1)
            cp_weight(src_layer.output.dense,       tar_encoder.layers[layer_num].fc2) 

            cp_weight(src_layer.attention.output.LayerNorm, tar_encoder.layers[layer_num].self_attn_layer_norm, include_eps=True) 
            cp_weight(src_layer.output.LayerNorm,           tar_encoder.layers[layer_num].final_layer_norm, include_eps=True) 

        return tar_encoder

    def copy_weights_from_huggingface(self, hg_bert):
        self.embeddings.load_state_dict( hg_bert.embeddings.state_dict() ) 
        self.pooler.load_state_dict( hg_bert.pooler.state_dict() ) 

        self.encoder = self.cp_encoder_block_weights_from_huggingface(
                                            src_encoder=hg_bert.encoder,
                                            tar_encoder=self.encoder 
                                          )

# 3 GPT2

class Conv1D(nn.Module):
    def __init__(self, nx, nf):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias   = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x

def gelu_new(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class GPT2MLP(nn.Module):
    def __init__(self, d_model, nx, dropout):
        super().__init__()
        self.c_fc    = Conv1D(d_model, nx) 
        self.c_proj  = Conv1D(nx, d_model)
        self.act     = gelu_new 
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))

class GPT2MLP_linear_version(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super(GPT2MLP, self).__init__()
        self.feedforward_1 = nn.Linear(d_model, dim_feedforward)
        self.act_function  = nn.GELU()
        self.feedforward_2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.feedforward_1(x)
        x = self.act_function(x)
        x = self.feedforward_2(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, bias=True):
        super().__init__()
        self.n_head  = n_head
        self.d_model = d_model
        self.c_attn  = Conv1D(d_model, d_model*3)  

        self.dropout = nn.Dropout(0.1)
        self.c_proj  = Conv1D(d_model, d_model)    #
        
        assert d_model % n_head == 0
        self.d_k = d_model // self.n_head  

    def split_heads(self, x):
        new_shape = x.size()[:-1] + (self.n_head, self.d_k) 
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3) 

    def _attn(self, q, k, v, mask=None):
        scores  = torch.matmul(q, k.transpose(-2, -1))
        scores  = scores/math.sqrt(v.size(-1))  
        nd, ns  = scores.size(-2), scores.size(-1)

        if mask != None:
            mask = (1.0 - mask) * -1e4 
            scores = scores + mask   

        scores  = F.softmax(scores, dim=-1) 
        scores  = self.dropout(scores)
        outputs = torch.matmul(scores, v)
        return outputs, scores
    
    def merge_heads(self, x):
        x         = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (x.size(-2)*x.size(-1),)
        return x.view(*new_shape)
        
    def forward(self, x, attention_mask):
        x        = self.c_attn(x) 
        q, k, v  = x.split(self.d_model, dim=2)
        q, k, v  = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        out, scores = self._attn(q, k, v, attention_mask)
        out      = self.merge_heads(out)
        out      = self.c_proj(out)
        return out, scores

class GPT2_TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward, dropout=0.1):
        super(GPT2_TransformerBlock, self).__init__()
        self.attn        = MultiHeadAttention(d_model=d_model, n_head=n_head, bias=True)
        self.mlp         = GPT2MLP(d_model=d_model, nx=dim_feedforward, dropout=dropout)
        self.ln_1        = nn.LayerNorm(d_model)
        self.ln_2        = nn.LayerNorm(d_model)
                
    def forward(self, x, look_ahead_mask):
        nx = self.ln_1(x)
        a, attn_scores = self.attn(nx, attention_mask=look_ahead_mask) 
        x = x + a 

        m = self.mlp( self.ln_2(x) )
        x = x + m 

        return x, attn_scores

class GPT2Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dim_feedforward=None):
        super(GPT2Decoder, self).__init__()
        self.num_layers = num_layers
        if dim_feedforward == None: dim_feedforward = 4*d_model 
        
        a_layer = GPT2_TransformerBlock(d_model=d_model, n_head=num_heads, dim_feedforward=dim_feedforward)

        self.layers = clones(a_layer, self.num_layers)
        
    def forward(self, x, look_ahead_mask=None):
        layers_attn_scores = []
        for layer in self.layers:
            x, attn_scores = layer(x, look_ahead_mask)
            layers_attn_scores.append(attn_scores)

        return x, layers_attn_scores

class GPT2(nn.Module):
    def __init__(self, 
                 vocab_size,   
                 num_layers,    
                 emb_dim,      
                 d_model,       
                 num_heads,
                 max_seq_length,
                 ):
        super().__init__()
        self.max_seq_len = max_seq_length
        self.dropout_rate = 0.1 
        self.dim_feedforward = 4 * d_model  

        self.tokens = 0

        self.wte = nn.Embedding(vocab_size, emb_dim)       
        self.wpe = nn.Embedding(self.max_seq_len, emb_dim) 
        self.emb_dropout = nn.Dropout(self.dropout_rate)
        self.register_buffer("position_ids", torch.arange(self.max_seq_len).expand((1, -1)))

        self.blocks = GPT2Decoder(
                                        num_layers=num_layers,
                                        d_model=d_model,
                                        num_heads=num_heads,
                                        dim_feedforward=self.dim_feedforward
                                 )
        self.ln_f   = nn.LayerNorm(d_model) 

        self.head = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, input_ids):
        B, seq_len = input_ids.size()
        assert seq_len <= self.max_seq_len
        
        token_embeddings = self.wte(input_ids)
        seq_length = input_ids.shape[1]
        position_ids = self.position_ids[:, :seq_length]
        position_embeddings = self.wpe(position_ids)
        x = self.emb_dropout(token_embeddings + position_embeddings)
        
        lookahead_mask = self.look_ahead_mask(seq_len).to(x.device) 
        x, layer_attn_scores = self.blocks(x, look_ahead_mask=lookahead_mask)
        x = self.ln_f(x)  

        logits = self.head(x)

        return logits

    def look_ahead_mask(self, tgt_len:int) -> torch.FloatTensor:  
        mask = torch.triu(torch.ones(tgt_len, tgt_len, dtype=torch.int), diagonal=1)
        mask = 1 - mask 
        return mask
        
    

def cp_weight(src, tar, copy_bias=True, include_eps=False):
    assert tar.weight.size() == src.weight.size()
    tar.load_state_dict( src.state_dict() )
    
    if include_eps:
        with torch.no_grad():
            tar.eps = src.eps  

def cp_gpt2_transformer_block_weights(src, tar):
    cp_weight(src.transformer.ln_f, tar.ln_f, include_eps=True) 

    for layer_num, src_block in enumerate(src.transformer.h):
        cp_weight(src_block.attn.c_attn,        tar.blocks.layers[layer_num].attn.c_attn) 
        cp_weight(src_block.attn.c_proj,        tar.blocks.layers[layer_num].attn.c_proj) 

        cp_weight(src_block.mlp.c_fc,       tar.blocks.layers[layer_num].mlp.c_fc)
        cp_weight(src_block.mlp.c_proj,     tar.blocks.layers[layer_num].mlp.c_proj)

        cp_weight(src_block.ln_1, tar.blocks.layers[layer_num].ln_1, include_eps=True)
        cp_weight(src_block.ln_2, tar.blocks.layers[layer_num].ln_2, include_eps=True)

    return tar