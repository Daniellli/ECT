'''
Author:   "  "
Date: 2022-06-20 20:59:06
LastEditTime: 2023-08-06 22:04:01
LastEditors: daniel
Description: 

FilePath: /Cerberus-main/model/decoder.py
email:  
'''




import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor



class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None,
            return_intermediate=False,
            return_attention = False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.return_attention = return_attention

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []
        attentions = []
        for layer in self.layers:
            if self.return_attention:
                output ,attention= layer(output, memory, tgt_mask=tgt_mask,
                            memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask,
                            pos=pos, query_pos=query_pos)
                attentions.append(attention)         
            else:
                output = layer(output, memory, tgt_mask=tgt_mask,
                            memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask,
                            pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        """
        Whether to perform normalization on the output of the last layer?
        """
        if self.norm is not None:
            output = self.norm(output)  
            if self.return_intermediate: 
                intermediate.pop() #* replace last output
                intermediate.append(output)

        if self.return_intermediate:
            if self.return_attention:
                return output.unsqueeze(0),torch.stack(attentions)
            else :
                return torch.stack(intermediate)
        

        if self.return_attention:
            return output.unsqueeze(0),torch.stack(attentions)
        else :
            return output.unsqueeze(0)




class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,return_attention=False):
        super().__init__()



        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        #!======================================================================================================
        self.self_attn_learnable_embedding1  = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        #* for learnable embedding 
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.return_attention = return_attention
        

        #!======================================================================================================
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

   

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    #* tgt is Q , memory is  KV
    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        #* Q feature shape can not be 768 

        q = k = self.with_pos_embed(tgt, query_pos)#?  decoder 这个query position  从哪里生成? 

        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        #todo: Memory refers to the keys and values of the cross-attention.
        #! Therefore, what I need to do is to apply self-attention to the memory as well.
        #* memory: learnable embedding 
        #*==============================================================================
        memory = memory + self.dropout4(self.self_attn_learnable_embedding1(memory, memory, value=memory)[0])
        memory = self.norm4(memory)
        #*============================================================================== learnable embedding interaction version 2 
        # depth_query = memory[0].unsqueeze(0)
        # normal_query = memory[1].unsqueeze(0)
        # reflectance_query = memory[2].unsqueeze(0)
        # illumination_query = memory[3].unsqueeze(0)

        # depth_query = depth_query + \
        #     self.dropout4(self.self_attn_learnable_embedding1(depth_query, depth_query, value=depth_query)[0])
        # depth_query = self.norm4(depth_query)

        # normal_query = normal_query + \
        #     self.dropout5(self.self_attn_learnable_embedding2(normal_query, normal_query, value=normal_query)[0])
        # normal_query = self.norm5(normal_query)


        # reflectance_query = reflectance_query + \
        #     self.dropout6(self.self_attn_learnable_embedding3(reflectance_query, reflectance_query, value=reflectance_query)[0])
        # reflectance_query = self.norm6(reflectance_query)

        # illumination_query = illumination_query + \
        #     self.dropout7(self.self_attn_learnable_embedding4(illumination_query, illumination_query, value=illumination_query)[0])
        # illumination_query = self.norm7(illumination_query)

        # memory = torch.cat([depth_query,normal_query,reflectance_query,illumination_query])

        #*==============================================================================
        """
        multihead_attn returns two parameters.
         The first one is the attention output, and the second one is the weight, representing the attention for this iteration.
        """

        if self.return_attention :
            tgt2,attention= self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)

            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)
            return tgt,attention
            
        else :
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]


            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)

            return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)



def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
