import math
from pprint import pprint
import torch
import torch.nn.functional as F

import torchvision
from torchvision.transforms import InterpolationMode, Resize 
import xformers
TOKENSCON = 77
TOKENS = 75


def _memory_efficient_attention_xformers(module, query, key, value):
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    hidden_states = xformers.ops.memory_efficient_attention(query, key, value,attn_bias=None)
    hidden_states = module.batch_to_head_dim(hidden_states)
    return hidden_states

def main_forward_diffusers(module,hidden_states,encoder_hidden_states,divide,userpp = False,tokens=[],width = 64,height = 64,step = 0, isxl = False, inhr = None):
    context = encoder_hidden_states
    query = module.to_q(hidden_states)
    # cond, uncond =query.chunk(2)
    # query=torch.cat([cond,uncond])
    key = module.to_k(context)
    value = module.to_v(context)

    print("query.shape: ", query.shape) #[2, 4096, 1536]
    print("key.shape: ", key.shape) #[2, 154, 1536]
    print("value.shape: ", value.shape) #[2, 154, 1536]

    query = module.head_to_batch_dim(query) #[48, 4096, 64]
    key = module.head_to_batch_dim(key) #[48, 154, 64]
    value = module.head_to_batch_dim(value) #[48, 154, 64]

    print("query.shape: ", query.shape)
    print("key.shape: ", key.shape)
    print("value.shape: ", value.shape)


    # TODO:
    hidden_states=_memory_efficient_attention_xformers(module, query, key, value)
    hidden_states = hidden_states.to(query.dtype) #[2, 4096, 1536]

    print("hidden_states.shape: ", hidden_states.shape)

    # linear proj
    hidden_states = module.to_out[0](hidden_states)
    # dropout
    hidden_states = module.to_out[1](hidden_states)

    print("hidden_states.shape: ", hidden_states.shape)

    return hidden_states #[1, 4096, 1536]


def main_forward_diffusers_sd3(module,hidden_states,encoder_hidden_states):
    residual = hidden_states

    input_ndim = hidden_states.ndim
    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
    context_input_ndim = encoder_hidden_states.ndim
    if context_input_ndim == 4:
        batch_size, channel, height, width = encoder_hidden_states.shape
        encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size = encoder_hidden_states.shape[0]

    # `sample` projections.
    query = module.to_q(hidden_states)
    key = module.to_k(hidden_states)
    value = module.to_v(hidden_states)

    # `context` projections.
    encoder_hidden_states_query_proj = module.add_q_proj(encoder_hidden_states)
    encoder_hidden_states_key_proj = module.add_k_proj(encoder_hidden_states)
    encoder_hidden_states_value_proj = module.add_v_proj(encoder_hidden_states)

    # attention
    query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
    key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
    value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // module.heads
    query = query.view(batch_size, -1, module.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, module.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, module.heads, head_dim).transpose(1, 2)

    #print("query.shape: ", query.shape) # [2, 24, 4250, 64]
    #print("key.shape: ", key.shape) # [2, 24, 4250, 64]
    #print("value.shape: ", value.shape) # [2, 24, 4250, 64]

    hidden_states = hidden_states = F.scaled_dot_product_attention(
        query, key, value, dropout_p=0.0, is_causal=False
    )
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, module.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

    # Split the attention outputs.
    hidden_states, encoder_hidden_states = (
        hidden_states[:, : residual.shape[1]],
        hidden_states[:, residual.shape[1] :],
    )

    # linear proj
    hidden_states = module.to_out[0](hidden_states)
    # dropout
    hidden_states = module.to_out[1](hidden_states)
    if not module.context_pre_only:
        encoder_hidden_states = module.to_add_out(encoder_hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
    if context_input_ndim == 4:
        encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    #print("hidden_states.shape: ", hidden_states.shape) #[2, 4096, 1536]
    #print("encoder_hidden_states.shape: ", encoder_hidden_states.shape)

    return hidden_states, encoder_hidden_states

    
def hook_forwards(self, root_module: torch.nn.Module):
    for name, module in root_module.named_modules():
        #print(name, module.__class__.__name__)
        if "attn" in name and module.__class__.__name__ == "Attention":
            #print(f"Attaching hook to {name}")
            module.forward = hook_forward(self, module)           
        
        #if module.__class__.__name__ == "JointTransformerBlock":
        #    print(f"Attaching hook to {name}")
        #    module.forward = hook_forward_joint_block(self, module)       



def hook_forward_joint_block(self, module):
    def forward(hidden_states, encoder_hidden_states, temb):
        print("**In hook_forward_joint_block()")

        # TODO:
        if self.fusion == False:
            self.block_hidden_states[self.cur_obj_idx].append(hidden_states)
        else:
            bbox_objs = {0: [350, 400, 200, 200] , 1: [700, 300, 200, 400]}

            height = 1024
            width = 1024

            latent_w = round(math.sqrt(hidden_states.size()[1]))
            latent_h = round(math.sqrt(hidden_states.size()[1]))

            scale = latent_w/height
            
            print("latent_h: ", latent_h)
            print("latent_w: ", latent_w)
            
            hidden_states_obj0 = self.block_hidden_states[0][self.cur_fusion_layer]
            hidden_states_obj1 = self.block_hidden_states[1][self.cur_fusion_layer]

            # reshape
            hidden_states = hidden_states.reshape((hidden_states.shape[0], latent_w, latent_h, hidden_states.shape[2]))
            hidden_states_obj0 = hidden_states_obj0.reshape((hidden_states_obj0.shape[0], latent_w, latent_h, hidden_states_obj0.shape[2]))
            hidden_states_obj1 = hidden_states_obj1.reshape((hidden_states_obj1.shape[0], latent_w, latent_h, hidden_states_obj1.shape[2]))


            bbox = bbox_objs[0]
            x1, y1, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            x2, y2 = x1+w, y1+h
            x1, y1, x2, y2 = int(x1*scale), int(y1*scale), int(x2*scale), int(y2*scale)
            print("x1, y1, x2, y2: ", x1, y1, x2, y2)

            w = 0
            hidden_states[:, x1:x2, y1:y2, :] = w*hidden_states[:, x1:x2, y1:y2, :] + (1-w)*hidden_states_obj0[:, x1:x2, y1:y2, :]

            bbox = bbox_objs[1]
            x1, y1, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            x2, y2 = x1+w, y1+h
            x1, y1, x2, y2 = int(x1*scale), int(y1*scale), int(x2*scale), int(y2*scale)
            print("x1, y1, x2, y2: ", x1, y1, x2, y2)

            w = 0
            hidden_states[:, x1:x2, y1:y2, :] = w*hidden_states[:, x1:x2, y1:y2, :] + (1-w)*hidden_states_obj1[:, x1:x2, y1:y2, :]

            hidden_states = hidden_states.reshape((hidden_states.shape[0], latent_w*latent_h, hidden_states.shape[3]))
            hidden_states_obj0 = hidden_states_obj0.reshape((hidden_states_obj0.shape[0], latent_w*latent_h, hidden_states_obj0.shape[3]))
            hidden_states_obj1 = hidden_states_obj1.reshape((hidden_states_obj1.shape[0], latent_w*latent_h, hidden_states_obj1.shape[3]))


        #print("temb.shape: ", temb.shape) #[2, 1536]

        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = module.norm1(hidden_states, emb=temb)
        
        if module.context_pre_only:
            norm_encoder_hidden_states = module.norm1_context(encoder_hidden_states, temb)
        else:
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = module.norm1_context(
                encoder_hidden_states, emb=temb
            )

        # Attention.
        attn_output, context_attn_output = module.attn(
            hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states
        )
        #print("gate_msa.unsqueeze(1).shape: ", gate_msa.unsqueeze(1).shape) # [2, 1, 1536]

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        print("hidden_states.shape: ", hidden_states.shape) # [2, 4096, 1536]

        norm_hidden_states = module.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None] #[2, 1, 1536]

        print("norm_hidden_states.shape: ", norm_hidden_states.shape) # [2, 4096, 1536]
        print("scale_mlp.shape: ", scale_mlp.shape) #[2, 1536]
        print("shift_mlp.shape: ", shift_mlp.shape) #[2, 1536]

        if module._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(module.ff, norm_hidden_states, module._chunk_dim, module._chunk_size)
        else:
            ff_output = module.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        if module.context_pre_only:
            encoder_hidden_states = None
        else:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = module.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            if module._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                context_ff_output = _chunked_feed_forward(
                    module.ff_context, norm_encoder_hidden_states, module._chunk_dim, module._chunk_size
                )
            else:
                context_ff_output = module.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return encoder_hidden_states, hidden_states

    return forward


def hook_forward(self, module):

    def bbox_to_xy(bbox, hidden_states, height, latent_h):
        x= hidden_states

        scale = latent_h/height

        # TODO: somehow, cross-attention layer is diagonally flipped
        x1, y1, w, h = bbox[1], bbox[0], bbox[3], bbox[2] 
        x2, y2 = x1+w, y1+h
        x1, y1, x2, y2 = int(x1*scale), int(y1*scale), int(x2*scale), int(y2*scale)
        #print("x1, y1, x2, y2: ", x1, y1, x2, y2)

        return x1, y1, x2, y2


    def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        #print("In hook_forward")
        x= hidden_states # SD3: [2, 4096, 1536] = [2, 64x64, 1536] = latent_model_input
        context= encoder_hidden_states # SD3: [2, 154, 1536] = prompt_embeds

        # Crop_Fusion
        height = weight = 1024
        latent_h = latent_w = round(math.sqrt(x.size()[1]))

        # OBJ: No fusion
        if self.fusion == False:

            # masked attention, no use
            if self.cur_step > self.stop_resize and self.cur_step < self.stop_local_attn and False:
                x_in = x.reshape(x.size()[0], latent_h, latent_w ,x.size()[2])
                x_zero = torch.zeros_like(x_in)

                x1, y1, x2, y2 = bbox_to_xy(self.bboxes[self.cur_obj_idx], x, height, latent_h)

                x_zero[:, x1:x2, y1:y2 ,:] = x_in[:, x1:x2, y1:y2 ,:]
                x_zero = x_zero.reshape([x.size()[0], -1 ,x.size()[2]])

                out_obj, context_attn_out = main_forward_diffusers_sd3(module, x_zero, context) # [2, 4096, 1536] = x.shape
                out_obj = out_obj.reshape(x.size()[0], latent_h, latent_w, x.size()[2])

            # local attention
            if self.cur_step > self.stop_resize and self.cur_step < self.stop_local_attn:
                x_in = x.reshape(x.size()[0], latent_h, latent_w ,x.size()[2])
                x1, y1, x2, y2 = bbox_to_xy(self.bboxes[self.cur_obj_idx], x, height, latent_h)

                x_bbox = x_in[:, x1:x2, y1:y2 ,:]
                x_bbox = x_bbox.reshape([x.size()[0], -1 ,x.size()[2]])

                out_bbox, context_attn_out = main_forward_diffusers_sd3(module, x_bbox, context) # [2, 4096, 1536] = x.shape
                out_bbox = out_bbox.reshape(x.size()[0], x2-x1, y2-y1, x.size()[2])

                out_obj = x_in.clone()
                out_obj[:, x1:x2, y1:y2 ,:] = out_bbox

            # global attention
            else:
                out_obj, context_attn_out = main_forward_diffusers_sd3(module, x, context) # [2, 4096, 1536] = x.shape
                out_obj = out_obj.reshape(x.size()[0], latent_h, latent_w, x.size()[2])

            ### saving per-layer attention map of each object
            self.attention_maps[self.cur_obj_idx].append(out_obj)

            out_x = out_obj

        # Base: Regional Feature Fusion 
        else:
            out_base, context_attn_out = main_forward_diffusers_sd3(module, x, context)
            out_base = out_base.reshape(x.size()[0],latent_h,latent_w,x.size()[2])
            
            # FIXME:
            out_obj0 = self.attention_maps[0][self.cur_fusion_layer]
            out_obj1 = self.attention_maps[1][self.cur_fusion_layer]

            # TODO: attention map visualization
            if False and self.cur_step == self.stop_resize-1 and self.cur_fusion_layer == len(self.attention_maps[0])-1:
                import matplotlib.pyplot as plt
                import numpy as np
                
                map_type = 'avg'
                if map_type == 'avg':
                    avg_map_0 = torch.zeros_like(self.attention_maps[0][0])
                    for attn_map in self.attention_maps[0]:
                        avg_map_0 += attn_map
                    avg_map_0 = avg_map_0/len(self.attention_maps[0])
                    avg_map_0 = avg_map_0.mean(dim=3).mean(dim=0)

                    plt.imshow(avg_map_0.detach().cpu(), cmap='hot', interpolation='nearest')
                    plt.savefig(f'/home/jovyan/r2f/attention_maps/{self.decomposed_prompts[0][1]}_avg_map_step{self.cur_step}.png')

                    avg_map_1 = torch.zeros_like(self.attention_maps[1][0])
                    for attn_map in self.attention_maps[1]:
                        avg_map_1 += attn_map
                    avg_map_1 = avg_map_1/len(self.attention_maps[1])
                    avg_map_1 = avg_map_1.mean(dim=3).mean(dim=0)

                    plt.imshow(avg_map_1.detach().cpu(), cmap='hot', interpolation='nearest')
                    plt.savefig(f'/home/jovyan/r2f/attention_maps/{self.decomposed_prompts[1][1]}_avg_map_step{self.cur_step}.png')
                
                elif map_type == 'last':
                    last_map_0 = torch.zeros_like(self.attention_maps[0][-1])
                    last_map_0 = last_map_0.mean(dim=3).mean(dim=0)

                    plt.imshow(last_map_0.detach().cpu(), cmap='hot', interpolation='nearest')
                    plt.savefig(f'/home/jovyan/r2f/attention_maps/{self.decomposed_prompts[0][0]}_last_map_step{self.cur_step}.png')

                    last_map_1 = torch.zeros_like(self.attention_maps[1][-1])
                    last_map_1 = last_map_1.mean(dim=3).mean(dim=0)

                    plt.imshow(last_map_1.detach().cpu(), cmap='hot', interpolation='nearest')
                    plt.savefig(f'/home/jovyan/r2f/attention_maps/{self.decomposed_prompts[1][0]}_last_map_step{self.cur_step}.png')

            # Resize
            w = self.w
            if self.cur_step == self.stop_resize-1: # FIXME: 0
                print("RESIZE!!!!")
                x1, y1, x2, y2 = bbox_to_xy(self.bboxes[0], x, height, latent_h)
                out_obj0 = out_obj0.transpose(1,2).transpose(1,3)
                out_obj0 = F.interpolate(out_obj0, size=(x2-x1, y2-y1), mode='bilinear')
                out_obj0 = out_obj0.transpose(2,3).transpose(1,3)
                out_base[:, x1:x2, y1:y2, :] = w*out_base[:, x1:x2, y1:y2, :] + (1-w)*out_obj0

                x1, y1, x2, y2 = bbox_to_xy(self.bboxes[1], x, height, latent_h)
                out_obj1 = out_obj1.transpose(1,2).transpose(1,3)
                out_obj1 = F.interpolate(out_obj1, size=(x2-x1, y2-y1), mode='bilinear')
                out_obj1 = out_obj1.transpose(2,3).transpose(1,3)
                out_base[:, x1:x2, y1:y2, :] = w*out_base[:, x1:x2, y1:y2, :] + (1-w)*out_obj1
                
                out_x = out_base
            # Crop
            else: 
                x1, y1, x2, y2 = bbox_to_xy(self.bboxes[0], x, height, latent_h)
                out_base[:, x1:x2, y1:y2, :] = w*out_base[:, x1:x2, y1:y2, :] + (1-w)*out_obj0[:, x1:x2, y1:y2, :]

                x1, y1, x2, y2 = bbox_to_xy(self.bboxes[1], x, height, latent_h)
                out_base[:, x1:x2, y1:y2, :] = w*out_base[:, x1:x2, y1:y2, :] + (1-w)*out_obj1[:, x1:x2, y1:y2, :]
                
                out_x = out_base

            self.cur_fusion_layer += 1

        out_x = out_x.reshape(x.size()[0], x.size()[1], x.size()[2]) # Restore to 3d source. #[2, 4096, 1536] 
            
        return out_x, context_attn_out

    return forward




def hook_forward_backup_no_resize(self, module):

    def bbox_to_xy(bbox, hidden_states, height, latent_h):
        x= hidden_states

        scale = latent_h/height

        x1, y1, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        x2, y2 = x1+w, y1+h
        x1, y1, x2, y2 = int(x1*scale), int(y1*scale), int(x2*scale), int(y2*scale)
        print("x1, y1, x2, y2: ", x1, y1, x2, y2)

        return x1, y1, x2, y2


    def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        print("In hook_forward")

        x= hidden_states # SD3: [2, 4096, 1536] = [2, 64x64, 1536] = latent_model_input
        context= encoder_hidden_states # SD3: [2, 154, 1536] = prompt_embeds

        height = 1024
        width = 1024

        latent_h = round(math.sqrt(x.size()[1]))
        latent_w = round(math.sqrt(x.size()[1]))

        if self.fusion == False:
            print("Fusion is False")
            print("obj_idx: ", self.cur_obj_idx)

            # TODO: Crop X before attention
            if self.cur_step < 5:
                x_in = x.reshape(x.size()[0], latent_h, latent_w ,x.size()[2])
                x1, y1, x2, y2 = bbox_to_xy(self.bbox_objs[self.cur_obj_idx], x, height, latent_h)

                x_bbox = x_in[:, x1:x2, y1:y2 ,:]
                x_bbox = x_bbox.reshape([x.size()[0], -1 ,x.size()[2]])

                out_bbox, context_attn_out = main_forward_diffusers_sd3(module, x_bbox, context) # [2, 4096, 1536] = x.shape
                out_bbox = out_bbox.reshape(x.size()[0], x2-x1, y2-y1, x.size()[2])

                out_obj = x_in.clone()
                out_obj[:, x1:x2, y1:y2 ,:] = out_bbox
            # Normal
            else:
                out_obj, context_attn_out = main_forward_diffusers_sd3(module, x, context) # [2, 4096, 1536] = x.shape
                out_obj = out_obj.reshape(x.size()[0], latent_h, latent_w, x.size()[2])

            ### saving per-layer attention map of each object
            self.attention_maps[self.cur_obj_idx].append(out_obj)

            out_x = out_obj
        else:
            print("Fusion is True")
            print("self.cur_fusion_layer: ", self.cur_fusion_layer)
            
            # TODO: Regional Feature Fusion !!   # TODO: Resize!!!!!!!
            out_base, context_attn_out = main_forward_diffusers_sd3(module, x, context)
            out_base = out_base.reshape(x.size()[0],latent_h,latent_w,x.size()[2])
            
            out_obj0 = self.attention_maps[0][self.cur_fusion_layer]
            out_obj1 = self.attention_maps[1][self.cur_fusion_layer]

            print("**attention shapes: ", out_base.shape, out_obj0.shape, out_obj1.shape)

            # TODO:
            if self.cur_fusion_layer == len(self.attention_maps[0])-1 and False:
                avg_map_0 = torch.zeros_like(self.attention_maps[0][0])
                for attn_map in self.attention_maps[0]:
                    avg_map_0 += attn_map
                
                avg_map_0 = avg_map_0/len(self.attention_maps[0])

                import matplotlib.pyplot as plt
                import numpy as np
                avg_map_0 = avg_map_0.mean(dim=3).mean(dim=0)
                plt.imshow(avg_map_0.detach().cpu(), cmap='hot', interpolation='nearest')
                plt.savefig(f'attention_maps/avg_map0_step{self.diffusion_step}.png')


            # Fusion
            w = 0.5 #min(self.cur_step/5, 1) # 0~30: linear up, 30~50: w=1

            x1, y1, x2, y2 = bbox_to_xy(self.bbox_objs[0], x, height, latent_h)
            out_base[:, x1:x2, y1:y2, :] = w*out_base[:, x1:x2, y1:y2, :] + (1-w)*out_obj0[:, x1:x2, y1:y2, :]

            x1, y1, x2, y2 = bbox_to_xy(self.bbox_objs[1], x, height, latent_h)
            out_base[:, x1:x2, y1:y2, :] = w*out_base[:, x1:x2, y1:y2, :] + (1-w)*out_obj1[:, x1:x2, y1:y2, :]
            
            out_x = out_base

            self.cur_fusion_layer += 1

        out_x = out_x.reshape(x.size()[0], x.size()[1], x.size()[2]) # Restore to 3d source. #[2, 4096, 1536] 
            
        return out_x, context_attn_out

    return forward


def hook_forward_backup(self, module):
    def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        x= hidden_states # SD3: [2, 4096, 1536] = [2, 64x64, 1536] = latent_model_input
        context= encoder_hidden_states # SD3: [2, 154, 1536] = prompt_embeds
        
        print("In hook_forward")

        print("input : ", hidden_states.size())
        print("tokens : ", context.size())

        if self.fusion == False:
            print("Fusion is False")
            print(self.cur_obj_idx)

            out_obj, context_attn_out = main_forward_diffusers_sd3(module, x, context) 
            print("out_obj.shape: ", out_obj.shape) # [2, 4096, 1536] = x.shape

            ### caching attention maps of the decomposed prompt for each object
            self.attention_maps[self.cur_obj_idx].append(out_obj)
            out_x = out_obj
        else:
            print("Fusion is True")
            print("self.cur_fusion_layer: ", self.cur_fusion_layer)

            # TODO: input reformatting
            bbox_objs = {0: [350, 400, 200, 200] , 1: [700, 300, 200, 400]}

            height = 1024
            width = 1024

            latent_w = round(math.sqrt(x.size()[1]))
            latent_h = round(math.sqrt(x.size()[1]))

            scale = latent_w/height
            
            print("latent_h: ", latent_h)
            print("latent_w: ", latent_w)
            
            # TODO: Regional Feature Fusion !!   # TODO: Resize!!!!!!!
            out_base, context_attn_out = main_forward_diffusers_sd3(module, x, context)
            
            out_obj0 = self.attention_maps[0][self.cur_fusion_layer]
            out_obj1 = self.attention_maps[1][self.cur_fusion_layer]

            # reshape to (latent_h x letent_w)
            out_base = out_base.reshape(x.size()[0],latent_h,latent_w,x.size()[2])
            out_obj0 = out_obj0.reshape(x.size()[0],latent_h,latent_w,x.size()[2]) 
            out_obj1 = out_obj1.reshape(x.size()[0],latent_h,latent_w,x.size()[2]) 

            print("**attention shapes: ", out_base.shape, out_obj0.shape, out_obj1.shape)

            import matplotlib.pyplot as plt
            import numpy as np
            map_0 = out_obj0.mean(dim=3).mean(dim=0)
            plt.imshow(map_0.detach().cpu(), cmap='hot', interpolation='nearest')
            plt.savefig(f'attention_maps/map0_step{self.diffusion_step}_layer{self.cur_fusion_layer}.png')


            bbox = bbox_objs[0]
            x1, y1, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            x2, y2 = x1+w, y1+h
            x1, y1, x2, y2 = int(x1*scale), int(y1*scale), int(x2*scale), int(y2*scale)
            print("x1, y1, x2, y2: ", x1, y1, x2, y2)

            w = 0
            out_base[:, x1:x2, y1:y2, :] = w*out_base[:, x1:x2, y1:y2, :] + (1-w)*out_obj0[:, x1:x2, y1:y2, :]

            bbox = bbox_objs[1]
            x1, y1, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            x2, y2 = x1+w, y1+h
            x1, y1, x2, y2 = int(x1*scale), int(y1*scale), int(x2*scale), int(y2*scale)
            print("x1, y1, x2, y2: ", x1, y1, x2, y2)

            w = 0
            out_base[:, x1:x2, y1:y2, :] = w*out_base[:, x1:x2, y1:y2, :] + (1-w)*out_obj1[:, x1:x2, y1:y2, :]
            
            out_base = out_base.reshape(x.size()[0],x.size()[1],x.size()[2]) # Restore to 3d source. #[2, 4096, 1536] 
            out_x = out_base

            self.cur_fusion_layer += 1
        
        return out_x, context_attn_out

    return forward



def split_dims(x_t, height, width, self=None):
    """Split an attention layer dimension to height + width.
    The original estimate was latent_h = sqrt(hw_ratio*x_t),
    rounding to the nearest value. However, this proved inaccurate.
    The actual operation seems to be as follows:
    - Divide h,w by 8, rounding DOWN.
    - For every new layer (of 4), divide both by 2 and round UP (then back up).
    - Multiply h*w to yield x_t.
    There is no inverse function to this set of operations,
    so instead we mimic them without the multiplication part using the original h+w.
    It's worth noting that no known checkpoints follow a different system of layering,
    but it's theoretically possible. Please report if encountered.
    """
    scale = math.ceil(math.log2(math.sqrt(height * width / x_t)))
    latent_h = repeat_div(height, scale)
    latent_w = repeat_div(width, scale)
    if x_t > latent_h * latent_w and hasattr(self, "nei_multi"):
        latent_h, latent_w = self.nei_multi[1], self.nei_multi[0] 
        while latent_h * latent_w != x_t:
            latent_h, latent_w = latent_h // 2, latent_w // 2

    return latent_h, latent_w

def repeat_div(x,y):
    """Imitates dimension halving common in convolution operations.
    
    This is a pretty big assumption of the model,
    but then if some model doesn't work like that it will be easy to spot.
    """
    while y > 0:
        x = math.ceil(x / 2)
        y = y - 1
    return x