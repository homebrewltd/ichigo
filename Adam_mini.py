import torch
from torch.optim.optimizer import Optimizer
import math
import torch.distributed as dist
from torch.optim.optimizer import _dispatch_sqrt

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Adam_mini(Optimizer):
    def __init__(
        self,
        model=None,
        weight_decay=0.1,
        lr=1,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        zero_3=False,
        n_embd = 2048,
        n_head = 32,
        n_query_groups = None
    ):
        '''
        model: the model you are training.

        zero_3: set to True if you are using zero_3 in Deepspeed, or if you are using model parallelism with more than 1 GPU. Set to False if otherwise.
        
        n_embd: number of embedding dimensions. Could be unspecified if you are training non-transformer models.
        
        n_head: number of attention heads. Could be unspecified if you are training non-transformer models.
        
        n_query_groups: number of query groups in Group query Attention. If not specified, it will be equal to n_head. Could be unspecified if you are training non-transformer models.
        '''
       

        self.n_embd = n_embd
        self.n_head = n_head
        if n_query_groups is not None:
            self.n_query_groups = n_query_groups
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head

        self.model = model
        self.world_size = torch.cuda.device_count()
        self.zero_optimization_stage = 0
        if zero_3:
            self.zero_optimization_stage = 3
            print("Adam-mini is using zero_3")
        optim_groups = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                dic = {}
                dic["name"] = name
                dic["params"] = param
                if ("norm" in name or "ln_f" in name):
                    dic["weight_decay"] = 0
                else:
                    dic["weight_decay"] = weight_decay
                
                if ("self_attn.k_proj.weight" in name or "self_attn.q_proj.weight" in name):
                    dic["parameter_per_head"] = self.n_embd * self.n_embd // self.n_head
                
                if ("attn.attn.weight" in name or "attn.qkv.weight" in name):
                    dic["n_head"] = self.n_head
                    dic["q_per_kv"] = self.n_head // self.n_query_groups

                optim_groups.append(dic)

        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)

        super(Adam_mini, self).__init__(optim_groups, defaults)


    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        with torch.no_grad():
            for group in self.param_groups:
                beta1 = group["beta1"]
                beta2 = group["beta2"]
                lr = group["lr"]
                name = group["name"]
                epsilon = group["epsilon"]
                
                for p in group["params"]:
                    state = self.state[p]
                    if ("embed_tokens" in name or "wte" in name or "lm_head" in name):
                        if p.grad is None:
                            continue
                        if len(state) == 0:
                            state["m"] = torch.zeros_like(p.data).to(torch.float32)
                            state["iteration"] = 0
                            state["v"] = torch.zeros_like(p.data).to(torch.float32)

                        grad = p.grad.data.to(torch.float32)
                        state["v"].mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
                        state["iteration"] += 1
                        if group["weight_decay"] != 0:
                            p.data.mul_(1 - lr * group["weight_decay"])

                        state["m"].lerp_(grad, 1 - beta1)

                        bias_correction_1 = 1 - beta1 ** state["iteration"]
                        bias_correction_2 = 1 - beta2 ** state["iteration"]
                        bias_correction_2_sqrt = math.sqrt(bias_correction_2)

                        h = (state["v"].sqrt() / bias_correction_2_sqrt).add_(epsilon)
                        stepsize = lr/ bias_correction_1
                        p.addcdiv_(state["m"], h, value=-stepsize)

                    elif ("self_attn.k_proj.weight" in name or "self_attn.q_proj.weight" in name or "attn.wq.weight" in name or "attn.wk.weight" in name):
                        if p.grad is None:
                            continue
                        dim = group["parameter_per_head"]
                        if (len(state)==0):
                            state["m"]  =  torch.zeros_like(p.data).to(torch.float32)
                            state["m"] = state["m"].view(-1, dim)
                            state['head'] = state['m'].shape[0]
                            state["iteration"] = 0
                            state["vmean"] = torch.zeros(state['head']).to(device)

                        grad = p.grad.data.to(torch.float32)
                        head = state['head']
                        grad = grad.view(head, dim)

                        tmp_lr = torch.mean(grad*grad, dim = 1).to(device)
                        state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                        v = state["vmean"]

                        state["iteration"] += 1
                        if group["weight_decay"] != 0:
                            p.data.mul_(1 - lr * group["weight_decay"])

                        state["m"].lerp_(grad, 1 - beta1)

                        bias_correction_1 = 1 - beta1 ** state["iteration"]
                        bias_correction_2 = 1 - beta2 ** state["iteration"]
                        bias_correction_2_sqrt = math.sqrt(bias_correction_2)

                        h = (v.sqrt() / bias_correction_2_sqrt).add_(epsilon)    
                        stepsize = ((1/bias_correction_1) / h).view(head,1)

                        update = state["m"] * (stepsize.to(state['m'].device))

                        if p.dim() > 1:
                            d0, d1 = p.size()
                            update = update.view(d0, d1)
                        else: 
                            update = update.view(-1)

                        update.mul_(lr)
                        p.add_(-update)
                        
                    elif ("attn.attn.weight" in name or "attn.qkv.weight" in name): 
                        if p.grad is None:
                            continue
                        if (len(state)==0):
                            state["m"]  =  torch.zeros_like(p.data).to(torch.float32)
                            state["m"] = state["m"].view(group["n_head"], group["q_per_kv"] + 2, -1)
                            state["iteration"] = 0
                            state["vmean"] = torch.zeros(group["n_head"], group["q_per_kv"]+2).to(device)
                            

                        grad = p.grad.data.to(torch.float32)
                        grad = grad.view(group["n_head"], group["q_per_kv"] + 2, -1) 

                        tmp_lr = torch.mean(grad*grad, dim = 2).to(device)
                        state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                        v = state["vmean"]
                
                        state["iteration"] += 1
                        if group["weight_decay"] != 0:
                            p.data.mul_(1 - lr * group["weight_decay"])

                        state["m"].lerp_(grad, 1 - beta1)


                        bias_correction_1 = 1 - beta1 ** state["iteration"]
                        bias_correction_2 = 1 - beta2 ** state["iteration"]
                        bias_correction_2_sqrt = math.sqrt(bias_correction_2)

                        h = (v.sqrt() / bias_correction_2_sqrt).add_(epsilon)    
                        stepsize = ((1/bias_correction_1) / h).view(group["n_head"],group["q_per_kv"]+2,1)
                                                    
   
                        update = state["m"] * (stepsize.to(state['m'].device))
            
                        if p.dim() > 1:
                            d0, d1 = p.size()
                            update = update.view(d0, d1)
                        else: 
                            update = update.view(-1)
        
                        update.mul_(lr)
                        p.add_(-update)

                        
                    else:        
                        if (len(state)==0):                   
                            dimension = torch.tensor(p.data.numel()).to(device).to(torch.float32)
                            reduced = False
                            if (self.world_size > 1) and (self.zero_optimization_stage == 3):
                                tensor_list = [torch.zeros_like(dimension) for _ in range(self.world_size)]
                                dist.all_gather(tensor_list, dimension)
                                s = 0
                                dimension = 0
                                for d in tensor_list:
                                    if (d>0):
                                        s = s + 1
                                    dimension = dimension + d
                                if (s>=2):
                                    reduced = True
                            
                            state["m"] = torch.zeros_like(p.data).to(torch.float32)
                            state["iteration"] = 0
                            state["reduced"] = reduced
                            state["vmean"] = torch.tensor(0.0).to(device)                                
                            state["dimension"] = dimension.item()
                        if p.grad is None:
                            tmp_lr = torch.tensor(0.0).to(device)
                        else:
                            grad = p.grad.data.to(torch.float32)
                            tmp_lr = torch.sum(grad*grad).to(device)                              
                        if (state["reduced"]):
                            dist.all_reduce(tmp_lr, op=dist.ReduceOp.SUM)
                        if (p.grad is None):
                            continue
                        tmp_lr = tmp_lr / (state["dimension"])
                        tmp_lr = tmp_lr.to(grad.device)
                        
                        if group["weight_decay"] != 0:
                            p.data.mul_(1 - lr * group["weight_decay"])
                        state["iteration"] += 1
                        state["m"].lerp_(grad, 1 - beta1)

                        bias_correction_1 = 1 - beta1 ** state["iteration"]
                        bias_correction_2 = 1 - beta2 ** state["iteration"]
                        bias_correction_2_sqrt = math.sqrt(bias_correction_2)
                        state["vmean"] = (1 - beta2) * tmp_lr + beta2 * state["vmean"]
                        h = (state["vmean"].sqrt() / bias_correction_2_sqrt).add_(epsilon)    

                        stepsize = (1 / bias_correction_1) / h
                        update = state["m"] * (stepsize.to(state['m'].device))
                        update.mul_(lr)
                        p.add_(-update)    
                    
     