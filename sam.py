import torch

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group.get("adaptive", False) else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "old_p" not in self.state[p]: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
                
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        
        # First forward-backward pass to compute gradients
        loss = closure()  # This computes gradients
        
        # First step (perturb weights)
        self.first_step(zero_grad=True)
        
        # Second forward-backward pass with perturbed weights
        closure()
        
        # Second step (restore weights and take optimizer step)
        self.second_step()
        
        return loss
    
    def _grad_norm(self):
        # Safety check
        grad_list = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    grad_list.append(((torch.abs(p) if group.get("adaptive", False) else 1.0) * p.grad).norm(p=2))
        
        # Check if list is empty
        if not grad_list:
            return torch.tensor(0.0).to(self.param_groups[0]["params"][0].device)
            
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
                    torch.stack([g.to(shared_device) for g in grad_list]),
                    p=2
               )
        return norm