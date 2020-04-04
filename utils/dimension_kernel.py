import torch




class Trigonometric_kernel:
    def __init__(self, L = 10):

        self.L = L
        freq_bands = torch.linspace(2.**0., 2.**(L-1), L) 
        periodic_fns = [torch.sin, torch.cos]


        embed_fns = []
        for freq in freq_bands:
            for p_fn in periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                
        self.multipler = len(freq_bands)*len(periodic_fns)
        self.embed_fns = embed_fns

    '''
     x: input vectors (N,C) 

     OUPUT

     pos_kernel: (N, calc_dim(C) )
    '''
    def __call__(self, x):
        pos_kernel = torch.stack([fn(x) for fn in self.embed_fns],dim = -1)
        pos_kernel = torch.flatten(pos_kernel,start_dim=1)
        pos_kernel = torch.cat([x, pos_kernel], dim = 1)

        return pos_kernel

    def calc_dim(self, dims):
        return self.multipler*dims + dims