class Arguments:
    def __init__(self, task = 'MNIST'):

        self.device     = 'cpu'        
        self.clr        = 0.005
        self.qlr        = 0.01

        self.allowed_gates = ['Identity', 'RX', 'RY', 'RZ', 'C(U3)']

        self.n_qubits   = 5 
        
        self.epochs     = 1
        self.batch_size = 256 
        self.sampling = 5
        self.kernel      = 4

        self.n_layers = 4
        self.base_code = [self.n_layers, 2, 3, 4, 1]
        self.exploration = [0.001, 0.002, 0.003]

        self.backend    = 'tq'            
        self.digits_of_interest = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.file_single = 'search_space/search_space_mnist_half_single'
        self.file_enta   = 'search_space/search_space_mnist_half_enta'
        self.fold        = 2
        self.init_weight = 'init_weight_MNIST_10'

        
            
