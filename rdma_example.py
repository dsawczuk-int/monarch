import torch
from monarch.actor import Actor, endpoint, this_host
from monarch.rdma import RDMABuffer

class ParameterServer(Actor):
    def __init__(self):
        self.weights = torch.rand(1000, 1000)  # Large model weights

        # RDMABuffer does not copy the data. It just
        # creates a view of the data that can be passed to
        # other processes.
        self.weight_buffer = RDMABuffer(self.weights.view(torch.uint8).flatten())

    @endpoint
    def get_weights(self) -> RDMABuffer:
        return self.weight_buffer  # Small message: just a reference!


class Worker(Actor):
    def __init__(self):
        self.local_weights = torch.zeros(1000, 1000)

    @endpoint
    def sync_weights(self, server: ParameterServer):
        # Control plane: get lightweight reference
        weight_ref = server.get_weights.call_one().get()

        # Data plane: explicit bulk transfer when needed
        weight_ref.read_into(self.local_weights.view(torch.uint8).flatten()).get()


server_proc = this_host().spawn_procs(per_host={"gpus": 1})
worker_proc = this_host().spawn_procs(per_host={"gpus": 1})

server = server_proc.spawn("server", ParameterServer)
worker = worker_proc.spawn("worker", Worker)

worker.sync_weights.call_one(server).get()