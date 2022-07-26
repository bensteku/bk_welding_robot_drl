import torch

# class for Pytorch model of the agent
# inspired by the architecture of MeshCNN

class AgentModel:

    def __init__(self):
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')

        self.net = None
        self.criterion = None
        self.welding_mesh = None
        if self.train:
            self.optimizer = torch.optim.Adam()  # TODO
        

    def forward(self, ee_pos, ee_rot, base_pos, joints, weld_seam, weld_seam_normals, robot_state):
        return self.net(self.welding_mesh, ee_pos, ee_rot, base_pos, joints, weld_seam, weld_seam_normals, robot_state)

    def backward(self, out):
        self.loss = self.criterion()  # TODO
        self.loss.backward()

    def optimize(self):
        self.optimizer.zero_grad()
        out = self.forward()
        self.backward(out)
        self.optimizer.step()
    